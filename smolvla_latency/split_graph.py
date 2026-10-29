"""Split SmolVLA (LeRobot 0.6.1) into three ONNX-exportable graphs, with the flow-matching loop outside them.

    vision       export_vision.VisionEncoder: pixel_values [1, 3, H, W] -> image_tokens [1, T, D]
    prefix       PrefixGraph:     image tokens per camera, language tokens and mask padded to a fixed length,
                                  padded state -> prefix pad mask and the post-RoPE K/V of every VLM layer
    expert_step  ExpertStepGraph: x_t, t, prefix pad mask, K/V -> velocity v_t

The caller runs ``x <- x + dt * v_t`` for ``num_steps`` steps, as LeRobot's ``euler_integrate`` does.
docs/SPLIT_GRAPH.md is the exact contract.

The wrappers call LeRobot's own ``embed_prefix``, ``SmolVLMWithExpertModel.forward`` and ``denoise_step``, so
the expert's self-attention and cross-attention layers run LeRobot's code, not a copy of it. ``PrefixGraph``
takes image tokens, so it runs ``embed_prefix`` with ``embed_image`` as the identity. Beyond that, three things
are swapped while a wrapper runs; tests/test_split_graph.py checks each against LeRobot:

* Denoising reads the prefix through ``FunctionalKVCache`` instead of ``DynamicCache``, so an expert step
  never mutates its inputs.
* The timestep embedding is computed in float32. LeRobot computes it in float64, which TensorRT does not
  support, and a plain float32 version is off by up to 2e-4 because the phases reach ~1600 rad.
* ``apply_rope`` takes its timescale table from a buffer computed by torch. Otherwise the exporter
  constant-folds the ``pow`` with numpy, which is one ulp off for some entries and moves the rotation angles
  at late positions by up to ~1e-5 rad. It also joins the two halves with ``cat`` instead of slice
  assignment, which would export as ScatterND. In PyTorch it is bit-identical to LeRobot's.
"""

import inspect
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla import modeling_smolvla, smolvlm_with_expert
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch import Tensor, nn

from .loading import synthetic_observation

ROPE_MAX_WAVELENGTH = 10_000  # apply_rope's default; SmolVLMWithExpertModel never passes another value


@dataclass
class CacheLayer:
    keys: Tensor  # [batch, kv_heads, seq, head_dim], the DynamicCache layout
    values: Tensor


class FunctionalKVCache:
    """The part of transformers' DynamicCache that SmolVLMWithExpertModel uses while denoising, without state.

    Self-attention expert layers call ``update``, which returns ``[prefix; new]`` along the sequence axis and
    stores nothing. Cross-attention layers read ``layers[i].keys`` and ``.values``. LeRobot's ``crop`` after
    each step then has nothing to undo, so an expert step is a pure function of its inputs.
    """

    def __init__(self, keys: list[Tensor], values: list[Tensor]):
        self.layers = [CacheLayer(k, v) for k, v in zip(keys, values, strict=True)]

    def __len__(self) -> int:
        # forward_cross_attn_layer checks `not past_key_values`; a filled cache has to be truthy.
        return len(self.layers)

    def update(self, key_states: Tensor, value_states: Tensor, layer_idx: int, *args, **kwargs):
        layer = self.layers[layer_idx]
        return torch.cat([layer.keys, key_states], dim=-2), torch.cat([layer.values, value_states], dim=-2)

    def crop(self, max_length: int) -> None:
        pass  # update() never appended anything


@contextmanager
def patched(owner, name: str, value):
    """Replace ``owner.name`` (a module global or an instance attribute) for the duration of the block."""
    if not hasattr(owner, name):
        raise AttributeError(f"{owner!r} has no attribute {name!r}: LeRobot changed, update split_graph.py")
    own = vars(owner)
    had, saved = name in own, own.get(name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if had:
            setattr(owner, name, saved)
        else:
            delattr(owner, name)


class _Rope(nn.Module):
    """LeRobot's ``apply_rope`` with a precomputed timescale and the halves joined by ``cat``."""

    def __init__(self, head_dim: int):
        super().__init__()
        exponents = (2.0 / head_dim) * torch.arange(head_dim // 2, dtype=torch.float32)
        self.register_buffer("timescale", ROPE_MAX_WAVELENGTH**exponents, persistent=False)

    def forward(self, x: Tensor, positions: Tensor, max_wavelength: int = ROPE_MAX_WAVELENGTH) -> Tensor:
        d_half = self.timescale.shape[0]
        if x.shape[-1] != 2 * d_half or max_wavelength != ROPE_MAX_WAVELENGTH:
            raise ValueError(f"RoPE table built for head_dim {2 * d_half}, wavelength {ROPE_MAX_WAVELENGTH}")
        dtype = x.dtype
        x = x.to(torch.float32)
        radians = positions[..., None].to(torch.float32) / self.timescale[None, None, :]
        radians = radians[..., None, :]
        sin, cos = torch.sin(radians), torch.cos(radians)
        x1, x2 = x.split(d_half, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class _TimeEmbedding(nn.Module):
    """float32 stand-in for ``create_sinusoidal_pos_embedding`` (which works in float64), for t in [0, 1].

    The embedding is sin and cos of ``2 pi t / period`` with periods down to ``min_period``, so the phase
    reaches ~1600 rad, where one float32 ulp is ~1e-4 rad. Instead the phase is reduced modulo one period
    exactly: the frequency ``1 / period`` is split into ``hi`` (11 significant bits) and ``lo``, and t into
    ``t_hi`` (a multiple of 2^-12) and ``t_lo``. ``hi * t_hi`` is then exact in float32, so is its fractional
    part, and the other terms are below 0.2 cycles. The result is within ~5e-7 of LeRobot's.
    """

    T_HI_STEP = 2.0**-12

    def __init__(self, dimension: int, min_period: float, max_period: float):
        super().__init__()
        if dimension % 2:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")
        self.spec = (dimension, min_period, max_period)
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64)
        freq = 1.0 / (min_period * (max_period / min_period) ** fraction)
        mantissa, exponent = torch.frexp(freq)
        hi = torch.ldexp(torch.round(torch.ldexp(mantissa, torch.tensor(11))), exponent - 11)
        self.register_buffer("freq_hi", hi.float(), persistent=False)
        self.register_buffer("freq_lo", (freq - hi).float(), persistent=False)
        self.register_buffer("freq", freq.float(), persistent=False)

    def forward(self, time: Tensor, dimension: int, min_period: float, max_period: float, device=None) -> Tensor:
        if (dimension, min_period, max_period) != self.spec:
            raise ValueError(f"time embedding built for {self.spec}, called with "
                             f"{(dimension, min_period, max_period)}")
        if time.ndim != 1:
            raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
        t = time.to(torch.float32)[:, None]
        t_hi = torch.floor(t / self.T_HI_STEP) * self.T_HI_STEP
        t_lo = t - t_hi
        cycles = self.freq_hi * t_hi
        cycles = cycles - torch.floor(cycles)
        cycles = cycles + (self.freq_lo * t_hi + self.freq * t_lo)
        angle = (cycles - torch.round(cycles)) * (2 * math.pi)
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)


def _check_supported(cfg, model=None) -> None:
    """Raise NotImplementedError naming each setting of ``cfg`` the split graphs do not reproduce.

    ``model`` (the policy's VLAFlowMatching) adds the checks that need the built model: the number of expert
    layers depends on how many layers the VLM has.
    """
    unsupported = {
        "use_cache=False": not cfg.use_cache,
        "empty_cameras > 0": cfg.empty_cameras > 0,
        "adapt_to_pi_aloha": cfg.adapt_to_pi_aloha,
        "attention_mode without cross attention": "cross" not in cfg.attention_mode,
        "rtc_config enabled": cfg.rtc_config is not None and cfg.rtc_config.enabled,
        "add_image_special_tokens=True (image start and end tokens in the prefix)": cfg.add_image_special_tokens,
        f"prefix_length={cfg.prefix_length} (prefix padded to a fixed length)": cfg.prefix_length > 0,
    }
    if model is not None:
        vwe = model.vlm_with_expert
        expert = f"an action expert with {vwe.num_expert_layers} layers for {vwe.num_vlm_layers} VLM layers"
        unsupported[f"{expert} (num_expert_layers={cfg.num_expert_layers})"] = (
            vwe.num_expert_layers != vwe.num_vlm_layers)
    found = [k for k, v in unsupported.items() if v]
    if found:
        raise NotImplementedError(f"split-graph export does not handle {'; '.join(found)}")


def _identity_image_embedding(image_tokens: Tensor) -> Tensor:
    return image_tokens


class PrefixGraph(nn.Module):
    """Image tokens, language and state -> prefix pad mask and per-layer K/V, as prefill leaves them.

    Inputs are the raw ``embed_image`` output per camera (``embed_prefix`` applies the sqrt(D) scaling inside
    this graph), ``lang_tokens``/``lang_mask`` padded to ``tokenizer_max_length`` and the state padded to
    ``max_state_dim``. Every camera counts as present.
    """

    def __init__(self, policy: SmolVLAPolicy):
        super().__init__()
        _check_supported(policy.config, policy.model)
        self.model = policy.model
        self.num_cameras = len(policy.config.image_features)
        self.rope = _Rope(self.model.vlm_with_expert.config.text_config.head_dim)

    def forward(self, image_tokens: tuple[Tensor, ...], lang_tokens: Tensor, lang_mask: Tensor, state: Tensor):
        if len(image_tokens) != self.num_cameras:
            raise ValueError(f"expected image tokens for {self.num_cameras} cameras, got {len(image_tokens)}")
        vwe = self.model.vlm_with_expert
        img_masks = [torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device) for tokens in image_tokens]
        with (
            patched(vwe, "embed_image", _identity_image_embedding),
            patched(smolvlm_with_expert, "apply_rope", self.rope),
        ):
            # From here to the cache this is VLAFlowMatching.sample_actions up to the denoising loop.
            embs, pad_masks, att_masks = self.model.embed_prefix(
                list(image_tokens), img_masks, lang_tokens, lang_mask, state=state
            )
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            _, cache = vwe.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=True,
            )
        kv = [t for layer in cache.layers for t in (layer.keys, layer.values)]
        return (pad_masks, *kv)


class ExpertStepGraph(nn.Module):
    """One flow-matching velocity evaluation: ``VLAFlowMatching.denoise_step`` over a fixed prefix cache."""

    def __init__(self, policy: SmolVLAPolicy):
        super().__init__()
        cfg = policy.config
        _check_supported(cfg, policy.model)
        self.model = policy.model
        vwe = self.model.vlm_with_expert
        self.rope = _Rope(vwe.config.text_config.head_dim)
        self.time_embedding = _TimeEmbedding(vwe.expert_hidden_size, cfg.min_period, cfg.max_period)

    def forward(self, x_t: Tensor, t: Tensor, prefix_pad_mask: Tensor, *past_kv: Tensor) -> Tensor:
        cache = FunctionalKVCache(list(past_kv[0::2]), list(past_kv[1::2]))
        with (
            patched(modeling_smolvla, "create_sinusoidal_pos_embedding", self.time_embedding),
            patched(smolvlm_with_expert, "apply_rope", self.rope),
        ):
            return self.model.denoise_step(
                prefix_pad_masks=prefix_pad_mask, past_key_values=cache, x_t=x_t, timestep=t
            )


def graph_inputs(batch: dict, cfg) -> dict[str, np.ndarray]:
    """The graphs' inputs from a preprocessed batch, prepared as predict_action_chunk prepares them.

    Returns ``pixel_values_<c>`` per camera for vision ([1, 3, H, W] float32 in [-1, 1]) and ``lang_tokens``,
    ``lang_mask`` and ``state`` for prefix. The batch must come from the policy preprocessor with the language
    padded to max_length (``loading.load_processors(..., language_padding="max_length")``).
    """
    _check_supported(cfg)
    holder = SimpleNamespace(config=cfg)  # prepare_images and prepare_state only read self.config
    images, img_masks = SmolVLAPolicy.prepare_images(holder, batch)
    if len(images) != len(cfg.image_features) or not all(bool(m.all()) for m in img_masks):
        raise NotImplementedError("every camera must be present and unmasked")
    lang_tokens, lang_mask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    if lang_tokens.shape[-1] != cfg.tokenizer_max_length:
        raise ValueError(f"language is padded to {lang_tokens.shape[-1]} tokens; the prefix graph needs "
                         f"{cfg.tokenizer_max_length} (load the preprocessor with fixed-length padding)")
    out = {f"pixel_values_{c}": img.float().cpu().numpy() for c, img in enumerate(images)}
    out["lang_tokens"] = lang_tokens.to(torch.int64).cpu().numpy()
    out["lang_mask"] = lang_mask.to(torch.bool).cpu().numpy()
    out["state"] = SmolVLAPolicy.prepare_state(holder, batch).float().cpu().numpy()
    return out


def synthetic_batch(cfg, pre, task: str, seed: int) -> tuple[dict, Tensor]:
    """profile_smolvla.py's synthetic observation through ``pre``, and its flow-matching noise for ``seed``."""
    obs = synthetic_observation(cfg, task, seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g).to(cfg.device)
    return pre(dict(obs)), noise


@dataclass
class Reference:
    """What crosses the split points during one ``SmolVLAPolicy.predict_action_chunk`` call."""

    image_tokens: list[Tensor] = field(default_factory=list)  # embed_image output per camera, before scaling
    prefix_pad_mask: Tensor | None = None
    past_kv: list[Tensor] = field(default_factory=list)  # interleaved like split_contract.kv_names()
    x_t: list[Tensor] = field(default_factory=list)  # denoise_step input per Euler step
    t: list[Tensor] = field(default_factory=list)
    v_t: list[Tensor] = field(default_factory=list)  # denoise_step output per Euler step
    actions: Tensor | None = None  # predict_action_chunk output


def capture_reference(policy: SmolVLAPolicy, batch: dict, noise: Tensor) -> Reference:
    """Run LeRobot's ``predict_action_chunk`` unchanged and record the tensors at the split points."""
    model, vwe = policy.model, policy.model.vlm_with_expert
    ref = Reference()
    embed_image, embed_prefix, forward, denoise_step = (
        vwe.embed_image, model.embed_prefix, vwe.forward, model.denoise_step)

    def record_image(image):
        out = embed_image(image)
        ref.image_tokens.append(out.detach().clone())
        return out

    def record_prefix(*args, **kwargs):
        embs, pad_masks, att_masks = embed_prefix(*args, **kwargs)
        ref.prefix_pad_mask = pad_masks.detach().clone()
        return embs, pad_masks, att_masks

    forward_sig = inspect.signature(forward)

    def record_cache(*args, **kwargs):
        outputs, cache = forward(*args, **kwargs)
        if forward_sig.bind(*args, **kwargs).arguments.get("past_key_values") is None:  # the prefill call
            ref.past_kv = [t.detach().clone() for layer in cache.layers for t in (layer.keys, layer.values)]
        return outputs, cache

    denoise_sig = inspect.signature(denoise_step)

    def record_step(*args, **kwargs):
        bound = denoise_sig.bind(*args, **kwargs).arguments
        ref.x_t.append(bound["x_t"].detach().clone())
        ref.t.append(bound["timestep"].detach().clone())
        v_t = denoise_step(*args, **kwargs)
        ref.v_t.append(v_t.detach().clone())
        return v_t

    policy.reset()
    with (
        patched(vwe, "embed_image", record_image),
        patched(model, "embed_prefix", record_prefix),
        patched(vwe, "forward", record_cache),
        patched(model, "denoise_step", record_step),
        torch.no_grad(),
    ):
        ref.actions = policy.predict_action_chunk(batch, noise=noise.clone())
    return ref


@dataclass
class ReferenceRun:
    """One synthetic observation through LeRobot, with the language padded to the fixed length and as saved."""

    inputs: dict[str, np.ndarray]  # graph_inputs() of the fixed-length batch
    noise: Tensor
    fixed: Reference  # what the split graphs must reproduce
    saved: Reference  # the checkpoint's own padding (pad_language_to, "longest" by default)
    saved_lang_len: int


def reference_run(policy: SmolVLAPolicy, pre_fixed, pre_saved, task: str, seed: int) -> ReferenceRun:
    cfg = policy.config
    _check_supported(cfg, policy.model)
    batch, noise = synthetic_batch(cfg, pre_fixed, task, seed)
    batch_saved, _ = synthetic_batch(cfg, pre_saved, task, seed)
    return ReferenceRun(
        inputs=graph_inputs(batch, cfg),
        noise=noise,
        fixed=capture_reference(policy, batch, noise),
        saved=capture_reference(policy, batch_saved, noise),
        saved_lang_len=batch_saved[OBS_LANGUAGE_TOKENS].shape[1],
    )


def split_config(cfg, ref: Reference) -> dict:
    """The sizes a runtime needs, as split_manifest.json and bundles record them."""
    key, num_layers = ref.past_kv[0], len(ref.past_kv) // 2
    return {
        "camera_keys": list(cfg.image_features),
        "image_hw": [cfg.resize_imgs_with_padding[1], cfg.resize_imgs_with_padding[0]],
        "image_tokens": ref.image_tokens[0].shape[1],
        "lang_len": cfg.tokenizer_max_length,
        "state_dim": cfg.robot_state_feature.shape[0],
        "max_state_dim": cfg.max_state_dim,
        "action_dim": cfg.action_feature.shape[0],
        "max_action_dim": cfg.max_action_dim,
        "chunk_size": cfg.chunk_size,
        "num_steps": cfg.num_steps,
        "num_layers": num_layers,
        # As SmolVLMWithExpertModel builds it; _check_supported only lets through one expert layer per VLM layer.
        "expert_layers": cfg.num_expert_layers if cfg.num_expert_layers > 0 else num_layers,
        "kv_heads": key.shape[1],
        "head_dim": key.shape[3],
        "prefix_len": ref.prefix_pad_mask.shape[1],
    }

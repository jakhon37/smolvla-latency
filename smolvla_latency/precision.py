"""Precision variants for SmolVLA and a report of what dtypes actually run.

What LeRobot 0.6.1 builds (with transformers 5.4-5.5):

* ``load_vlm_weights=True`` (what LeRobot's training commands pass, so it is saved in most
  fine-tuned checkpoints): the VLM is loaded from the hub with ``torch_dtype="bfloat16"``,
  which sets ``config.text_config.dtype = bf16``. The action expert is built from a copy of
  that text config, so it is bf16 too. The exception is the cross-attention layers'
  ``k_proj``/``v_proj``: SmolVLA replaces them with plain ``nn.Linear`` after construction,
  so they are fp32. The checkpoint is then copied into those parameters.
  ``tests/check_lerobot_dtypes.py`` checks this split against LeRobot itself.
* ``load_vlm_weights=False`` builds from the base VLM's config without weights; the dtype then
  follows that config. ``loading.load_policy`` always forces fp32 for the reference.

The small projections (state_proj, action_in_proj, action_out_proj, action_time_mlp_*) are
fp32 in both cases, and ``denoise_step`` casts to fp32 before ``action_out_proj``, so a plain
``policy.half()`` fails with a dtype mismatch. The variants below only cast the VLM and the
action expert.
"""

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class Variant:
    vlm: str
    expert: str
    autocast: str | None = None
    lerobot_layout: bool = False  # keep cross-attn k/v fp32 as LeRobot does
    lerobot_backend: bool = False  # cudnn.benchmark + TF32 matmul, as lerobot_eval.py sets
    note: str = ""


VARIANTS = {
    "fp32": Variant("fp32", "fp32", note="reference; strict fp32 (no TF32)"),
    "lerobot": Variant("bf16", "bf16", lerobot_layout=True, lerobot_backend=True,
                       note="what lerobot-eval runs for a load_vlm_weights=true checkpoint"),
    "lerobot_amp": Variant("bf16", "bf16", "fp16", lerobot_layout=True, lerobot_backend=True,
                           note="same, with --policy.use_amp=true (fp16 autocast over bf16 weights)"),
    "bf16vlm": Variant("bf16", "fp32", note="bf16 VLM, fp32 expert (not a layout LeRobot builds)"),
    "fp16vlm": Variant("fp16", "fp32"),
    "fp16all": Variant("fp16", "fp16"),
    "bf16all": Variant("bf16", "bf16", note="bf16 everywhere including cross-attn k/v"),
    "amp": Variant("fp32", "fp32", "fp16",
                   note="fp16 autocast over fp32 weights (lerobot-eval --policy.use_amp=true on a "
                        "load_vlm_weights=false checkpoint, minus its backend flags)"),
}
DEFAULT_VARIANTS = "fp32,lerobot,lerobot_amp,bf16vlm,fp16vlm,fp16all,amp"


def _cross_attn_layers(vwe):
    if "cross" not in vwe.attention_mode:
        return []
    n = vwe.self_attn_every_n_layers
    return [layer for i, layer in enumerate(vwe.lm_expert.layers) if not (n > 0 and i % n == 0)]


def _cast_params(module: torch.nn.Module, dtype: torch.dtype, skip: set[int] = frozenset()) -> None:
    """Cast parameters only, in place. Buffers (unused rotary tables) stay fp32, as in LeRobot's
    own build, and skipped tensors are never rounded through another dtype."""
    for p in module.parameters():
        if id(p) not in skip and p.dtype != dtype:
            p.data = p.data.to(dtype)


def apply_variant(policy, name: str) -> None:
    """Call on an all-fp32 policy (the profiler reloads the fp32 reference before each variant)."""
    v = VARIANTS[name]
    vwe = policy.model.vlm_with_expert
    keep_fp32 = set()
    if v.lerobot_layout:
        for layer in _cross_attn_layers(vwe):
            keep_fp32.update(id(p) for p in layer.self_attn.k_proj.parameters())
            keep_fp32.update(id(p) for p in layer.self_attn.v_proj.parameters())
    _cast_params(vwe.vlm, DTYPES[v.vlm])
    _cast_params(vwe.lm_expert, DTYPES[v.expert], keep_fp32)


def autocast_ctx(name: str, device: torch.device):
    amp = VARIANTS[name].autocast
    if amp is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=DTYPES[amp])


@contextmanager
def backend_flags(name: str):
    """Strict fp32 for our variants; lerobot_eval.py's flags for the lerobot* variants. Restores on exit."""
    lerobot = VARIANTS[name].lerobot_backend
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32 = lerobot
    torch.backends.cudnn.allow_tf32 = lerobot  # torch default is True; lerobot_eval leaves it so
    torch.backends.cudnn.benchmark = lerobot
    try:
        yield {"matmul_allow_tf32": lerobot, "cudnn_allow_tf32": lerobot, "cudnn_benchmark": lerobot}
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = saved


def param_dtypes(module: torch.nn.Module) -> dict[str, str]:
    return {n: str(p.dtype).replace("torch.", "") for n, p in module.named_parameters()}


def dtype_report(policy) -> dict[str, dict[str, int]]:
    """Parameter dtype counts per submodule, read from the live model."""
    m = policy.model
    vlm = m.vlm_with_expert.get_vlm_model()
    parts = {
        "vision_model": vlm.vision_model,
        "connector": vlm.connector,
        "text_model": vlm.text_model,
        "lm_expert": m.vlm_with_expert.lm_expert,
        "state_proj": m.state_proj,
        "action_in_proj": m.action_in_proj,
        "action_out_proj": m.action_out_proj,
        "action_time_mlp": torch.nn.ModuleList([m.action_time_mlp_in, m.action_time_mlp_out]),
    }
    return {
        name: dict(Counter(str(p.dtype).replace("torch.", "") for p in mod.parameters()))
        for name, mod in parts.items()
    }

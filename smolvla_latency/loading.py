"""Load a SmolVLA checkpoint in full fp32 and build synthetic observations."""

import os
from collections import Counter, defaultdict

import torch
from huggingface_hub import hf_hub_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE
from safetensors import safe_open
from safetensors.torch import load_model as load_model_as_safetensor

from .precision import dtype_report

DEFAULT_TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def checkpoint_file(path: str, revision: str | None = None) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "model.safetensors")
    return hf_hub_download(repo_id=path, filename="model.safetensors", revision=revision)


def _group(key: str) -> str:
    for g in ("vision_model", "connector", "text_model", "lm_expert"):
        if f".{g}." in key:
            return g
    return "other"


def inspect_checkpoint(path: str, revision: str | None = None) -> dict:
    """Stored dtypes per submodule, read from the safetensors header (no tensors loaded)."""
    f = checkpoint_file(path, revision)
    by_group: dict[str, Counter] = defaultdict(Counter)
    keys = []
    with safe_open(f, framework="pt") as st:
        for k in st.keys():
            keys.append(k)
            by_group[_group(k)][st.get_slice(k).get_dtype()] += 1
    return {"file": f, "n_tensors": len(keys), "stored_dtypes": {g: dict(c) for g, c in by_group.items()}, "keys": keys}


def load_config(path: str, device: str, revision: str | None = None):
    cfg = PreTrainedConfig.from_pretrained(path, revision=revision)
    if cfg.type != "smolvla":
        raise ValueError(f"{path} is a '{cfg.type}' policy; this tool only handles smolvla")
    cfg.device = device
    return cfg


def load_processors(cfg, path: str, device: str, revision: str | None = None, language_padding: str | None = None):
    """The checkpoint's pre/post-processors. ``language_padding="max_length"`` pads the task to
    ``tokenizer_max_length`` instead of the saved ``pad_language_to`` (usually "longest")."""
    overrides = {"device_processor": {"device": device}}
    if language_padding is not None:
        overrides["tokenizer_processor"] = {"padding": language_padding}
    return make_pre_post_processors(
        cfg,
        pretrained_path=path,
        pretrained_revision=revision,
        preprocessor_overrides=overrides,
    )


def load_policy(path: str, device: str, revision: str | None = None):
    """Build SmolVLA with every parameter in fp32, then copy the checkpoint in.

    ``load_vlm_weights=False`` skips LeRobot's bf16 pre-load of the base VLM (the checkpoint
    already contains the VLM). The model is cast to fp32 *before* the weights are copied, so
    the reference holds the stored values exactly even if the base VLM's config carries a
    bf16 dtype. The dtypes the constructor produced are reported as ``built_dtypes``.
    """
    cfg = load_config(path, device, revision)
    saved_flag = bool(cfg.load_vlm_weights)
    cfg.load_vlm_weights = False

    policy = SmolVLAPolicy(cfg)
    built = dtype_report(policy)
    policy.float()
    file = checkpoint_file(path, revision)
    missing, unexpected = load_model_as_safetensor(policy, file, strict=False, device="cpu")
    policy.to(device)
    policy.eval()

    info = inspect_checkpoint(path, revision)
    info.pop("keys")
    info.update(
        saved_load_vlm_weights=saved_flag,
        built_dtypes=built,
        missing_in_checkpoint=sorted(missing),
        unexpected_in_checkpoint=sorted(unexpected),
    )

    pre, post = load_processors(cfg, path, device, revision)
    return policy, pre, post, info


def synthetic_observation(cfg, task: str = DEFAULT_TASK, seed: int = 0) -> dict:
    """One un-batched observation shaped like the policy's input features.

    Images are float in [0, 1], channel-first, as LeRobot's env processors produce them.
    Content does not affect latency; for accuracy work use real frames.
    """
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for key, feat in cfg.image_features.items():
        obs[key] = torch.rand(*feat.shape, generator=g)
    if cfg.robot_state_feature is not None:
        obs[OBS_STATE] = torch.randn(*cfg.robot_state_feature.shape, generator=g)
    obs["task"] = task
    return obs

"""Wrap SmolVLA (LeRobot 0.6.1) methods so each call is split into non-overlapping components.

Inclusive regions recorded by the wrappers:

    preprocess        policy preprocessor pipeline (tokenise, normalise, to device)
    policy_total      SmolVLAPolicy.predict_action_chunk
      prepare_images    resize-with-pad to 512x512 and scale to [-1, 1]
      sample_actions    VLAFlowMatching.sample_actions
        embed_prefix      image + language + state embeddings
          vision_encoder    SigLIP vision tower + connector, once per camera
        vlm_prefix        VLM forward over the prefix, fills the KV cache
        denoise_step      one flow-matching step, num_steps times
          expert_forward    action-expert forward reading the KV cache
    postprocess       unnormalise, to CPU

``exclusive()`` turns these into components that add up to the end-to-end time.
"""

from collections.abc import Callable

from .timing import RegionTimer

COMPONENTS = [
    "preprocess",
    "prepare_images",
    "vision_encoder",
    "embed_prefix_other",
    "vlm_prefix",
    "expert_forward",
    "denoise_other",
    "sample_other",
    "policy_other",
    "postprocess",
]


def _wrap(obj, name: str, timer: RegionTimer, region: str | Callable) -> None:
    original = getattr(obj, name)

    def wrapped(*args, **kwargs):
        label = region(*args, **kwargs) if callable(region) else region
        with timer.region(label):
            return original(*args, **kwargs)

    setattr(obj, name, wrapped)


def _vlm_forward_label(*args, **kwargs) -> str:
    inputs_embeds = kwargs.get("inputs_embeds")
    if inputs_embeds is None and len(args) >= 4:
        inputs_embeds = args[3]
    prefix = inputs_embeds[0] if inputs_embeds else None
    return "vlm_prefix" if prefix is not None else "expert_forward"


def instrument_policy(policy, timer: RegionTimer) -> None:
    """Patch the policy instance in place. Instance attributes shadow the class methods."""
    model = policy.model
    vwe = model.vlm_with_expert
    _wrap(policy, "predict_action_chunk", timer, "policy_total")
    _wrap(policy, "prepare_images", timer, "prepare_images")
    _wrap(model, "sample_actions", timer, "sample_actions")
    _wrap(model, "embed_prefix", timer, "embed_prefix")
    _wrap(model, "denoise_step", timer, "denoise_step")
    _wrap(vwe, "embed_image", timer, "vision_encoder")
    _wrap(vwe, "forward", timer, _vlm_forward_label)


def exclusive(row: dict[str, float]) -> dict[str, float]:
    g = lambda k: row.get(k, 0.0)  # noqa: E731
    out = {
        "preprocess": g("preprocess"),
        "prepare_images": g("prepare_images"),
        "vision_encoder": g("vision_encoder"),
        "embed_prefix_other": g("embed_prefix") - g("vision_encoder"),
        "vlm_prefix": g("vlm_prefix"),
        "expert_forward": g("expert_forward"),
        "denoise_other": g("denoise_step") - g("expert_forward"),
        "sample_other": g("sample_actions") - g("embed_prefix") - g("vlm_prefix") - g("denoise_step"),
        "policy_other": g("policy_total") - g("prepare_images") - g("sample_actions"),
        "postprocess": g("postprocess"),
    }
    out["end_to_end"] = g("preprocess") + g("policy_total") + g("postprocess")
    return out

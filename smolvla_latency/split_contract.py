"""Graph and tensor names and the Euler schedule of the split SmolVLA graphs.

numpy only, so a runtime can use it without torch or LeRobot. docs/SPLIT_GRAPH.md is the full contract.
"""

import numpy as np

MANIFEST = "split_manifest.json"
GRAPHS = ("vision", "prefix", "expert_step")
VISION_INPUT_NAMES = ["pixel_values"]
VISION_OUTPUT_NAMES = ["image_tokens"]
EXPERT_OUTPUT_NAMES = ["v_t"]


def kv_names(num_layers: int) -> list[str]:
    """K/V tensor names, interleaved per layer: past_key_0, past_value_0, past_key_1, ..."""
    return [name for i in range(num_layers) for name in (f"past_key_{i}", f"past_value_{i}")]


def prefix_input_names(num_cameras: int) -> list[str]:
    return [f"image_tokens_{c}" for c in range(num_cameras)] + ["lang_tokens", "lang_mask", "state"]


def prefix_output_names(num_layers: int) -> list[str]:
    return ["prefix_pad_mask", *kv_names(num_layers)]


def expert_input_names(num_layers: int) -> list[str]:
    """The prefix outputs, in the same order, follow x_t and t."""
    return ["x_t", "t", *prefix_output_names(num_layers)]


def euler_schedule(num_steps: int) -> tuple[np.float32, list[np.ndarray]]:
    """LeRobot's ``euler_integrate`` schedule: ``dt = -1/num_steps``, ``t_k = 1 + k dt`` in float64, then float32.

    Its update ``x_t + dt * v_t`` (a Python float times a float32 tensor) multiplies by float32(dt).
    """
    dt = -1.0 / num_steps
    return np.float32(dt), [np.array([1.0 + k * dt], dtype=np.float32) for k in range(num_steps)]

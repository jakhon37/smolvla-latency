"""The C++ runtime (cpp/, Python module smolvla_rt) against ort_flow.py and the bundle's PyTorch actions.

Runs when both of these are set, and is skipped otherwise; tests/cpp_parity.sh sets them:
    SMOLVLA_RT_PYTHON_DIR   the directory holding the module (cpp/build/python, built with -DSMOLVLA_RT_BUILD_PYTHON=ON)
    SMOLVLA_RT_FIXTURE_DIR  an export_split.py output directory with bundle.bin from tools/dump_bundle.py
Besides the module it needs only numpy, onnx and onnxruntime.
"""

import importlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime
import pytest

from ort_flow import SplitRuntime, bundle_problem
from smolvla_latency.metrics import compare

MODULE_DIR = os.environ.get("SMOLVLA_RT_PYTHON_DIR")
FIXTURE_DIR = os.environ.get("SMOLVLA_RT_FIXTURE_DIR")
pytestmark = pytest.mark.skipif(not (MODULE_DIR and FIXTURE_DIR),
                                reason="needs SMOLVLA_RT_PYTHON_DIR and SMOLVLA_RT_FIXTURE_DIR (tests/cpp_parity.sh)")


@pytest.fixture(scope="module")
def smolvla_rt():
    sys.path.insert(0, MODULE_DIR)
    try:
        return importlib.import_module("smolvla_rt")
    finally:
        sys.path.remove(MODULE_DIR)


@pytest.fixture(scope="module")
def problem():
    config = SplitRuntime.from_dir(FIXTURE_DIR, "fp32").config
    return bundle_problem(Path(FIXTURE_DIR) / "bundle.bin", config)


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_matches_ort_flow_bit_for_bit(smolvla_rt, problem, precision):
    """Same ONNX Runtime, graphs, inputs and Euler rule: the C++ and numpy loops give identical chunks."""
    if smolvla_rt.onnxruntime_version != onnxruntime.__version__:  # other kernels may round differently
        pytest.skip(f"the module runs on ONNX Runtime {smolvla_rt.onnxruntime_version} and ort_flow.py on "
                    f"{onnxruntime.__version__}; identical chunks need the same version (setup_env.sh pins the "
                    "Python package to the C++ SDK's)")
    inputs, noise, _ = problem
    expected = SplitRuntime.from_dir(FIXTURE_DIR, precision).predict_action_chunk(inputs, noise)
    rt = smolvla_rt.Runtime(FIXTURE_DIR, precision=precision)
    actions = rt.predict_action_chunk(inputs, noise)
    assert actions.dtype == np.float32 and actions.shape == expected.shape
    assert np.array_equal(actions, expected), compare(actions, expected)


def test_matches_pytorch(smolvla_rt, problem):
    inputs, noise, expected = problem
    rt = smolvla_rt.Runtime(FIXTURE_DIR)
    diff = compare(rt.predict_action_chunk(inputs, noise), expected["bundle_actions"])
    print(f"\nC++ {rt.backend} fp32 chunk vs PyTorch: max_abs {diff['max_abs']:.2e}")
    assert diff["max_abs"] < 1e-5 and diff["nonfinite"] == 0  # as tests/test_split_graph.py gates ort_flow.py


def test_config_and_timings(smolvla_rt, problem):
    inputs, noise, _ = problem
    rt = smolvla_rt.Runtime(FIXTURE_DIR, threads=1)
    rt.predict_action_chunk(inputs, noise)
    assert rt.config == SplitRuntime.from_dir(FIXTURE_DIR, "fp32").config
    t = rt.last_timings
    assert len(t["vision"]) == len(rt.config["camera_keys"]) and len(t["expert_step"]) == rt.config["num_steps"]
    assert t["total"] >= t["prefix"] + sum(t["vision"]) + sum(t["expert_step"])


def test_concurrent_calls_on_one_runtime(smolvla_rt, problem):
    """predict_action_chunk releases the GIL, and every call works in the runtime's one set of buffers."""
    inputs, noise, _ = problem
    rt = smolvla_rt.Runtime(FIXTURE_DIR, threads=1)
    expected = rt.predict_action_chunk(inputs, noise)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: rt.predict_action_chunk(inputs, noise), range(8)))
    assert all(np.array_equal(r, expected) for r in results)


def test_rejects_bad_inputs(smolvla_rt, problem):
    inputs, noise, _ = problem
    rt = smolvla_rt.Runtime(FIXTURE_DIR)
    with pytest.raises(ValueError, match="lang_mask"):
        rt.predict_action_chunk({k: v for k, v in inputs.items() if k != "lang_mask"}, noise)
    with pytest.raises(ValueError, match="state"):
        rt.predict_action_chunk({**inputs, "state": inputs["state"][:, :-1]}, noise)
    with pytest.raises(ValueError, match="noise"):
        rt.predict_action_chunk(inputs, noise[:, :-1])
    with pytest.raises(IndexError, match="no int8 graphs"):
        smolvla_rt.Runtime(FIXTURE_DIR, precision="int8")

"""Split-graph export of SmolVLA: the wrappers, the ONNX graphs and the ONNX Runtime flow loop vs LeRobot.

Offline on CPU with the tiny random checkpoint from make_fake_policy.py with two key/value heads, so a K/V
layout mistake changes the numbers (about three minutes). The fp32 action chunk is asserted at 1e-5 max abs, each
graph's output at 1e-4. fp16 numbers are printed (run with -s to see them) and only sanity-checked.
"""

import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
from lerobot.policies.smolvla import modeling_smolvla, smolvlm_with_expert
from transformers import DynamicCache

from export_split import check_runtime, export_precision, write_manifest
from ort_flow import SplitRuntime
from smolvla_latency.bundle import read_bundle, write_bundle
from smolvla_latency.loading import DEFAULT_TASK, load_policy, load_processors
from smolvla_latency.metrics import compare
from smolvla_latency.split_contract import (
    euler_schedule,
    expert_input_names,
    prefix_input_names,
    prefix_output_names,
)
from smolvla_latency.split_graph import (
    ExpertStepGraph,
    FunctionalKVCache,
    PrefixGraph,
    _check_supported,
    _Rope,
    _TimeEmbedding,
    reference_run,
)

ROOT = Path(__file__).resolve().parents[1]
FP32_TOL = 1e-4  # one graph's output vs LeRobot's value at that split point
CHUNK_TOL = 1e-5  # the fp32 action chunk vs predict_action_chunk, on the CPU


def run_script(*args) -> None:
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny")
    run_script("tests/make_fake_smolvlm.py", work / "vlm", "--tiny", "--kv-heads", "2")
    run_script("tests/make_fake_policy.py", "--vlm", work / "vlm", "--out", work / "policy")
    path = str(work / "policy")
    policy, pre_saved, _, _ = load_policy(path, "cpu")
    pre_fixed, _ = load_processors(policy.config, path, "cpu", language_padding="max_length")
    run = reference_run(policy, pre_fixed, pre_saved, DEFAULT_TASK, seed=0)
    return SimpleNamespace(dir=work / "policy", policy=policy, run=run)


def export(policy, run, out: Path, precision: str) -> SimpleNamespace:
    exported = {precision: export_precision(policy, run, out, precision, opset=18)}
    manifest = write_manifest(out, policy, run, exported, "tiny")
    return SimpleNamespace(dir=out, manifest=manifest, exported=exported[precision],
                           rt=SplitRuntime.from_dir(out, precision))


@pytest.fixture(scope="module")
def fp32(tiny, tmp_path_factory):
    return export(tiny.policy, tiny.run, tmp_path_factory.mktemp("onnx_fp32"), "fp32")


@pytest.fixture(scope="module")
def fp16(tiny, tmp_path_factory):
    policy, _, _, _ = load_policy(str(tiny.dir), "cpu")  # its own copy: the fp16 export casts it in place
    return export(policy, tiny.run, tmp_path_factory.mktemp("onnx_fp16"), "fp16")


def test_functional_cache_concatenates_without_mutating():
    g = torch.Generator().manual_seed(0)
    keys = [torch.randn(1, 2, 5, 4, generator=g) for _ in range(3)]
    values = [torch.randn(1, 2, 5, 4, generator=g) for _ in range(3)]
    cache, dynamic = FunctionalKVCache(keys, values), DynamicCache()
    for i, (k, v) in enumerate(zip(keys, values, strict=True)):
        dynamic.update(k, v, i)
    new_k, new_v = torch.randn(1, 2, 3, 4, generator=g), torch.randn(1, 2, 3, 4, generator=g)

    k, v = cache.update(new_k, new_v, 1)
    expected_k, expected_v = dynamic.update(new_k, new_v, 1)
    assert torch.equal(k, expected_k) and torch.equal(v, expected_v)
    assert torch.equal(cache.update(new_k, new_v, 1)[0], k)  # a second step starts from the prefix again
    cache.crop(5)
    assert len(cache) == 3 and cache  # LeRobot's cross-attention branch tests truthiness
    assert all(layer.keys is k and layer.values is v for layer, k, v in zip(cache.layers, keys, values, strict=True))


def test_rope_is_bit_identical_to_lerobot():
    rope, positions = _Rope(64), torch.arange(177)[None]
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(1, 177, 5, 64, generator=torch.Generator().manual_seed(0)).to(dtype)
        assert torch.equal(rope(x, positions), smolvlm_with_expert.apply_rope(x, positions))


def test_time_embedding_float32_matches_lerobot_float64():
    _, schedule = euler_schedule(10)
    t = torch.cat([torch.from_numpy(np.concatenate(schedule)), torch.tensor([0.0, 1.0, 1e-7]),
                   torch.rand(5000, generator=torch.Generator().manual_seed(0))])
    for dim in (144, 720):
        expected = create_sinusoidal_pos_embedding(t, dim, 4e-3, 4.0, device=torch.device("cpu"))
        assert expected.dtype == torch.float64  # what TensorRT cannot run
        got = _TimeEmbedding(dim, 4e-3, 4.0)(t, dim, 4e-3, 4.0)
        assert got.dtype == torch.float32
        assert (got - expected).abs().max().item() < 1e-6


def test_euler_loop_matches_lerobot_bitwise():
    noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(0))

    def velocity(x, t):
        return torch.sin(3 * x) * t[:, None, None] - x

    expected = euler_integrate(velocity, noise, 10)
    dt, times = euler_schedule(10)
    x = noise.numpy()
    for t in times:
        x = x + dt * velocity(torch.from_numpy(x), torch.from_numpy(t)).numpy()
    assert np.array_equal(x, expected.numpy())


def test_fixed_language_length_leaves_actions_unchanged(tiny):
    run, lang_len = tiny.run, tiny.policy.config.tokenizer_max_length
    assert run.saved_lang_len < lang_len
    assert run.fixed.prefix_pad_mask.shape[1] - run.saved.prefix_pad_mask.shape[1] == lang_len - run.saved_lang_len
    diff = compare(run.fixed.actions.numpy(), run.saved.actions.numpy())
    print(f"\nlanguage padded to {lang_len} instead of {run.saved_lang_len} tokens: "
          f"actions max_abs {diff['max_abs']:.2e}")
    assert diff["max_abs"] < 1e-5


def test_prefix_wrapper_reproduces_lerobot_prefill(tiny):
    ref, inputs = tiny.run.fixed, tiny.run.inputs
    with torch.no_grad():
        out = PrefixGraph(tiny.policy)(tuple(ref.image_tokens), *(torch.from_numpy(inputs[k])
                                                                  for k in ("lang_tokens", "lang_mask", "state")))
    assert torch.equal(out[0], ref.prefix_pad_mask)
    assert len(out[1:]) == len(ref.past_kv) == 2 * tiny.policy.model.vlm_with_expert.num_vlm_layers
    assert all(torch.equal(a, b) for a, b in zip(out[1:], ref.past_kv, strict=True))


def test_expert_wrapper_reproduces_denoise_step(tiny):
    ref = tiny.run.fixed
    expert = ExpertStepGraph(tiny.policy)
    with torch.no_grad():
        err = max((expert(x_t, t, ref.prefix_pad_mask, *ref.past_kv) - v_t).abs().max().item()
                  for x_t, t, v_t in zip(ref.x_t, ref.t, ref.v_t, strict=True))
    assert err < 1e-5  # only the float32 timestep embedding differs


def test_wrappers_restore_lerobot(tiny):
    originals = (smolvlm_with_expert.apply_rope, modeling_smolvla.create_sinusoidal_pos_embedding)
    ref, inputs = tiny.run.fixed, tiny.run.inputs
    with torch.no_grad():
        PrefixGraph(tiny.policy)(tuple(ref.image_tokens), *(torch.from_numpy(inputs[k])
                                                            for k in ("lang_tokens", "lang_mask", "state")))
        ExpertStepGraph(tiny.policy)(ref.x_t[0], ref.t[0], ref.prefix_pad_mask, *ref.past_kv)
    assert (smolvlm_with_expert.apply_rope, modeling_smolvla.create_sinusoidal_pos_embedding) == originals
    model = tiny.policy.model
    assert not {"embed_image", "forward"} & set(vars(model.vlm_with_expert))
    assert not {"embed_prefix", "denoise_step"} & set(vars(model))


def test_onnx_graphs_fp32_match_lerobot(tiny, fp32):
    res = check_runtime(fp32.rt, tiny.run, fp32.exported)
    print()
    for name, g in res["graphs"].items():
        print(f"fp32 {name:<11} ORT vs LeRobot max_abs {g['ort_vs_lerobot_fp32']['max_abs']:.2e}")
        assert g["ort_vs_lerobot_fp32"]["max_abs"] < FP32_TOL, (name, g["ort_vs_lerobot_fp32"])
        assert g["ort_vs_torch_same_precision"]["max_abs"] < FP32_TOL, (name, g["ort_vs_torch_same_precision"])
    assert res["graphs"]["prefix"]["prefix_pad_mask_equal"]


def test_ort_flow_matches_predict_action_chunk_fp32(tiny, fp32):
    run = tiny.run
    actions = fp32.rt.predict_action_chunk(run.inputs, run.noise.numpy())
    assert actions.dtype == np.float32 and actions.shape == tuple(run.fixed.actions.shape)
    print()
    for name, ref in (("fixed-length language", run.fixed), ("saved padding", run.saved)):
        diff = compare(actions, ref.actions.numpy())
        print(f"fp32 action chunk vs LeRobot ({name}): max_abs {diff['max_abs']:.2e}")
        assert diff["max_abs"] < CHUNK_TOL


def test_manifest_is_the_graph_contract(tiny, fp32):
    cfg, graphs = fp32.manifest["config"], fp32.manifest["variants"]["fp32"]
    n_cams, n_layers, p_len = len(cfg["camera_keys"]), cfg["num_layers"], cfg["prefix_len"]
    assert p_len == n_cams * cfg["image_tokens"] + cfg["lang_len"] + 1  # images, language, state
    assert cfg["expert_layers"] == n_layers
    assert cfg["kv_heads"] == 2  # with one, K/V read in [seq, heads] order instead of [heads, seq] look the same
    names = {(g, kind): [x["name"] for x in graphs[g][kind]] for g in graphs for kind in ("inputs", "outputs")}
    shapes = {(g, x["name"]): x["shape"] for g in graphs for kind in ("inputs", "outputs") for x in graphs[g][kind]}
    assert names["prefix", "inputs"] == prefix_input_names(n_cams)
    assert names["prefix", "outputs"] == prefix_output_names(n_layers)
    assert names["expert_step", "inputs"] == expert_input_names(n_layers)
    assert names["expert_step", "inputs"][2:] == names["prefix", "outputs"]
    assert names["expert_step", "outputs"] == ["v_t"]
    h, w = cfg["image_hw"]
    assert shapes["vision", "pixel_values"] == [1, 3, h, w]
    assert shapes["vision", "image_tokens"] == shapes["prefix", "image_tokens_0"]
    assert shapes["prefix", "lang_tokens"] == shapes["prefix", "lang_mask"] == [1, cfg["lang_len"]]
    assert shapes["prefix", "state"] == [1, cfg["max_state_dim"]]
    assert shapes["prefix", "prefix_pad_mask"] == [1, p_len]
    assert shapes["prefix", "past_value_0"] == [1, cfg["kv_heads"], p_len, cfg["head_dim"]]
    assert shapes["expert_step", "x_t"] == shapes["expert_step", "v_t"] == [1, cfg["chunk_size"], cfg["max_action_dim"]]
    assert shapes["expert_step", "t"] == [1]
    for g in graphs.values():
        assert g["files"] == [g["file"], g["file"] + ".data"] and g["opset"] == {"ai.onnx": 18}
        assert all((fp32.dir / f).exists() for f in g["files"])
        model = onnx.load(str(fp32.dir / g["file"]), load_external_data=False)
        typed = [*model.graph.input, *model.graph.output, *model.graph.value_info]
        assert onnx.TensorProto.DOUBLE not in {v.type.tensor_type.elem_type for v in typed}
        assert onnx.TensorProto.DOUBLE not in {t.data_type for t in model.graph.initializer}
        assert "ScatterND" not in {n.op_type for n in model.graph.node}


def test_fp16_graphs_strongly_typed_and_close(tiny, fp16):
    graphs = fp16.manifest["variants"]["fp16"]
    dtypes = {(g, x["name"]): x["dtype"] for g in graphs for kind in ("inputs", "outputs") for x in graphs[g][kind]}
    assert dtypes["vision", "pixel_values"] == dtypes["vision", "image_tokens"] == "float16"
    assert dtypes["prefix", "image_tokens_0"] == dtypes["prefix", "past_key_0"] == "float16"
    assert dtypes["prefix", "state"] == dtypes["expert_step", "x_t"] == dtypes["expert_step", "t"] == "float32"
    assert dtypes["expert_step", "past_value_0"] == "float16" and dtypes["expert_step", "v_t"] == "float32"
    for g in graphs.values():
        model = onnx.load(str(fp16.dir / g["file"]), load_external_data=False)
        assert onnx.TensorProto.FLOAT16 in {t.data_type for t in model.graph.initializer}

    res = check_runtime(fp16.rt, tiny.run, fp16.exported)
    print()
    for name, g in res["graphs"].items():
        print(f"fp16 {name:<11} ORT vs LeRobot fp32 max_abs {g['ort_vs_lerobot_fp32']['max_abs']:.2e}  "
              f"ORT vs torch fp16 max_abs {g['ort_vs_torch_same_precision']['max_abs']:.2e}")
    a = res["actions_vs_lerobot_fp32"]
    print(f"fp16 action chunk vs LeRobot fp32: max_abs {a['max_abs']:.2e}  cos {a['cosine']:.6f}")
    assert a["nonfinite"] == 0 and a["cosine"] > 0.999


def test_bundle_round_trip_and_layout(tmp_path):
    tensors = {
        "x": np.arange(7, dtype=np.float32).reshape(1, 7),
        "mask": np.array([[True, False, True]]),
        "ids": np.array([[1, -2, 3]], dtype=np.int64),
        "half": np.full((2, 3), 0.5, dtype=np.float16),
        "t": np.array([0.9], dtype=np.float32),
    }
    path = tmp_path / "b.bin"
    write_bundle(path, tensors, {"note": "test", "dims": [1, 2]})
    got, meta = read_bundle(path)
    assert meta == {"note": "test", "dims": [1, 2]} and list(got) == list(tensors)
    for name, array in tensors.items():
        assert got[name].dtype == array.dtype and np.array_equal(got[name], array)

    raw = path.read_bytes()
    assert raw[:8] == b"SVLABNDL"
    (size,) = struct.unpack_from("<Q", raw, 8)
    assert (16 + size) % 64 == 0
    header = json.loads(raw[16:16 + size])
    for entry in header["tensors"]:
        start = 16 + size + entry["offset"]
        assert entry["offset"] % 64 == 0
        expected = tensors[entry["name"]]
        assert raw[start:start + entry["nbytes"]] == expected.astype(expected.dtype.newbyteorder("<")).tobytes()

    with pytest.raises(TypeError):
        write_bundle(tmp_path / "bad.bin", {"d": np.zeros(2)}, {})  # float64 is not a bundle dtype


def bundle_header(entries: list, **changes) -> dict:
    return {"format": "smolvla-split-bundle", "version": 1, "meta": {}, "tensors": entries, **changes}


def raw_bundle(path: Path, header, data: bytes, aligned: bool = True) -> Path:
    """A bundle with exactly this header (any JSON value), space-padded as write_bundle pads a valid one, or one
    space past that when not ``aligned``."""
    text = json.dumps(header).encode()
    text += b" " * (-(16 + len(text)) % 64 + (0 if aligned else 1))
    path.write_bytes(b"SVLABNDL" + struct.pack("<Q", len(text)) + text + data)
    return path


@pytest.mark.parametrize("shape, nbytes, offset, problem", [
    ([2**32, 2**32], 0, 0, "needs 73786976294838206464"),  # 2^64 elements: int64 arithmetic wraps to 0
    ([2**62 + 1, 4], 16, 0, "needs"),  # wraps to 4 elements in int64, which these 16 bytes would hold
    ([-1, 4], 16, 0, "non-negative integers"),
    ([1.5, 4], 16, 0, "non-negative integers"),
    ("4", 16, 0, "non-negative integers"),
    ([1, 4], 16, 2**64, "past the end"),
    ([1, 4], 16, -64, "past the end"),
])
def test_bundle_rejects_absurd_shapes(tmp_path, shape, nbytes, offset, problem):
    entry = {"name": "x", "dtype": "float32", "shape": shape, "offset": offset, "nbytes": nbytes}
    with pytest.raises(ValueError, match=problem) as error:
        read_bundle(raw_bundle(tmp_path / "absurd.bin", bundle_header([entry]), bytes(64)))
    assert "absurd.bin: x " in str(error.value)


X = {"name": "x", "dtype": "float32", "shape": [1, 4], "offset": 0, "nbytes": 16}


@pytest.mark.parametrize("header, problem", [
    ([1, 2], "unsupported bundle format"),
    (bundle_header([X], format=7), "unsupported bundle format"),
    (bundle_header([X], version="1"), "unsupported bundle format"),
    (bundle_header([X], version=True), "unsupported bundle format"),
    (bundle_header([X], version=2**32 + 1), "unsupported bundle format"),
    (bundle_header([X], meta=[]), '"meta" must be a JSON object'),
    (bundle_header({"x": X}), '"tensors" must be a JSON array'),
    (bundle_header([X], meta={"euler": {}}), "meta.euler must be"),
    (bundle_header([X], meta={"euler": {"t": [1.0, 0.5]}}), "meta.euler must be"),
    (bundle_header([X], meta={"euler": None}), "meta.euler must be"),
    (bundle_header([{k: v for k, v in X.items() if k != "offset"}]), "tensor entry 0 needs"),
    (bundle_header([{**X, "name": 5}]), "tensor entry 0 needs"),
    (bundle_header([X, {**X, "offset": 64}]), "duplicate tensor x"),
    (bundle_header([{**X, "nbytes": 16.0}]), "x has 16.0 bytes"),
    (bundle_header([{**X, "dtype": ["float32"]}]), "x has dtype"),
])
def test_bundle_rejects_malformed_headers(tmp_path, header, problem):
    """What the C++ reader rejects too (cpp/tests/test_bundle.cpp), with a ValueError naming the file."""
    path = tmp_path / "bad.bin"
    with pytest.raises(ValueError, match=re.escape(problem)) as error:
        read_bundle(raw_bundle(path, header, bytes(128)))
    assert str(error.value).startswith(str(path))


def test_bundle_rejects_an_unaligned_data_section(tmp_path):
    read_bundle(raw_bundle(tmp_path / "ok.bin", bundle_header([X], meta={"euler": {"dt": -0.5, "t": [1.0, 0.5]}}),
                           bytes(64)))
    with pytest.raises(ValueError, match="not 64-byte aligned"):
        read_bundle(raw_bundle(tmp_path / "bad.bin", bundle_header([X]), bytes(64), aligned=False))


def test_unsupported_configs_are_named():
    ok = SimpleNamespace(use_cache=True, empty_cameras=0, adapt_to_pi_aloha=False, attention_mode="cross_attn",
                         rtc_config=None, add_image_special_tokens=False, prefix_length=-1, num_expert_layers=-1)
    model = SimpleNamespace(vlm_with_expert=SimpleNamespace(num_expert_layers=16, num_vlm_layers=16))
    _check_supported(ok, model)
    for change, problem in (({"add_image_special_tokens": True}, "add_image_special_tokens=True"),
                            ({"prefix_length": 200}, "prefix_length=200")):
        with pytest.raises(NotImplementedError, match=problem):
            _check_supported(SimpleNamespace(**{**vars(ok), **change}))
    fewer = SimpleNamespace(vlm_with_expert=SimpleNamespace(num_expert_layers=8, num_vlm_layers=16))
    with pytest.raises(NotImplementedError, match="an action expert with 8 layers for 16 VLM layers"):
        _check_supported(SimpleNamespace(**{**vars(ok), "num_expert_layers": 8}), fewer)


def checkpoint_with(tiny, tmp_path: Path, **changes) -> Path:
    """A copy of the tiny checkpoint whose config.json has these values; the weights are linked, not copied."""
    policy = tmp_path / "policy"
    shutil.copytree(tiny.dir, policy, ignore=shutil.ignore_patterns("model.safetensors"))
    (policy / "model.safetensors").symlink_to(tiny.dir / "model.safetensors")
    config = json.loads((policy / "config.json").read_text())
    (policy / "config.json").write_text(json.dumps({**config, **changes}))
    return policy


@pytest.mark.parametrize("changes, problem", [
    ({"add_image_special_tokens": True}, "add_image_special_tokens=True"),
    ({"prefix_length": 200}, "prefix_length=200"),
    # LeRobot builds this expert, but its own denoising then fails on a shape mismatch.
    ({"num_expert_layers": 2}, "an action expert with 2 layers for 4 VLM layers"),
])
def test_export_refuses_unsupported_checkpoints(tiny, tmp_path, changes, problem):
    out = tmp_path / "onnx"
    result = subprocess.run([sys.executable, "export_split.py", "--policy", checkpoint_with(tiny, tmp_path, **changes),
                             "--out-dir", out], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and problem in result.stderr, result.stderr[-2000:]
    assert not any(out.glob("*.onnx"))


def test_command_line_tools(tiny, tmp_path):
    """export_split.py, tools/dump_bundle.py and ort_flow.py as docs/SPLIT_GRAPH.md runs them."""
    onnx_dir, bundle, flow = tmp_path / "onnx", tmp_path / "bundle.bin", tmp_path / "flow.json"
    run_script("export_split.py", "--policy", tiny.dir, "--out-dir", onnx_dir, "--precisions", "fp32")
    report = json.loads((onnx_dir / "export_report.json").read_text())
    assert report["variants"]["fp32"]["actions_vs_lerobot_fp32"]["max_abs"] < CHUNK_TOL

    run_script("tools/dump_bundle.py", "--policy", tiny.dir, "--out", bundle)
    tensors, meta = read_bundle(bundle)
    np.testing.assert_allclose(tensors["actions"], tiny.run.fixed.actions.numpy(), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(tensors["noise"], tiny.run.noise.numpy())
    assert meta["config"] == json.loads((onnx_dir / "split_manifest.json").read_text())["config"]

    # --bundle must run where only numpy, onnx and onnxruntime are installed.
    without_torch = ("import runpy, sys; sys.modules.update(torch=None, transformers=None, lerobot=None); "
                     "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
    run_script("-c", without_torch, "ort_flow.py", "--onnx-dir", onnx_dir, "--bundle", bundle, "--out", flow)
    assert json.loads(flow.read_text())["vs_bundle_actions"]["max_abs"] < CHUNK_TOL
    run_script("ort_flow.py", "--onnx-dir", onnx_dir, "--policy", tiny.dir, "--check", "--out", flow)
    checked = json.loads(flow.read_text())
    assert checked["vs_torch_fixed_len"]["max_abs"] < CHUNK_TOL and checked["vs_torch_longest"]["max_abs"] < CHUNK_TOL

"""export_vision.py's ONNX Runtime parity settings and its TorchScript-exporter guard; no model is built."""

import subprocess
import sys
from pathlib import Path

import onnxruntime as ort
import pytest
import transformers

from export_vision import ort_providers

ROOT = Path(__file__).resolve().parents[1]


def test_cuda_parity_check_runs_without_tf32(monkeypatch):
    """export_vision.py and export_split.py check fp32 graphs on ONNX Runtime CUDA with TF32 off."""
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert ort_providers("cuda") == [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    assert ort_providers("cpu") == ["CPUExecutionProvider"]
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert ort_providers("cuda") == ["CPUExecutionProvider"]


@pytest.mark.skipif(int(transformers.__version__.split(".")[0]) < 5, reason="the guard is for transformers 5")
def test_legacy_exporter_fails_before_loading_anything(tmp_path):
    out = tmp_path / "onnx"
    result = subprocess.run([sys.executable, "export_vision.py", "--policy", str(tmp_path / "no-such-policy"),
                             "--out-dir", str(out), "--legacy-exporter"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert f"fails under transformers {transformers.__version__}" in result.stderr, result.stderr[-2000:]
    assert not out.exists()

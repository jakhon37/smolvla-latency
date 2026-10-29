"""What a report was produced with, recorded by every tools/real_* report under "code" and "system".

code_version()      the git commit and the sha256 of every code file that differs from it, so a report names the
                    exact code it came from even when that code is not committed yet
system_info()       CPU model and the instruction-set flags that decide bf16/fp16 arithmetic, the CPUs the process may
                    run on, ONNX Runtime's version and the size of its default intra-op thread pool on this machine
decoder_versions()  torchcodec and the FFmpeg libraries it decodes the dataset's videos with
hub_revision()      the commit a repository resolved to in the local Hugging Face cache (for repositories that are
                    fetched without a pinned revision, such as a checkpoint's base VLM)
"""

import hashlib
import importlib.metadata as md
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CODE_SUFFIXES = {".py", ".sh", ".c", ".cc", ".cpp", ".h", ".hpp", ".cmake", ".toml", ".cfg", ".yml", ".yaml"}
CODE_NAMES = {"CMakeLists.txt", "requirements.txt"}
CPU_FLAGS = ("avx2", "fma", "f16c", "avx512f", "avx512_vnni", "avx512_bf16", "avx512_fp16", "amx_tile", "amx_bf16")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a) -> str:
    """sha256 of an array's bytes in C order (dtype and shape are recorded next to it)."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def code_version() -> dict:
    """The repository's git commit, and the sha256 of each code file that is modified or untracked."""

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        status = git("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "note": f"{REPO} is not a git checkout"}
    changed = {}
    for line in status.splitlines():
        rel = line[3:].split(" -> ")[-1].strip('"')
        path = REPO / rel
        if path.suffix in CODE_SUFFIXES or path.name in CODE_NAMES:
            changed[rel] = sha256_file(path)[:16] if path.is_file() else "deleted"
    return {"commit": commit, "dirty": bool(changed), "changed_code_sha256_16": dict(sorted(changed.items()))}


def ort_default_threads() -> int | None:
    """Threads in ONNX Runtime's intra-op pool when a session sets none (intra_op_num_threads = 0), counted as the
    worker threads a session starts plus the calling thread, which also runs work."""
    try:
        import onnxruntime as ort
        from onnx import TensorProto, helper
    except ImportError:
        return None
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "identity", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    opts = ort.SessionOptions()  # intra_op_num_threads left at 0, ONNX Runtime's default
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")  # idle workers sleep instead of spinning
    try:
        before = len(os.listdir("/proc/self/task"))
        session = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
        workers = len(os.listdir("/proc/self/task")) - before
    except OSError:  # no /proc (not Linux)
        return None
    del session
    return workers + 1


def system_info() -> dict:
    model, flags = platform.processor() or platform.machine(), set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
            elif line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
    except OSError:
        pass
    info = {"cpu": model, "cpu_flags": {f: f in flags for f in CPU_FLAGS}, "logical_cpus": os.cpu_count(),
            "usable_cpus": sorted(os.sched_getaffinity(0))}
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["ort_default_intra_op_threads"] = ort_default_threads()
    except ImportError:
        info["onnxruntime"] = None
    info.update(numpy=np.__version__, python=platform.python_version(), platform=platform.platform(),
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return info


def decoder_versions() -> dict:
    out: dict = {}
    for pkg in ("torchcodec", "av"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    try:
        from torchcodec._core import get_ffmpeg_library_versions

        out["ffmpeg_libraries"] = {k: ".".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
                                   for k, v in get_ffmpeg_library_versions().items()}
    except Exception as e:  # noqa: BLE001 - a private torchcodec API; record why it is missing
        out["ffmpeg_libraries"] = f"unavailable: {type(e).__name__}: {e}"
    return out


def hub_revision(repo_id: str, filename: str, repo_type: str = "model") -> str | None:
    """The commit the local Hugging Face cache resolved ``repo_id``'s default branch to, read from the snapshot that
    holds ``filename`` (which must already be downloaded); None when it is not cached or ``repo_id`` is a path."""
    if os.path.isdir(repo_id):
        return None
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id, filename, repo_type=repo_type)
    if isinstance(path, str):
        parts = Path(path).parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
    return None

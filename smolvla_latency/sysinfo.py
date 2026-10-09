"""Record the software and GPU state a measurement was taken under."""

import importlib.metadata as md
import platform
import subprocess

import torch


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _gpu_id() -> str | None:
    """UUID of the device torch uses, so nvidia-smi reads the same GPU under CUDA_VISIBLE_DEVICES."""
    try:
        return f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    except Exception:
        return None


def _nvidia_smi(query: str) -> str | None:
    base = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]
    gpu = _gpu_id() if torch.cuda.is_available() else None
    attempts = ([base[:1] + [f"--id={gpu}"] + base[1:]] if gpu else []) + [base]
    for i, cmd in enumerate(attempts):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
            value = out.stdout.strip().splitlines()[0]
            # Falling back to the first listed GPU may read a different device than torch uses.
            return value if (i == 0 and gpu) or not gpu else f"{value} (first GPU; uuid lookup failed)"
        except Exception:
            continue
    return None


def gpu_clocks() -> dict:
    """Current clocks and throttle reasons; call right after a timed loop to see clocks under load."""
    return {
        "sm_mem_mhz": _nvidia_smi("clocks.sm,clocks.mem"),
        "temperature_c": _nvidia_smi("temperature.gpu"),
        "clock_event_reasons_active": _nvidia_smi("clocks_event_reasons.active")
        or _nvidia_smi("clocks_throttle_reasons.active"),
    }


def collect() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "packages": {p: _version(p) for p in
                     ("lerobot", "transformers", "safetensors", "onnx", "onnxruntime", "onnxruntime-gpu",
                      "tensorrt", "tensorrt-cu12", "tensorrt-cu13", "polygraphy")},
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(
            gpu=p.name,
            gpu_uuid=_gpu_id(),
            compute_capability=f"{p.major}.{p.minor}",
            gpu_mem_gb=round(p.total_memory / 2**30, 1),
            torch_arch_list=torch.cuda.get_arch_list(),
            bf16_native=torch.cuda.is_bf16_supported(including_emulation=False),
            bf16_with_emulation=torch.cuda.is_bf16_supported(including_emulation=True),
            driver=_nvidia_smi("driver_version"),
            clocks_idle=gpu_clocks(),
            max_clocks_sm_mem_mhz=_nvidia_smi("clocks.max.sm,clocks.max.mem"),
            power_limit=_nvidia_smi("power.limit"),
        )
    else:
        info.update(cpu=platform.processor() or platform.machine(), threads=torch.get_num_threads())
    return info

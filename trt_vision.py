"""Build a TensorRT engine for the exported vision encoder, time it, and check parity.

    python trt_vision.py --onnx onnx/vision_fp16.onnx                      # strongly typed FP16
    python trt_vision.py --onnx onnx/vision_fp32.onnx                      # FP32 engine
    python trt_vision.py --onnx onnx/vision_fp32.onnx --mode fp16-flag     # TensorRT 10 only
    python trt_vision.py --onnx onnx/vision_fp16.onnx --torch-baseline --policy lerobot/smolvla_libero

Needs an NVIDIA GPU and the ``tensorrt`` wheel. Latency is GPU time per call measured with CUDA
events on one stream (inputs already on the GPU, batch size 1), the same way the optional
PyTorch baseline is timed, so the two numbers are comparable. Parity is against the PyTorch
fp32 outputs saved by export_vision.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from smolvla_latency import sysinfo
from smolvla_latency.metrics import compare
from smolvla_latency.timing import summarize


def trt_to_torch(trt) -> dict:
    pairs = [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16),
             ("int32", torch.int32), ("int64", torch.int64), ("int8", torch.int8), ("bool", torch.bool)]
    return {getattr(trt, a): t for a, t in pairs if hasattr(trt, a)}


def build(trt, onnx_path: Path, mode: str, workspace_gb: int, tf32: bool) -> tuple[bytes, float]:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    strongly = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
    if mode == "strongly-typed" and strongly is not None:
        flags |= 1 << int(strongly)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):  # resolves <name>.onnx.data next to the graph
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if mode == "fp16-flag":
        if not hasattr(trt.BuilderFlag, "FP16"):
            raise SystemExit("This TensorRT has no weakly typed FP16 flag; export fp16 ONNX and use strongly-typed.")
        config.set_flag(trt.BuilderFlag.FP16)
    if not tf32 and hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    t = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed; rerun with a verbose logger")
    return bytes(blob), time.perf_counter() - t


class Engine:
    def __init__(self, trt, blob: bytes):
        self.trt = trt
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        self.ctx = self.engine.create_execution_context()
        to_torch = trt_to_torch(trt)
        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise ValueError(f"{name} has a dynamic shape {shape}; export with static shapes")
            buf = torch.empty(shape, dtype=to_torch[self.engine.get_tensor_dtype(name)], device="cuda")
            self.ctx.set_tensor_address(name, buf.data_ptr())
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_input else self.outputs)[name] = buf
        self.stream = torch.cuda.Stream()

    def enqueue(self) -> None:
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("execute_async_v3 returned False")

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        (inp,) = self.inputs.values()
        (out,) = self.outputs.values()
        with torch.cuda.stream(self.stream):
            inp.copy_(x.to(device="cuda", dtype=inp.dtype))
            self.enqueue()
            result = out.float().cpu()  # stream-ordered copy, then host sync
        return result


def time_gpu(fn, stream: torch.cuda.Stream, warmup: int, iters: int) -> dict:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
        stream.synchronize()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
        for start, end in events:
            start.record(stream)
            fn()
            end.record(stream)
        stream.synchronize()
    return summarize([s.elapsed_time(e) for s, e in events])


def torch_baseline(policy_path: str, revision: str | None, x: torch.Tensor, warmup: int, iters: int,
                   tf32: bool) -> dict:
    from export_vision import VisionEncoder
    from smolvla_latency.loading import load_policy

    # Match the engine: TF32 only if --tf32 was given for TensorRT too.
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32

    policy, _, _, _ = load_policy(policy_path, "cuda", revision)
    enc = VisionEncoder(policy.model.vlm_with_expert).eval()
    out = {}
    stream = torch.cuda.Stream()
    for name, dtype in (("torch_fp32", torch.float32), ("torch_fp16", torch.float16)):
        enc.to(dtype)
        xi = x.to("cuda", dtype)
        torch.cuda.synchronize()  # xi was written on the default stream

        def step(enc=enc, xi=xi):
            return enc(xi)

        with torch.no_grad():
            out[name] = time_gpu(step, stream, warmup, iters)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--mode", choices=["strongly-typed", "fp16-flag"], default="strongly-typed")
    ap.add_argument("--check-data", type=Path, default=None, help="default: check_data.npz next to --onnx")
    ap.add_argument("--workspace-gb", type=int, default=4)
    ap.add_argument("--tf32", action="store_true", help="allow TF32 for fp32 layers (Ampere+ only)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--torch-baseline", action="store_true", help="also time the PyTorch encoder (needs --policy)")
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("TensorRT needs a CUDA GPU")

    import tensorrt as trt

    check = np.load(args.check_data or args.onnx.with_name("check_data.npz"))
    x_all = torch.from_numpy(check["pixel_values"])
    ref = check["image_tokens_fp32"]

    blob, build_s = build(trt, args.onnx, args.mode, args.workspace_gb, args.tf32)
    tag = f"{args.onnx.stem}_{args.mode}"
    engine_path = args.onnx.with_name(f"{tag}.trt{trt.__version__.split('.')[0]}.engine")
    engine_path.write_bytes(blob)
    eng = Engine(trt, blob)
    io = {n: [list(b.shape), str(b.dtype)] for n, b in {**eng.inputs, **eng.outputs}.items()}
    print(f"built {engine_path.name} in {build_s:.0f}s ({len(blob) / 2**20:.0f} MB), io={io}")

    outs = np.concatenate([eng.infer(x_all[i:i + 1]).numpy() for i in range(len(x_all))])
    parity = compare(outs, ref)
    print(f"parity vs torch fp32 over {len(x_all)} inputs: {parity}")

    (inp,) = eng.inputs.values()
    inp.copy_(x_all[:1].to("cuda", inp.dtype))
    torch.cuda.synchronize()  # the copy ran on the default stream; timing uses eng.stream
    latency = time_gpu(eng.enqueue, eng.stream, args.warmup, args.iters)
    print(f"TensorRT p50 {latency['p50']:.3f} ms  p99 {latency['p99']:.3f} ms")

    report = {
        "tool": "smolvla-latency/trt_vision.py",
        "system": sysinfo.collect(),
        "tensorrt": trt.__version__,
        "onnx": str(args.onnx), "mode": args.mode, "tf32": args.tf32,
        "engine": engine_path.name, "engine_mb": round(len(blob) / 2**20, 1), "build_s": round(build_s, 1),
        "io": io, "parity_vs_torch_fp32": parity, "n_parity_inputs": len(x_all),
        "latency_ms": {"tensorrt": latency},
    }
    if args.torch_baseline:
        report["latency_ms"].update(torch_baseline(args.policy, args.revision, x_all[:1], args.warmup, args.iters,
                                                    args.tf32))
        for k, v in report["latency_ms"].items():
            print(f"  {k:<11} p50 {v['p50']:.3f} ms  p99 {v['p99']:.3f} ms")
    out = args.out or args.onnx.with_name(f"trt_report_{tag}.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Export SmolVLA's vision encoder (SigLIP tower + connector) to ONNX and check it against PyTorch.

    python export_vision.py --policy lerobot/smolvla_libero --out-dir onnx

Writes vision_fp32.onnx and vision_fp16.onnx (weights cast in PyTorch before export, so the
fp16 graph is strongly typed and builds on TensorRT 10 and 11 without the FP16 builder flag),
check_data.npz (inputs plus PyTorch fp32 outputs, used by trt_vision.py) and
export_report.json (ONNX Runtime parity).

The exported graph is exactly ``vlm_with_expert.embed_image``: pixel_values in [-1, 1] at the
policy's padded resolution in, image tokens out. The sqrt(dim) scaling that embed_prefix applies
afterwards is not part of it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from smolvla_latency import sysinfo
from smolvla_latency.loading import load_policy
from smolvla_latency.metrics import compare


class VisionEncoder(torch.nn.Module):
    def __init__(self, vlm_with_expert):
        super().__init__()
        self.vwe = vlm_with_expert

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vwe.embed_image(pixel_values)



def export(module, example, path: Path, opset: int, dynamo: bool) -> dict:
    t = time.perf_counter()
    torch.onnx.export(
        module, (example,), str(path),
        input_names=["pixel_values"], output_names=["image_tokens"],
        opset_version=opset, dynamo=dynamo,
    )
    # The dynamo exporter writes weights to <name>.onnx.data; keep both files together.
    files = [p for p in (path, path.with_name(path.name + ".data")) if p.exists()]
    return {"path": str(path), "files": [p.name for p in files], "seconds": round(time.perf_counter() - t, 1),
            "mb": round(sum(p.stat().st_size for p in files) / 2**20, 1), "dynamo": dynamo, "opset": opset}


def ort_providers(device: str) -> list:
    """ONNX Runtime providers for a parity check on ``device``: CUDA if asked for and available, else the CPU.

    The CUDA provider runs fp32 MatMul and Conv in TF32 unless told otherwise, and the reference is strict fp32.
    """
    import onnxruntime as ort

    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        return [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def ort_run(path: Path, x: np.ndarray, device: str) -> tuple[np.ndarray, str]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=ort_providers(device))
    feed = {sess.get_inputs()[0].name: x}
    outs = [sess.run(None, {k: v[i:i + 1] for k, v in feed.items()})[0] for i in range(x.shape[0])]
    return np.concatenate(outs), sess.get_providers()[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=Path("onnx"))
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--legacy-exporter", action="store_true",
                    help="use the TorchScript exporter (dynamo=False); it fails under transformers 5, which "
                         "LeRobot 0.6.1 requires")
    ap.add_argument("--num-check", type=int, default=16, help="random inputs used for parity")
    ap.add_argument("--skip-fp16", action="store_true")
    args = ap.parse_args()
    if args.legacy_exporter and int(transformers.__version__.split(".")[0]) >= 5:
        raise SystemExit(f"--legacy-exporter: the TorchScript exporter fails under transformers "
                         f"{transformers.__version__}, inside SmolVLM's vision attention mask (IndexError in "
                         "transformers.masking_utils.sdpa_mask). Use the default exporter, dynamo=True.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Strict fp32 reference: cuDNN convs (the SigLIP patch embedding) default to TF32 on Ampere+.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    policy, _, _, ckpt = load_policy(args.policy, args.device, args.revision)
    w, h = policy.config.resize_imgs_with_padding  # SmolVLA stores (width, height)
    enc = VisionEncoder(policy.model.vlm_with_expert).eval()

    g = torch.Generator().manual_seed(0)
    x = torch.rand(args.num_check, 3, h, w, generator=g) * 2 - 1  # SigLIP range
    with torch.no_grad():
        ref = torch.cat([enc(x[i:i + 1].to(args.device)).float().cpu() for i in range(len(x))]).numpy()
    np.savez(args.out_dir / "check_data.npz", pixel_values=x.numpy(), image_tokens_fp32=ref)
    print(f"input (1, 3, {h}, {w}) -> image tokens {tuple(ref.shape[1:])}")

    report = {"system": sysinfo.collect(), "policy": args.policy, "input_hw": [h, w],
              "output_shape": list(ref.shape[1:]), "checkpoint_stored_dtypes": ckpt["stored_dtypes"],
              "reference_tf32": False,
              "exports": {}}
    dynamo = not args.legacy_exporter
    example = x[:1].to(args.device)

    with torch.no_grad():
        r = export(enc, example, args.out_dir / "vision_fp32.onnx", args.opset, dynamo)
        out, provider = ort_run(Path(r["path"]), x.numpy(), args.device)
        r.update(ort_provider=provider, parity_vs_torch_fp32=compare(out, ref))
        report["exports"]["fp32"] = r
        print(f"fp32 onnx {r['mb']} MB, ORT[{provider}] vs torch: {r['parity_vs_torch_fp32']}")

        if not args.skip_fp16:
            enc.half()
            r = export(enc, example.half(), args.out_dir / "vision_fp16.onnx", args.opset, dynamo)
            torch_fp16 = torch.cat([enc(x[i:i + 1].to(args.device).half()).float().cpu()
                                    for i in range(len(x))]).numpy()
            r["torch_fp16_vs_torch_fp32"] = compare(torch_fp16, ref)
            try:
                out, provider = ort_run(Path(r["path"]), x.numpy().astype(np.float16), args.device)
                r.update(ort_provider=provider, parity_vs_torch_fp32=compare(out.astype(np.float32), ref))
            except Exception as e:  # ORT CPU lacks fp16 kernels for some ops; TensorRT checks it instead
                r["ort_error"] = f"{type(e).__name__}: {e}"[:500]
            report["exports"]["fp16"] = r
            print(f"fp16 onnx {r['mb']} MB, torch fp16 vs fp32: {r['torch_fp16_vs_torch_fp32']}")
            print(f"  ORT: {r.get('parity_vs_torch_fp32', r.get('ort_error'))}")

    (args.out_dir / "export_report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out_dir / 'export_report.json'}")


if __name__ == "__main__":
    main()

"""Export SmolVLA as three ONNX graphs (vision, prefix, expert_step) and check them against LeRobot.

    python export_split.py --policy lerobot/smolvla_libero --out-dir onnx_split

The 10-step Euler loop stays outside the graphs: ort_flow.py runs vision once per camera, prefix once and
expert_step num_steps times. docs/SPLIT_GRAPH.md is the contract.

Per precision it writes vision_<p>.onnx, prefix_<p>.onnx and expert_step_<p>.onnx, each with its weights in
<graph>.onnx.data next to it (whatever the size, so every graph loads the same way and none can reach the
2 GB protobuf limit), then:

* split_manifest.json: graph files and every input/output name, shape and dtype, plus the config values a
  runtime needs (chunk size, steps, action dims, camera order). ort_flow.py and the C++ runtime read it.
* export_report.json: ONNX Runtime parity of each graph, and of the whole action chunk, against LeRobot's
  fp32 predict_action_chunk on one synthetic observation with the same noise; also how much padding the
  language to the fixed length moves the PyTorch actions.

fp16 is strongly typed: the VLM and the action expert are cast to fp16 in PyTorch before export, as the
profiler's fp16all variant does, so the graph carries its own casts and needs no TensorRT FP16 flag. The
small projections (state, action in/out, time MLP) stay fp32, so state, x_t, t and v_t stay fp32.

Exporter: torch.onnx.export(dynamo=True), i.e. torch.export plus onnxscript, at opset 18 by default, within
what ONNX Runtime and the TensorRT 10 parser support. The TorchScript exporter is not offered: under
transformers 5.5 it fails inside the SigLIP attention-mask code.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from export_vision import VisionEncoder, ort_providers
from ort_flow import SplitRuntime, onnx_io
from smolvla_latency import sysinfo
from smolvla_latency.loading import DEFAULT_TASK, load_config, load_policy, load_processors
from smolvla_latency.metrics import compare
from smolvla_latency.precision import apply_variant
from smolvla_latency.split_contract import (
    EXPERT_OUTPUT_NAMES,
    GRAPHS,
    MANIFEST,
    VISION_INPUT_NAMES,
    VISION_OUTPUT_NAMES,
    expert_input_names,
    prefix_input_names,
    prefix_output_names,
)
from smolvla_latency.split_graph import (
    ExpertStepGraph,
    PrefixGraph,
    ReferenceRun,
    _check_supported,
    reference_run,
    split_config,
)

PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16}


def export_graph(module, args: tuple, path: Path, input_names, output_names, opset: int) -> dict:
    """Export with static shapes, then keep every weight in one <name>.onnx.data file next to the graph."""
    t = time.perf_counter()
    data = path.with_name(path.name + ".data")
    for stale in (path, data):
        stale.unlink(missing_ok=True)
    with torch.no_grad():
        torch.onnx.export(module, args, str(path), input_names=list(input_names), output_names=list(output_names),
                          opset_version=opset, dynamo=True)
    model = onnx.load(str(path))
    path.unlink()
    data.unlink(missing_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data.name, size_threshold=1024)
    files = [p for p in (path, data) if p.exists()]
    return {"file": path.name, "files": [p.name for p in files], "seconds": round(time.perf_counter() - t, 1),
            "mb": round(sum(p.stat().st_size for p in files) / 2**20, 1),
            "opset": {o.domain or "ai.onnx": o.version for o in model.opset_import}}


def flat(arrays) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def split_modules(policy, run: ReferenceRun, precision: str) -> dict[str, tuple]:
    """(module, example args, input names, output names) per graph. fp16 casts the policy in place."""
    dtype = PRECISIONS[precision]
    if precision == "fp16":
        apply_variant(policy, "fp16all")
    ref, device = run.fixed, policy.config.device
    n_cams, n_layers = len(ref.image_tokens), len(ref.past_kv) // 2

    def dev(x, cast=None):
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        return x.to(device=device, dtype=cast or x.dtype)

    return {
        "vision": (VisionEncoder(policy.model.vlm_with_expert).eval(),
                   (dev(run.inputs["pixel_values_0"], dtype),),
                   VISION_INPUT_NAMES, VISION_OUTPUT_NAMES),
        "prefix": (PrefixGraph(policy).to(device).eval(),
                   (tuple(dev(t, dtype) for t in ref.image_tokens), dev(run.inputs["lang_tokens"]),
                    dev(run.inputs["lang_mask"]), dev(run.inputs["state"])),
                   prefix_input_names(n_cams), prefix_output_names(n_layers)),
        "expert_step": (ExpertStepGraph(policy).to(device).eval(),
                        (dev(run.noise), dev(ref.t[0]), dev(ref.prefix_pad_mask),
                         *(dev(t, dtype) for t in ref.past_kv)),
                        expert_input_names(n_layers), EXPERT_OUTPUT_NAMES),
    }


def export_precision(policy, run: ReferenceRun, out_dir: Path, precision: str, opset: int) -> dict:
    """Export the three graphs; also keep the PyTorch outputs for the same example inputs."""
    results = {}
    for name, (module, example, input_names, output_names) in split_modules(policy, run, precision).items():
        results[name] = export_graph(module, example, out_dir / f"{name}_{precision}.onnx", input_names,
                                     output_names, opset)
        with torch.no_grad():
            out = module(*example)
        out = out if isinstance(out, tuple) else (out,)
        results[name]["torch_outputs"] = [o.float().cpu().numpy() for o in out]
        print(f"{precision} {name}: {results[name]['mb']} MB in {results[name]['seconds']} s")
    return results


def check_runtime(rt: SplitRuntime, run: ReferenceRun, exported: dict) -> dict:
    """ONNX Runtime vs LeRobot fp32: each graph fed LeRobot's own split-point inputs, then the whole chunk.

    ``ort_vs_torch_same_precision`` compares each graph with its PyTorch module on the export example, which
    separates export errors from precision loss in fp16.
    """
    ref, inputs = run.fixed, run.inputs
    kv_out = prefix_output_names(len(ref.past_kv) // 2)
    tokens = [rt.image_tokens(inputs[f"pixel_values_{c}"]) for c in range(len(ref.image_tokens))]
    prefix = rt.prefix([t.cpu().numpy() for t in ref.image_tokens], inputs["lang_tokens"], inputs["lang_mask"],
                       inputs["state"])
    ref_prefix = dict(zip(kv_out, [ref.prefix_pad_mask.cpu().numpy(), *(t.cpu().numpy() for t in ref.past_kv)],
                          strict=True))
    v_t = [rt.velocity(x.cpu().numpy(), t.cpu().numpy(), ref_prefix) for x, t in zip(ref.x_t, ref.t, strict=True)]
    actions = rt.predict_action_chunk(inputs, run.noise.cpu().numpy())

    ort_on_example = {"vision": tokens[:1], "prefix": [prefix[n] for n in kv_out], "expert_step": v_t[:1]}
    graphs = {}
    for name in GRAPHS:
        info = {k: v for k, v in exported[name].items() if k != "torch_outputs"}
        info["ort_vs_torch_same_precision"] = compare(flat(ort_on_example[name]),
                                                      flat(exported[name]["torch_outputs"]))
        graphs[name] = info
    graphs["vision"]["ort_vs_lerobot_fp32"] = compare(flat(tokens), flat(t.cpu() for t in ref.image_tokens))
    graphs["prefix"]["ort_vs_lerobot_fp32"] = compare(flat(prefix[n] for n in kv_out[1:]),
                                                      flat(t.cpu() for t in ref.past_kv))
    graphs["prefix"]["prefix_pad_mask_equal"] = bool(np.array_equal(prefix["prefix_pad_mask"],
                                                                     ref_prefix["prefix_pad_mask"]))
    graphs["expert_step"]["ort_vs_lerobot_fp32"] = compare(flat(v_t), flat(t.cpu() for t in ref.v_t))
    return {"graphs": graphs, "ort_provider": rt.provider,
            "actions_vs_lerobot_fp32": compare(actions, ref.actions.cpu().numpy())}


def write_manifest(out_dir: Path, policy, run: ReferenceRun, exported: dict[str, dict], policy_path: str) -> dict:
    manifest = {
        "format": "smolvla-split",
        "version": 1,
        "policy": policy_path,
        "exporter": "torch.onnx.export(dynamo=True)",
        "torch": torch.__version__,
        "packages": {k: v for k, v in sysinfo.collect()["packages"].items() if v},
        "config": split_config(policy.config, run.fixed),
        "variants": {
            precision: {name: {"file": info["file"], "files": info["files"], "opset": info["opset"],
                               **onnx_io(out_dir / info["file"])}
                        for name, info in graphs.items()}
            for precision, graphs in exported.items()
        },
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="lerobot/smolvla_libero", help="hub id or local checkpoint dir")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cpu", help="where PyTorch traces and computes the reference")
    ap.add_argument("--out-dir", type=Path, default=Path("onnx_split"))
    ap.add_argument("--precisions", default="fp32,fp16", help=f"comma list from {list(PRECISIONS)}")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true", help="skip the ONNX Runtime parity check")
    args = ap.parse_args()
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    if unknown := [p for p in precisions if p not in PRECISIONS]:
        raise SystemExit(f"unknown precisions {unknown}; choose from {list(PRECISIONS)}")
    precisions = [p for p in PRECISIONS if p in precisions]  # fp32 first: the fp16 export casts the policy
    # Settings the graphs cannot reproduce fail here, before any weights are read; reference_run then checks the
    # built model (its number of expert layers).
    _check_supported(load_config(args.policy, args.device, args.revision))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Strict fp32 reference, as export_vision.py.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    policy, pre_saved, _, _ = load_policy(args.policy, args.device, args.revision)
    pre_fixed, _ = load_processors(policy.config, args.policy, args.device, args.revision, "max_length")
    run = reference_run(policy, pre_fixed, pre_saved, args.task, args.seed)
    cfg = policy.config
    padding = {"fixed_len": cfg.tokenizer_max_length, "saved_padding": cfg.pad_language_to,
               "saved_len": run.saved_lang_len,
               "fixed_vs_saved_actions": compare(run.fixed.actions.cpu().numpy(), run.saved.actions.cpu().numpy())}
    print(f"language padded to {padding['fixed_len']} tokens instead of {padding['saved_len']} "
          f"({cfg.pad_language_to}) moves the PyTorch actions by max_abs "
          f"{padding['fixed_vs_saved_actions']['max_abs']:.2e}")

    exported = {p: export_precision(policy, run, args.out_dir, p, args.opset) for p in precisions}
    manifest = write_manifest(args.out_dir, policy, run, exported, args.policy)

    report = {"tool": "smolvla-latency/export_split.py", "system": sysinfo.collect(), "policy": args.policy,
              "task": args.task, "seed": args.seed, "manifest": MANIFEST, "reference_tf32": False,
              "language_padding": padding, "variants": {}}
    providers = ort_providers(args.device)  # with CUDA, TF32 off: the fp32 graphs meet the strict fp32 reference
    for p in precisions:
        if args.no_check:
            report["variants"][p] = {"graphs": {n: {k: v for k, v in g.items() if k != "torch_outputs"}
                                                for n, g in exported[p].items()}}
            continue
        rt = SplitRuntime.from_dir(args.out_dir, p, providers=providers, manifest=manifest)
        report["variants"][p] = res = check_runtime(rt, run, exported[p])
        for name, g in res["graphs"].items():
            print(f"{p} {name:<11} ORT vs LeRobot fp32 max_abs {g['ort_vs_lerobot_fp32']['max_abs']:.2e}  "
                  f"vs torch {p} max_abs {g['ort_vs_torch_same_precision']['max_abs']:.2e}")
        a = res["actions_vs_lerobot_fp32"]
        print(f"{p} action chunk ORT vs LeRobot fp32: max_abs {a['max_abs']:.2e}  cos {a['cosine']:.8f}")
    (args.out_dir / "export_report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out_dir / MANIFEST} and {args.out_dir / 'export_report.json'}")


if __name__ == "__main__":
    main()

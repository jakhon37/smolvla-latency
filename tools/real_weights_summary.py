"""Collect a scripts/run_real_weights_cpu.sh run into one small JSON file (results/real_weights_cpu.json).

    python tools/real_weights_summary.py WORK_DIR --out results/real_weights_cpu.json [--rerun RERUN_DIR]

Every number is copied from the run's own reports, named in "sources" (paths relative to WORK_DIR), or computed here
from the arrays the run saved (the inputs' hashes, the cross-checks). Worst cases are over the 16 observations: the
largest max_abs, the smallest cosine, the total of non-finite values.

--rerun names a second run of some of the steps (scripts/run_real_weights_cpu.sh RERUN_DIR STEP ...), typically with
newer tools. Its reports are compared with WORK_DIR's, value by value, leaving out timings, paths and the machine and
code records; the comparison, and the provenance the rerun recorded that the first run did not, go under
"rerun". The numbers everywhere else still come from WORK_DIR.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_info import code_version, sha256_array  # noqa: E402
from smolvla_latency.metrics import compare  # noqa: E402


def dumps(obj, indent: int = 0) -> str:
    """JSON with one key per line, but lists and objects of plain values on one line."""
    pad, inner = " " * indent, " " * (indent + 1)
    flat = lambda v: not isinstance(v, (dict, list)) or not v  # noqa: E731
    if isinstance(obj, dict) and obj and not all(flat(v) for v in obj.values()):
        items = [f"{inner}{json.dumps(k)}: {dumps(v, indent + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(obj, list) and obj and not all(flat(v) for v in obj):
        return "[\n" + ",\n".join(f"{inner}{dumps(v, indent + 1)}" for v in obj) + f"\n{pad}]"
    return json.dumps(obj)


def load(work: Path, rel: str, sources: list[str]):
    path = work / rel
    if not path.exists():
        return None
    sources.append(rel)
    return json.loads(path.read_text())


def short_scope(scope: str) -> str:
    """The deepest module path and the node's name from the exporter's scope list ('...layers.11.mlp / gelu_11')."""
    try:
        parts = [p for p in ast.literal_eval(scope) if p] if scope.startswith("[") else [scope]
    except (ValueError, SyntaxError):
        return scope
    return " / ".join(parts[-2:])


def slim_tensor(t: dict | None) -> dict | None:
    if t is None:
        return None
    out = {k: t[k] for k in ("op", "dtype", "shape", "max", "min", "max_abs", "nonfinite") if k in t}
    return {"scope": short_scope(t.get("scope", "")), **out}


def probe_summary(r: dict) -> dict:
    """The probe's numbers for one graph, without its long tensor lists.

    SigLIP's tanh GELU ("activation_fn") holds the largest fp16 tensors: its cube x ** 3, which overflows, and the
    three nodes after it. They are listed apart from the largest fp16 tensor outside the GELU.
    """
    outside_gelu = next((t for t in r["top_fp16"] if "activation_fn" not in t["scope"]), None)
    inside_gelu = [t for t in r["top_fp16"] if "activation_fn" in t["scope"] and not t["nonfinite"]]
    by_op = r.get("by_op", {})
    return {
        "calls": r["calls"], "tensors": r["tensors"], "fp16_tensors": r["fp16_tensors"],
        "coverage": r.get("coverage"),
        "largest_fp16": slim_tensor(r["fp16_max_abs_tensor"]),
        "largest_fp16_outside_gelu": slim_tensor(outside_gelu) if r["fp16_tensors"] else None,
        "largest_finite_fp16_inside_gelu": [{**slim_tensor(t), "consumers": t["consumers"]} for t in inside_gelu[:6]],
        "largest_float32_tensors": [slim_tensor(t) for t in r["top"] if t["dtype"] == "float32"][:3],
        "cast_nodes_in_graph": {"to_float32": by_op.get("Cast (float32)", {}).get("tensors", 0),
                                "to_float16": by_op.get("Cast (float16)", {}).get("tensors", 0)},
        "fp16_entries_at_65504": r["fp16_entries_at_65504"],
        "nonfinite_tensors": [{**slim_tensor(t), "consumers": t["consumers"]} for t in r["nonfinite_tensors"]],
        "attention_scores_softmax_inputs": r["softmax_inputs"],
        "layernorm_inputs": r["layernorm_inputs"],
        "mask_fill_tensors": {"count": len(r["mask_fill_tensors"]),
                              "ops": sorted({t["op"] for t in r["mask_fill_tensors"]})},
        "largest_overall_excluding_mask_fill": slim_tensor(r["max_abs_tensor"]),
        "probe_vs_plain_outputs_max_abs": r["probe_vs_plain_outputs_max_abs"],
        "per_layer_max_abs": r["per_layer_max_abs"],
    }


def load_npz(work: Path, rel: str, sources: list[str]):
    path = work / rel
    if not path.exists():
        return None
    sources.append(rel)
    return dict(np.load(path))


def cross_checks(w: Path, sources: list[str]) -> dict:
    """Comparisons between the runs' action chunks (normalised), from the .actions.npz files next to the reports."""
    ref = load_npz(w, "bundles_actions.npz", sources)["actions"]
    out = {}
    p0 = load_npz(w, "p0_cpu.actions.npz", sources) or {}
    ort = {p: (load_npz(w, f"check_{p}.actions.npz", sources) or {}).get("actions") for p in ("fp32", "fp16")}
    if "fp32" in p0:
        mine = ref[p0["fp32_observations"]].reshape(p0["fp32"].shape)
        out["p0_fp32_vs_bundle_reference"] = {"bitwise_equal": bool(np.array_equal(p0["fp32"], mine)),
                                              "max_abs": float(np.abs(p0["fp32"] - mine).max())}
    if ort["fp16"] is not None:
        for v in ("fp16all", "fp16vlm", "amp", "lerobot", "bf16all", "bf16vlm"):
            if v in p0:
                idx = p0[f"{v}_observations"]
                out[f"ort_fp16_vs_torch_{v}"] = {"observations": idx.tolist(),
                                                 **compare(ort["fp16"][idx], p0[v].reshape(len(idx), *ref.shape[1:]))}
    if "fp32" in p0:
        fp32 = dict(zip(p0["fp32_observations"].tolist(), p0["fp32"].reshape(-1, *ref.shape[1:]), strict=True))
        per_frame = {}
        for v in sorted(k for k in p0 if not k.endswith("_observations")):
            idx = p0[f"{v}_observations"].tolist()
            chunks = p0[v].reshape(len(idx), *ref.shape[1:])
            per_frame[v] = {i: compare(c, fp32[i]) for i, c in zip(idx, chunks, strict=True)}
        # The P0 variants' errors, recomputed here from the chunks the run saved, for comparison with the report.
        out["p0_vs_fp32_recomputed_from_saved_chunks"] = {
            v: {"max_abs": max(e["max_abs"] for e in f.values()), "cosine_min": min(e["cosine"] for e in f.values()),
                "nonfinite": sum(e["nonfinite"] for e in f.values()), "n": len(f)} for v, f in per_frame.items()}
        # How far bf16 moves the chunk against fp16, frame by frame, where both ran.
        out["max_abs_ratio_per_frame"] = {
            f"{b}_over_{h}": {str(i): round(per_frame[b][i]["max_abs"] / per_frame[h][i]["max_abs"], 2)
                              for i in sorted(per_frame[h]) if i in per_frame[b] and per_frame[h][i]["max_abs"]}
            for b in ("lerobot", "bf16vlm", "bf16all") for h in ("fp16all", "fp16vlm", "amp")
            if b in per_frame and h in per_frame}
    return out


def input_hashes(work: Path, sources: list[str]) -> dict | None:
    """sha256 of every array of the run's inputs file: the decoded frames, the states, the demonstrated actions."""
    data = load_npz(work, "inputs/real_inputs.npz", sources)
    if data is None:
        return None
    return {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": sha256_array(v)} for k, v in data.items()}


# Keys whose values differ between two runs of the same computation: timings, paths, the machine and code records.
VOLATILE = {"seconds", "predict_seconds", "chunk_seconds", "chunk_seconds_p50", "latency_ms", "ms_per_call", "utc",
            "system", "machine", "code", "file", "onnx_dir", "inputs", "load_seconds", "formula", "tool"}


def differences(a, b, path: str = "") -> tuple[list[str], list[str]]:
    """Paths whose values differ between two reports, and paths present in only one of them."""
    diff, only = [], []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(a.keys() | b.keys()):
            if k in VOLATILE:
                continue
            if k not in a or k not in b:
                only.append(f"{path}.{k}")
                continue
            d, o = differences(a[k], b[k], f"{path}.{k}")
            diff += d
            only += o
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d, o = differences(x, y, f"{path}[{i}]")
            diff += d
            only += o
    elif a != b:
        diff.append(path or ".")
    return diff, only


def compare_rerun(work: Path, rerun: Path) -> dict:
    """Every report and array the rerun shares with the first run, compared value by value (see VOLATILE)."""
    reports, arrays = {}, {}
    for path in sorted(rerun.rglob("*")):
        rel = str(path.relative_to(rerun))
        mine = work / rel
        if not mine.is_file() or "logs" in path.parts or path.name == "real_weights_cpu.json":
            continue
        if path.suffix == ".json":
            diff, only = differences(json.loads(mine.read_text()), json.loads(path.read_text()))
            reports[rel] = {"identical": not diff, "differing": diff[:20], "n_differing": len(diff),
                            "keys_in_one_run_only": len(only)}
        elif path.suffix == ".npz":
            a, b = dict(np.load(mine)), dict(np.load(path))
            arrays[rel] = {"bit_identical": a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a),
                           "arrays": sorted(a)}
    return {"what": "the rerun's reports and arrays against the first run's, value by value; timings, paths and the "
                    "machine and code records left out",
            "all_identical": all(r["identical"] for r in reports.values())
            and all(a["bit_identical"] for a in arrays.values()),
            "reports": reports, "arrays": arrays}


def rerun_provenance(rerun: Path) -> dict:
    """What the rerun's reports recorded beyond the first run's: code, machine, decoder, base VLM, token ids, the
    probe's coverage and, if the rerun was measured (peak_memory.json), its memory use."""
    def read(rel):
        path = rerun / rel
        return json.loads(path.read_text()) if path.exists() else None

    out = {}
    selection = read("inputs/real_inputs.json")
    if selection:
        out["inputs_decoder"] = selection.get("decoder")
    bundles = read("bundles.json")
    if bundles:
        out["base_vlm"] = bundles.get("base_vlm")
        ids = {b["bundle"]: b.get("lang_token_ids") for b in bundles["bundles"]}
        out["language_token_ids"] = ids if all(v is not None for v in ids.values()) else None
        out["machine"] = bundles.get("machine")
    # The code each report recorded, one entry per distinct record with the reports that carry it.
    codes: dict[str, dict] = {}
    for path in sorted(rerun.rglob("*.json")):
        code = (json.loads(path.read_text()) or {}).get("code") if "logs" not in path.parts else None
        if isinstance(code, dict):
            entry = codes.setdefault(json.dumps(code, sort_keys=True), {"code": code, "reports": []})
            entry["reports"].append(str(path.relative_to(rerun)))
    out["code"] = list(codes.values())
    # The probe's account of every node output (reports written before it existed have none).
    coverage = {p: {g: r.get("coverage") for g, r in probe["graphs"].items()}
                for p in ("fp32", "fp16") for probe in [read(f"probe_{p}.json")] if probe}
    if coverage:
        out["probe_coverage"] = coverage
    memory = read("peak_memory.json")
    if memory:
        out["peak_memory"] = memory
    return out


def worst(items: list[dict]) -> dict:
    return {"max_abs": max(i["max_abs"] for i in items), "cosine_min": min(i["cosine"] for i in items),
            "rel_l2_max": max(i["rel_l2"] for i in items), "nonfinite": sum(i["nonfinite"] for i in items),
            "n": len(items)}


def latency(d: dict | None, keys) -> dict | None:
    if not d:
        return None
    return {k: {"p50": round(d[k]["p50"], 1), "n": d[k]["n"]} for k in keys if k in d}


def split_runtime(work: Path, p: str, sources: list[str]) -> dict | None:
    check = load(work, f"check_{p}.json", sources)
    if check is None:
        return None
    s = check["summary"]
    out = {
        "ort_threads": check["ort_threads"], "provider": check["provider"],
        "ort_flow": {
            "actions_vs_lerobot_fp32": s["actions_normalised"],
            "actions_denormalised_vs_lerobot_fp32": s["actions_denormalised"],
            "actions_max_abs_over_action_std": s["actions_max_abs_over_std"],
            "actions_max_abs_over_action_std_per_dim": s["actions_max_abs_over_std_per_dim"],
            "graphs_on_lerobot_inputs": s["graphs_on_lerobot_inputs"],
            "prefix_pad_mask_equal": s["prefix_pad_mask_equal"],
            "end_to_end_stages_vs_lerobot": s["end_to_end_stages"],
            "graph_outputs_max_abs_and_nonfinite": s["graph_outputs"],
            "lerobot_fp32_outputs_max_abs": s["lerobot_fp32_outputs"],
            "staged_run_equals_predict_action_chunk": s.get("staged_equals_predict_action_chunk"),
        },
        "actions_max_abs_per_frame": [b["actions"]["normalised"]["max_abs"] for b in check["bundles"]],
        "latency_ms_1_thread": latency(check.get("latency_ms"), ("chunk", "vision", "prefix", "expert_step")),
    }
    # ort_flow.py --bundle on every bundle: the repository's entry point, the same runtime.
    cli = [load(work, f"ort_flow_{p}/{b['bundle'].replace('.bin', '.json')}", sources) for b in check["bundles"]]
    if all(cli):
        same = all(c["vs_bundle_actions"]["max_abs"] == b["actions"]["normalised"]["max_abs"]
                   and c["vs_bundle_actions"]["cosine"] == b["actions"]["normalised"]["cosine"]
                   for c, b in zip(cli, check["bundles"], strict=True))
        out["ort_flow_cli"] = {"actions_vs_lerobot_fp32": worst([c["vs_bundle_actions"] for c in cli]),
                               "nonfinite": sum(c["nonfinite"] for c in cli),
                               "identical_to_check": same}
    cpp = [load(work, f"cpp_{p}/{b['bundle'].replace('.bin', '.json')}", sources) for b in check["bundles"]]
    if all(cpp):
        stages = [c["parity"]["pipeline_vs_lerobot"] for c in cpp]
        out["cpp"] = {
            "threads": cpp[0]["threads"], "backend": cpp[0]["backend"],
            "actions_vs_lerobot_fp32": worst([c["parity"]["actions"] for c in cpp]),
            "stages_vs_lerobot": {k: worst([s[k] for s in stages])
                                  for k in ("past_key_values", "x_t_all_steps", "v_t_all_steps")},
            "image_tokens_vs_lerobot": worst([s[f"image_tokens_{c}"] for s in stages for c in range(2)]),
            "prefix_pad_mask_equal": all(s["prefix_pad_mask_equal"] for s in stages),
            "rerun_identical": all(c["parity"]["rerun_identical"] for c in cpp),
            "euler_schedule_matches_bundle": all(c["euler_schedule_matches_bundle"] is True for c in cpp),
            # The bench writes metrics, not the chunk. Against the same reference on every bundle: the same max_abs
            # (a max of exact differences) and non-finite count, and the largest gap in mean_abs and cosine, which
            # C++ and numpy sum in different orders.
            "max_abs_and_nonfinite_equal_to_ort_flow": all(
                c["parity"]["actions"][k] == b["actions"]["normalised"][k]
                for c, b in zip(cpp, check["bundles"], strict=True) for k in ("max_abs", "nonfinite")),
            "mean_abs_and_cosine_largest_gap_to_ort_flow": max(
                abs(c["parity"]["actions"][k] - b["actions"]["normalised"][k])
                for c, b in zip(cpp, check["bundles"], strict=True) for k in ("mean_abs", "cosine")),
        }
    lat = load(work, f"cpp_latency_{p}.json", sources)
    if lat:
        out["cpp_latency_ms_1_thread"] = latency(lat["latency_ms"], ("chunk", "vision", "prefix", "expert_step",
                                                                      "inputs", "euler"))
    export = load(work, f"export_report_{p}.json", sources)
    if export:
        v = export["variants"][p]
        out["export"] = {g: {"mb": v["graphs"][g]["mb"], "seconds": v["graphs"][g]["seconds"]}
                         for g in v["graphs"]}
        out["export_report_synthetic_input"] = {
            "what": "export_split.py's own check: profile_smolvla.py's synthetic observation, seed 0",
            "ort_threads": "ONNX Runtime's default intra-op pool: export_split.py sets no thread count (see "
                           "system.ort_default_intra_op_threads)",
            "actions_vs_lerobot_fp32": v.get("actions_vs_lerobot_fp32"),
            "graphs_vs_lerobot_fp32_max_abs": {g: v["graphs"][g].get("ort_vs_lerobot_fp32", {}).get("max_abs")
                                               for g in v["graphs"]},
        }
    probe = load(work, f"probe_{p}.json", sources)
    prof = load(work, f"ort_profile_{p}.json", sources)
    if prof:
        out["ort_kernels_per_call"] = {}
        for g, r in prof["graphs"].items():
            ops = r["by_op_and_input_dtype"]
            count = lambda cond: round(sum(v["per_call"] for k, v in ops.items() if cond(k)), 1)  # noqa: E731
            matmul = ("MatMul ", "Gemm ", "FusedMatMul ")
            out["ort_kernels_per_call"][g] = {
                "all": round(r["kernels_per_call"], 1),
                "on_float16_inputs": count(lambda k: k.endswith("[float16]")),
                "on_float32_inputs": count(lambda k: k.endswith("[float]")),
                "cast": count(lambda k: k.startswith("Cast ")),
                # Float-output Cast nodes the exported graph itself holds (from the probe); the profiler's other Cast
                # kernels are casts ONNX Runtime inserted, or integer casts.
                "cast_nodes_in_graph": (sum(probe_summary(probe["graphs"][g])["cast_nodes_in_graph"].values())
                                        if probe and g in probe["graphs"] else None),
                "matmul_gemm_on_float16_inputs": count(lambda k: k.startswith(matmul) and k.endswith("[float16]")),
                "matmul_gemm_on_float32_inputs": count(lambda k: k.startswith(matmul) and k.endswith("[float]")),
                "by_op_and_input_dtype_per_call_and_ms": {k: [v["per_call"], v["ms_per_call"]] for k, v in ops.items()},
            }
    if probe:
        out["probe"] = {"bundles": probe["bundles"],
                        "graphs": {g: probe_summary(r) for g, r in probe["graphs"].items()}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rerun", type=Path, default=None, help="a second run of some steps, compared with WORK_DIR")
    args = ap.parse_args()
    w, sources = args.work, []

    bundles = load(w, "bundles.json", sources)
    sel = bundles["selection"]
    summary = {
        "what": "lerobot/smolvla_libero on 16 frames of lerobot/libero, CPU only: split-graph parity (ONNX Runtime "
                "from Python and C++) and PyTorch precision variants against LeRobot's fp32 predict_action_chunk, "
                "same noise per frame",
        "checkpoint": {"policy": bundles["policy"], "revision": bundles["revision"],
                       "stored_dtypes": bundles["checkpoint"]["stored_dtypes"],
                       "saved_load_vlm_weights": bundles["checkpoint"]["saved_load_vlm_weights"],
                       "missing_in_checkpoint": bundles["checkpoint"]["missing_in_checkpoint"]},
        "inputs": {
            "dataset": sel["dataset"], "revision": sel["revision"], "video_backend": sel["video_backend"],
            "frame_rule": sel["frame_rule"],
            "frames": [{k: f[k] for k in ("episode", "frame_index", "episode_length", "task")} for f in sel["frames"]],
            "noise": "observation i: torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(i + 1))",
            "language": "the checkpoint's preprocessor pads to max_length (48 tokens), as the prefix graph needs",
            "same_tokens_as_saved_preprocessor": all(b["same_language_tokens_as_saved_preprocessor"]
                                                     for b in bundles["bundles"]),
            "language_token_ids": ({b["bundle"]: b["lang_token_ids"] for b in bundles["bundles"]}
                                   if all("lang_token_ids" in b for b in bundles["bundles"]) else None),
            "arrays_sha256": input_hashes(w, sources),
            "decoder": sel.get("decoder"),
        },
        "action_std": bundles["action_std"],
        "reference": {"what": "LeRobot predict_action_chunk, strict fp32 (loading.load_policy), torch threads",
                      "torch_threads": bundles["torch_threads"],
                      "nonfinite": sum(b["actions_nonfinite"] for b in bundles["bundles"])},
        "sanity_fp32_vs_demonstration": {
            "what": "mean |a - demo| / action std over the first 10 actions of each chunk, after the postprocessor, "
                    "and the same for the dataset's mean action; a check that weights and preprocessing are right",
            "policy_first10": float(np.mean([b["vs_demo"]["mean_abs_over_std_first10"] for b in bundles["bundles"]])),
            "mean_action_first10": float(np.mean([b["vs_demo"]["dataset_mean_action_baseline_first10"]
                                                  for b in bundles["bundles"]])),
            "per_frame": [[round(b["vs_demo"]["mean_abs_over_std_first10"], 3),
                           round(b["vs_demo"]["dataset_mean_action_baseline_first10"], 3)] for b in bundles["bundles"]],
        },
        "split_graphs": {p: split_runtime(w, p, sources) for p in ("fp32", "fp16")},
    }
    gelu = load(w, "gelu_fp16.json", sources)
    if gelu:
        summary["split_graphs"]["fp16"]["gelu_tanh_in_fp16"] = {k: v for k, v in gelu.items() if k != "tool"}
    weights = load(w, "weights_fp16_range.json", sources)
    if weights:
        summary["stored_weights_in_fp16"] = {"note": weights["note"], "groups": weights["groups"]}

    p0 = load(w, "p0_cpu.json", sources)
    if p0:
        variants = {}
        for name, v in p0["variants"].items():
            if "error" in v:
                variants[name] = {"error": v["error"].splitlines()[0][:300]}
                continue
            variants[name] = {
                "observations": len(v["per_observation"]),
                "actions_dtype": v["per_observation"][0]["actions_dtype"],
                "vs_fp32": v["parity_vs_fp32"]["normalised"],
                "vs_fp32_max_abs_over_action_std": v["parity_vs_fp32"]["max_abs_over_std"],
                "vs_fp32_max_abs_over_action_std_per_dim": v["parity_vs_fp32"]["max_abs_over_std_per_dim"],
                "max_abs_per_frame": {str(r["observation"]): r["vs_fp32"]["normalised"]["max_abs"]
                                      for r in v["per_observation"]},
                "latency_ms_p50": {c: round(q["p50"], 1) for c, q in v["components_ms"].items()},
                "calls_per_chunk": v["calls_per_chunk"],
                "end_to_end_ms_p50": round(v["end_to_end_instrumented_ms"]["p50"], 1),
            }
        summary["p0_pytorch_cpu"] = {"torch_threads": p0["torch_threads"], "device": p0["device"],
                                     "note": "CPU latencies, one thread: not comparable to any GPU",
                                     "variants": variants}
    summary["cross_checks"] = cross_checks(w, sources)
    recomputed = summary["cross_checks"].get("p0_vs_fp32_recomputed_from_saved_chunks", {})
    for name, v in (p0 or {}).get("variants", {}).items():
        if name in recomputed and "parity_vs_fp32" in v:
            reported = v["parity_vs_fp32"]["normalised"]
            recomputed[name]["equals_report"] = all(recomputed[name][k] == reported[k]
                                                    for k in ("max_abs", "cosine_min", "nonfinite"))
    # The machine, as the run's own reports recorded it.
    machine = bundles.get("machine") or (load(w, "check_fp32.json", []) or {}).get("system", {})
    summary["system"] = {**{k: bundles["system"].get(k) for k in ("python", "torch", "packages")},
                         **{k: v for k, v in machine.items() if k != "utc"}}
    rerun = None
    if args.rerun:
        rerun = {"dir": args.rerun.name, **compare_rerun(w, args.rerun), **rerun_provenance(args.rerun)}
        # What the first run did not record, from the rerun where that is the same thing: the same CPU, the same
        # decoded inputs, the same reference actions.
        again = rerun.get("machine") or {}
        if again.get("cpu") == machine.get("cpu"):
            added = {k: v for k, v in again.items() if k not in summary["system"] and k != "utc"}
            summary["system"].update(added, recorded_by_the_rerun_on_the_same_cpu=sorted(added))
        if (rerun["arrays"].get("inputs/real_inputs.npz", {}).get("bit_identical") and rerun.get("inputs_decoder")
                and not summary["inputs"]["decoder"]):
            summary["inputs"]["decoder"] = rerun["inputs_decoder"]
            summary["inputs"]["decoder_recorded_by"] = "the rerun, whose decoded inputs are bit-identical to these"
        if (rerun["arrays"].get("bundles_actions.npz", {}).get("bit_identical") and rerun.get("language_token_ids")
                and not summary["inputs"]["language_token_ids"]):
            summary["inputs"]["language_token_ids"] = rerun["language_token_ids"]
            summary["inputs"]["language_token_ids_recorded_by"] = (
                "the rerun: same inputs, checkpoint and tokenizer revision, bit-identical reference actions")
        # The GELU check needs no model: where the rerun's report equals this run's on every value both hold, show
        # the rerun's, which may check more (and states the formula as the rerun's tool evaluates it).
        gelu_again = args.rerun / "gelu_fp16.json"
        if gelu_again.exists() and rerun["reports"].get("gelu_fp16.json", {}).get("identical"):
            summary["split_graphs"]["fp16"]["gelu_tanh_in_fp16"] = {
                **{k: v for k, v in json.loads(gelu_again.read_text()).items() if k not in ("tool", "code")},
                "recorded_by": "the rerun; its values equal this run's wherever both reports have them"}
    summary["code"] = {"summary": code_version(),
                       "reports": {s: r["code"] for s in sources if s.endswith(".json")
                                   for r in [load(w, s, [])] if isinstance(r, dict) and r.get("code")}}
    if rerun:
        summary["rerun"] = rerun
    summary["sources"] = sources
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(summary) + "\n")
    print(f"wrote {args.out} from {len(sources)} reports")


if __name__ == "__main__":
    main()

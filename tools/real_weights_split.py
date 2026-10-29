"""Split graphs on real weights and real observations: bundles, ONNX Runtime parity, and the range of every tensor.

    python tools/real_weights_split.py bundles --inputs real_inputs.npz --out-dir bundles
    python tools/real_weights_split.py weights --out weights.json
    python tools/real_weights_split.py check --onnx-dir onnx_fp32 --precision fp32 --bundles bundles --out check.json
    python tools/real_weights_split.py probe --onnx-dir onnx_fp16 --precision fp16 --bundles bundles --out probe.json

bundles  One bundle per observation from tools/real_inputs.py, in tools/dump_bundle.py's format: the graph inputs, the
         noise for seed i (profile_smolvla.py's convention: torch.Generator().manual_seed(i + 1)), LeRobot's fp32
         predict_action_chunk and its tensors at the split points. ort_flow.py --bundle and smolvla_rt_bench read
         them. Also writes bundles.json: the actions after the checkpoint's postprocessor, and how they compare with
         the dataset's demonstrated actions (a check that the weights and the preprocessing are right).
weights  The checkpoint's stored weights cast to fp16, per submodule: largest magnitude, values beyond fp16's range,
         nonzero values that become zero or subnormal.
check    ort_flow.SplitRuntime on every bundle: the action chunk against LeRobot's, normalised and after the
         checkpoint's postprocessor relative to each action dimension's std; each graph fed LeRobot's own inputs at
         its split point; each stage of the end-to-end run against LeRobot's; the largest magnitude and the
         non-finite count of every graph output.
probe    The same end-to-end run with every floating-point intermediate tensor of the three graphs added as a graph
         output: the largest magnitude and the non-finite count of each, over all bundles, cameras and steps. Each
         fp16 tensor is read as it rounds to fp16, but ONNX Runtime's CPU provider computes most nodes in fp32 and may
         hand a node's fp32 result to the next node, so an overflow seen here need not propagate as it would in an
         engine that keeps every node in fp16. Fails unless every float node output was probed (no output without an
         inferred type, no subgraphs).
ort-profile  ONNX Runtime's profiler on one chunk: which kernels run, on which input dtype, and for how long.
gelu-fp16    The tanh GELU node by node as the exporter writes it, each node's output in fp16, over every fp16 input up
             to --limit and over every finite fp16 input, against the exact function: what the overflow of its x ** 3
             term in fp16 does to the result in an engine that keeps every node in fp16.

ONNX Runtime and torch run on --threads threads (default 1). Every report records the code it came from (git commit
and the hashes of changed files) and the machine (tools/run_info.py).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ort_flow import ONNX_DTYPES, SplitRuntime, percentiles  # noqa: E402
from run_info import code_version, hub_revision, sha256_array  # noqa: E402
from run_info import system_info as run_system_info  # noqa: E402
from smolvla_latency.bundle import read_bundle, write_bundle  # noqa: E402
from smolvla_latency.metrics import compare  # noqa: E402
from smolvla_latency.split_contract import GRAPHS, MANIFEST, euler_schedule, kv_names  # noqa: E402

FP16_MAX = 65504.0
# Attention masks fill masked scores with float32's lowest value (about -3.4e38); the probe counts such entries
# apart and leaves them out of the magnitudes, which are about activations.
MASK_FILL = 1e30


def bundle_paths(directory: Path) -> list[Path]:
    paths = sorted(Path(directory).glob("bundle_*.bin"))
    if not paths:
        raise SystemExit(f"no bundle_*.bin in {directory}")
    return paths


def action_stats(post) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the actions in the postprocessor's unnormaliser (the checkpoint's normalisation stats)."""
    for step in post.steps:
        stats = getattr(step, "stats", None) or {}
        if "action" in stats:
            return (np.asarray(stats["action"]["mean"], dtype=np.float64),
                    np.asarray(stats["action"]["std"], dtype=np.float64))
    raise SystemExit("the postprocessor has no action statistics")


def postprocessor(policy: str, revision: str | None):
    """The checkpoint's postprocessor and its action std."""
    from smolvla_latency.loading import load_config, load_processors

    _, post = load_processors(load_config(policy, "cpu", revision), policy, "cpu", revision)
    return post, action_stats(post)[1]


def denormalise(post, actions: np.ndarray) -> np.ndarray:
    import torch

    return post(torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))).numpy()


def action_errors(post, std: np.ndarray, out: np.ndarray, ref: np.ndarray) -> dict:
    """Normalised comparison, then after the postprocessor: max abs per action dimension, also divided by its std."""
    res = {"normalised": compare(out, ref)}
    d_out, d_ref = denormalise(post, out), denormalise(post, ref)
    res["denormalised"] = compare(d_out, d_ref)
    per_dim = np.abs(d_out.astype(np.float64) - d_ref).reshape(-1, d_ref.shape[-1]).max(0)
    res["denormalised_max_abs_per_dim"] = per_dim.tolist()
    res["max_abs_over_std_per_dim"] = (per_dim / std).tolist()
    res["max_abs_over_std"] = float((per_dim / std).max())
    return res


def magnitude(arrays) -> dict:
    a = [np.asarray(x, dtype=np.float64) for x in arrays]
    finite = [x[np.isfinite(x)] for x in a]
    return {"max_abs": float(max((np.abs(x).max() for x in finite if x.size), default=0.0)),
            "nonfinite": int(sum((~np.isfinite(x)).sum() for x in a))}


# ---------------------------------------------------------------- bundles


def cmd_bundles(args) -> None:
    import torch
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    from real_inputs import load_observations
    from smolvla_latency import sysinfo
    from smolvla_latency.loading import load_policy, load_processors
    from smolvla_latency.split_graph import capture_reference, graph_inputs, split_config

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    observations, selection = load_observations(args.inputs)
    policy, pre_saved, post, ckpt = load_policy(args.policy, "cpu", args.revision)
    cfg = policy.config
    pre_fixed, _ = load_processors(cfg, args.policy, "cpu", args.revision, "max_length")
    mean, std = action_stats(post)
    demo_actions, demo_valid = np.load(args.inputs)["demo_actions"], np.load(args.inputs)["demo_valid"]
    dt, times = euler_schedule(cfg.num_steps)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records, reference_actions = [], []
    for i, obs in enumerate(observations):
        frame = selection["frames"][i]
        batch, batch_saved = pre_fixed(dict(obs)), pre_saved(dict(obs))
        same_language = all(torch.equal(batch[k], batch_saved[k])
                            for k in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK))
        inputs = graph_inputs(batch, cfg)
        seed = args.seed_offset + i
        noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(seed + 1))
        t0 = time.perf_counter()
        ref = capture_reference(policy, batch, noise)
        seconds = time.perf_counter() - t0
        actions = ref.actions.numpy()
        reference_actions.append(actions)

        tensors = {**inputs, "noise": noise.numpy(), "actions": actions}
        tensors.update({f"ref.image_tokens_{c}": t.numpy() for c, t in enumerate(ref.image_tokens)})
        tensors["ref.prefix_pad_mask"] = ref.prefix_pad_mask.numpy()
        names = kv_names(len(ref.past_kv) // 2)
        tensors.update({f"ref.{n}": t.numpy() for n, t in zip(names, ref.past_kv, strict=True)})
        for k, (x_t, v_t) in enumerate(zip(ref.x_t, ref.v_t, strict=True)):
            tensors[f"ref.x_t_{k}"] = x_t.numpy()
            tensors[f"ref.v_t_{k}"] = v_t.numpy()
        meta = {
            "policy": args.policy, "revision": args.revision, "task": obs["task"], "seed": seed,
            "observation": {k: frame[k] for k in ("episode", "frame_index", "dataset_index", "task_index")},
            "dataset": {"repo_id": selection["dataset"], "revision": selection["revision"]},
            "config": split_config(cfg, ref),
            "euler": {"dt": float(dt), "t": [float(t[0]) for t in times]},
            "actions": "SmolVLAPolicy.predict_action_chunk on these inputs: [1, chunk_size, action_dim], normalised",
            "torch_threads": torch.get_num_threads(),
            "packages": {k: v for k, v in sysinfo.collect()["packages"].items() if v},
            "torch": torch.__version__,
        }
        path = args.out_dir / f"bundle_{i:02d}.bin"
        write_bundle(path, tensors, meta)

        # The fp32 chunk after the postprocessor against the demonstrated actions that follow the frame, and the
        # dataset's mean action as a baseline: a check that the weights and the preprocessing are right.
        denorm = post(ref.actions.clone()).numpy()[0]
        demo, valid = demo_actions[i], demo_valid[i]
        err = np.abs(denorm[valid] - demo[valid]) / std
        base = np.abs(mean - demo[valid]) / std
        mask = np.asarray(inputs["lang_mask"]).reshape(-1).astype(bool)
        records.append({
            "index": i, "bundle": path.name, **{k: frame[k] for k in ("episode", "frame_index", "task")},
            "seed": seed, "same_language_tokens_as_saved_preprocessor": bool(same_language),
            "lang_tokens_real": int(mask.sum()),
            "lang_token_ids": np.asarray(inputs["lang_tokens"]).reshape(-1)[mask].tolist(),
            "graph_inputs_sha256": {k: sha256_array(np.asarray(v)) for k, v in sorted(inputs.items())},
            "noise_sha256": sha256_array(noise.numpy()),
            "predict_seconds": round(seconds, 2),
            "actions_normalised_max_abs": float(np.abs(actions).max()),
            "actions_nonfinite": int((~np.isfinite(actions)).sum()),
            "vs_demo": {"steps": int(valid.sum()),
                        "mean_abs_over_std_first10": float(err[:10].mean()),
                        "mean_abs_over_std_all": float(err.mean()),
                        "dataset_mean_action_baseline_first10": float(base[:10].mean()),
                        "dataset_mean_action_baseline_all": float(base.mean())},
        })
        print(f"{path.name}: episode {frame['episode']} frame {frame['frame_index']}, {seconds:.1f} s, "
              f"vs demo first 10 steps {err[:10].mean():.3f} std (mean-action baseline {base[:10].mean():.3f})")

    # The base VLM's config and tokenizer come from cfg.vlm_model_name, which the checkpoint names without a
    # revision, so they resolve to that repository's main branch: record the commit they came from.
    tokenizer = next((getattr(s, "tokenizer_name", None) for s in pre_saved.steps
                      if getattr(s, "tokenizer_name", None)), cfg.vlm_model_name)
    base_vlm = {"config_repo": cfg.vlm_model_name, "config_revision": hub_revision(cfg.vlm_model_name, "config.json"),
                "tokenizer_repo": tokenizer, "tokenizer_revision": hub_revision(tokenizer, "tokenizer_config.json"),
                "pinned": False}
    report = {"tool": "tools/real_weights_split.py bundles", "policy": args.policy, "revision": args.revision,
              "inputs": str(args.inputs), "selection": selection, "torch_threads": torch.get_num_threads(),
              "checkpoint": {k: v for k, v in ckpt.items() if k != "keys"}, "base_vlm": base_vlm,
              "action_std": std.tolist(), "system": sysinfo.collect(), "machine": run_system_info(),
              "code": code_version(), "bundles": records}
    (args.out_dir / "bundles.json").write_text(json.dumps(report, indent=1))
    np.savez(args.out_dir / "bundles_actions.npz", actions=np.concatenate(reference_actions))
    print(f"wrote {len(records)} bundles, {args.out_dir / 'bundles.json'} and bundles_actions.npz")


# ---------------------------------------------------------------- check


def staged_chunk(rt: SplitRuntime, inputs: dict, noise: np.ndarray):
    """SplitRuntime.predict_action_chunk, the same calls in the same order, keeping what crosses the split points."""
    tokens = [rt.image_tokens(inputs[f"pixel_values_{c}"]) for c in range(len(rt.config["camera_keys"]))]
    prefix = rt.prefix(tokens, inputs["lang_tokens"], inputs["lang_mask"], inputs["state"])
    x_t = np.asarray(noise, dtype=np.float32)
    xs, vs = [], []
    for t in rt.times:
        v_t = rt.velocity(x_t, t, prefix)
        xs.append(x_t)
        vs.append(v_t)
        x_t = x_t + rt.dt * v_t
    return x_t[:, :, : rt.config["action_dim"]], tokens, prefix, xs, vs


def bundle_problem(path: Path, rt: SplitRuntime):
    tensors, meta = read_bundle(path)
    if meta.get("config") != rt.config:
        raise SystemExit(f"{path} was written for another export: {meta.get('config')} vs {rt.config}")
    dt, times = euler_schedule(rt.config["num_steps"])
    if meta["euler"] != {"dt": float(dt), "t": [float(t[0]) for t in times]}:
        raise SystemExit(f"{path}: Euler schedule differs from the runtime's")
    cams = len(rt.config["camera_keys"])
    names = [f"pixel_values_{c}" for c in range(cams)] + ["lang_tokens", "lang_mask", "state"]
    return {k: tensors[k] for k in names}, tensors, meta


def cmd_check(args) -> None:
    rt = SplitRuntime.from_dir(args.onnx_dir, args.precision, threads=args.threads)
    post, std = postprocessor(args.policy, args.revision)
    n_layers = rt.config["num_layers"]
    kv = kv_names(n_layers)
    rows, all_actions = [], []
    for b, path in enumerate(bundle_paths(args.bundles)):
        inputs, tensors, meta = bundle_problem(path, rt)
        cams = len(rt.config["camera_keys"])
        t0 = time.perf_counter()
        actions, tokens, prefix, xs, vs = staged_chunk(rt, inputs, tensors["noise"])
        seconds = time.perf_counter() - t0
        all_actions.append(actions)
        row = {"bundle": path.name, "observation": meta.get("observation"), "seed": meta.get("seed"),
               "chunk_seconds": round(seconds, 2)}
        if b == 0:  # the staged run is SplitRuntime.predict_action_chunk, bit for bit
            row["staged_equals_predict_action_chunk"] = bool(
                np.array_equal(actions, rt.predict_action_chunk(inputs, tensors["noise"])))
        row["actions"] = action_errors(post, std, actions, tensors["actions"])

        ref_tokens = [tensors[f"ref.image_tokens_{c}"] for c in range(cams)]
        ref_prefix = {"prefix_pad_mask": tensors["ref.prefix_pad_mask"], **{n: tensors[f"ref.{n}"] for n in kv}}
        # Each graph fed LeRobot's own inputs at its split point (vision's input is the observation itself).
        prefix_on_ref = rt.prefix(ref_tokens, inputs["lang_tokens"], inputs["lang_mask"], inputs["state"])
        v_on_ref = [rt.velocity(tensors[f"ref.x_t_{k}"], t, ref_prefix) for k, t in enumerate(rt.times)]
        per_layer = [compare(np.concatenate([prefix_on_ref[f"past_key_{i}"].ravel(),
                                             prefix_on_ref[f"past_value_{i}"].ravel()]).astype(np.float64),
                             np.concatenate([tensors[f"ref.past_key_{i}"].ravel(),
                                             tensors[f"ref.past_value_{i}"].ravel()]))["max_abs"]
                     for i in range(n_layers)]
        row["graphs_on_lerobot_inputs"] = {
            "vision": compare(np.concatenate([t.ravel() for t in tokens]).astype(np.float64),
                              np.concatenate([t.ravel() for t in ref_tokens])),
            "prefix": compare(np.concatenate([prefix_on_ref[n].ravel() for n in kv]).astype(np.float64),
                              np.concatenate([tensors[f"ref.{n}"].ravel() for n in kv])),
            "prefix_max_abs_per_layer": per_layer,
            "prefix_pad_mask_equal": bool(np.array_equal(prefix_on_ref["prefix_pad_mask"],
                                                         tensors["ref.prefix_pad_mask"])),
            "expert_step": compare(np.concatenate([v.ravel() for v in v_on_ref]).astype(np.float64),
                                   np.concatenate([tensors[f"ref.v_t_{k}"].ravel() for k in range(len(rt.times))])),
        }
        # Stages of the end-to-end run, each carrying what upstream stages passed on.
        row["end_to_end_stages"] = {
            "image_tokens": row["graphs_on_lerobot_inputs"]["vision"],
            "past_key_values": compare(np.concatenate([prefix[n].ravel() for n in kv]).astype(np.float64),
                                       np.concatenate([tensors[f"ref.{n}"].ravel() for n in kv])),
            "x_t_all_steps": compare(np.concatenate([x.ravel() for x in xs]).astype(np.float64),
                                     np.concatenate([tensors[f"ref.x_t_{k}"].ravel() for k in range(len(xs))])),
            "v_t_all_steps": compare(np.concatenate([v.ravel() for v in vs]).astype(np.float64),
                                     np.concatenate([tensors[f"ref.v_t_{k}"].ravel() for k in range(len(vs))])),
        }
        row["graph_outputs"] = {
            "image_tokens": magnitude(tokens),
            "past_key": magnitude(prefix[f"past_key_{i}"] for i in range(n_layers)),
            "past_value": magnitude(prefix[f"past_value_{i}"] for i in range(n_layers)),
            "v_t": magnitude(vs),
            "actions": magnitude([actions]),
        }
        row["lerobot_fp32"] = {"image_tokens": magnitude(ref_tokens),
                               "past_key": magnitude(tensors[f"ref.past_key_{i}"] for i in range(n_layers)),
                               "past_value": magnitude(tensors[f"ref.past_value_{i}"] for i in range(n_layers))}
        rows.append(row)
        a = row["actions"]
        print(f"{path.name}: {seconds:.1f} s  chunk max_abs {a['normalised']['max_abs']:.2e} "
              f"cos {a['normalised']['cosine']:.9f}  max_abs/std {a['max_abs_over_std']:.2e}  "
              f"nonfinite {a['normalised']['nonfinite']}")

    report = {"tool": "tools/real_weights_split.py check", "onnx_dir": str(args.onnx_dir),
              "precision": args.precision, "provider": rt.provider, "ort_threads": args.threads,
              "policy": args.policy, "action_std": std.tolist(), "config": rt.config,
              "summary": summarise(rows), "bundles": rows}
    if args.iters:
        report["latency_ms"] = time_chunks(rt, bundle_paths(args.bundles)[0], args.warmup, args.iters)
    report["system"], report["code"] = run_system_info(), code_version()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    np.savez(args.out.with_suffix(".actions.npz"), actions=np.concatenate(all_actions))
    print(json.dumps(report["summary"], indent=1))
    print(f"wrote {args.out} and its .actions.npz")


def summarise(rows: list[dict]) -> dict:
    """Worst case over the bundles: largest max_abs and rel_l2, smallest cosine, total non-finite counts."""

    def worst(get) -> dict:
        items = [get(r) for r in rows]
        return {"max_abs": max(i["max_abs"] for i in items), "rel_l2_max": max(i["rel_l2"] for i in items),
                "cosine_min": min(i["cosine"] for i in items), "nonfinite": sum(i["nonfinite"] for i in items),
                "n": len(items)}

    out = {
        "actions_normalised": worst(lambda r: r["actions"]["normalised"]),
        "actions_denormalised": worst(lambda r: r["actions"]["denormalised"]),
        "actions_max_abs_over_std": max(r["actions"]["max_abs_over_std"] for r in rows),
        "actions_max_abs_over_std_per_dim": np.max([r["actions"]["max_abs_over_std_per_dim"] for r in rows],
                                                   0).tolist(),
        "graphs_on_lerobot_inputs": {g: worst(lambda r, g=g: r["graphs_on_lerobot_inputs"][g])
                                     for g in GRAPHS},
        "prefix_pad_mask_equal": all(r["graphs_on_lerobot_inputs"]["prefix_pad_mask_equal"] for r in rows),
        "end_to_end_stages": {s: worst(lambda r, s=s: r["end_to_end_stages"][s])
                              for s in ("image_tokens", "past_key_values", "x_t_all_steps", "v_t_all_steps")},
        "graph_outputs": {k: {"max_abs": max(r["graph_outputs"][k]["max_abs"] for r in rows),
                              "nonfinite": sum(r["graph_outputs"][k]["nonfinite"] for r in rows)}
                          for k in rows[0]["graph_outputs"]},
        "lerobot_fp32_outputs": {k: {"max_abs": max(r["lerobot_fp32"][k]["max_abs"] for r in rows)}
                                 for k in rows[0]["lerobot_fp32"]},
        "chunk_seconds_p50": float(np.median([r["chunk_seconds"] for r in rows])),
    }
    if "staged_equals_predict_action_chunk" in rows[0]:
        out["staged_equals_predict_action_chunk"] = rows[0]["staged_equals_predict_action_chunk"]
    return out


def time_chunks(rt: SplitRuntime, path: Path, warmup: int, iters: int) -> dict:
    inputs, tensors, _ = bundle_problem(path, rt)
    for _ in range(warmup):
        rt.predict_action_chunk(inputs, tensors["noise"])
    rt.timings, chunk_ms = {}, []
    for _ in range(iters):
        t = time.perf_counter()
        rt.predict_action_chunk(inputs, tensors["noise"])
        chunk_ms.append((time.perf_counter() - t) * 1e3)
    timings, rt.timings = rt.timings, None
    return {"bundle": path.name, "warmup": warmup, "chunk": percentiles(chunk_ms),
            **{k: percentiles(v) for k, v in timings.items()}}


# ---------------------------------------------------------------- weights


def cmd_weights(args) -> None:
    """How the checkpoint's stored weights fare when cast to fp16, per submodule: largest magnitude, values beyond
    fp16's range, nonzero values that flush to zero or become subnormal. Read tensor by tensor from the file."""
    import torch
    from safetensors import safe_open

    from smolvla_latency.loading import _group, checkpoint_file

    tiny, normal = 2.0**-24, 2.0**-14  # fp16's smallest subnormal and smallest normal
    groups: dict[str, dict] = {}
    with safe_open(checkpoint_file(args.policy, args.revision), framework="pt") as f:
        for key in f.keys():
            w = f.get_tensor(key)
            g = groups.setdefault(f"{_group(key)} ({str(w.dtype).replace('torch.', '')})",
                                  {"tensors": 0, "values": 0, "max_abs": 0.0, "max_abs_tensor": "",
                                   "beyond_fp16_max": 0, "flush_to_zero_in_fp16": 0, "subnormal_in_fp16": 0})
            a = w.float().abs()
            nonzero = a > 0
            g["tensors"] += 1
            g["values"] += a.numel()
            if float(a.max()) > g["max_abs"]:
                g["max_abs"], g["max_abs_tensor"] = float(a.max()), key
            g["beyond_fp16_max"] += int((a > FP16_MAX).sum())
            g["flush_to_zero_in_fp16"] += int((nonzero & (a < tiny / 2)).sum())
            g["subnormal_in_fp16"] += int((nonzero & (a >= tiny / 2) & (a < normal)).sum())
            del w, a, nonzero
    report = {"tool": "tools/real_weights_split.py weights", "policy": args.policy, "revision": args.revision,
              "note": "flush_to_zero: nonzero |w| < 2^-25, which rounds to 0 in fp16; subnormal: below 2^-14",
              "torch": torch.__version__, "code": code_version(), "groups": groups}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    for name, g in groups.items():
        print(f"{name:<28} {g['values']:>11,} values  max |w| {g['max_abs']:.4g}  beyond fp16 {g['beyond_fp16_max']}  "
              f"to zero {g['flush_to_zero_in_fp16']}  subnormal {g['subnormal_in_fp16']}")
    print(f"wrote {args.out}")


# ---------------------------------------------------------------- ort-profile


def cmd_ort_profile(args) -> None:
    """Which kernels ONNX Runtime runs, and on which input dtype, for one chunk on the first bundle (its profiler).

    For fp16 graphs this shows where the CPU provider computes in fp16 and where it inserted casts to run a node in
    fp32 because it has no fp16 kernel for it.
    """
    from collections import Counter

    import onnxruntime as ort

    manifest = json.loads((args.onnx_dir / MANIFEST).read_text())
    graphs = manifest["variants"][args.precision]
    prof_dir = args.out.parent / f"{args.out.stem}_raw"
    prof_dir.mkdir(parents=True, exist_ok=True)
    sessions = {}
    for name in GRAPHS:
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.enable_profiling = True
        so.profile_file_prefix = str(prof_dir / name)
        sessions[name] = ort.InferenceSession(str(args.onnx_dir / graphs[name]["file"]), so,
                                              providers=["CPUExecutionProvider"])
    rt = SplitRuntime(sessions, manifest["config"])
    path = bundle_paths(args.bundles)[0]
    inputs, tensors, _ = bundle_problem(path, rt)
    rt.predict_action_chunk(inputs, tensors["noise"])
    calls = {"vision": len(rt.config["camera_keys"]), "prefix": 1, "expert_step": rt.config["num_steps"]}
    report = {"tool": "tools/real_weights_split.py ort-profile", "onnx_dir": str(args.onnx_dir),
              "precision": args.precision, "ort_threads": args.threads, "bundle": path.name, "graphs": {}}
    for name, sess in sessions.items():
        raw = Path(sess.end_profiling())
        kernels: Counter = Counter()
        ms: Counter = Counter()
        for e in json.loads(raw.read_text()):
            if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time"):
                continue
            a = e.get("args", {})
            dtypes = [next(iter(d)) for d in a.get("input_type_shape", []) if d]
            floats = [d for d in dtypes if d in ("float", "float16")]
            key = f"{a.get('op_name')} [{floats[0] if floats else 'other'}]"
            kernels[key] += 1
            ms[key] += e.get("dur", 0) / 1e3
        raw.unlink()
        n = calls[name]
        report["graphs"][name] = {
            "calls": n, "kernels_per_call": sum(kernels.values()) / n,
            "by_op_and_input_dtype": {k: {"per_call": kernels[k] / n, "ms_per_call": round(ms[k] / n, 3)}
                                      for k in sorted(kernels, key=lambda k: -ms[k])},
        }
        print(f"{name}: {sum(kernels.values()) / n:.0f} kernels per call; heaviest: " + ", ".join(
            f"{k} {ms[k] / n:.1f} ms" for k in sorted(kernels, key=lambda k: -ms[k])[:6]))
    prof_dir.rmdir()
    report["system"], report["code"] = run_system_info(), code_version()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


# ---------------------------------------------------------------- gelu-fp16


GELU_TANH_NODES = "x * (0.5 * (tanh(0.7978845608028654 * (x + 0.044715 * x**3)) + 1))"


def gelu_tanh_fp16(x: np.ndarray) -> np.ndarray:
    """The tanh GELU node by node as the exporter writes it, every node's output in fp16: onnxscript's aten_gelu with
    approximate="tanh" emits Pow, Mul, Add, Mul, Tanh, Add, Mul, Mul in this order, GELU_TANH_NODES."""
    h = np.float16
    with np.errstate(over="ignore"):
        cubed = np.power(x, h(3))
        inner = h(0.044715) * cubed
        inner = x + inner
        inner = h(0.7978845608028654) * inner
        inner = np.tanh(inner)
        inner = inner + h(1)
        inner = h(0.5) * inner
        return x * inner


def cmd_gelu_fp16(args) -> None:
    """Every fp16 value in [-limit, limit] through the exported tanh GELU in fp16, against the exact tanh GELU.

    In the vision tower the cube x ** 3 overflows fp16 for |x| above ~40.3; this shows what that does to the result.
    """
    every = np.arange(np.iinfo(np.uint16).max + 1, dtype=np.uint32).astype(np.uint16).view(np.float16)
    every = every[np.isfinite(every)]
    out_every = gelu_tanh_fp16(every)
    within = np.abs(every) <= args.limit
    x, out = every[within], out_every[within]
    xd = x.astype(np.float64)
    exact = 0.5 * xd * (1 + np.tanh(np.sqrt(2 / np.pi) * (xd + 0.044715 * xd**3)))
    with np.errstate(over="ignore"):
        overflow = ~np.isfinite(np.power(x, np.float16(3)))
    err = np.abs(out.astype(np.float64) - exact)
    report = {
        "tool": "tools/real_weights_split.py gelu-fp16", "limit": args.limit, "values": int(x.size),
        "formula": f"{GELU_TANH_NODES}, node by node, each node's output in fp16 (numpy float16)",
        "cube_overflows_for_abs_x_from": float(np.abs(x[overflow]).min()) if overflow.any() else None,
        "values_whose_cube_overflows": int(overflow.sum()),
        "nonfinite_outputs": int((~np.isfinite(out)).sum()),
        "max_abs_error": float(err.max()),
        "max_abs_error_where_cube_overflows": float(err[overflow].max()) if overflow.any() else None,
        "positive_overflow_outputs_equal_x": bool(np.all(out[overflow & (x > 0)] == x[overflow & (x > 0)])),
        "negative_overflow_outputs_are_zero": bool(np.all(out[overflow & (x < 0)] == 0)),
        "finite_fp16_inputs": int(every.size),
        "nonfinite_outputs_any_finite_input": int((~np.isfinite(out_every)).sum()),
        "code": code_version(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# ---------------------------------------------------------------- probe

FLOAT_BYTES = {"float32": 4, "float16": 2}
# ONNX element types that are floating point but that the probe cannot read: double, bfloat16, the float8 and float4
# types. None occurs in the exported graphs; the probe fails if one does.
OTHER_FLOAT_TYPES = {11, 16, 17, 18, 19, 20, 23}


def node_scope(node) -> str:
    """The module scopes the exporter recorded for a node, as it wrote them (a list ending with the node's name)."""
    return next((p.value for p in node.metadata_props if p.key == "pkg.torch.onnx.name_scopes"), "")


def float_intermediates(path: Path):
    """The graph at ``path`` without its weights; every float tensor a node computes that is not a graph output; and
    an account of every node output, so the probe can show that it read every float intermediate tensor.

    Not probed, by design: Constant nodes (fixed values), graph outputs (the check reads them) and integer or bool
    tensors. Gaps, which make the probe fail: a node output without an inferred tensor type, a float type the probe
    cannot read, a node holding a subgraph (If, Loop, Scan) and a call to a model-local function, whose inner
    tensors are not graph values.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    info = {v.name: v for v in [*inferred.graph.value_info, *inferred.graph.output]}
    outputs = {o.name for o in model.graph.output}
    functions = {(f.domain, f.name) for f in model.functions}
    consumers: dict[str, set] = {}
    for node in model.graph.node:
        for i in node.input:
            consumers.setdefault(i, set()).add(node.op_type)
    gaps: dict[str, list] = {"untyped_outputs": [], "other_float_outputs": [], "subgraph_nodes": [],
                             "function_calls": []}
    counts = {"nodes": len(model.graph.node), "constant_nodes": 0, "node_outputs": 0, "graph_outputs": 0,
              "non_float_outputs": 0, "probed_float_outputs": 0}
    tensors = []
    for node in model.graph.node:
        if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for a in node.attribute):
            gaps["subgraph_nodes"].append(f"{node.op_type} {node.name}")
        if (node.domain, node.op_type) in functions:
            gaps["function_calls"].append(f"{node.domain}.{node.op_type} {node.name}")
        if node.op_type == "Constant":
            counts["constant_nodes"] += 1
            continue
        for out in node.output:
            if not out:  # an optional output the node does not produce
                continue
            counts["node_outputs"] += 1
            if out in outputs:
                counts["graph_outputs"] += 1
                continue
            vi = info.get(out)
            elem = vi.type.tensor_type.elem_type if vi is not None and vi.type.HasField("tensor_type") else 0
            if elem == 0:
                gaps["untyped_outputs"].append(out)
                continue
            if elem in OTHER_FLOAT_TYPES:
                gaps["other_float_outputs"].append(out)
                continue
            if ONNX_DTYPES.get(elem) not in FLOAT_BYTES:
                counts["non_float_outputs"] += 1
                continue
            dtype = ONNX_DTYPES[elem]
            dims = [d.dim_value if d.HasField("dim_value") else 0 for d in vi.type.tensor_type.shape.dim]
            tensors.append({"name": out, "op": node.op_type, "scope": node_scope(node), "dtype": dtype, "shape": dims,
                            "consumers": sorted(consumers.get(out, ())),
                            "bytes": FLOAT_BYTES[dtype] * int(np.prod(dims)), "value_info": vi})
            counts["probed_float_outputs"] += 1
    coverage = {**counts, **{k: len(v) for k, v in gaps.items()},
                "gap_examples": {k: v[:10] for k, v in gaps.items() if v}, "model_local_functions": len(functions)}
    coverage["complete"] = not any(gaps.values())
    return model, tensors, coverage


def groups_by_bytes(tensors: list[dict], budget: int) -> list[list[dict]]:
    groups, cur, size = [], [], 0
    for t in tensors:
        if cur and size + t["bytes"] > budget:
            groups.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += t["bytes"]
    return groups + ([cur] if cur else [])


def cmd_probe(args) -> None:
    import onnx
    import onnxruntime as ort

    graphs = json.loads((args.onnx_dir / MANIFEST).read_text())["variants"][args.precision]
    # 0. Which tensors the probe reads, before anything runs: every float node output, or the probe stops here.
    coverage = {name: float_intermediates(args.onnx_dir / graphs[name]["file"])[2] for name in GRAPHS}
    for name, c in coverage.items():
        print(f"{name}: {c['probed_float_outputs']} float node outputs to probe; {c['node_outputs']} node outputs, "
              f"{c['graph_outputs']} graph outputs, {c['non_float_outputs']} integer or bool, "
              f"{c['constant_nodes']} Constant nodes; complete: {c['complete']}")
    if not all(c["complete"] for c in coverage.values()):
        raise SystemExit(f"the probe would miss float tensors: {json.dumps(coverage, indent=1)}")
    rt = SplitRuntime.from_dir(args.onnx_dir, args.precision, threads=args.threads)
    paths = bundle_paths(args.bundles)
    if args.bundle_indices:
        paths = [paths[int(i)] for i in args.bundle_indices.split(",")]

    # 1. The plain end-to-end run on each bundle, keeping every graph call's inputs and outputs.
    calls: dict[str, list] = {name: [] for name in GRAPHS}
    plain_run, dtypes = rt._run, rt.input_dtypes

    def keep(name, feed):
        outs = plain_run(name, feed)
        calls[name].append(({k: np.asarray(v, dtype=dtypes[name][k]) for k, v in feed.items()}, outs))
        return outs

    rt._run = keep
    chunks = []
    for path in paths:
        inputs, tensors, _ = bundle_problem(path, rt)
        actions = rt.predict_action_chunk(inputs, tensors["noise"])
        chunks.append({"bundle": path.name, "vs_lerobot_fp32": compare(actions, tensors["actions"]),
                       "actions": magnitude([actions])})
    rt.sessions.clear()  # free the plain graphs before the probe copies load

    # 2. Per graph, the same calls again on copies of the graph that also output a group of intermediate tensors.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    report = {"tool": "tools/real_weights_split.py probe", "onnx_dir": str(args.onnx_dir),
              "precision": args.precision, "ort_threads": args.threads, "bundles": [p.name for p in paths],
              "chunks": chunks, "fp16_max": FP16_MAX, "graphs": {}}
    every = {}
    for name in GRAPHS:
        path = args.onnx_dir / graphs[name]["file"]
        model, tensors, _ = float_intermediates(path)
        stats = {t["name"]: {"max": -np.inf, "min": np.inf, "max_abs": 0.0, "nonfinite": 0, "mask_fill": 0,
                             "at_fp16_max": 0, "calls": 0} for t in tensors}
        same_outputs = 0.0
        groups = groups_by_bytes(tensors, args.group_mb * 2**20)
        t0 = time.perf_counter()
        for g, group in enumerate(groups):
            probe = onnx.ModelProto()
            probe.CopyFrom(model)
            probe.graph.output.extend(t["value_info"] for t in group)
            probe_path = path.with_name(f"{path.stem}.probe{g}.onnx")  # next to the graph: same .onnx.data file
            onnx.save_model(probe, str(probe_path))
            try:
                sess = ort.InferenceSession(str(probe_path), opts, providers=["CPUExecutionProvider"])
                names = [o.name for o in sess.get_outputs()]
                for feed, plain in calls[name]:
                    outs = dict(zip(names, sess.run(None, feed), strict=True))
                    for k, v in plain.items():
                        if v.dtype != np.bool_:
                            same_outputs = max(same_outputs, float(np.abs(outs[k].astype(np.float64) - v).max()))
                    for t in group:
                        x = outs[t["name"]].astype(np.float32)
                        s = stats[t["name"]]
                        finite = np.isfinite(x)
                        s["nonfinite"] += int(x.size - finite.sum())
                        s["calls"] += 1
                        xf = x[finite]
                        fill = np.abs(xf) >= MASK_FILL
                        s["mask_fill"] += int(fill.sum())
                        if t["dtype"] == "float16":
                            s["at_fp16_max"] += int((np.abs(xf) >= FP16_MAX).sum())
                        xv = xf[~fill]
                        if xv.size:
                            s["max"], s["min"] = max(s["max"], float(xv.max())), min(s["min"], float(xv.min()))
                            s["max_abs"] = max(s["max_abs"], float(np.abs(xv).max()))
                del sess
            finally:
                probe_path.unlink(missing_ok=True)
        rows = [{**{k: v for k, v in t.items() if k != "value_info"}, **stats[t["name"]]} for t in tensors]
        every[name] = rows
        report["graphs"][name] = {"calls": len(calls[name]), "groups": len(groups),
                                  "seconds": round(time.perf_counter() - t0, 1), "coverage": coverage[name],
                                  "probe_vs_plain_outputs_max_abs": same_outputs, **summarise_probe(rows, args.top)}
        g = report["graphs"][name]
        fp16 = f"{g['fp16_max_abs']:.4g}" if g["fp16_max_abs"] is not None else "none"
        print(f"{name}: {g['tensors']} tensors over {g['calls']} calls, largest |x| {g['max_abs']:.4g} "
              f"({g['max_abs_tensor']['op']}), largest fp16 |x| {fp16}, "
              f"non-finite in {len(g['nonfinite_tensors'])} tensors, probe vs plain outputs {same_outputs:.2e}")
    report["system"], report["code"] = run_system_info(), code_version()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float))
    if args.keep_tensors:
        args.out.with_name(args.out.stem + "_tensors.json").write_text(json.dumps(every, default=float))
    print(f"wrote {args.out}")


LAYER = re.compile(r"layers\.(\d+)")


def summarise_probe(rows: list[dict], top: int) -> dict:
    fp16 = [r for r in rows if r["dtype"] == "float16"]
    by_op: dict[str, dict] = {}
    for r in rows:
        o = by_op.setdefault(f"{r['op']} ({r['dtype']})", {"tensors": 0, "max_abs": 0.0, "nonfinite": 0})
        o["tensors"] += 1
        o["max_abs"] = max(o["max_abs"], r["max_abs"])
        o["nonfinite"] += r["nonfinite"]
    biggest = max(rows, key=lambda r: r["max_abs"])
    biggest16 = max(fp16, key=lambda r: r["max_abs"]) if fp16 else None

    def fed_to(op: str) -> dict:
        sel = [r for r in rows if op in r["consumers"]]
        return {"tensors": len(sel), "dtypes": sorted({r["dtype"] for r in sel}),
                "max": max((r["max"] for r in sel), default=None), "min": min((r["min"] for r in sel), default=None),
                "max_abs": max((r["max_abs"] for r in sel), default=None),
                "nonfinite": sum(r["nonfinite"] for r in sel)}

    return {
        "tensors": len(rows), "fp16_tensors": len(fp16),
        "max_abs": biggest["max_abs"], "max_abs_tensor": slim(biggest),
        "fp16_max_abs": biggest16["max_abs"] if biggest16 else None,
        "fp16_max_abs_tensor": slim(biggest16) if biggest16 else None,
        "fp16_headroom": FP16_MAX / biggest16["max_abs"] if biggest16 and biggest16["max_abs"] else None,
        "fp16_entries_at_65504": sum(r["at_fp16_max"] for r in fp16),
        "nonfinite_tensors": [slim(r) for r in rows if r["nonfinite"]],
        "mask_fill_tensors": [{k: r[k] for k in ("name", "op", "dtype", "consumers", "mask_fill")}
                              for r in rows if r["mask_fill"]],
        "softmax_inputs": fed_to("Softmax"),
        "layernorm_inputs": fed_to("LayerNormalization"),
        "by_op": dict(sorted(by_op.items(), key=lambda kv: -kv[1]["max_abs"])),
        "top": [slim(r) for r in sorted(rows, key=lambda r: -r["max_abs"])[:top]],
        "top_fp16": [slim(r) for r in sorted(fp16, key=lambda r: -r["max_abs"])[:top]],
        "per_layer_max_abs": per_layer(rows),
    }


def per_layer(rows: list[dict]) -> dict:
    layers: dict[str, float] = {}
    for r in rows:
        m = LAYER.search(r["scope"] or "")
        if m:
            layers[m.group(1)] = max(layers.get(m.group(1), 0.0), r["max_abs"])
    return dict(sorted(layers.items(), key=lambda kv: int(kv[0])))


def slim(r: dict) -> dict:
    return {k: r[k] for k in ("name", "op", "scope", "dtype", "shape", "consumers", "max", "min", "max_abs",
                              "nonfinite", "mask_fill", "at_fp16_max", "calls")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundles", help="LeRobot fp32 references on the real observations")
    b.add_argument("--inputs", type=Path, required=True, help="tools/real_inputs.py output (.npz)")
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--seed-offset", type=int, default=0, help="observation i uses seed offset + i")
    wt = sub.add_parser("weights", help="the stored weights' range in fp16")
    wt.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check", help="ort_flow.SplitRuntime against the bundles")
    p = sub.add_parser("probe", help="largest magnitude of every intermediate tensor")
    op = sub.add_parser("ort-profile", help="ONNX Runtime's kernels and their input dtypes for one chunk")
    ge = sub.add_parser("gelu-fp16", help="the exported tanh GELU evaluated in fp16 over its whole input range")
    ge.add_argument("--limit", type=float, default=1024.0, help="largest |x| checked")
    ge.add_argument("--out", type=Path, required=True)
    for s in (c, p, op):
        s.add_argument("--onnx-dir", type=Path, required=True)
        s.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
        s.add_argument("--bundles", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
    c.add_argument("--iters", type=int, default=0, help="then time this many chunks on the first bundle")
    c.add_argument("--warmup", type=int, default=1)
    p.add_argument("--bundle-indices", default="", help="comma list of bundle positions (default: all)")
    p.add_argument("--group-mb", type=int, default=768, help="intermediate outputs fetched per session, MB")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--keep-tensors", action="store_true", help="also write every tensor's numbers")
    for s in (b, wt, c, p, op):
        s.add_argument("--policy", default="lerobot/smolvla_libero")
        s.add_argument("--revision", default=None)
        s.add_argument("--threads", type=int, default=1, help="torch and ONNX Runtime intra-op threads")
    args = ap.parse_args()
    {"bundles": cmd_bundles, "weights": cmd_weights, "check": cmd_check, "probe": cmd_probe,
     "ort-profile": cmd_ort_profile, "gelu-fp16": cmd_gelu_fp16}[args.cmd](args)


if __name__ == "__main__":
    main()

"""The P0 precision variants on real weights and real observations, on the CPU: deviation from fp32 and latency.

    python tools/real_weights_p0.py --policy lerobot/smolvla_libero --inputs real_inputs.npz --out p0_cpu.json

profile_smolvla.py's method, on the observations from tools/real_inputs.py instead of its synthetic one:
- the fp32 reference is loading.load_policy's strict fp32 build, and every variant is cast from it by
  precision.apply_variant, with precision.autocast_ctx and precision.backend_flags around the call;
- observation i gets the same noise in every variant, torch.Generator().manual_seed(seed_offset + i + 1);
- instrument.instrument_policy times the components, which instrument.exclusive makes add up.
Each variant runs --warmup untimed calls on the first observation, then every observation once; the component
times are over those calls. The --subset-variants run only --subset-observations, without warm-up: meant for the
slow ones, as torch's fp16 matrix products are far slower than fp32 on CPUs without AVX512-FP16 or AMX.
Errors are against the fp32 chunk of the same observation: normalised, and after the checkpoint's postprocessor
relative to each action dimension's std. A variant that raises is recorded with its error and the run goes on.

These are CPU numbers, torch on --threads threads (default 1): they say nothing about a GPU.
"""

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_inputs import load_observations  # noqa: E402
from real_weights_split import action_errors, action_stats  # noqa: E402
from run_info import code_version, system_info  # noqa: E402
from smolvla_latency import sysinfo  # noqa: E402
from smolvla_latency.instrument import COMPONENTS, exclusive, instrument_policy  # noqa: E402
from smolvla_latency.loading import load_policy  # noqa: E402
from smolvla_latency.precision import VARIANTS, apply_variant, autocast_ctx, backend_flags, dtype_report  # noqa: E402
from smolvla_latency.timing import RegionTimer, summarize  # noqa: E402

CPU_VARIANTS = "fp32,lerobot,bf16vlm,bf16all,fp16vlm,fp16all,amp,lerobot_amp"


def worst(items: list[dict]) -> dict:
    return {"max_abs": max(i["max_abs"] for i in items), "rel_l2_max": max(i["rel_l2"] for i in items),
            "cosine_min": min(i["cosine"] for i in items), "nonfinite": sum(i["nonfinite"] for i in items),
            "n": len(items)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="lerobot/smolvla_libero")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--inputs", type=Path, required=True, help="tools/real_inputs.py output (.npz)")
    ap.add_argument("--variants", default=CPU_VARIANTS, help="comma list; fp32 always runs first")
    ap.add_argument("--observations", type=int, default=0, help="only the first N observations (default: all)")
    ap.add_argument("--subset-variants", default="", help="variants that run only --subset-observations")
    ap.add_argument("--subset-observations", default="1,5,9,13", help="comma list of observation indices")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1, help="torch intra-op threads")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if unknown := [v for v in variants if v not in VARIANTS]:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    variants = ["fp32"] + [v for v in variants if v != "fp32"]
    subset_variants = {v.strip() for v in args.subset_variants.split(",") if v.strip()}
    if "fp32" in subset_variants:
        raise SystemExit("fp32 is the reference for every observation; it cannot be a subset variant")
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    observations, selection = load_observations(args.inputs)
    if args.observations:
        observations = observations[: args.observations]
    t0 = time.perf_counter()
    policy, pre, post, ckpt = load_policy(args.policy, "cpu", args.revision)
    load_seconds = time.perf_counter() - t0
    cfg = policy.config
    _, std = action_stats(post)
    ref_state = {k: v.detach().to("cpu", copy=True) for k, v in policy.state_dict().items()}
    noises = [torch.randn(1, cfg.chunk_size, cfg.max_action_dim,
                          generator=torch.Generator().manual_seed(args.seed_offset + i + 1))
              for i in range(len(observations))]

    timer = RegionTimer(device)
    instrument_policy(policy, timer)

    def call(variant: str, i: int) -> torch.Tensor:
        policy.reset()
        with timer.region("preprocess"):
            batch = pre(dict(observations[i]))
        with torch.no_grad(), autocast_ctx(variant, device):
            actions = policy.predict_action_chunk(batch, noise=noises[i].clone())
        with timer.region("postprocess"):
            post(actions)
        return actions

    payload = {
        "tool": "tools/real_weights_p0.py", "args": {k: str(v) for k, v in vars(args).items()},
        "device": "cpu", "torch_threads": torch.get_num_threads(), "load_seconds": round(load_seconds, 1),
        "system": sysinfo.collect(), "machine": system_info(), "code": code_version(),
        "selection": selection, "action_std": std.tolist(),
        "checkpoint": ckpt, "variants": {},
    }
    ref_actions: dict[int, np.ndarray] = {}
    saved_actions: dict[str, np.ndarray] = {}
    for variant in variants:
        print(f"\n== {variant}: {VARIANTS[variant]}", flush=True)
        res: dict = {"variant": asdict(VARIANTS[variant])}
        try:
            with backend_flags(variant) as flags:
                res["backend_flags"] = flags
                policy.to(torch.float32)
                policy.load_state_dict(ref_state, strict=True)
                apply_variant(policy, variant)
                res["live_dtypes"] = dtype_report(policy)
                timer.enabled = True
                timer.rows.clear()
                subset = variant in subset_variants
                indices = ([int(i) for i in args.subset_observations.split(",")] if subset
                           else list(range(len(observations))))
                res["observations"], res["warmup"] = indices, 0 if subset else args.warmup
                for _ in range(res["warmup"]):
                    call(variant, 0)
                    timer.end_call(keep=False)
                per_obs = []
                for i in indices:
                    t = time.perf_counter()
                    actions = call(variant, i)
                    wall = (time.perf_counter() - t) * 1e3
                    timer.end_call(keep=True)
                    a = actions.float().cpu().numpy()
                    if variant == "fp32":
                        ref_actions[i] = a
                    row = {"observation": i, "wall_ms": wall, "actions_dtype": str(actions.dtype),
                           "vs_fp32": action_errors(post, std, a, ref_actions[i]), "actions": a}
                    per_obs.append(row)
                    e = row["vs_fp32"]
                    print(f"  obs {i:2d}: {wall / 1e3:6.1f} s  max_abs {e['normalised']['max_abs']:.2e}  "
                          f"cos {e['normalised']['cosine']:.7f}  max_abs/std {e['max_abs_over_std']:.2e}  "
                          f"nonfinite {e['normalised']['nonfinite']}", flush=True)
            rows = [exclusive(r) for r in timer.rows]
            res["components_ms"] = {c: summarize([r[c] for r in rows]) for c in COMPONENTS}
            res["end_to_end_instrumented_ms"] = summarize([r["end_to_end"] for r in rows])
            res["calls_per_chunk"] = {k[2:]: timer.rows[0][k] for k in timer.rows[0] if k.startswith("n_")}
            res["parity_vs_fp32"] = {
                "normalised": worst([r["vs_fp32"]["normalised"] for r in per_obs]),
                "denormalised": worst([r["vs_fp32"]["denormalised"] for r in per_obs]),
                "max_abs_over_std": max(r["vs_fp32"]["max_abs_over_std"] for r in per_obs),
                "max_abs_over_std_per_dim": np.max([r["vs_fp32"]["max_abs_over_std_per_dim"] for r in per_obs],
                                                   0).tolist(),
            }
            res["per_observation"] = per_obs
            saved_actions[variant] = np.concatenate([r.pop("actions") for r in per_obs])
            saved_actions[f"{variant}_observations"] = np.asarray(indices)
            e2e = res["end_to_end_instrumented_ms"]
            print(f"  end-to-end p50 {e2e['p50'] / 1e3:.2f} s; " + ", ".join(
                f"{c} {res['components_ms'][c]['p50']:.0f}" for c in COMPONENTS) + " ms (p50)", flush=True)
        except Exception as e:  # keep going, as profile_smolvla.py does
            res["error"] = f"{type(e).__name__}: {e}"
            res["traceback"] = traceback.format_exc()
            print(f"FAILED: {res['error']}", flush=True)
            if variant == "fp32":
                raise
        payload["variants"][variant] = res
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=1))  # after every variant, so a stopped run keeps the rest
        np.savez(args.out.with_suffix(".actions.npz"), **saved_actions)  # normalised chunks, [observations, C, A]
    print(f"\nwrote {args.out} and its .actions.npz")


if __name__ == "__main__":
    main()

"""Per-component latency of SmolVLA across precision variants, with action-chunk parity vs fp32.

    python profile_smolvla.py --policy lerobot/smolvla_libero --iters 1000

Batch size 1, one action chunk per call, same observation and same flow-matching noise for
every variant. Writes one JSON file with system info, stored checkpoint dtypes, live dtypes
per variant, per-component p50/p90/p99, uninstrumented end-to-end latency, peak memory, GPU
clocks under load and the action-chunk error of each variant against the fp32 reference.
"""

import argparse
import json
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from smolvla_latency import sysinfo
from smolvla_latency.instrument import COMPONENTS, exclusive, instrument_policy
from smolvla_latency.loading import DEFAULT_TASK, load_policy, synthetic_observation
from smolvla_latency.metrics import compare
from smolvla_latency.precision import (
    DEFAULT_VARIANTS,
    VARIANTS,
    apply_variant,
    autocast_ctx,
    backend_flags,
    dtype_report,
)
from smolvla_latency.timing import RegionTimer, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="lerobot/smolvla_libero", help="hub id or local checkpoint dir")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help=f"comma list from {list(VARIANTS)}; fp32 always runs first as the reference")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    variants = ["fp32"] + [v for v in variants if v != "fp32"]

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    meta = sysinfo.collect()
    print(json.dumps({k: meta[k] for k in meta if k != "packages"}, indent=1))

    t0 = time.perf_counter()
    policy, pre, post, ckpt = load_policy(args.policy, args.device, args.revision)
    print(f"loaded {args.policy} in {time.perf_counter() - t0:.1f}s; "
          f"saved load_vlm_weights={ckpt['saved_load_vlm_weights']}; stored dtypes={ckpt['stored_dtypes']}")
    if ckpt["missing_in_checkpoint"]:
        print(f"WARNING: {len(ckpt['missing_in_checkpoint'])} model keys missing from checkpoint, "
              f"e.g. {ckpt['missing_in_checkpoint'][:3]}")

    cfg = policy.config
    ref_state = {k: v.detach().to("cpu", copy=True) for k, v in policy.state_dict().items()}
    obs = synthetic_observation(cfg, args.task, args.seed)
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g).to(device)

    timer = RegionTimer(device)
    instrument_policy(policy, timer)

    def call(variant: str) -> torch.Tensor:
        policy.reset()
        with timer.region("preprocess"):
            batch = pre(dict(obs))
        with torch.no_grad(), autocast_ctx(variant, device):
            actions = policy.predict_action_chunk(batch, noise=noise.clone())
        with timer.region("postprocess"):
            post(actions)
        return actions

    def run_variant(variant: str, res: dict) -> torch.Tensor:
        policy.to(torch.float32)
        policy.load_state_dict(ref_state, strict=True)
        apply_variant(policy, variant)
        res["live_dtypes"] = dtype_report(policy)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        timer.enabled = True
        timer.rows.clear()
        for _ in range(args.warmup):
            call(variant)
            timer.end_call(keep=False)
        actions = None
        for _ in range(args.iters):
            actions = call(variant)
            timer.end_call(keep=True)
        rows = [exclusive(r) for r in timer.rows]
        res["components_ms"] = {c: summarize([r[c] for r in rows]) for c in COMPONENTS}
        res["end_to_end_instrumented_ms"] = summarize([r["end_to_end"] for r in rows])
        res["calls_per_chunk"] = {k[2:]: timer.rows[0][k] for k in timer.rows[0] if k.startswith("n_")}

        timer.enabled = False
        clean = []
        for _ in range(args.iters):
            timer.sync()
            t = time.perf_counter()
            call(variant)
            timer.sync()
            clean.append((time.perf_counter() - t) * 1e3)
        res["end_to_end_ms"] = summarize(clean)
        res["instrumentation_overhead_p50_ms"] = res["end_to_end_instrumented_ms"]["p50"] - res["end_to_end_ms"]["p50"]
        if device.type == "cuda":
            res["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            res["gpu_clocks_after"] = sysinfo.gpu_clocks()
        return actions

    results = {}
    ref_actions = None
    for variant in variants:
        print(f"\n== {variant}: {VARIANTS[variant]}")
        res: dict = {"variant": asdict(VARIANTS[variant])}
        try:
            with backend_flags(variant) as flags:
                res["backend_flags"] = flags
                actions = run_variant(variant, res)
            if variant == "fp32":
                ref_actions = actions
            res["parity_vs_fp32"] = compare(actions.float().cpu().numpy(), ref_actions.float().cpu().numpy())
            e2e = res["end_to_end_ms"]
            print(f"end-to-end p50 {e2e['p50']:.2f} ms  p99 {e2e['p99']:.2f} ms  "
                  f"max|da| {res['parity_vs_fp32']['max_abs']:.2e}  cos {res['parity_vs_fp32']['cosine']:.6f}")
            for c in COMPONENTS:
                q = res["components_ms"][c]
                print(f"  {c:<20} p50 {q['p50']:9.3f}  p99 {q['p99']:9.3f}")
        except Exception as e:  # keep going so one unsupported dtype does not lose the run
            res["error"] = f"{type(e).__name__}: {e}"
            res["traceback"] = traceback.format_exc()
            print(f"FAILED: {res['error']}")
            if variant == "fp32":
                raise
        results[variant] = res

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gpu = (meta.get("gpu") or "cpu").replace(" ", "_")
    out = args.out or Path("results") / f"profile_{gpu}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "smolvla-latency/profile_smolvla.py",
        "utc": stamp,
        "args": {k: str(v) for k, v in vars(args).items()},
        "system": meta,
        "policy_config": {
            "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps, "num_steps": cfg.num_steps,
            "num_vlm_layers": cfg.num_vlm_layers, "resize_imgs_with_padding": list(cfg.resize_imgs_with_padding),
            "image_features": {k: list(v.shape) for k, v in cfg.image_features.items()},
            "vlm_model_name": cfg.vlm_model_name,
        },
        "checkpoint": ckpt,
        "variants": results,
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

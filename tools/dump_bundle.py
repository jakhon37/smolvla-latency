"""Write split-graph runtime inputs, flow-matching noise and LeRobot's reference actions to a bundle file.

    python tools/dump_bundle.py --policy lerobot/smolvla_libero --out bundle.bin

The bundle is what a runtime other than Python needs to check itself against PyTorch: it feeds
pixel_values_<c>, lang_tokens, lang_mask and state to the graphs from export_split.py, runs the Euler loop
from `noise`, and compares with `actions`, which is SmolVLAPolicy.predict_action_chunk on the same inputs
(normalised action space, before the postprocessor). The ref.* tensors are LeRobot's values at the split
points (image tokens, prefix K/V, x_t and v_t of every step), to locate a mismatch.
The observation is profile_smolvla.py's synthetic one for --seed, with its noise convention.

File layout: smolvla_latency/bundle.py. Tensor list: docs/SPLIT_GRAPH.md. Check a bundle with
    python ort_flow.py --onnx-dir onnx_split --bundle bundle.bin
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smolvla_latency import sysinfo  # noqa: E402
from smolvla_latency.bundle import write_bundle  # noqa: E402
from smolvla_latency.loading import DEFAULT_TASK, load_policy, load_processors  # noqa: E402
from smolvla_latency.metrics import compare  # noqa: E402
from smolvla_latency.split_contract import euler_schedule, kv_names  # noqa: E402
from smolvla_latency.split_graph import reference_run, split_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="lerobot/smolvla_libero", help="hub id or local checkpoint dir")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-intermediates", action="store_true", help="omit the ref.* tensors")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy, pre_saved, _, _ = load_policy(args.policy, "cpu", args.revision)
    cfg = policy.config
    pre_fixed, _ = load_processors(cfg, args.policy, "cpu", args.revision, "max_length")
    run = reference_run(policy, pre_fixed, pre_saved, args.task, args.seed)
    ref = run.fixed

    tensors = {**run.inputs, "noise": run.noise.numpy(), "actions": ref.actions.numpy()}
    if not args.no_intermediates:
        tensors.update({f"ref.image_tokens_{c}": t.numpy() for c, t in enumerate(ref.image_tokens)})
        tensors["ref.prefix_pad_mask"] = ref.prefix_pad_mask.numpy()
        names = kv_names(len(ref.past_kv) // 2)
        tensors.update({f"ref.{name}": t.numpy() for name, t in zip(names, ref.past_kv, strict=True)})
        for k, (x_t, v_t) in enumerate(zip(ref.x_t, ref.v_t, strict=True)):
            tensors[f"ref.x_t_{k}"] = x_t.numpy()
            tensors[f"ref.v_t_{k}"] = v_t.numpy()

    dt, times = euler_schedule(cfg.num_steps)
    meta = {
        "policy": args.policy,
        "task": args.task,
        "seed": args.seed,
        "config": split_config(cfg, ref),
        "euler": {"dt": float(dt), "t": [float(t[0]) for t in times]},
        "actions": "SmolVLAPolicy.predict_action_chunk on these inputs: [1, chunk_size, action_dim], normalised",
        "fixed_vs_saved_padding_actions": {"saved_padding": cfg.pad_language_to, "saved_len": run.saved_lang_len,
                                           **compare(ref.actions.numpy(), run.saved.actions.numpy())},
        "packages": {k: v for k, v in sysinfo.collect()["packages"].items() if v},
        "torch": torch.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_bundle(args.out, tensors, meta)
    print(f"wrote {args.out}: {len(tensors)} tensors, {args.out.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()

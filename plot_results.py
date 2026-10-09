"""Latency breakdown chart and markdown table from one or more profile_smolvla.py results.

    python plot_results.py results/profile_*.json --out results/breakdown

Writes <out>.png, <out>.svg and <out>.md. Each bar is one (GPU, variant); segments are the
median time of each component, so segments need not add up exactly to the end-to-end median
printed at the bar's end (that number comes from the uninstrumented run).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical slots 1-5 (validated adjacent, light surface). Three sit below 3:1
# contrast, so every value also appears in the markdown table written next to the chart.
GROUPS = [
    ("Vision encoder", ["vision_encoder"], "#2a78d6"),
    ("VLM prefix (KV cache)", ["vlm_prefix"], "#eb6834"),
    ("Action expert, 10 flow steps", ["expert_forward", "denoise_other"], "#1baf7a"),
    ("Pre/post-processing", ["preprocess", "prepare_images", "postprocess"], "#eda100"),
    ("Other glue", ["embed_prefix_other", "sample_other", "policy_other"], "#e87ba4"),
]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def rows_from(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        d = json.loads(p.read_text())
        gpu = d["system"].get("gpu") or "CPU"
        for name, v in d["variants"].items():
            if "error" in v:
                rows.append({"gpu": gpu, "variant": name, "error": v["error"]})
                continue
            comp = {c: s["p50"] for c, s in v["components_ms"].items()}
            row = {"gpu": gpu, "variant": name,
                   "e2e_p50": v["end_to_end_ms"]["p50"], "e2e_p99": v["end_to_end_ms"]["p99"],
                   "cos": v["parity_vs_fp32"]["cosine"], "max_abs": v["parity_vs_fp32"]["max_abs"],
                   "peak_mb": v.get("peak_mem_mb")}
            for label, keys, _ in GROUPS:
                row[label] = sum(comp.get(k, 0.0) for k in keys)
            rows.append(row)
    return rows


def table(rows: list[dict]) -> str:
    head = ["GPU", "variant", "e2e p50 ms", "e2e p99 ms", *[g[0] for g in GROUPS], "cos vs fp32", "max abs", "peak MB"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['gpu']} | {r['variant']} | failed: {r['error'][:80]} |" + " |" * (len(head) - 3))
            continue
        cells = [r["gpu"], r["variant"], f"{r['e2e_p50']:.1f}", f"{r['e2e_p99']:.1f}",
                 *[f"{r[g[0]]:.1f}" for g in GROUPS], f"{r['cos']:.6f}", f"{r['max_abs']:.2e}",
                 "" if r["peak_mb"] is None else f"{r['peak_mb']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot(rows: list[dict], out: Path) -> None:
    ok = [r for r in rows if "error" not in r]
    multi_gpu = len({r["gpu"] for r in ok}) > 1
    labels = [f"{r['gpu']} · {r['variant']}" if multi_gpu else r["variant"] for r in ok]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(ok) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = list(range(len(ok)))[::-1]
    left = [0.0] * len(ok)
    for label, _, color in GROUPS:
        widths = [r[label] for r in ok]
        ax.barh(y, widths, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        left = [a + b for a, b in zip(left, widths)]
    for yi, r, end in zip(y, ok, left):
        ax.text(end, yi, f"  {r['e2e_p50']:.1f} ms", va="center", ha="left", color=INK, fontsize=9)
    ax.set_yticks(y, labels, color=INK, fontsize=9)
    ax.set_xlabel("milliseconds per action chunk (median per component, batch 1)", color=INK_2, fontsize=9)
    ax.set_xlim(0, max(left) * 1.18)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    gpus = sorted({r["gpu"] for r in ok})
    ax.set_title(f"SmolVLA latency by component · {', '.join(gpus)}", loc="left", color=INK, fontsize=11)
    leg = ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18 - 0.02 * len(ok)))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=160, facecolor=SURFACE, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("results/breakdown"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = rows_from(args.results)
    md = table(rows)
    args.out.with_suffix(".md").write_text(md)
    print(md)
    plot(rows, args.out)
    print(f"wrote {args.out}.png/.svg/.md")


if __name__ == "__main__":
    main()

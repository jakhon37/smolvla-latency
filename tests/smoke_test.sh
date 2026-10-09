#!/usr/bin/env bash
# Offline end-to-end check on CPU: no Hub access, no GPU, random weights.
#
#   bash tests/smoke_test.sh [PYTHON] [--full]
#
# --full uses the real SmolVLM2-500M layer sizes (slower; peak about 5-6 GB RAM) so the ONNX export
# sees production shapes; the default tiny model takes about a minute.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-$HOME/venvs/smolvla/bin/python}
SIZE=--tiny
[ "${2:-}" = "--full" ] && SIZE=""
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export HF_HUB_OFFLINE=1

"$PY" tests/make_fake_smolvlm.py "$WORK/vlm" $SIZE
"$PY" tests/make_fake_policy.py --vlm "$WORK/vlm" --out "$WORK/policy" --bf16-flag
# lerobot_amp is CUDA-only: CPU autocast leaves the bf16 LayerNorm weights uncast ("mixed dtype").
"$PY" profile_smolvla.py --policy "$WORK/policy" --warmup 1 --iters 3 --out "$WORK/profile.json" \
  --variants fp32,lerobot,bf16vlm,fp16vlm,fp16all,bf16all,amp
"$PY" plot_results.py "$WORK/profile.json" --out "$WORK/breakdown"
"$PY" export_vision.py --policy "$WORK/policy" --out-dir "$WORK/onnx" --num-check 2
"$PY" - "$WORK" <<'EOF'
import json, sys
w = sys.argv[1]
p = json.load(open(f"{w}/profile.json"))
assert p["checkpoint"]["saved_load_vlm_weights"] is True
assert not p["checkpoint"]["missing_in_checkpoint"], p["checkpoint"]["missing_in_checkpoint"]
for name, v in p["variants"].items():
    assert "error" not in v, (name, v.get("error"))
    assert v["calls_per_chunk"]["expert_forward"] == 10.0, v["calls_per_chunk"]
    assert v["calls_per_chunk"]["vision_encoder"] == 2.0, v["calls_per_chunk"]
    assert v["parity_vs_fp32"]["nonfinite"] == 0
assert p["variants"]["fp32"]["parity_vs_fp32"]["max_abs"] == 0.0
v = p["variants"]
assert v["bf16vlm"]["live_dtypes"]["text_model"].keys() == {"bfloat16"}
assert v["bf16vlm"]["live_dtypes"]["lm_expert"].keys() == {"float32"}
assert v["lerobot"]["live_dtypes"]["lm_expert"].keys() == {"bfloat16", "float32"}  # cross-attn k/v stay fp32
assert v["lerobot"]["backend_flags"]["cudnn_benchmark"] is True
assert v["fp32"]["backend_flags"]["matmul_allow_tf32"] is False
e = json.load(open(f"{w}/onnx/export_report.json"))
assert e["exports"]["fp32"]["parity_vs_torch_fp32"]["cosine"] > 0.99999, e["exports"]["fp32"]
print("profile/export checks passed")
EOF
"$PY" tests/check_lerobot_dtypes.py "$WORK/dtypes"
echo "smoke test passed"

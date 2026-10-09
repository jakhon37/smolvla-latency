#!/usr/bin/env bash
# Run the whole P0 measurement on one GPU machine (Kaggle, Colab, a lab server).
#
#   bash scripts/run_p0.sh [POLICY] [ITERS]
#
# POLICY defaults to lerobot/smolvla_libero, ITERS to 1000. Uses $PY (default
# $HOME/venvs/smolvla/bin/python from setup_env.sh). Everything lands in results/<gpu>_<utc>/.
set -euo pipefail
cd "$(dirname "$0")/.."
POLICY=${1:-lerobot/smolvla_libero}
ITERS=${2:-1000}
PY=${PY:-$HOME/venvs/smolvla/bin/python}
# Name the folder after the device torch will use (respects CUDA_VISIBLE_DEVICES).
GPU=$("$PY" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" | tr ' ' '_')
OUT=results/${GPU:-cpu}_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
echo "policy=$POLICY iters=$ITERS out=$OUT"

bash scripts/env_check.sh "$PY" | tee "$OUT/env_check.txt"

"$PY" profile_smolvla.py --policy "$POLICY" --iters "$ITERS" --out "$OUT/profile.json" | tee "$OUT/profile.log"
"$PY" plot_results.py "$OUT/profile.json" --out "$OUT/breakdown"

"$PY" export_vision.py --policy "$POLICY" --out-dir "$OUT/onnx" | tee "$OUT/export.log"
if "$PY" -c "import tensorrt" 2>/dev/null; then
  "$PY" trt_vision.py --onnx "$OUT/onnx/vision_fp16.onnx" --iters "$ITERS" --torch-baseline --policy "$POLICY" \
    | tee "$OUT/trt_fp16.log"
  "$PY" trt_vision.py --onnx "$OUT/onnx/vision_fp32.onnx" --iters "$ITERS" | tee "$OUT/trt_fp32.log"
else
  echo "tensorrt not installed; skipping engine builds" | tee "$OUT/trt_skipped.txt"
fi
# Engines and ONNX weights are large and machine-specific; keep the reports, drop the blobs.
find "$OUT/onnx" \( -name '*.engine' -o -name '*.onnx.data' \) -delete
echo "done: $OUT"

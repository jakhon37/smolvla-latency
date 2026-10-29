#!/usr/bin/env bash
# Build the C++ runtime and check it against PyTorch on the tiny random test policy, offline on CPU.
#
#   bash tests/cpp_parity.sh [PYTHON]
#
# The policy's VLM has two key/value heads, so a K/V layout mistake changes the actions. Exports its split graphs
# (fp32 and fp16), dumps a bundle with LeRobot's reference actions, then runs the C++ unit and parity tests
# (ctest), smolvla_rt_bench with a 1e-5 max-abs gate on the fp32 chunk, and, when PYTHON has pybind11, the Python
# module's tests against ort_flow.py. Needs cmake, a C++17 compiler and network access for ONNX Runtime and the
# CMake-fetched test dependencies on the first run.
#   ONNXRUNTIME_ROOT       use this ONNX Runtime SDK instead of fetching one into cpp/third_party/onnxruntime
#   SMOLVLA_RT_BUILD_DIR   build directory (default cpp/build)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-$HOME/venvs/smolvla/bin/python}
BUILD=${SMOLVLA_RT_BUILD_DIR:-cpp/build}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export HF_HUB_OFFLINE=1

if [ -n "${ONNXRUNTIME_ROOT:-}" ]; then
  ORT_ROOT=$ONNXRUNTIME_ROOT
else
  ORT_ROOT=cpp/third_party/onnxruntime
  bash cpp/scripts/fetch_onnxruntime.sh --dest "$ORT_ROOT"  # returns at once when that version is already there
fi
ORT_ROOT=$(cd "$ORT_ROOT" && pwd)

"$PY" tests/make_fake_smolvlm.py "$WORK/vlm" --tiny --kv-heads 2
"$PY" tests/make_fake_policy.py --vlm "$WORK/vlm" --out "$WORK/policy"
"$PY" export_split.py --policy "$WORK/policy" --out-dir "$WORK/onnx" --precisions fp32,fp16
"$PY" tools/dump_bundle.py --policy "$WORK/policy" --out "$WORK/onnx/bundle.bin"

PYTHON_MODULE=OFF
if "$PY" -c "import pybind11" 2>/dev/null; then
  PYTHON_MODULE=ON
else
  echo "pybind11 is not installed for $PY: building without the Python module"
fi
cmake -S cpp -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$ORT_ROOT" -DSMOLVLA_RT_WERROR=ON \
  -DSMOLVLA_RT_BUILD_PYTHON=$PYTHON_MODULE -DPython_EXECUTABLE="$PY"
cmake --build "$BUILD" -j"$(nproc)"
BUILD=$(cd "$BUILD" && pwd)

export SMOLVLA_RT_FIXTURE_DIR="$WORK/onnx"
(cd "$BUILD" && ctest --output-on-failure --no-tests=error)
# ctest reports a parity test that found no fixture as skipped, not failed; here every one of them has to run.
PARITY=$("$BUILD/tests/smolvla_rt_parity_tests")
echo "$PARITY" | grep -E "max_abs|PASSED"
if echo "$PARITY" | grep -q SKIPPED; then
  echo "parity tests were skipped" >&2
  exit 1
fi

"$BUILD/smolvla_rt_bench" --onnx-dir "$WORK/onnx" --bundle "$WORK/onnx/bundle.bin" --max-abs 1e-5 \
  --warmup 2 --iters 10 --out "$WORK/bench_fp32.json"
"$BUILD/smolvla_rt_bench" --onnx-dir "$WORK/onnx" --bundle "$WORK/onnx/bundle.bin" --precision fp16 \
  --min-cosine 0.999 --iters 0 --out "$WORK/bench_fp16.json"
"$PY" - "$WORK" <<'EOF'
import json, sys
w = sys.argv[1]
fp32 = json.load(open(f"{w}/bench_fp32.json"))
assert fp32["passed"] and fp32["euler_schedule_matches_bundle"] is True, fp32
assert fp32["config"]["kv_heads"] == 2, "with one K/V head a K/V layout mistake leaves the actions unchanged"
assert fp32["parity"]["rerun_identical"] and fp32["parity"]["pipeline_vs_lerobot"]["prefix_pad_mask_equal"]
assert fp32["latency_ms"]["expert_step"]["n"] == 10 * fp32["config"]["num_steps"]
assert json.load(open(f"{w}/bench_fp16.json"))["passed"]
print("bench reports checked")
EOF

if [ "$PYTHON_MODULE" = ON ]; then
  SMOLVLA_RT_PYTHON_DIR="$BUILD/python" "$PY" -m pytest tests/test_cpp_runtime.py -q -rs -p no:cacheprovider
fi
echo "cpp parity passed"

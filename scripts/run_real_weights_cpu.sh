#!/usr/bin/env bash
# Real weights on the CPU: lerobot/smolvla_libero on 16 frames of lerobot/libero, the run behind
# results/REAL_WEIGHTS_CPU.md. The checkpoint and the dataset come from the Hugging Face Hub at pinned revisions. The
# checkpoint's base VLM repository (HuggingFaceTB/SmolVLM2-500M-Video-Instruct: its config and tokenizer) is named
# without a revision, so it comes from that repository's main branch; bundles.json records the commit it resolved to.
#
#   bash scripts/run_real_weights_cpu.sh PYTHON WORK_DIR [STEP ...]
#
# Steps, in this order by default:
#   inputs   tools/real_inputs.py: 4 episodes x 4 frames of lerobot/libero (both cameras, state, task)
#   bundles  tools/real_weights_split.py bundles: LeRobot's fp32 predict_action_chunk on each frame, one bundle each;
#            and real_weights_split.py weights: the stored weights' range in fp16
#   cpp      fetch ONNX Runtime 1.31.0 and build smolvla_rt_bench into WORK_DIR (skipped if SMOLVLA_RT_BENCH is set)
#   fp32     export_split.py --precisions fp32, then on every bundle: real_weights_split.py check (ort_flow.py's
#            runtime), ort_flow.py --bundle and smolvla_rt_bench; real_weights_split.py ort-profile on one bundle
#            and probe on four; then the export is deleted, so only one export is on disk at a time (1.5 GB fp32,
#            0.76 GB fp16)
#   fp16     the same with the fp16 graphs, the probe on every bundle; and real_weights_split.py gelu-fp16: the
#            vision tower's tanh GELU, node by node as exported, evaluated in fp16 over every finite fp16 input
#   p0       tools/real_weights_p0.py: the precision variants in PyTorch on the same frames and noise; the ones
#            that compute in fp16 on four frames only (torch's fp16 matmuls are slow on CPUs without AVX512-FP16)
#   summary  tools/real_weights_summary.py: WORK_DIR/real_weights_cpu.json
#   clean    delete SCRATCH_DIR (not run by default)
# torch and ONNX Runtime run on one thread, except export_split.py's own parity check: it sets no thread count, so
# ONNX Runtime uses its default intra-op pool, as many threads as cores, whose worker threads pin themselves to
# cores and so escape taskset. (The run behind results/ ran under taskset -c 3 with an LD_PRELOAD shim that limits
# affinity requests to the starting CPU set, which kept those workers on core 3.) Every report records the git
# commit and the hashes of changed code files, and the machine. Reports and logs go to WORK_DIR; the bulky files
# (the dataset's video files while the frames are extracted, the exports, the bundles, the C++ build) to
# SCRATCH_DIR, default WORK_DIR/scratch, at most about 2 GB at a time (a tmpfs such as /dev/shm keeps them off the
# disk). Besides that, the Hub cache holds the 0.9 GB checkpoint. Peak memory: see results/REAL_WEIGHTS_CPU.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=$1
W=$(mkdir -p "$2" && cd "$2" && pwd)
shift 2
T=$(mkdir -p "${SCRATCH_DIR:-$W/scratch}" && cd "${SCRATCH_DIR:-$W/scratch}" && pwd)
STEPS=${*:-inputs bundles cpp fp32 fp16 p0 summary}
POLICY=${POLICY:-lerobot/smolvla_libero}
REVISION=${REVISION-a24f1f380461b665158d52acd140152c79e8b36d}  # the checkpoint's revision; empty for a local dir
REV=()
[ -n "$REVISION" ] && REV=(--revision "$REVISION")
PROBE_FP32=1,5,9,13                           # one frame from each episode for the fp32 probe
BENCH=${SMOLVLA_RT_BENCH:-$T/cpp_build/smolvla_rt_bench}
mkdir -p "$W/logs"

run_precision() {
  local P=$1 D=$T/onnx_$1
  "$PY" export_split.py --policy "$POLICY" "${REV[@]}" --precisions "$P" --out-dir "$D" \
    2>&1 | tee "$W/logs/export_$P.log"
  cp "$D/export_report.json" "$W/export_report_$P.json"
  cp "$D/split_manifest.json" "$W/split_manifest_$P.json"
  "$PY" tools/real_weights_split.py check --policy "$POLICY" "${REV[@]}" --onnx-dir "$D" --precision "$P" \
    --bundles "$T/bundles" --out "$W/check_$P.json" --iters 3 --warmup 1 2>&1 | tee "$W/logs/check_$P.log"
  mkdir -p "$W/ort_flow_$P" "$W/cpp_$P"
  for b in "$T"/bundles/bundle_*.bin; do
    "$PY" ort_flow.py --onnx-dir "$D" --bundle "$b" --precision "$P" --threads 1 \
      --out "$W/ort_flow_$P/$(basename "$b" .bin).json"
    if [ -x "$BENCH" ]; then
      # No --max-abs gate here: the reports carry the numbers, and the summary checks them.
      "$BENCH" --onnx-dir "$D" --bundle "$b" --precision "$P" --threads 1 --iters 0 --max-abs -1 \
        --out "$W/cpp_$P/$(basename "$b" .bin).json"
    fi
  done 2>&1 | tee "$W/logs/runtimes_$P.log"
  if [ -x "$BENCH" ]; then
    "$BENCH" --onnx-dir "$D" --bundle "$T/bundles/bundle_00.bin" --precision "$P" --threads 1 --max-abs -1 \
      --warmup 1 --iters 3 --out "$W/cpp_latency_$P.json" 2>&1 | tee "$W/logs/cpp_latency_$P.log"
  fi
  "$PY" tools/real_weights_split.py ort-profile --policy "$POLICY" "${REV[@]}" --onnx-dir "$D" --precision "$P" \
    --bundles "$T/bundles" --out "$W/ort_profile_$P.json" 2>&1 | tee "$W/logs/ort_profile_$P.log"
  local SUBSET=""
  if [ "$P" = fp32 ]; then
    SUBSET="--bundle-indices $PROBE_FP32"
  else
    "$PY" tools/real_weights_split.py gelu-fp16 --out "$W/gelu_fp16.json" > "$W/logs/gelu_fp16.log"
  fi
  # shellcheck disable=SC2086
  "$PY" tools/real_weights_split.py probe --policy "$POLICY" "${REV[@]}" --onnx-dir "$D" --precision "$P" \
    --bundles "$T/bundles" --out "$W/probe_$P.json" $SUBSET 2>&1 | tee "$W/logs/probe_$P.log"
  rm -rf "$D"
}

for STEP in $STEPS; do
  echo "== $STEP $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  case $STEP in
    inputs)
      "$PY" tools/real_inputs.py --root "$T/dataset" --out "$W/inputs/real_inputs.npz" 2>&1 | tee "$W/logs/inputs.log"
      rm -rf "$T/dataset"  # the frames are in the npz; the video files are no longer needed
      ;;
    bundles)
      "$PY" tools/real_weights_split.py bundles --policy "$POLICY" "${REV[@]}" \
        --inputs "$W/inputs/real_inputs.npz" --out-dir "$T/bundles" 2>&1 | tee "$W/logs/bundles.log"
      cp "$T/bundles/bundles.json" "$T/bundles/bundles_actions.npz" "$W/"
      "$PY" tools/real_weights_split.py weights --policy "$POLICY" "${REV[@]}" --out "$W/weights_fp16_range.json" \
        2>&1 | tee "$W/logs/weights.log"
      ;;
    cpp)
      if [ -z "${SMOLVLA_RT_BENCH:-}" ]; then
        bash cpp/scripts/fetch_onnxruntime.sh --dest "$T/deps/onnxruntime"
        cmake -S cpp -B "$T/cpp_build" -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$T/deps/onnxruntime" \
          -DSMOLVLA_RT_BUILD_TESTS=OFF -DSMOLVLA_RT_WERROR=ON
        cmake --build "$T/cpp_build" --target smolvla_rt_bench -j"${JOBS:-1}"
      fi
      ;;
    fp32 | fp16)
      run_precision "$STEP"
      ;;
    p0)
      "$PY" tools/real_weights_p0.py --policy "$POLICY" "${REV[@]}" --inputs "$W/inputs/real_inputs.npz" \
        --variants fp32,lerobot,bf16vlm,bf16all,fp16vlm,fp16all,amp,lerobot_amp \
        --subset-variants fp16vlm,fp16all,amp,lerobot_amp --subset-observations 1,5,9,13 \
        --out "$W/p0_cpu.json" 2>&1 | tee "$W/logs/p0.log"
      ;;
    summary)
      "$PY" tools/real_weights_summary.py "$W" --out "$W/real_weights_cpu.json"
      ;;
    clean)
      rm -rf "$T"
      ;;
    *)
      echo "unknown step $STEP" >&2
      exit 2
      ;;
  esac
done
echo "done: $W"

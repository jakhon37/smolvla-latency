# smolvla-latency

Where does [SmolVLA](https://huggingface.co/papers/2506.01844)'s time go on a small GPU, and what do lower precisions and TensorRT buy without changing its actions?

SmolVLA is Hugging Face's 450M-parameter vision-language-action policy in [LeRobot](https://github.com/huggingface/lerobot). This repo measures one action-chunk call at batch size 1, split into components:

| Component | What it is |
|---|---|
| Vision encoder | SigLIP tower + connector, once per camera |
| VLM prefix | 16 SmolVLM layers over image, language and state tokens; fills the KV cache |
| Action expert | 10 flow-matching steps of the smaller expert transformer reading that cache |
| Pre/post-processing | tokenise, normalise, resize-with-pad, unnormalise |
| Other glue | masks, Euler updates, Python overhead |

The study then:
- runs each precision variant on the same observation and the same flow-matching noise;
- reports latency (p50/p90/p99) and the action-chunk error against fp32;
- exports the vision encoder to ONNX and TensorRT, checking parity before claiming any speed-up.

Status: tooling complete and smoke-tested offline on CPU. Checked on the CPU with the real `lerobot/smolvla_libero` weights and LIBERO frames: see [Real weights (CPU)](#real-weights-cpu). GPU numbers pending.

## What the code says before any measurement (LeRobot 0.6.1, transformers 5.5)

These come from the source, not from measurement.
- `tests/check_lerobot_dtypes.py` confirms the dtype layout against LeRobot itself: same per-parameter dtypes, and bit-identical actions on a test model.
- The dtype rules depend on `load_vlm_weights`. LeRobot's LIBERO, Meta-World and RoboCasa training commands pass `true`, so fine-tuned checkpoints usually save it.

1. **With `load_vlm_weights=true`, the VLM runs in bf16.**
   - `SmolVLMWithExpertModel` first loads the base VLM with `torch_dtype="bfloat16"` ([smolvlm_with_expert.py#L90-L96](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L90-L96)).
   - The checkpoint is then copied into those bf16 parameters.
2. **The action expert follows the VLM into bf16, with one exception.**
   - It is built from a copy of the VLM's text config. Under transformers 5, `from_config` uses that config's dtype, so the expert is bf16 too.
   - The exception: the cross-attention layers' `k_proj`/`v_proj` are replaced with plain `nn.Linear` after construction, so they stay fp32.
3. **The small projections are always fp32.** `state_proj`, `action_in_proj`, `action_out_proj` and the time MLP are fp32, and `denoise_step` casts to fp32 before `action_out_proj`.
   - A plain `policy.half()` therefore crashes.
   - The variants below cast only the VLM and the expert.
4. **`lerobot-eval` sets some global flags.** It always sets `cudnn.benchmark=True` and TF32 matmul. With `--policy.use_amp=true` it adds fp16 autocast, which on such checkpoints runs over the bf16 weights.
5. **On Turing (T4), bf16 has no hardware support.** On compute capability 7.5, `torch.cuda.is_bf16_supported(including_emulation=False)` is `False`. How much the bf16 VLM and expert cost there is the first thing this repo measures.

## Variants

| Name | VLM | Expert | Note |
|---|---|---|---|
| `fp32` | fp32 | fp32 | Reference for parity. Strict fp32: TF32 off |
| `lerobot` | bf16 | bf16, cross-attn k/v fp32 | What `lerobot-eval` runs for a `load_vlm_weights=true` checkpoint, with its backend flags |
| `lerobot_amp` | same | same | Plus fp16 autocast (`--policy.use_amp=true`). CUDA only: on CPU, autocast leaves the bf16 LayerNorm weights uncast |
| `bf16vlm` | bf16 | fp32 | Not a layout LeRobot builds |
| `fp16vlm` | fp16 | fp32 | |
| `fp16all` | fp16 | fp16 | |
| `bf16all` | bf16 | bf16 | Opt-in. Everything bf16, including cross-attn k/v |
| `amp` | fp32 + autocast fp16 | fp32 + autocast fp16 | `use_amp` on a `load_vlm_weights=false` checkpoint, without LeRobot's backend flags |

## Run it

### On a GPU machine (lab server, Colab, Kaggle)

```bash
git clone https://github.com/jakhon37/smolvla-latency && cd smolvla-latency
bash scripts/setup_env.sh                 # Python 3.12 venv in ~/venvs/smolvla: torch cu128, LeRobot 0.6.1, TensorRT 10.16
bash scripts/run_p0.sh                    # env check, profile, ONNX export, TensorRT fp16 + fp32
```

Everything lands in `results/<gpu>_<utc>/`:

| File | Contents |
|---|---|
| `env_check.txt` | Machine, driver, bf16 support, headless MuJoCo rendering |
| `profile.json` | Full per-component and per-variant results |
| `breakdown.png`, `breakdown.svg`, `breakdown.md` | Chart plus the same numbers as a table |
| `onnx/export_report.json` | ONNX export and ONNX Runtime parity |
| `onnx/trt_report_*.json` | TensorRT build, parity and latency vs PyTorch |

On a shared server:
- Pick an idle GPU first, e.g. `CUDA_VISIBLE_DEVICES=1 bash scripts/run_p0.sh`.
- Another job on the same GPU shows up in p99.
- `setup_env.sh` picks cu126 torch wheels for GPUs below compute capability 7.5 (P100, V100), because the cu128 wheels do not cover them. TensorRT 10 targets Turing and newer, so on such GPUs only the PyTorch profile applies.

On Kaggle: turn on internet and a GPU, then run [`notebooks/kaggle_p0.ipynb`](notebooks/kaggle_p0.ipynb). It puts the venv in `/tmp`, so it doesn't count against the 20 GB output limit.

### Individual steps

```bash
PY=~/venvs/smolvla/bin/python
$PY profile_smolvla.py --policy lerobot/smolvla_libero --iters 1000   # default: fp32,lerobot,lerobot_amp,bf16vlm,fp16vlm,fp16all,amp
$PY export_vision.py   --policy lerobot/smolvla_libero --out-dir onnx
$PY trt_vision.py      --onnx onnx/vision_fp16.onnx --torch-baseline      # strongly typed FP16 (TensorRT 10 and 11)
$PY trt_vision.py      --onnx onnx/vision_fp32.onnx --mode fp16-flag       # weakly typed FP16, TensorRT 10 only
$PY plot_results.py    results/*/profile.json --out results/compare        # several GPUs on one chart
```

### Offline smoke test (no GPU, no Hub)

```bash
bash tests/smoke_test.sh ~/venvs/smolvla/bin/python          # tiny random model, a few minutes
bash tests/smoke_test.sh ~/venvs/smolvla/bin/python --full   # SmolVLM2-500M layer sizes, about 5-6 GB RAM
~/venvs/smolvla/bin/python -m pytest tests                   # split-graph parity tests, tiny model, about 3 minutes
bash tests/cpp_parity.sh ~/venvs/smolvla/bin/python          # C++ runtime vs PyTorch, tiny model, a few minutes
```

GitHub Actions runs the first, third and fourth on a CPU runner for every push ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Split-graph export

`export_split.py` exports SmolVLA as three ONNX graphs. A runtime outside Python (C++, TensorRT) can then run a whole action chunk, with the 10-step Euler loop in its own code:

| Graph | Runs | In → out (SmolVLM2-500M sizes, two LIBERO cameras) |
|---|---|---|
| `vision` | once per camera | `pixel_values [1,3,512,512]` → image tokens `[1,64,960]` |
| `prefix` | once per chunk | image tokens, language padded to 48 tokens, state → prefix mask and post-RoPE K/V of the 16 VLM layers |
| `expert_step` | 10 times per chunk | `x_t`, `t`, prefix mask, K/V → velocity `v_t [1,50,32]` |

```bash
$PY export_split.py --policy lerobot/smolvla_libero --out-dir onnx_split       # fp32 + strongly typed fp16, parity report
$PY ort_flow.py --onnx-dir onnx_split --policy lerobot/smolvla_libero --check   # ONNX Runtime chunk vs predict_action_chunk
$PY tools/dump_bundle.py --policy lerobot/smolvla_libero --out onnx_split/bundle.bin   # inputs, noise, reference actions for other runtimes
```

- `ort_flow.py` is the reference runtime: ONNX Runtime for the graphs, numpy for LeRobot's Euler update, the same noise as PyTorch.
- With random weights, offline, the fp32 action chunk matches `predict_action_chunk` to 4.8e-7 on the tiny test model and to 7e-7 at SmolVLM2-500M sizes. The tests require 1e-5.
- With the real `lerobot/smolvla_libero` weights on 16 LIBERO frames, it matches to 6.3e-6, and the fp16 graphs to 2.0e-3. See [Real weights (CPU)](#real-weights-cpu).
- The tiny test model has two key/value heads (`make_fake_smolvlm.py --tiny --kv-heads 2`). With one, K/V read in the wrong memory layout would be the same numbers and pass every check.
- `export_split.py` refuses settings the graphs do not reproduce, before it reads any weights where it can: `add_image_special_tokens`, `prefix_length > 0`, an action expert with fewer layers than the VLM, and the others listed in the contract.
- [docs/SPLIT_GRAPH.md](docs/SPLIT_GRAPH.md) is the contract. It covers:
  - graph and tensor names, shapes, dtypes and layouts;
  - the Euler rule;
  - the bundle format;
  - the three places the export deviates from LeRobot's code, and why: a float32 timestep embedding instead of float64, a functional KV cache, and a precomputed RoPE table.

## C++ runtime

[`cpp/`](cpp) runs the split graphs from C++17, with no Python at inference time. Per action chunk it runs `vision` for each camera, `prefix` once, then the 10-step Euler loop over `expert_step`, accumulating in float32 with LeRobot's update rule. Every buffer is allocated and bound when the runtime is built, and `expert_step` reads the prefix K/V in place.

| Part | What it is |
|---|---|
| `smolvla_rt` library | `Graph`, an interface over one graph with named, statically shaped inputs and outputs. `OrtGraph` implements it with the ONNX Runtime C++ API, on the CPU or the CUDA execution provider. `TrtGraph` implements it with the TensorRT 10 runtime API (`deserializeCudaEngine`, `setTensorAddress`, `enqueueV3`) and is built only when CMake finds TensorRT. `SmolVLARuntime` checks the three graphs against the contract and runs the chunk. Also readers for `split_manifest.json` and for bundles |
| `smolvla_rt_bench` | Checks the chunk against a bundle's PyTorch actions (max abs, cosine) and, stage by stage, against LeRobot's intermediate tensors. Times every component (p50/p90/p99 after warm-up). Optionally runs the control-loop harness. Writes a JSON report |
| Control-loop harness | A 50 Hz control thread takes one action per tick from a chunk buffer, which an inference thread refills when half a chunk is left. It reports wake-up latency, period error, missed deadlines and starved ticks per 10k ticks |
| `smolvla_rt` Python module | Optional pybind11 build with `ort_flow.SplitRuntime`'s `predict_action_chunk(inputs, noise)` |

**Build**

```bash
bash cpp/scripts/fetch_onnxruntime.sh                 # ONNX Runtime 1.31.0 C/C++ SDK into cpp/third_party/onnxruntime
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release  # fetches GoogleTest and nlohmann/json if not installed
cmake --build cpp/build -j
```

- `fetch_onnxruntime.sh` downloads the official release archive from GitHub. If GitHub release downloads are blocked, it falls back to the NuGet package, which has the same CPU library. `--flavor cuda12` or `--flavor cuda13` fetches a GPU build.
- `-DONNXRUNTIME_ROOT=<dir>` uses another SDK. `-DTENSORRT_ROOT=<dir>` adds `TrtGraph` (TensorRT 10 and the CUDA toolkit).
- `-DSMOLVLA_RT_BUILD_PYTHON=ON -DPython_EXECUTABLE=$PY` builds the Python module for the venv's Python, with the pybind11 that `setup_env.sh` installs there. Without `Python_EXECUTABLE`, CMake builds for the first Python it finds. The module lands in `cpp/build/python`: `PYTHONPATH=cpp/build/python $PY -c "import smolvla_rt; rt = smolvla_rt.Runtime('onnx_split')"`.

**Run**

```bash
B=cpp/build/smolvla_rt_bench
$B --onnx-dir onnx_split --bundle onnx_split/bundle.bin --iters 100 --out onnx_split/bench.json   # parity + latency, fp32, CPU
$B --onnx-dir onnx_split --bundle onnx_split/bundle.bin --precision fp16 --provider cuda
$B --onnx-dir onnx_split --bundle onnx_split/bundle.bin --precision fp16 --backend trt  # <graph>_fp16.engine files
$B --onnx-dir onnx_split --bundle onnx_split/bundle.bin --threads 1 --control-loop-ticks 10000 --control-cpu 2 --inference-cpu 3
bash tests/cpp_parity.sh ~/venvs/smolvla/bin/python   # build, export the tiny test policy, check C++ against PyTorch
```

- The bench exits with status 1 if the fp32 chunk differs from the bundle's PyTorch actions by more than `--max-abs`, 1e-4 by default. The tests hold the CPU build to 1e-5; the default leaves room for GPU kernels, which add up in a different order than PyTorch on the CPU. fp16 has no default gate; `--min-cosine` sets one.
- TensorRT engines come from the ONNX graphs, for example `trtexec --onnx=onnx_split/prefix_fp16.onnx --stronglyTyped --saveEngine=onnx_split/prefix_fp16.engine`, once per graph. Build fp32 engines with `--noTF32` as well: TensorRT computes fp32 layers in TF32 unless told not to.
- `--control-cpu` and `--inference-cpu` pin the control thread and the thread that calls `predict_action_chunk`, not ONNX Runtime's worker threads. With the default `--threads 0`, ONNX Runtime starts its own worker threads and pins them to cores itself, ignoring `taskset` and these options, so a worker can share the control thread's CPU; the bench warns about it. With `--threads 1`, as in the measurements below, ONNX Runtime has no workers and runs the graphs on the inference thread.

In C++:

```cpp
#include "smolvla_rt/loader.h"

auto rt = smolvla_rt::load_runtime(smolvla_rt::Manifest::read("onnx_split"), {});  // ONNX Runtime, fp32, CPU
smolvla_rt::Observation obs{{pixels_cam0, pixels_cam1}, lang_tokens, lang_mask, state};  // preprocessed, as in docs/SPLIT_GRAPH.md
rt->predict_action_chunk(obs, noise, actions);  // noise [1, 50, 32] -> actions [1, 50, 7], normalised
```

**Measured** on a 4-vCPU Xeon cloud VM (CPU only), ONNX Runtime 1.31.0, with the tiny random test policy that `tests/cpp_parity.sh` exports: `make_fake_smolvlm.py --tiny --kv-heads 2`, so 4 VLM layers with 2 key/value heads. The parity numbers were taken with the bundle's PyTorch reference computed on 1 and on 4 threads, and ONNX Runtime on 1 and on 4 threads. A range means the PyTorch thread count moved the last digits; the ONNX Runtime thread count changed nothing.

| Check | Result |
|---|---|
| fp32 action chunk vs PyTorch `predict_action_chunk` | max abs 4.8e-7; 9.5e-7 on the GitHub Actions ubuntu-24.04 runner (gate in the tests: 1e-5) |
| fp32 and fp16 chunks vs `ort_flow.py`, same inputs | bit-identical |
| Pipeline stages vs LeRobot: image tokens, prefix K/V, `v_t` over 10 steps | max abs 3.5e-6, 1.1e-6 to 1.2e-6, 1.2e-6 to 1.4e-6; pad mask equal |
| LeRobot's own `x_t` trajectory through the C++ Euler update | bit-identical at all 10 steps, the last one giving the action chunk |
| fp16 action chunk vs PyTorch fp32 | max abs 1.9e-4 to 2.0e-4, cosine 0.9999999974 to 0.9999999978 |

With one key/value head, K/V read in the wrong memory layout would be the same numbers. With two, a C++ runtime that read the prefix K/V as `[1, P, Hkv, Dh]` instead of `[1, Hkv, P, Dh]` (injected on purpose) moved the fp32 chunk by 0.24 and failed the tests and the bench. On a one-head model the same mistake passes every numeric check, so the parity tests also require more than one head.

Latency on one core (`--threads 1`, pinned to CPU 3, five alternating rounds of 30 chunks against `ort_flow.py`; median of the rounds' p50s, and their range):

| | C++ | `ort_flow.py` |
|---|---|---|
| Action chunk | 288 ms (274-302) | 295 ms (271-305) |
| `vision`, per camera | 94.0 ms (90.5-101.6) | 96.8 ms (90.1-99.6) |
| `prefix` | 14.8 ms (14.3-16.0) | 15.5 ms (15.1-16.3) |
| `expert_step`, per step | 7.32 ms (6.89-7.81) | 7.44 ms (7.11-7.77) |

- ONNX Runtime's graph execution dominates, so on this model C++ and Python are within noise of each other: the rounds spread more than they differ.
- The C++ runtime's own work per chunk is copying the inputs into its buffers, 2.4 ms for two 512x512 cameras, plus the Euler updates, 0.03 ms. Removing the camera-pixel copy, for example by letting the caller write into the runtime's input buffers, is a known follow-up.
- The tiny model only shows runtime overhead. For the full-size graphs with real weights, see [Real weights (CPU)](#real-weights-cpu): there C++ and Python took about the same time, roughly 10 s per chunk on one core.

Control-loop harness: 10k ticks at 50 Hz on the same model (`--threads 1`), with the next chunk requested when 25 actions are left:

| | Control thread on CPU 2, inference on CPU 3 | Both on CPU 3 |
|---|---|---|
| Period error p50 / p99 / max | 0.04 / 3.4 / 78 ms | 0.01 / 2.3 / 34 ms |
| Wake-up latency p50 / p99 / max | 0.20 / 1.9 / 78 ms | 0.14 / 1.3 / 34 ms |
| Missed deadlines (woke a full period late), per 10k ticks | 9 | 4 |
| Ticks without an action, per 10k | 0 | 0 |
| Inference per chunk, p50 | 287 ms | 291 ms |

- This is soft real-time: a stock kernel (Linux 6.18, `PREEMPT_DYNAMIC`, not `PREEMPT_RT`) in a 4-vCPU VM shared with other jobs, the process at nice 15, no `SCHED_FIFO`.
- One run per layout, back to back, on a VM where CPU 2 also served other jobs. Here, and in an earlier pair of runs on the one-key/value-head test model, the split layout had the larger tails; these runs cannot say whether that comes from the layout or from the load. A dedicated machine with `--rt-priority` and isolated cores would do better, but nothing here bounds the worst case.
- Inference took about 290 ms per chunk, and each request left 500 ms of actions, so no tick went without one.

**Not tested**
- `TrtGraph` has never run. It compiles, with `-Wall -Wextra -Wpedantic -Wshadow -Werror`, against the TensorRT 10.16 headers from NVIDIA's OSS repository and the CUDA 12.9 runtime headers, but was never linked to TensorRT or given an engine: there is no GPU or TensorRT here. This configuration repeats that check; it needs neither ONNX Runtime nor GoogleTest:
  `cmake -S cpp -B cpp/build-trt -DSMOLVLA_RT_TRT_COMPILE_CHECK=ON -DSMOLVLA_RT_WERROR=ON -DSMOLVLA_RT_WITH_ORT=OFF -DSMOLVLA_RT_BUILD_TESTS=OFF -DTENSORRT_ROOT=<TensorRT/include parent> -DSMOLVLA_RT_CUDA_INCLUDE_DIR=<dir with cuda_runtime_api.h>`, then `cmake --build cpp/build-trt`.
- Several GPUs. The current CUDA device belongs to each thread, and the runtime can be driven from threads other than the one that built it: the control loop's inference thread, or Python threads. `CudaDevice` therefore selects its device on the calling thread before every allocation, copy and free. `cpp/tests/test_cuda_device.cpp` checks that on the CPU against a stand-in for the CUDA runtime, one operation per new thread. `TrtGraph` selects its device in `run()` and in its destructor; that part is compile-checked only.
- The real CUDA calls behind `CudaDevice` (`cpp/src/cuda_api.cpp`) were linked against CUDA 12.9's `libcudart`, but only the "no CUDA driver" error path ran.
- ONNX Runtime's CUDA execution provider: the SDK used here is CPU-only. With it, `OrtGraph` keeps buffers in host memory, so ONNX Runtime copies the prefix K/V to the GPU on every expert step; `TrtGraph` keeps them on the device.
- The guard against a fused multiply-add in the Euler update (`Contract.EulerUpdateRoundsTheProductSeparately`, for the build's `-ffp-contract=off`) can only fail where the compiler may emit FMA instructions: aarch64, as on Jetson, or x86-64 built with `-march=native` on a CPU that has them. The default x86-64 target has no FMA, so on this VM and on the CI runner the test would pass without the flag too.
- No GPU latency of any kind. The numbers in this section come from random weights. Real weights were checked on the CPU only, in [Real weights (CPU)](#real-weights-cpu). There the fp16 `vision` graph's attention scores stayed within ±48. The only overflow, inside the vision tower's GELU, did no harm on ONNX Runtime's CPU provider, nor with every node evaluated in IEEE fp16 (numpy); GPU kernels were not tested.
- No aarch64 build or run, so nothing on Jetson either: everything here ran on x86-64 with GCC 13. The aarch64 branch of `fetch_onnxruntime.sh` is untested.

## Real weights (CPU)

Every measured number above comes from random weights. This run used the real [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) checkpoint and 16 real frames of [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero), the dataset its training config names: 4 episodes, one task from each LIBERO suite, 4 frames each, both cameras, state and task. Everything went through the checkpoint's own pre- and postprocessors, and each frame got the same noise in every runtime.

CPU only, on a 4-vCPU Xeon VM. Every measurement ran on one core and one thread. The exception is `export_split.py`'s own check, which used ONNX Runtime's default pool of 4 threads on that core.
- [`results/REAL_WEIGHTS_CPU.md`](results/REAL_WEIGHTS_CPU.md) has the details and [`results/real_weights_cpu.json`](results/real_weights_cpu.json) the numbers.
- `bash scripts/run_real_weights_cpu.sh PYTHON WORK_DIR` reproduces the run from the Hub. A rerun of its `inputs`, `bundles`, `fp32` and `fp16` steps with the committed tools reproduced those steps' numbers bit for bit, timings aside.
- The reference is LeRobot's `predict_action_chunk` in fp32. Its actions are 0.07 std from the dataset's demonstrated actions on average over the first 10 steps, against 0.72 for the mean action, so weights and preprocessing are right.

Against that reference, worst case over the 16 frames, in normalised action space (units of each action dimension's std):

| | Max abs | Cosine, smallest | Non-finite |
|---|---|---|---|
| Split graphs fp32, `ort_flow.py` and the C++ runtime (same numbers) | 6.3e-6 | 0.9999999999996 | 0 |
| Split graphs fp16, `ort_flow.py` and the C++ runtime (same numbers) | 2.0e-3 | 0.99999992 | 0 |
| PyTorch `fp16all`, `fp16vlm`, `amp` (4 of the frames) | 2.9e-3 | 0.99999975 | 0 |
| PyTorch `lerobot`: bf16, what LeRobot runs for this checkpoint | 4.9e-2 | 0.99995 | 0 |
| PyTorch `bf16vlm`, `bf16all` | 3.7e-2, 3.2e-2 | 0.99996 | 0 |

`lerobot_amp` fails on the CPU, as expected (`mixed dtype`).

**fp16 is safe for this checkpoint on these frames**, under the two conditions below:
- **The probe.** Every floating-point intermediate tensor of the three fp16 graphs was made a graph output and read on all 16 frames, each fp16 tensor as it rounds to fp16. One overflowed: the cube `x^3` inside the tanh GELU of the last two SigLIP layers, which the exporter writes out node by node. ONNX Runtime's CPU provider handed the next nodes the fp32 cube. An engine that keeps every node in fp16 overflows the three nodes after the cube too.
- **The overflow is harmless where it was checked.** tanh saturates. Evaluated node by node in IEEE fp16 (numpy), the formula still returns x or 0, exactly, wherever the cube overflows, and no finite fp16 input gives a non-finite output. On ONNX Runtime's CPU provider the chunk stays finite. GPU kernels were not tested.
- **The margins.** Outside the GELU, the largest fp16 tensors are the vision residual stream at 663.5 and the VLM residual stream at 7732, 8.5 times below fp16's 65504. Inside the GELU, the nodes after the cube reached 50912 in this run, 1.29 times below, and the cube itself reached 26768 in a layer where it did not overflow. All of these feed tanh. The vision attention scores, the only fp16 ones, stay within ±48.
- **Condition 1: the fp32 parts stay fp32.** The fp16 graphs are strongly typed and keep some tensors in fp32: the RMSNorm statistics (the VLM's residual stream squared reaches 6.0e7), the VLM's and the expert's attention scores, softmax and masks, RoPE, and the expert's time embedding (values up to 4096). The vision tower's LayerNorm statistics must stay fp32 too (ONNX's default `stash_type`), since its inputs reach 663.5. A runtime has to honour these types.
- **Condition 2: 16 frames are a small sample.** The margins come from these frames only.
- **bf16 moves the actions more than fp16.** On the four frames where both ran, LeRobot's own bf16 moved the chunk 4 to 11 times as far as PyTorch's `fp16all`, and up to 15 times as far as `fp16vlm` and 12 as far as `amp`.
- **The CPU's fp16 chunk is optimistic.** ONNX Runtime's CPU provider computes the fp16 graphs' matrix products in fp32, so its fp16 chunk is likely closer to fp32 than a GPU's. On the same frames it gave 1.2e-3, against 2.9e-3 for PyTorch's `fp16all`, which rounds every fp16 operation's output.

CPU latency on one thread, p50: a chunk takes 9.4 s in PyTorch fp32, 10.1 s with the fp32 split graphs from Python and 9.95 s from C++. bf16 and fp16 are slower on this CPU, which has no arithmetic for either: PyTorch bf16 24.6 s, PyTorch fp16 64.7 s, fp16 graphs 11.2 to 12.1 s. These say nothing about a GPU.

**Not tested:**
- any GPU, and TensorRT engines;
- whether TensorRT keeps the LayerNorm statistics and the other fp32-typed tensors in fp32, in particular in a weakly typed build (`trt_vision.py --mode fp16-flag`), where the builder chooses each layer's precision;
- how TensorRT or a fused GPU GELU kernel treats the GELU's infinite intermediate;
- other checkpoints, more frames, and closed-loop success.

## Method notes

**Timing**
- Every region boundary calls `torch.cuda.synchronize()`, so GPU work is charged to the region that launched it.
- That costs a little, so each variant also runs uninstrumented. The headline end-to-end number comes from that run; the instrumentation overhead is reported.
- `trt_vision.py` times with CUDA events on one stream, inputs already on the GPU. Its optional PyTorch baseline uses the same method, so the two are comparable.

**Inputs**
- Synthetic, shaped like the checkpoint's input features: images in [0, 1], 256×256, two cameras for LIBERO.
- Image content does not change latency.
- Language token count does change it, so `--task` is a real LIBERO instruction.

**Parity**
- Each variant's raw action chunk (normalised space, before unnormalising) is compared with the fp32 chunk under identical noise: max-abs, mean-abs, relative L2, cosine.
- The fp32 reference holds the checkpoint's stored values exactly. The policy is built with `load_vlm_weights=false` and cast to fp32 *before* the weights are copied in, so a bf16 dtype in the base VLM's config cannot round them.
- The dtypes the constructor produced are recorded as `built_dtypes`.
- Every variant starts from that reference and casts parameters directly. Nothing is rounded through a dtype and back.

**Backend flags**
- Our variants run with TF32 and `cudnn.benchmark` off.
- The `lerobot*` variants turn both on, as `lerobot_eval.py` does.
- Each result records the flags it ran with.
- The ONNX parity reference is computed with TF32 off.

**Not claimed here**
- Closed-loop task success. Numerical parity is not closed-loop parity; paired simulator rollouts are the next project.

**Record clocks and versions**
- GPU clocks vary on cloud machines. `sysinfo` records clocks and throttle reasons at idle and after each variant's timed loop, plus driver, CUDA, TensorRT and package versions.
- It reads them from the device torch actually uses, so `CUDA_VISIBLE_DEVICES` is respected.

## Prior work

**Tether**
- Exports SmolVLA and other VLAs to ONNX and TensorRT. It reports 19.49 ms for SmolVLA with the TensorRT execution provider vs 108.11 ms with ONNX Runtime CUDA, on an A10G.
- It is BSL-1.1 licensed. Nothing here is derived from its code.

**This repo adds**
- Per-component timing on the original PyTorch stack.
- A precision layout that matches what LeRobot actually runs, checked against LeRobot.
- A strongly typed FP16 path that survives TensorRT 11's removal of weak typing.

## Licence

Apache-2.0. SmolVLA checkpoints and SmolVLM2 weights keep their own licences.

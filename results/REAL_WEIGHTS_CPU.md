# Real weights on the CPU: `lerobot/smolvla_libero` on LIBERO frames

Every other number in this repository comes from random weights. These come from the real checkpoint and real observations, on a CPU only.
- [`real_weights_cpu.json`](real_weights_cpu.json) holds the numbers. Each is copied from one of the run's own reports, which its `sources` list names, or computed from the arrays the run saved.
- `bash scripts/run_real_weights_cpu.sh PYTHON WORK_DIR` reproduces the run from the Hugging Face Hub. The checkpoint and the dataset come at pinned revisions; the checkpoint's base VLM repository, which supplies the config and the tokenizer, comes from its main branch (`7b375e1` for this run). The run took about 2 hours on one core.
- [Provenance](#provenance) says which code produced the numbers, and how a rerun with the committed tools compared.

## What ran

| | |
|---|---|
| Checkpoint | `lerobot/smolvla_libero` at `a24f1f3` (2026-10-12, a metadata fix on top of the weights uploaded on 2026-03-24). The VLM and the action expert are stored in bf16; the expert's cross-attention k/v and the small projections in fp32. Its config says `load_vlm_weights=true`; the reference skips that pre-load of the base VLM, since the checkpoint holds every weight |
| Observations | 16 frames of `lerobot/libero` at `a1aaacb`. The checkpoint's `train_config.json` names this dataset but no revision. `a1aaacb` was its latest revision when the run started (2026-06-08, "Fix episode indexing in metadata to match data files"). It is newer than the weights, so which revision they were trained on is unknown. Episodes 0, 423, 846 and 1269, whose tasks come from the four LIBERO suites (libero_10, goal, object, spatial), frames `floor(k L / 4)` for k = 0..3, with L the episode's length. Both cameras as `LeRobotDataset` decodes them (torchcodec 0.11.1 on FFmpeg 6.1.1), the 8-d state and the task. `tools/real_inputs.py` picks them; the JSON records the sha256 of every array |
| Preprocessing | The checkpoint's own pre- and postprocessors. They pad the task to 48 tokens, the length the prefix graph needs, so the graphs see exactly LeRobot's inputs. The tasks use 10 to 22 of the 48 |
| Noise | Observation i: `torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(i + 1))`. The same in every runtime and variant |
| Reference | LeRobot's `predict_action_chunk` with every parameter in fp32 (`loading.load_policy`), torch on one thread |
| Machine | Intel Xeon @ 2.80 GHz: AVX-512, but no AVX512-BF16, AVX512-FP16 or AMX, so no bf16 or fp16 arithmetic. A 4-vCPU VM. Every measurement ran pinned to core 3, on one thread. The one exception is `export_split.py`'s own check (below): it sets no thread count, so ONNX Runtime used its default pool, 4 threads on this VM, all kept on core 3. torch 2.11, LeRobot 0.6.1, transformers 5.5.4, ONNX Runtime 1.31.0 (Python package and C++ SDK) |
| Memory | Measured in the rerun of the `inputs`, `bundles`, `fp32` and `fp16` steps ([Provenance](#provenance)), not in `cpp` or `p0`: the largest single process peaked at 6.3 GB, in the fp16 step. /dev/shm, which held the Hub cache, the bundles and one export, peaked at 2.5 GB (sampled every 10 s) |

Errors are max abs over the 50 x 7 action chunk and over all 16 frames, in normalised action space. Normalised actions are in units of each dimension's std from the checkpoint's normalisation stats, so these are also the errors after the postprocessor divided by that std. The JSON has both, per dimension.

**The reference is right.** On the first 10 actions of each chunk, LeRobot's fp32 actions are 0.071 std from the dataset's own demonstrated actions on average (0.044 to 0.138 per frame), against 0.725 for the dataset's mean action.

## Split graphs against LeRobot

`export_split.py` exported each precision; `ort_flow.py`'s runtime (Python) and `smolvla_rt_bench` (C++) ran them on every frame with ONNX Runtime's CPU provider on one thread. Worst case over the 16 frames:

| | fp32 graphs | fp16 graphs |
|---|---|---|
| Action chunk vs LeRobot fp32: max abs | 6.3e-6 | 2.0e-3 |
| cosine, smallest | 0.9999999999996 | 0.99999992 |
| Largest per action dimension (max abs / std) | 1.7e-6 to 6.3e-6 | 0.9e-3 to 2.0e-3 |
| Non-finite values in the chunk, in `image_tokens`, K/V and `v_t` | 0 | 0 |
| Each graph on LeRobot's own inputs, max abs (relative L2): `vision` | 3.7e-4 (1.3e-6) | 0.17 (7.8e-4) |
| `prefix` K/V | 3.1e-5 (4.1e-7) | 1.9e-2 (3.7e-4) |
| `expert_step`, 10 steps | 1.3e-5 (5.2e-7) | 1.7e-2 (2.5e-4) |
| C++ against `ort_flow.py` | same max abs on every frame | same max abs on every frame |

- The image tokens reach 245 in magnitude, so their absolute errors are larger than the other graphs'.
- `ort_flow.py --bundle`, the command-line entry point, gave the same numbers as the runtime used here, bundle by bundle.
- The C++ bench also matched LeRobot stage by stage: the prefix pad mask was equal, and reruns were bit-identical. This is the first full-size run of the C++ runtime.
- `export_split.py`'s own check, on `profile_smolvla.py`'s synthetic observation (random pixels), gave 3.5e-6 (fp32) and 4.3e-3 (fp16) on the chunk, with ONNX Runtime's default pool of 4 threads: the real frames came out closer to fp32 in fp16.

## fp16: safe for this checkpoint on these frames, with two conditions

Outside SigLIP's GELU, every fp16 tensor in the three graphs stays at least 8.5 times below fp16's largest value, 65504, on all 16 frames. Inside the GELU, the cube x^3 overflows and the nodes after it come within 1.29 times of the limit; all of them feed tanh, which saturates, so the GELU's output is unaffected. The chunk stays finite and within 2.0e-3 std of fp32.

The check: `tools/real_weights_split.py probe` made every floating-point intermediate tensor of the exported graphs a graph output and ran all 16 frames through them.
- Each fp16 tensor is read as it rounds to fp16.
- ONNX Runtime's CPU provider computes most nodes in fp32, though, and can hand the next node the unrounded fp32 value (see [What the CPU run does not show](#what-the-cpu-run-does-not-show)). So an overflow does not propagate here as it would in an engine that keeps every node in fp16.
- The probe checks that it reads every float node output: none lacks an inferred type, and no node holds a subgraph or calls a local function. The committed tool checked this in the rerun: it read all 516, 1180 and 1227 float node outputs of the fp16 `vision`, `prefix` and `expert_step` graphs, and 516, 1007 and 1063 of the fp32 ones. The other node outputs are the graph outputs, which the check reads, and integer or bool tensors.

Largest magnitude per graph:

| Graph | Largest fp16 tensor outside the GELU (headroom to 65504) | Inside the GELU | Attention scores before softmax | Non-finite |
|---|---|---|---|---|
| `vision` (SigLIP) | 663.5, the residual stream into the last LayerNorm (99x) | the cube overflows in layers 10 and 11; finite fp16 tensors up to 50912 (1.29x) | fp16, within [-48.0, 44.4] | the GELU cube only |
| `prefix` (VLM) | 7732, the residual stream (8.5x) | no GELU | fp32 in the graph, within [-35.5, 31.1] | none |
| `expert_step` | 19.1 (3400x) | no GELU | fp32 in the graph, within [-31.8, 45.8] | none |

**The exception: SigLIP's GELU.** It is the tanh approximation, which the exporter writes node by node as `x * (0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))))`: Pow, Mul, Add, Mul, Tanh, Add, Mul, Mul.
- **Where it overflows.** x^3 overflows fp16 above |x| = 40.3. In the last two vision layers the MLP's first projection goes beyond that, to 104 on the four frames of the fp32 probe. Over the 32 camera images, 335 and 115,060 values of these two [1, 1024, 3072] cubes became infinite.
- **What came close.** ONNX Runtime passed the fp32 cube on, so the next three nodes stayed finite in this run, but large: in layer 11, `0.044715 x^3` reached 50784, the sum `x + 0.044715 x^3` 50912 (1.29 times below 65504) and its product with sqrt(2/pi) 40608. In layers where it did not overflow, the cube reached 26768 (layer 5, 2.4 times below) and 9256 (layer 9).
- **Under strict fp16.** In an engine that keeps every node in fp16, those three nodes overflow wherever the cube does.

That is still harmless where it was checked:
- tanh saturates: tanh(+inf) = 1 and tanh(-inf) = -1, so the GELU returns exactly x for large positive x and 0 for large negative x;
- evaluated node by node in IEEE fp16 (numpy), in the exporter's order, the formula gives no non-finite output for any finite fp16 input, and for every input up to |x| = 1024 it is exact wherever the cube overflows (`tools/real_weights_split.py gelu-fp16`);
- ONNX Runtime's CPU provider computes this chain of nodes in fp32 (its profiler shows the cube's `Pow` on fp32 inputs), so the plain run never rounds the cube to fp16.

GPU kernels were not tested; a fused GELU kernel may compute the cube in fp16.

**Condition 1: the fp32 parts stay fp32.** The fp16 graphs are strongly typed, and some of their tensors are fp32 or must be computed in fp32, because fp16 would not hold them:
- **SigLIP's LayerNorm statistics.** Its inputs reach 663.5, whose square is beyond fp16. ONNX's `LayerNormalization` keeps its statistics in fp32 by default (`stash_type` 1). ONNX Runtime's fp16 LayerNorm kernel does: its outputs stay finite and the vision output matches fp32.
- **The VLM's RMSNorm statistics.** Its residual stream reaches 7732, which squared is 5.98e7. The graph computes them in fp32, because transformers and LeRobot upcast there.
- **Attention, RoPE and the time embedding.** The VLM's and the expert's attention scores and softmax are fp32, and their masks fill with float32's lowest value (15 and 16 `Where` nodes). RoPE is fp32 too. So is the expert's time embedding, whose intermediate values reach 4096, where fp16's spacing is 4.

A runtime has to keep every fp32-typed tensor in fp32 and the LayerNorm statistics in fp32. A strongly typed TensorRT build is meant to follow the graph's types; a weakly typed one (`trt_vision.py --mode fp16-flag`) chooses each layer's precision itself. Neither was tested.

**Condition 2: the margins come from 16 frames.** Outside the GELU, the VLM's residual stream is the closest tensor to the limit, at 7732 (8.5x). It is about 7000 at every one of the 16 layers: on the four frames of the fp32 probe, each layer's largest value, its RMSNorm's square, is 4.9e7 to 5.1e7. Other observations, other tasks, or a checkpoint fine-tuned further could come closer to 65504.

The stored weights are no obstacle either:
- none is beyond fp16's range: the largest is 12.1, the vision position embedding;
- 0.26% of the vision weights and 0.16% of the expert's become fp16 subnormals, and at most 109 per submodule round to zero.

### What the CPU run does not show

ONNX Runtime's CPU provider has no fp16 kernels for most of these nodes:
- **Matrix products in fp32.** Its profiler shows every MatMul and Gemm of the fp16 graphs running on fp32 inputs, with about 195 Cast kernels per graph call. In `vision`, which holds no Cast node, ONNX Runtime inserted all of them. In `prefix` and `expert_step`, most are the graph's own Cast nodes, 174 and 166, around the parts kept in fp32.
- **Values pass between nodes in fp32.** Between two such nodes the value can stay in fp32. In the probe, the GELU's next node received the finite fp32 cube although the cube rounded to fp16 was infinite.
- **The probe moved the outputs.** Making every intermediate a graph output changes which values ONNX Runtime rounds to fp16 between nodes and which nodes it fuses. That moved the graphs' outputs against the plain run by up to 0.125 (vision), 0.039 (prefix K/V) and 0.0054 (`v_t`). Rounding every node to fp16 was not measured.
- **So the CPU's fp16 error is optimistic.** The 2.0e-3 chunk error is likely smaller than what an fp16 GPU engine gives. PyTorch's `fp16all` below is closer to a GPU: it rounds the output of every operation in the fp16 layout to fp16, keeping the same parts in fp32 as the graphs. On the four frames it ran, it gave up to 2.9e-3, where these graphs gave up to 1.2e-3.
- **The ranges outside the GELU should carry over.** They are values computed with less rounding than a strict fp16 engine applies, and the output shifts just above are small next to an 8.5x margin. Inside the GELU, a strict engine overflows more nodes, as described.

## Precision variants in PyTorch

`tools/real_weights_p0.py` runs `profile_smolvla.py`'s variants through the library on the same frames and noise. Each variant is cast from the fp32 reference. Against fp32, worst case over the frames:

| Variant | Frames | Max abs vs fp32 (= / std) | Cosine, smallest | Non-finite |
|---|---|---|---|---|
| `lerobot`: bf16 VLM and expert, cross-attention k/v fp32. What LeRobot runs for this checkpoint | 16 | 4.9e-2 | 0.99995 | 0 |
| `bf16vlm` | 16 | 3.7e-2 | 0.99996 | 0 |
| `bf16all` | 16 | 3.2e-2 | 0.99996 | 0 |
| `fp16vlm` | 4 | 2.9e-3 | 0.9999998 | 0 |
| `fp16all`, the fp16 graphs' layout | 4 | 2.9e-3 | 0.9999998 | 0 |
| `amp`, fp16 autocast over fp32 weights | 4 | 2.9e-3 | 0.99999975 | 0 |
| `lerobot_amp` | 0 | fails: `RuntimeError: mixed dtype (CPU)`, as the README says for the CPU | | |

- The variants that compute in fp16 ran on frames 1, 5, 9 and 13 only, one per episode, without warm-up, because on this CPU an fp16 chunk takes about 7 times as long as an fp32 one (see below).
- **bf16, which LeRobot itself runs for this checkpoint, moves the chunk 4 to 11 times as far as `fp16all` does** on the four frames where both ran: up to 0.020 std (`lerobot`), against 0.0029 (`fp16all`). Against `fp16vlm` and `amp` the ratio reaches 15 and 12. Over all 16 frames bf16 reaches 0.049.
- On those four frames, PyTorch's `fp16all` gives up to 2.9e-3 and the fp16 graphs on ONNX Runtime up to 1.2e-3; the two differ by up to 2.6e-3.
- The fp32 variant here is bit-identical to the reference in the bundles: the same code, the same inputs and noise, one thread.
- The summary recomputes every variant's error from the chunks the run saved; all equal the report's.

## CPU latency, one thread

Not comparable to any GPU. ONNX Runtime's numbers are the p50 of 3 chunks on frame 0, after 1 warm-up. PyTorch's are the p50 over the frames each variant ran: 16 after 1 warm-up call, or 4 for `fp16all`.

| | PyTorch fp32 | PyTorch `lerobot` (bf16) | PyTorch `fp16all` | ONNX Runtime fp32: Python / C++ | ONNX Runtime fp16: Python / C++ |
|---|---|---|---|---|---|
| Action chunk | 9.39 s | 24.6 s | 64.7 s | 10.1 s / 9.95 s | 12.1 s / 11.2 s |
| Vision, per camera | 2.89 s | 8.19 s | 22.5 s | 3.79 s / 3.67 s | 4.59 s / 4.27 s |
| VLM prefix | 0.84 s | 2.51 s | 6.08 s | 0.65 s / 0.63 s | 0.76 s / 0.68 s |
| Action expert, per step | 271 ms | 565 ms | 1363 ms | 179.6 ms / 182.7 ms | 196.5 ms / 191.4 ms |

- PyTorch's columns come from `instrument.py`'s components. Its expert step leaves out `denoise_other`, about 50 ms per chunk for the time embedding and the action projections, which the `expert_step` graph includes.
- With no bf16 or fp16 arithmetic on this CPU, both lower precisions are slower than fp32 in PyTorch: bf16 2.6 times, fp16 6.9 times. ONNX Runtime runs the fp16 graphs mostly in fp32, with casts around the nodes, and is 1.1 to 1.2 times slower than with the fp32 graphs.
- C++ and Python run the same ONNX Runtime graphs. C++ took 1.4% (fp32) and 7% (fp16) less time per chunk, from 3 chunks each, measured minutes apart: too few to separate the runtimes.
- In the `vision` graph, ONNX Runtime's profiler puts 0.50 s per camera (fp32) and 0.88 s (fp16) in its `Where` nodes: 12 of them, one per layer, follow an `IsNaN` on the [1, 12, 1024, 1024] attention probabilities. That is the exporter's attention decomposition guarding against fully masked rows. With every image patch present no row is masked, so a vision graph without them is a possible speed-up; it was not tried here.

## Provenance

- **The first run.** The reports behind these numbers were written on 2026-10-29 between 15:34 and 18:49 UTC by the tools as they were then. Some of the code changed after the steps that used it, and those reports did not record their code:
  - `scripts/run_real_weights_cpu.sh` changed at 18:38, after the whole run. Its fp16 step gained the `gelu-fp16` call, which had run on its own at 18:39.
  - `tools/real_weights_split.py` changed at 18:39, after its last check, probe and profile.
  - For this write-up, all the tools changed again: the GELU check now multiplies in the exporter's order, the probe checks its coverage, every report records its code and machine, and the inputs and bundles record their hashes and token ids.
- **The rerun.** With the committed tools, the `inputs`, `bundles`, `fp32` and `fp16` steps ran again on the same machine; the C++ bench was not rebuilt. `tools/real_weights_summary.py --rerun` compared their reports and saved arrays with the first run's, value by value, leaving out timings and paths.
  - Every value they share with the first run's reports, and every saved array, came out identical (`rerun.all_identical` in the JSON).
  - That covers the decoded frames, the 16 reference chunks, the stored weights' fp16 table and both exports' own checks.
  - It also covers every graph error and chunk from `ort_flow.py`'s runtime and its command line, the ONNX Runtime kernel counts, the GELU check and the probe's range of every tensor.
  - `tools/real_weights_summary.py` changed once more after the rerun, to carry the rerun's probe coverage and GELU check into the JSON; no rerun step runs it.
- **The P0 step** was not rerun. `tools/real_weights_p0.py` has since changed only to record its code and machine. The summary recomputes its numbers from the chunks it saved, and they equal the report's.
- **What the JSON records.** Every report now records the git commit and the hashes of changed code files (`tools/run_info.py`). The JSON's `code` and `rerun.code` hold them. `inputs` holds the sha256 of every input array, the language token ids and the decoder versions. `rerun.base_vlm` holds the base VLM's commit, `rerun.probe_coverage` the probe's account of every node output and `rerun.peak_memory` the memory use.

## Not tested here

- Any GPU: ONNX Runtime's CUDA provider, TensorRT engines built from these graphs, and fp16 GPU kernels, which may accumulate differently from both runtimes here.
- Whether TensorRT keeps the LayerNorm statistics and the other fp32-typed tensors in fp32, in particular in a weakly typed build, and how TensorRT or a fused GELU kernel treats the GELU's infinite intermediate.
- Other checkpoints (`lerobot/smolvla_base`, `lerobot/smolvla_libero_plus`), more frames, other datasets.
- Closed-loop task success: numerical parity is not closed-loop parity.

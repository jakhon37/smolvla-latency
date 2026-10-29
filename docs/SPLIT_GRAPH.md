# Split-graph SmolVLA: the contract

`export_split.py` turns one SmolVLA checkpoint (LeRobot 0.6.1) into three ONNX graphs. A runtime calls them in this order:

```
for each camera c:   vision(pixel_values_c)                          -> image_tokens_c
once per chunk:      prefix(image_tokens_*, lang_tokens, lang_mask, state)
                                                                     -> prefix_pad_mask, past_key_i, past_value_i
num_steps times:     expert_step(x_t, t, prefix_pad_mask, past_key_i, past_value_i)
                                                                     -> v_t ;  x_t <- x_t + dt * v_t
```

Together they reproduce `SmolVLAPolicy.predict_action_chunk` after preprocessing. The Euler loop stays in the runtime, so the number of steps, the schedule and the noise remain under its control.
- `ort_flow.py` is the reference implementation: ONNX Runtime for the graphs, numpy for the loop. With `--bundle` it needs only numpy, onnx and onnxruntime, so it also runs on a device without torch.
- `smolvla_latency/split_contract.py` holds the tensor names and the Euler schedule, also numpy only.
- `tools/dump_bundle.py` writes test vectors for other runtimes.
- `cpp/` implements this contract in C++17, with ONNX Runtime or TensorRT for the graphs and the loop in C++, and checks itself against those test vectors. See the README's "C++ runtime" section.

All shapes are static and the batch size is 1. Sizes below use the names in `split_manifest.json`'s `config`. The last column is the full-size test fixture: SmolVLM2-500M layer sizes with LIBERO-shaped inputs (two 256x256 cameras, 8-d state, 7-d actions).

| Name | Meaning | Full-size fixture |
|---|---|---|
| `H`, `W` (`image_hw`) | padded image size | 512, 512 |
| `T` (`image_tokens`) | tokens per camera | 64 |
| `D` | VLM hidden size | 960 |
| `L` (`lang_len`) | `tokenizer_max_length` | 48 |
| `S` (`max_state_dim`) | padded state size | 32 |
| `N` (`num_layers`) | VLM layers kept (`num_vlm_layers`) | 16 |
| `expert_layers` | action expert layers; the export needs one per VLM layer, so always `N` | 16 |
| `Hkv` (`kv_heads`), `Dh` (`head_dim`) | key/value heads and head size | 5, 64 |
| `P` (`prefix_len`) | prefix length, `cameras * T + L + 1` | 177 with 2 cameras |
| `C` (`chunk_size`) | actions per chunk | 50 |
| `A` (`max_action_dim`) | padded action size | 32 |
| `action_dim` | real action size, kept after the loop | 7 |
| `num_steps` | Euler steps | 10 |

## Files

`export_split.py --policy <ckpt> --out-dir <dir>` writes, for each precision `<p>` in `fp32`, `fp16`:

| File | Contents |
|---|---|
| `vision_<p>.onnx`, `prefix_<p>.onnx`, `expert_step_<p>.onnx` | The graphs |
| `<graph>.onnx.data` | That graph's weights, next to it. ONNX Runtime and TensorRT's `parse_from_file` resolve them relative to the graph |
| `split_manifest.json` | For each graph: files, opset, every input and output name, dtype and shape. Also the `config` sizes above and the camera order. Generated from the graph files |
| `export_report.json` | ONNX Runtime parity of each graph and of the whole action chunk against LeRobot fp32. Also the language-padding check |

**Exporter**
- `torch.onnx.export(dynamo=True)`, i.e. torch.export plus onnxscript.
- Opset 18, IR version 10, within what ONNX Runtime and the TensorRT 10 parser support; `--opset` changes it.
- The TorchScript exporter is not offered: under transformers 5.5 it fails inside the SigLIP attention-mask code.

**Weights**
- Every graph stores its weights in one external-data file, whatever its size.
- At full size the prefix alone is about 0.8 GB in fp32, so the 2 GB protobuf limit is never a concern.
- Keep each `.onnx` next to its `.onnx.data`.

## Graphs

Layouts are row-major, as in PyTorch. "fp32 / fp16" gives the dtype in each precision variant.

### vision: once per camera

| | Name | dtype | Shape | Content |
|---|---|---|---|---|
| in | `pixel_values` | fp32 / fp16 | `[1, 3, H, W]` | NCHW RGB in [-1, 1], prepared as in [Inputs](#inputs) |
| out | `image_tokens` | fp32 / fp16 | `[1, T, D]` | `SmolVLMWithExpertModel.embed_image`: SigLIP tower and connector |

This is the P0 graph from `export_vision.py`. Its output is **before** the `sqrt(D)` scaling, which happens inside `prefix`.

### prefix: once per chunk

| | Name | dtype | Shape | Content |
|---|---|---|---|---|
| in | `image_tokens_<c>` | fp32 / fp16 | `[1, T, D]` | vision output for camera `c`, in `camera_keys` order |
| in | `lang_tokens` | int64 | `[1, L]` | token ids, right-padded to exactly `L` |
| in | `lang_mask` | bool | `[1, L]` | true for real tokens, false for padding |
| in | `state` | fp32 | `[1, S]` | normalised state, zero-padded to `S` |
| out | `prefix_pad_mask` | bool | `[1, P]` | true where a prefix position is a real token |
| out | `past_key_<i>`, `past_value_<i>` for i = 0..N-1 | fp32 / fp16 | `[1, Hkv, P, Dh]` | layer i's keys and values over the prefix |

**What the graph computes**
- `VLAFlowMatching.embed_prefix` with every camera present: image tokens times `sqrt(D)`, the language embedding lookup times `sqrt(D)`, and `state_proj`.
- The attention masks and position ids.
- The VLM prefill with `use_cache=True`.

**Prefix order** (`P` positions): camera 0's `T` tokens, camera 1's `T` tokens, ..., then the `L` language positions, then the state token.

**Outputs**
- The keys and values are exactly what `SmolVLMWithExpertModel.forward_attn_layer` stores in its `DynamicCache` during prefill.
- They are post-RoPE, in the cache layout `[batch, kv_heads, seq, head_dim]`.
- Padding positions hold finite values that `prefix_pad_mask` masks out downstream.
- Outputs are interleaved per layer: `past_key_0, past_value_0, past_key_1, ...`, after `prefix_pad_mask`.

### expert_step: `num_steps` times per chunk

| | Name | dtype | Shape | Content |
|---|---|---|---|---|
| in | `x_t` | fp32 | `[1, C, A]` | current sample; the noise at the first step |
| in | `t` | fp32 | `[1]` | flow time in [0, 1], see [Euler loop](#euler-loop) |
| in | `prefix_pad_mask`, `past_key_<i>`, `past_value_<i>` | as `prefix` outputs | | the prefix outputs, unchanged |
| out | `v_t` | fp32 | `[1, C, A]` | velocity, `VLAFlowMatching.denoise_step` |

**Binding.** `expert_step`'s inputs after `x_t` and `t` have the same names, order, shapes and dtypes as `prefix`'s outputs. A runtime can bind the prefix output buffers directly and keep them on the device for all steps.

**Purity.** The step is a pure function of its inputs: nothing in the cache is modified, so the same buffers serve every step.

**Layers.** Expert layers alternate as LeRobot builds them (`self_attn_every_n_layers=2`):
- Even layers are self-attention. They attend over `[prefix K/V; own K/V]`.
- Odd layers are cross-attention. They project the prefix K/V of the same VLM layer through their own `k_proj`/`v_proj`.

## Euler loop

This is LeRobot's `euler_integrate`, bit for bit:

```
dt   = float32(-1.0 / num_steps)
x_t  = noise                                  # [1, C, A] float32, standard normal
for k in 0 .. num_steps-1:
    t    = float32(1.0 + k * (-1.0 / num_steps))   # computed in float64, then rounded once
    v_t  = expert_step(x_t, [t], prefix outputs)
    x_t  = x_t + dt * v_t                     # float32 multiply, then float32 add
actions = x_t[:, :, :action_dim]              # normalised; the postprocessor unnormalises
```

With `num_steps = 10`, `dt` is float32(-0.1) and `t` runs 1.0, 0.9, ..., 0.1, each rounded to float32. `tests/test_split_graph.py` checks that this loop reproduces `euler_integrate` exactly.

## Inputs

These are the steps of LeRobot's preprocessor and `predict_action_chunk` that sit before the graphs. `split_graph.graph_inputs` implements them from a preprocessed batch.

**Images**
1. Float RGB in [0, 1], CHW. Visual features use identity normalisation.
2. Resize to fit `H x W`, keeping the aspect ratio: bilinear, `align_corners=False`.
3. Pad on the **left and top** with 0.
4. Map to [-1, 1] with `x * 2 - 1`.

A 256x256 LIBERO frame is upsampled to 512x512 without padding.

**Language**
1. The task string gets a trailing `"\n"`.
2. It is tokenised with the VLM tokenizer, truncated to `L`, and right-padded to exactly `L` with the pad token.
3. `lang_mask` is true for real tokens.

LeRobot's saved preprocessors pad to the longest sequence instead (`pad_language_to="longest"`; the default task is 18 tokens with the test fixtures' toy tokenizer). With `load_processors(..., language_padding="max_length")` the same pipeline produces the fixed length. Padding to `L` leaves the actions unchanged up to float rounding; see [Verification](#verification).

**State**
- Normalised with the checkpoint's statistics (`MEAN_STD`), then zero-padded to `S`.

**Noise**
- `[1, C, A]` standard normal.
- The tools here draw it as `torch.randn(1, C, A, generator=torch.Generator().manual_seed(seed + 1))`, the convention of `profile_smolvla.py`.

## Precision variants

| | fp32 | fp16 |
|---|---|---|
| Vision, connector, VLM layers, action expert | fp32 | fp16 weights and activations |
| `state_proj`, `action_in_proj`, `action_out_proj`, time MLP | fp32 | fp32 |
| In the VLM layers (`prefix`) and the action expert: attention scores and softmax, RoPE, RMSNorm statistics | fp32 | fp32 (LeRobot and transformers upcast) |
| In `vision` (SigLIP): attention scores and softmax | fp32 | fp16: transformers' SigLIP attention (SDPA) does not upcast them |
| `pixel_values`, `image_tokens`, `past_key_*`, `past_value_*` | fp32 | fp16 |
| `state`, `x_t`, `t`, `v_t` | fp32 | fp32 |

The fp16 graphs are strongly typed. The weights are cast in PyTorch before export, the profiler's `fp16all` layout, so every cast is explicit in the graph. TensorRT should build them with `STRONGLY_TYPED` and no FP16 builder flag, which TensorRT 11 removes. That is the path `trt_vision.py` takes for the vision graph. No TensorRT engine has been built from any of these graphs yet: the test machine has no GPU.

## What differs from LeRobot's code, and why

The wrappers in `smolvla_latency/split_graph.py` call LeRobot's own `embed_prefix`, `SmolVLMWithExpertModel.forward` and `denoise_step`. The prefix wrapper runs `embed_prefix` with `embed_image` as the identity, because its inputs are the vision graph's outputs. Beyond that, three things are swapped while the wrappers run:

1. **`FunctionalKVCache` replaces `DynamicCache` during denoising.**
   - `update(k, v, layer_idx)` returns `[prefix; new]` without storing anything.
   - `.layers[i].keys` and `.values` serve the cross-attention layers.
   - `crop()` has nothing to undo.
2. **The timestep embedding is computed in float32.**
   - LeRobot's `create_sinusoidal_pos_embedding` works in float64, which TensorRT does not support.
   - Its phases reach about 1600 rad, so a plain float32 version is off by up to 2e-4.
   - The replacement reduces the phase modulo one period exactly: frequency and time are split so the large product is exact in float32.
   - It is within 5e-7 of the float64 result for t in [0, 1].
3. **RoPE takes its timescale table from a buffer computed by torch.**
   - When the exporter folds `10000 ** (2i / Dh)` itself, numpy's float32 `pow` is one ulp off for some entries. That moves the rotation angles at position ~170 by ~1e-5 rad.
   - The halves are joined with `cat` instead of slice assignment, which avoids ScatterND.
   - In PyTorch it is bit-identical to LeRobot's `apply_rope`.

The tests check each one against LeRobot.

**Not supported** (the export raises `NotImplementedError` naming the setting):
- `empty_cameras > 0` or masked cameras;
- `adapt_to_pi_aloha`;
- `use_cache=False`;
- an `attention_mode` without cross-attention;
- real-time chunking (RTC);
- `add_image_special_tokens=True`, which puts image start and end tokens in the prefix;
- `prefix_length > 0`, which pads the prefix to a fixed length;
- an action expert with fewer layers than the VLM (`num_expert_layers`). With LeRobot's default interleaving, LeRobot's own denoising fails on such a checkpoint.

`export_split.py` checks the config before it reads any weights, and the model before it runs anything. A manifest describing one of the last three anyway is refused by the C++ loader with the setting's name: `prefix_len` then differs from `cameras * T + L + 1`, or `expert_layers` from `num_layers`.

## Running it

```bash
PY=~/venvs/smolvla/bin/python
$PY export_split.py --policy lerobot/smolvla_libero --out-dir onnx_split          # fp32 + fp16, with parity report
$PY ort_flow.py --onnx-dir onnx_split --policy lerobot/smolvla_libero --check     # chunk vs PyTorch
$PY tools/dump_bundle.py --policy lerobot/smolvla_libero --out onnx_split/bundle.bin
$PY ort_flow.py --onnx-dir onnx_split --bundle onnx_split/bundle.bin [--precision fp16] [--iters 20]
```

In Python:

```python
from ort_flow import SplitRuntime
rt = SplitRuntime.from_dir("onnx_split", "fp32")
actions = rt.predict_action_chunk(inputs, noise)   # inputs: pixel_values_<c>, lang_tokens, lang_mask, state
```

## Test bundle

`tools/dump_bundle.py` writes one synthetic observation in a flat binary file; `smolvla_latency/bundle.py` reads and writes it.

**Layout** (all integers little-endian):

| Offset | Size | Content |
|---|---|---|
| 0 | 8 | ASCII `SVLABNDL` |
| 8 | 8 | uint64 `Hn`, header length in bytes |
| 16 | `Hn` | UTF-8 JSON header, padded with spaces so `16 + Hn` is a multiple of 64 |
| `16 + Hn` | | data section |

**Header**
- `{"format": "smolvla-split-bundle", "version": 1, "meta": {...}, "tensors": [...]}`.
- Each tensor entry is `{"name", "dtype", "shape", "offset", "nbytes"}`.
- `offset` counts from the start of the data section and is a multiple of 64.
- Data is C-order, little-endian. dtypes: `float32`, `float16`, `int64`, `bool` (one byte, 0 or 1).
- The readers, `smolvla_latency/bundle.py` and the C++ `Bundle`, check every shape, size and offset without overflow. They also reject duplicate tensor names, a data section that does not start on a 64-byte boundary, and header fields of the wrong JSON type, `meta.euler` included. Their errors name the file, and the tensor where there is one.

**Tensors**

| Name | dtype | Shape | Content |
|---|---|---|---|
| `pixel_values_<c>` | float32 | `[1, 3, H, W]` | vision input per camera (cast to fp16 for the fp16 graph) |
| `lang_tokens`, `lang_mask` | int64, bool | `[1, L]` | prefix inputs |
| `state` | float32 | `[1, S]` | prefix input |
| `noise` | float32 | `[1, C, A]` | initial `x_t` |
| `actions` | float32 | `[1, C, action_dim]` | `predict_action_chunk` on exactly these inputs, normalised |
| `ref.image_tokens_<c>` | float32 | `[1, T, D]` | LeRobot's vision output |
| `ref.prefix_pad_mask`, `ref.past_key_<i>`, `ref.past_value_<i>` | | as `prefix` outputs | LeRobot's prefill cache |
| `ref.x_t_<k>`, `ref.v_t_<k>` | float32 | `[1, C, A]` | `denoise_step` input and output at step k |

`--no-intermediates` omits the `ref.*` tensors. `meta` holds:
- `config`, the same as the manifest's;
- the Euler `dt` and `t` values, as `"euler": {"dt": number, "t": [number, ...]}`;
- the task and seed;
- how much the fixed language padding moves the PyTorch actions.

## Verification

`tests/test_split_graph.py` runs offline on CPU with the tiny random checkpoint from `tests/make_fake_policy.py` on `tests/make_fake_smolvlm.py --tiny --kv-heads 2`: 2 cameras, 4 VLM layers, 2 key/value heads (6 query heads), `P = 177`. With a single key/value head, K/V read in `[seq, heads]` order instead of `[heads, seq]` would be the same numbers; with two, that mistake moves the action chunk by 0.24. The same checks were run once on the full-size fixture: SmolVLM2-500M layer sizes, 450M parameters, random weights, 2 cameras, 16 VLM layers, 5 key/value heads, `P = 177`.

Every graph is fed LeRobot's own inputs at its split point, and its output is compared with what LeRobot computed there (max abs; ONNX Runtime 1.31, CPU). The tiny column was measured with 1 and 4 PyTorch threads for the reference and 1 and 4 ONNX Runtime threads; where the PyTorch thread count changes the last digits, it gives the range. The ONNX Runtime thread count changed nothing.

| Check | Tiny | Full size |
|---|---|---|
| Prefix wrapper vs LeRobot prefill, in PyTorch | bit-identical keys, values and mask | not run |
| Expert wrapper vs `denoise_step`, in PyTorch, 10 steps | 1.2e-6 to 1.4e-6 (the float32 timestep embedding) | not run |
| Language padded to 48 vs to the longest (18), PyTorch actions | 4.8e-7 | 6.0e-7 |
| fp32 `vision` | 3.5e-6 | 1.4e-5 (relative L2 1.2e-6) |
| fp32 `prefix` keys and values | 4.2e-7 to 4.8e-7, mask equal | 3.2e-6, mask equal |
| fp32 `expert_step`, 10 steps | 1.1e-6 to 1.3e-6 | 2.4e-6 |
| fp32 action chunk: `ort_flow` vs `predict_action_chunk`, same noise | 4.8e-7 (test asserts < 1e-5) | 7.2e-7 |
| fp16 action chunk vs LeRobot fp32 | 1.9e-4 to 2.0e-4, cosine 0.9999999974 to 0.9999999978 | 1.0e-3, cosine 0.99999994 |
| fp16 per graph vs LeRobot fp32: vision, prefix, expert_step | 3.3e-3, 8.5e-4, 5.2e-4 | 9.1e-3, 2.6e-3, 1.3e-3 |

The test asserts each fp32 graph at 1e-4 and the fp32 chunk at 1e-5. fp16 is reported, not asserted tightly, and comes from random weights: a checkpoint trained in bf16 can hold activations beyond fp16's range (65504), so check its fp16 chunk for non-finite values before relying on it. The fp16 `vision` graph's attention scores, fp16 tensors (see [Precision variants](#precision-variants)), are the first place to look.

**Full-size graph files**

| Graph | fp32 | fp16 |
|---|---|---|
| vision | 376 MB | 189 MB |
| prefix | 747 MB | 374 MB |
| expert_step | 382 MB | 195 MB |

Inputs and outputs had the shapes in the tables above, with `D = 960`, `Hkv = 5` and `Dh = 64`.

**Not tested here**
- TensorRT engines built from these graphs; there is no GPU on the test machine.
- ONNX Runtime's CUDA execution provider.
- Real checkpoint weights; the Hub is unreachable from the test machine.

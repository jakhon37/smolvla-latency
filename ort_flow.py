"""Reference runtime for the split SmolVLA graphs: vision per camera, prefix once, Euler loop over expert_step.

    python ort_flow.py --onnx-dir onnx_split --policy lerobot/smolvla_libero --check
    python ort_flow.py --onnx-dir onnx_split --bundle bundle.bin [--precision fp16]

Does what SmolVLAPolicy.predict_action_chunk does after preprocessing: ONNX Runtime runs the graphs and
numpy runs LeRobot's Euler update (x <- x + dt * v_t, dt = -1/num_steps, t_k = 1 + k dt), so the same noise
gives the same action chunk. The output is in normalised action space, unpadded to the action dimension,
before the policy postprocessor. docs/SPLIT_GRAPH.md is the contract.

--policy builds one synthetic observation (profile_smolvla.py's inputs and noise convention) through the
checkpoint's preprocessor with the language padded to tokenizer_max_length; no weights are loaded unless
--check asks for the PyTorch comparison. --bundle takes inputs, noise and reference actions from a file
written by tools/dump_bundle.py, the same file a C++ runtime checks itself against. With --bundle only
numpy, onnx and onnxruntime are needed, so the graphs can be checked on a device without torch.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from smolvla_latency.bundle import read_bundle
from smolvla_latency.metrics import compare
from smolvla_latency.split_contract import GRAPHS, MANIFEST, euler_schedule, prefix_input_names

ONNX_DTYPES = {onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.FLOAT16: "float16",
               onnx.TensorProto.INT64: "int64", onnx.TensorProto.BOOL: "bool"}
ORT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int64)": np.int64,
              "tensor(bool)": np.bool_}


def onnx_io(path: Path) -> dict:
    """Input and output names, dtypes and shapes as stored in the graph; the weights are not read."""
    graph = onnx.load(str(path), load_external_data=False).graph

    def describe(values) -> list[dict]:
        return [{"name": v.name, "dtype": ONNX_DTYPES[v.type.tensor_type.elem_type],
                 "shape": [d.dim_value if d.HasField("dim_value") else d.dim_param
                           for d in v.type.tensor_type.shape.dim]} for v in values]

    return {"inputs": describe(graph.input), "outputs": describe(graph.output)}


class SplitRuntime:
    """The three graphs of one precision plus the Euler loop. Inputs are cast to each graph's declared dtype."""

    def __init__(self, sessions: dict[str, ort.InferenceSession], config: dict):
        self.sessions = sessions
        self.config = config
        self.dt, self.times = euler_schedule(config["num_steps"])
        self.input_dtypes = {n: {i.name: ORT_DTYPES[i.type] for i in s.get_inputs()} for n, s in sessions.items()}
        self.output_names = {n: [o.name for o in s.get_outputs()] for n, s in sessions.items()}
        self.timings: dict[str, list[float]] | None = None  # per-graph run times (ms) while not None

    @classmethod
    def from_dir(cls, onnx_dir: Path, precision: str = "fp32", providers: list[str] | None = None,
                 threads: int = 0, manifest: dict | None = None) -> "SplitRuntime":
        manifest = manifest or json.loads((Path(onnx_dir) / MANIFEST).read_text())
        if precision not in manifest["variants"]:
            raise SystemExit(f"{onnx_dir} has no {precision} graphs; it has {list(manifest['variants'])}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        graphs = manifest["variants"][precision]
        sessions = {name: ort.InferenceSession(str(Path(onnx_dir) / graphs[name]["file"]), opts,
                                               providers=providers or ["CPUExecutionProvider"])
                    for name in GRAPHS}
        return cls(sessions, manifest["config"])

    @property
    def provider(self) -> str:
        return self.sessions["expert_step"].get_providers()[0]

    def _run(self, name: str, feed: dict) -> dict[str, np.ndarray]:
        dtypes = self.input_dtypes[name]
        feed = {k: np.asarray(v, dtype=dtypes[k]) for k, v in feed.items()}
        t = time.perf_counter()
        outs = self.sessions[name].run(None, feed)
        if self.timings is not None:
            self.timings.setdefault(name, []).append((time.perf_counter() - t) * 1e3)
        return dict(zip(self.output_names[name], outs, strict=True))

    def image_tokens(self, pixel_values: np.ndarray) -> np.ndarray:
        return self._run("vision", {"pixel_values": pixel_values})["image_tokens"]

    def prefix(self, image_tokens: list[np.ndarray], lang_tokens, lang_mask, state) -> dict[str, np.ndarray]:
        """prefix_pad_mask and past_key_<i> / past_value_<i>, keyed by name: exactly expert_step's extra inputs."""
        names = prefix_input_names(len(image_tokens))
        return self._run("prefix", dict(zip(names, [*image_tokens, lang_tokens, lang_mask, state], strict=True)))

    def velocity(self, x_t: np.ndarray, t: np.ndarray, prefix: dict[str, np.ndarray]) -> np.ndarray:
        return self._run("expert_step", {"x_t": x_t, "t": t, **prefix})["v_t"]

    def predict_action_chunk(self, inputs: dict[str, np.ndarray], noise: np.ndarray) -> np.ndarray:
        """``inputs`` as split_graph.graph_inputs returns them; ``noise`` [1, chunk_size, max_action_dim]."""
        cams = range(len(self.config["camera_keys"]))
        tokens = [self.image_tokens(inputs[f"pixel_values_{c}"]) for c in cams]
        prefix = self.prefix(tokens, inputs["lang_tokens"], inputs["lang_mask"], inputs["state"])
        x_t = np.asarray(noise, dtype=np.float32)
        for t in self.times:
            x_t = x_t + self.dt * self.velocity(x_t, t, prefix)
        return x_t[:, :, : self.config["action_dim"]]


def policy_problem(args) -> tuple[dict, np.ndarray, dict]:
    """Graph inputs and noise for one synthetic observation, and the PyTorch actions when --check is set."""
    from smolvla_latency.loading import DEFAULT_TASK, load_config, load_policy, load_processors
    from smolvla_latency.split_graph import graph_inputs, reference_run, synthetic_batch

    task = args.task or DEFAULT_TASK
    if args.check:
        policy, pre_saved, _, _ = load_policy(args.policy, "cpu", args.revision)
        pre_fixed, _ = load_processors(policy.config, args.policy, "cpu", args.revision, "max_length")
        run = reference_run(policy, pre_fixed, pre_saved, task, args.seed)
        expected = {"torch_fixed_len": run.fixed.actions.numpy(),
                    f"torch_{policy.config.pad_language_to}": run.saved.actions.numpy()}
        return run.inputs, run.noise.numpy(), expected
    cfg = load_config(args.policy, "cpu", args.revision)
    pre_fixed, _ = load_processors(cfg, args.policy, "cpu", args.revision, "max_length")
    batch, noise = synthetic_batch(cfg, pre_fixed, task, args.seed)
    return graph_inputs(batch, cfg), noise.numpy(), {}


def bundle_problem(path: Path, config: dict) -> tuple[dict, np.ndarray, dict]:
    tensors, meta = read_bundle(path)
    if meta.get("config") != config:
        raise SystemExit(f"{path} was written for another policy or export: {meta.get('config')} vs {config}")
    names = [f"pixel_values_{c}" for c in range(len(config["camera_keys"]))] + ["lang_tokens", "lang_mask", "state"]
    return {k: tensors[k] for k in names}, tensors["noise"], {"bundle_actions": tensors["actions"]}


def percentiles(values: list[float]) -> dict[str, float]:
    # Not smolvla_latency.timing.summarize: that module needs torch, and --bundle runs without it.
    a = np.asarray(values)
    return {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "n": int(a.size)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", type=Path, required=True, help="output of export_split.py")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy", help="checkpoint whose config and preprocessor build the inputs")
    src.add_argument("--bundle", type=Path, help="inputs, noise and reference actions from tools/dump_bundle.py")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--task", default=None, help="default: smolvla_latency.loading.DEFAULT_TASK")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="with --policy: also run PyTorch predict_action_chunk")
    ap.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0: its default)")
    ap.add_argument("--iters", type=int, default=0, help="also time this many chunks")
    ap.add_argument("--out", type=Path, default=None, help="write a JSON report here")
    args = ap.parse_args()

    rt = SplitRuntime.from_dir(args.onnx_dir, args.precision, threads=args.threads)
    if args.policy:
        inputs, noise, expected = policy_problem(args)
    else:
        inputs, noise, expected = bundle_problem(args.bundle, rt.config)
    actions = rt.predict_action_chunk(inputs, noise)
    report = {"onnx_dir": str(args.onnx_dir), "precision": args.precision, "provider": rt.provider,
              "actions_shape": list(actions.shape), "nonfinite": int((~np.isfinite(actions)).sum())}
    print(f"{args.precision} action chunk {tuple(actions.shape)} via {rt.provider}")
    for name, ref in expected.items():
        report[f"vs_{name}"] = compare(actions, ref)
        print(f"  vs {name:<16} max_abs {report[f'vs_{name}']['max_abs']:.2e}  "
              f"cos {report[f'vs_{name}']['cosine']:.8f}")

    if args.iters:
        rt.timings = {}
        chunk_ms = []
        for _ in range(args.iters):
            t = time.perf_counter()
            rt.predict_action_chunk(inputs, noise)
            chunk_ms.append((time.perf_counter() - t) * 1e3)
        report["latency_ms"] = {"chunk": percentiles(chunk_ms), **{k: percentiles(v) for k, v in rt.timings.items()}}
        for k, v in report["latency_ms"].items():
            print(f"  {k:<11} p50 {v['p50']:9.2f} ms  p99 {v['p99']:9.2f} ms  (n={v['n']})")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Check this repo's precision handling against LeRobot itself (offline, CPU, tiny random model).

    python tests/check_lerobot_dtypes.py WORKDIR

1. The ``lerobot`` variant must give every parameter the same dtype as
   ``SmolVLAPolicy.from_pretrained`` (the path lerobot-eval takes) on a checkpoint saved with
   load_vlm_weights=true, and the same actions for the same observation and noise.
2. ``load_policy`` must hold the checkpoint's values exactly in fp32 even when the base VLM
   config carries torch_dtype=bfloat16.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402

from smolvla_latency.loading import load_policy, synthetic_observation  # noqa: E402
from smolvla_latency.precision import apply_variant, param_dtypes  # noqa: E402


def sh(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def main(work: Path) -> None:
    vlm, pol = work / "vlm-w", work / "policy-w"
    sh("tests/make_fake_smolvlm.py", str(vlm), "--tiny", "--weights")
    sh("tests/make_fake_policy.py", "--vlm", str(vlm), "--out", str(pol), "--bf16-flag")

    native = SmolVLAPolicy.from_pretrained(str(pol)).eval()  # what make_policy / lerobot-eval build
    ours, pre, _, _ = load_policy(str(pol), "cpu")
    apply_variant(ours, "lerobot")
    nd, od = param_dtypes(native), param_dtypes(ours)
    assert nd.keys() == od.keys(), set(nd) ^ set(od)
    diff = {k: (nd[k], od[k]) for k in nd if nd[k] != od[k]}
    assert not diff, f"{len(diff)} dtype mismatches, e.g. {list(diff.items())[:5]}"
    n_fp32_expert = sum(1 for k, v in nd.items() if "lm_expert" in k and v == "float32")
    assert n_fp32_expert > 0, "expected fp32 cross-attn k/v in the expert"

    obs = synthetic_observation(ours.config, seed=0)
    cfg = ours.config
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        acts = []
        for p in (native, ours):
            p.reset()
            acts.append(p.predict_action_chunk(pre(dict(obs)), noise=noise.clone()))
    max_abs = (acts[0] - acts[1]).abs().max().item()
    assert max_abs < 1e-6, f"lerobot variant actions differ from LeRobot's own load: {max_abs}"
    print(f"lerobot variant == LeRobot load: {len(nd)} params match dtypes "
          f"({n_fp32_expert} fp32 expert tensors), action max_abs {max_abs:.1e}")

    vlm16, pol16 = work / "vlm-bf16cfg", work / "policy-bf16cfg"
    sh("tests/make_fake_smolvlm.py", str(vlm16), "--tiny", "--config-dtype", "bfloat16")
    sh("tests/make_fake_policy.py", "--vlm", str(vlm16), "--out", str(pol16))
    ref, _, _, info = load_policy(str(pol16), "cpu")
    stored = load_file(str(pol16 / "model.safetensors"))
    state = ref.state_dict()
    bad = [k for k, v in stored.items()
           if state[k].dtype != torch.float32 or not torch.equal(state[k], v.to(torch.float32))]
    assert not bad, f"{len(bad)} tensors not exact fp32 copies, e.g. {bad[:3]}"
    print(f"fp32 reference exact with a bf16 base config ({len(stored)} tensors); "
          f"constructor built: {json.dumps(info['built_dtypes']['text_model'])}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

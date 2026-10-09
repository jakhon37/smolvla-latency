"""Save a random-weight SmolVLA checkpoint (LIBERO-shaped inputs) for offline smoke tests.

    python tests/make_fake_policy.py --vlm /tmp/fake-smolvlm --out /tmp/fake-policy [--bf16-flag]

The directory looks like a hub checkpoint: config.json, model.safetensors and the
pre/post-processor files, so every script here takes ``--policy /tmp/fake-policy``.
"""

import argparse
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", type=Path, required=True, help="dir from make_fake_smolvlm.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bf16-flag", action="store_true",
                    help="save load_vlm_weights=true, as LeRobot's training recipes do")
    a = ap.parse_args()

    torch.manual_seed(0)
    cfg = SmolVLAConfig(
        vlm_model_name=str(a.vlm.resolve()),
        device="cpu",
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    policy = SmolVLAPolicy(cfg)
    stats = {
        "observation.state": {"mean": torch.zeros(8), "std": torch.ones(8)},
        "action": {"mean": torch.zeros(7), "std": torch.ones(7)},
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    cfg.load_vlm_weights = a.bf16_flag
    policy.save_pretrained(a.out)
    pre.save_pretrained(a.out)
    post.save_pretrained(a.out)
    print(f"fake policy written to {a.out}")


if __name__ == "__main__":
    main()

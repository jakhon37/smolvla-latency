"""Build an offline stand-in for HuggingFaceTB/SmolVLM2-500M-Video-Instruct.

Only for smoke-testing this repo where the Hugging Face Hub is unreachable.
Weights are random and the tokenizer is a toy BPE, so any timing or accuracy
from it is meaningless. Layer sizes approximate the real 500M model so that
shapes, exports and code paths match.

    python tests/make_fake_smolvlm.py /tmp/fake-smolvlm [--tiny] [--weights] [--config-dtype bfloat16]

--weights also saves random VLM weights, so LeRobot's own load_vlm_weights=true path can run.
--config-dtype writes a torch_dtype into the config, as hub configs usually carry one.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (
    PreTrainedTokenizerFast,
    SmolVLMConfig,
    SmolVLMForConditionalGeneration,
    SmolVLMImageProcessor,
    SmolVLMProcessor,
    SmolVLMVideoProcessor,
)

SPECIAL = {
    "fake_image_token": "<fake_token_around_image>",
    "global_image_token": "<global-img>",
    "image_token": "<image>",
    "end_of_utterance_token": "<end_of_utterance>",
}

CORPUS = [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "put both the alphabet soup and the tomato sauce in the basket",
    "open the middle drawer of the cabinet",
    "turn on the stove and put the moka pot on it",
] * 50


def build(out: Path, tiny: bool, weights: bool = False, config_dtype: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    specials = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", *SPECIAL.values()]
    tok.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=600, special_tokens=specials))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        bos_token="<|im_start|>",
        pad_token="<|endoftext|>",
        extra_special_tokens=SPECIAL,
    )

    text = dict(
        hidden_size=960,
        intermediate_size=2560,
        num_hidden_layers=32,
        num_attention_heads=15,
        num_key_value_heads=5,
        head_dim=64,
        vocab_size=49280,
        rope_theta=100000.0,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
    )
    vision = dict(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=512,
        patch_size=16,
    )
    if tiny:
        text.update(num_hidden_layers=4, hidden_size=192, intermediate_size=512, num_attention_heads=3,
                    num_key_value_heads=1)
        vision.update(num_hidden_layers=2, hidden_size=192, intermediate_size=384, num_attention_heads=3)

    config = SmolVLMConfig(
        text_config=text,
        vision_config=vision,
        scale_factor=4,
        image_token_id=tokenizer.convert_tokens_to_ids(SPECIAL["image_token"]),
    )
    if config_dtype:
        config.dtype = config_dtype
        config.text_config.dtype = config_dtype
        config.vision_config.dtype = config_dtype
    config.save_pretrained(out)
    if weights:
        SmolVLMForConditionalGeneration(config).float().save_pretrained(out)

    processor = SmolVLMProcessor(
        image_processor=SmolVLMImageProcessor(),
        tokenizer=tokenizer,
        video_processor=SmolVLMVideoProcessor(),
        image_seq_len=64,
    )
    processor.save_pretrained(out)
    print(f"fake SmolVLM written to {out} (tiny={tiny}, weights={weights}, config_dtype={config_dtype})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--tiny", action="store_true", help="2-4 layers, for fast CI")
    p.add_argument("--weights", action="store_true", help="also save random weights")
    p.add_argument("--config-dtype", default=None, help="e.g. bfloat16")
    a = p.parse_args()
    build(a.out, a.tiny, a.weights, a.config_dtype)

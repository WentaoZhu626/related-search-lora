"""构造一个“玩具版” Qwen3（随机初始化的极小模型 + 字节级 tokenizer + Qwen 风格 chat template）。

只用于在没有 GPU、没有网络、没有真实模型权重的机器上冒烟测试整条流水线（训练 / 评估 / 体验脚本）。
它的输出毫无意义，只检查代码能否跑通、张量形状和损失是否正常。

    python tests/make_toy_model.py --output /tmp/toy_qwen3
"""

from __future__ import annotations

import argparse
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def build(output: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {token: index for index, token in enumerate(sorted(alphabet))}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    fast.chat_template = CHAT_TEMPLATE
    output.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(output)

    config = Qwen3Config(
        vocab_size=len(fast),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
        bos_token_id=None,
    )
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(output, safe_serialization=True)
    print(f"toy model written to {output} (vocab {len(fast)}, params {sum(p.numel() for p in model.parameters()):,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    build(Path(parser.parse_args().output))

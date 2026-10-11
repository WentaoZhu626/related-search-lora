"""推理：加载 base 模型 + LoRA adapter + 判断头，批量计算门控概率、批量生成相关词。

与训练完全对齐：
  * 门控：右侧 padding，取“prompt 最后一个 token”（位置 = 该样本真实长度 - 1）的最后一层隐藏状态，
    与训练时 ``gate_position`` 的含义一致；
  * 生成：左侧 padding + ``generate``（HF 会按 attention_mask 正确处理 position_ids）；
  * system prompt 从 adapter 目录下的 ``system_prompt.txt`` 读取（训练时保存的那一份），
    没有该文件才回退到 rs_prompt.SYSTEM_PROMPT。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from rs_common import from_pretrained_compat, parse_terms
from rs_model import backbone_last_hidden, build_gate_head, gather_positions
from rs_prompt import MAX_TERMS, SYSTEM_PROMPT, prompt_text


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    gate_head: torch.nn.Module
    system_prompt: str
    threshold: float
    gate_config: dict
    device: str

    @property
    def eos_token_ids(self) -> list[int]:
        ids = [self.tokenizer.eos_token_id]
        extra = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if isinstance(extra, int) and extra >= 0 and extra != self.tokenizer.unk_token_id and extra not in ids:
            ids.append(extra)
        return [i for i in ids if i is not None]


def load_model(
    adapter_dir: str | Path,
    base_model: str | None = None,
    device: str | None = None,
    merge: bool = False,
) -> LoadedModel:
    """``base_model`` 缺省时读取 adapter_config.json 里记录的训练时 base 路径。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gate_config = json.loads((adapter_dir / "gate_config.json").read_text(encoding="utf-8"))
    prompt_path = adapter_dir / "system_prompt.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else SYSTEM_PROMPT
    if base_model is None:
        base_model = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base = from_pretrained_compat(
        AutoModelForCausalLM, base_model, dtype, attn_implementation="sdpa", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    if merge:
        model = model.merge_and_unload()
    model.to(device).eval()
    model.config.use_cache = True

    gate_head = build_gate_head(int(gate_config["hidden_size"]), float(gate_config.get("gate_head_dropout", 0.1)))
    gate_head.load_state_dict(torch.load(adapter_dir / "gate_head.pt", map_location="cpu", weights_only=True))
    gate_head.to(device).eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        gate_head=gate_head,
        system_prompt=system_prompt,
        threshold=float(gate_config["threshold"]),
        gate_config=gate_config,
        device=device,
    )


def _encode_prompts(lm: LoadedModel, queries: list[str], padding_side: str) -> dict[str, torch.Tensor]:
    texts = [prompt_text(lm.tokenizer, query, lm.system_prompt) for query in queries]
    lm.tokenizer.padding_side = padding_side
    batch = lm.tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    # 只保留模型需要的两项（有的 tokenizer 还会返回 token_type_ids，generate 会因此报错）。
    return {key: batch[key].to(lm.device) for key in ("input_ids", "attention_mask")}


def _length_order(queries: list[str]) -> list[int]:
    return sorted(range(len(queries)), key=lambda index: len(queries[index]))


@torch.inference_mode()
def gate_probabilities(lm: LoadedModel, queries: list[str], batch_size: int = 64) -> list[float]:
    """P(应展示相关搜索)；返回顺序与 queries 一致。"""
    order = _length_order(queries)
    result = [0.0] * len(queries)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = _encode_prompts(lm, [queries[i] for i in indices], "right")
        _, hidden = backbone_last_hidden(lm.model, batch["input_ids"], batch["attention_mask"])
        positions = batch["attention_mask"].sum(dim=1) - 1
        logits = lm.gate_head(gather_positions(hidden, positions).float()).squeeze(-1)
        for index, value in zip(indices, torch.sigmoid(logits).cpu().tolist()):
            result[index] = value
    return result


@torch.inference_mode()
def generate_raw(
    lm: LoadedModel,
    queries: list[str],
    batch_size: int = 32,
    max_new_tokens: int = 96,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
) -> list[str]:
    """贪心（或 beam）解码，返回模型原始输出文本；顺序与 queries 一致。"""
    order = _length_order(queries)
    result = [""] * len(queries)
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = _encode_prompts(lm, [queries[i] for i in indices], "left")
        generated = lm.model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            eos_token_id=lm.eos_token_ids,
            pad_token_id=lm.tokenizer.pad_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
        )
        new_tokens = generated[:, batch["input_ids"].size(1) :]
        texts = lm.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for index, text in zip(indices, texts):
            result[index] = text.strip()
    return result


def generate_terms(lm: LoadedModel, queries: list[str], **kwargs) -> tuple[list[str], list[list[str]]]:
    raw = generate_raw(lm, queries, **kwargs)
    return raw, [parse_terms(text, MAX_TERMS) for text in raw]

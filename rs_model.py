"""模型公共部分：判断头、只取 backbone 隐藏状态的前向。

判断头读取“assistant 开始生成之前的最后一个 prompt token”的最后一层隐藏状态（经过最终 RMSNorm）。
训练和推理都走同一条路径，因此 ``gate_position`` 的含义完全一致。
"""

from __future__ import annotations

import torch
from torch import nn


def build_gate_head(hidden_size: int, dropout: float = 0.1) -> nn.Module:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))


def unwrap_causal_lm(model):
    """PeftModel -> 内部的 *ForCausalLM（LoRA 层已经原地注入其中）。"""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def backbone_last_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """只跑 transformer 主干，返回 (causal_lm, last_hidden_state)。

    这样可以避免在所有 prompt 位置上计算 15 万词表大小的 logits：
    判断头只需要隐藏状态，生成损失只需要答案位置的 logits（见训练脚本）。
    """
    causal = unwrap_causal_lm(model)
    hidden = causal.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    return causal, hidden


def gather_positions(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    index = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[index, positions]

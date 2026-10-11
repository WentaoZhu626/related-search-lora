"""Qwen3-4B-Instruct-2507 + LoRA + 判断头（gate head）联合训练。

保留旧版 train_qwen_instruct_gate_lora.py 的主体设计：
  * L = generation_loss_weight * L_gen + gate_loss_weight * L_gate；
  * 正样本：生成 loss（只算答案 token）+ 判断头 BCE；负样本：只算判断头 BCE；
  * 判断头读 prompt 最后一个 token 的隐藏状态；BCE 的 pos_weight = 负样本数 / 正样本数；
  * 验证集上用 balanced accuracy 选择判断阈值，写入 gate_config.json。

与旧版的差别（都不改变训练目标）：
  1. 使用 rs_prompt.py 里的新 system prompt，并把它随 adapter 一起保存；
  2. 手工拼接“prompt + 答案 + <|im_end|>”并分别 tokenize，兼容不同 transformers 版本；
     超长样本直接丢弃并计数，而不是截断（截断会让答案缺少结束符）；
  3. 只在“答案位置”计算 lm_head/logits，不再对所有 prompt 位置算 15 万词表的 logits，省显存和时间；
  4. 评估时 prediction_loss_only=True，避免 Trainer 把整批 logits 累积到显存里；
  5. 每个 checkpoint 也保存 gate_head.pt，支持 --resume-from-checkpoint；
  6. 数据目录换成 prepare_ood_split.py 的输出（train / validation；test_id / test_ood 只在评估脚本里用）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from rs_common import from_pretrained_compat, read_jsonl, runtime_metadata, set_seed, target_text
from rs_metrics import binary_metrics
from rs_model import backbone_last_hidden, build_gate_head, gather_positions
from rs_prompt import PROMPT_VERSION, SYSTEM_PROMPT, check_template_consistency, encode_example

DEFAULT_ROOT = "/root/autodl-tmp/related_research"


class ListDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


@dataclass
class GateCollator: 
    # 这个类负责把多条训练样本整理成一个 batch：把不同长度的文本补齐，并转换成模型可以使用的张量
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # 确定这一批要补齐到多长
        max_len = max(len(item["input_ids"]) for item in features)
        max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "gate_positions": torch.tensor([item["gate_position"] for item in features], dtype=torch.long),
            "gate_labels": torch.tensor([item["gate_label"] for item in features], dtype=torch.float32),
            "sample_weights": torch.tensor([item.get("sample_weight", 1.0) for item in features], dtype=torch.float32),
        }


class GateTrainer(Trainer):
    def __init__(
        self,
        *args,
        generation_loss_weight: float = 1.0,    # 这个参数是生成损失前面的乘数，用于衡量生成损失衡量模型生成的相关搜索词，与训练答案有多大差距的重要性
        gate_loss_weight: float = 1.0,  # 判断损失衡量模型对“应该展示／不应该展示”的预测是否正确。这个参数是判断损失前面的乘数
        gate_pos_weight: float = 1.0,   # 判断任务内部，正样本的权重
        label_smoothing: float = 0.0,   # 生成任务的标签平滑程度
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generation_loss_weight = generation_loss_weight
        self.gate_loss_weight = gate_loss_weight
        self.gate_pos_weight = gate_pos_weight
        self.label_smoothing = label_smoothing
        self._parts: dict[str, list[float]] = {"lm_loss": [], "gate_loss": []}

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        # 取出正确答案和辅助信息
        labels = inputs.pop("labels")   # 生成任务的正确 token
        gate_positions = inputs.pop("gate_positions")   # 判断头应该读取哪个 token 位置
        gate_labels = inputs.pop("gate_labels") # 是否应该展示
        sample_weights = inputs.pop("sample_weights")   # 每条样本的训练权重

        causal, hidden = backbone_last_hidden(model, inputs["input_ids"], inputs["attention_mask"])

        # 生成损失：位置 t 的隐藏状态预测 labels[t+1]；只对有标签（答案）的位置计算 logits。
        shift_labels = labels[:, 1:]
        answer_mask = shift_labels.ne(-100)
        if bool(answer_mask.any()):
            answer_hidden = hidden[:, :-1][answer_mask]
            logits = causal.lm_head(answer_hidden).float()
            token_losses = F.cross_entropy(
                logits, shift_labels[answer_mask], reduction="none", label_smoothing=self.label_smoothing
            )
            token_weights = sample_weights[:, None].expand_as(shift_labels)[answer_mask]
            lm_loss = (token_losses * token_weights).sum() / token_weights.sum().clamp_min(1.0)
        else:
            lm_loss = hidden.sum() * 0.0

        gate_logits = model.gate_head(gather_positions(hidden, gate_positions).float()).squeeze(-1)
        gate_losses = F.binary_cross_entropy_with_logits(
            gate_logits,
            gate_labels,
            pos_weight=torch.tensor(self.gate_pos_weight, device=gate_logits.device),
            reduction="none",
        )
        gate_loss = (gate_losses * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
        loss = self.generation_loss_weight * lm_loss + self.gate_loss_weight * gate_loss

        if model.training:
            self._parts["lm_loss"].append(float(lm_loss.detach()))
            self._parts["gate_loss"].append(float(gate_loss.detach()))
        if return_outputs:
            return loss, {"gate_logits": gate_logits.detach(), "lm_loss": lm_loss.detach(), "gate_loss": gate_loss.detach()}
        return loss

    def log(self, logs: dict, *args, **kwargs):
        # 把两项损失分开记录，方便区分“生成变差”还是“门控变差”。
        if self._parts["lm_loss"] and "loss" in logs:
            for name, values in self._parts.items():
                logs[name] = round(sum(values) / len(values), 6)
                values.clear()
        super().log(logs, *args, **kwargs)


class GateHeadCheckpointCallback(TrainerCallback):
    """Trainer 的 checkpoint 只保存 LoRA；这里把判断头也存进去。"""

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "gate_head"):
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint_dir.exists():
                torch.save(model.gate_head.state_dict(), checkpoint_dir / "gate_head.pt")
        return control


def tensorboard_or_none() -> list[str]:
    """装了 tensorboard 就写日志，没装也不报错。"""
    try:
        import tensorboard  # noqa: F401

        return ["tensorboard"]
    except ModuleNotFoundError:
        try:
            import tensorboardX  # noqa: F401

            return ["tensorboard"]
        except ModuleNotFoundError:
            print("未安装 tensorboard，跳过 TensorBoard 日志（pip install tensorboard 可开启）。", flush=True)
            return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{DEFAULT_ROOT}/models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data-dir", default=f"{DEFAULT_ROOT}/data/ood_split_v3", help="prepare_ood_split.py 的输出目录")
    parser.add_argument("--output-dir", default=f"{DEFAULT_ROOT}/outputs/ood_v3_qwen3_4b_gate_lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8, help="单卡 batch；24GB 显存放不下就降到 4 并把梯度累积加倍")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="有效 batch = batch-size × 该值（默认 32）")
    parser.add_argument("--max-length", type=int, default=384, help="prompt + 答案的 token 上限，超长样本丢弃")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="显存够时关掉可提速 20～30%")
    parser.add_argument("--generation-loss-weight", type=float, default=1.0)
    parser.add_argument("--gate-loss-weight", type=float, default=1.0)
    parser.add_argument("--gate-pos-weight", type=float)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gate-head-dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--smoke", action="store_true", help="冒烟测试：每类 16 条、只训练 5 步")
    return parser.parse_args()

# 推理模式，关闭梯度计算
@torch.inference_mode()
def validation_gate_probabilities(model, dataset: Dataset, collator: GateCollator, batch_size: int):
    # 在验证集上，让判断头给每条样本打分，返回真实标签和“应该展示相关搜索”的预测概率
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    labels: list[int] = []
    probabilities: list[float] = []
    model.eval()
    device = next(model.parameters()).device
    for batch in loader:
        _, hidden = backbone_last_hidden(model, batch["input_ids"].to(device), batch["attention_mask"].to(device))
        # 让判断头输出分数，从每条样本中取出 prompt 最后一个 token 的隐藏向量，交给判断头，得到每条样本一个原始分数
        logits = model.gate_head(gather_positions(hidden, batch["gate_positions"].to(device)).float()).squeeze(-1)
        # 把分数变成概率，并收集真实标签
        probabilities.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(batch["gate_labels"].long().tolist())
    return labels, probabilities


def build_dataset(rows: list[dict], tokenizer, args, name: str) -> tuple[ListDataset, dict]:
    # 把原始数据转换成模型可以训练的样本，并统计转换后的数据情况
    items, dropped = [], 0
    for row in rows:
        should_show = int(row.get("should_show", bool(row.get("related_searches"))))    # 读取每条数据的展示标签
        answer = target_text(row["related_searches"]) if should_show and row.get("related_searches") else None  # 如果应该展示，而且有相关词，就调用 target_text() 将列表整理成答案字符串
        encoded = encode_example(tokenizer, str(row["query"]), answer, args.max_length, SYSTEM_PROMPT)  # 把文字编码成 token 和训练标签
        # 正样本输入：prompt + 答案 + 结束符    生成 labels：prompt 部分为 -100，答案和结束符有有效标签
        # 负样本输入：只有 prompt       生成 labels：全部为 -100
        if encoded is None:
            # 丢弃超长样本
            dropped += 1
            continue
        encoded["gate_label"] = float(should_show)
        encoded["sample_weight"] = float(row.get("sample_weight", 1.0))
        items.append(encoded)
    # 统计保留下来的数据
    lengths = [len(item["input_ids"]) for item in items]
    stats = {
        "split": name,
        "kept": len(items),
        "dropped_too_long": dropped,
        "positive": int(sum(item["gate_label"] for item in items)),
        "max_tokens": max(lengths) if lengths else 0,
        "mean_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
    }
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return ListDataset(items), stats


def main() -> None:
    # 读取配置
    args = parse_args()
    # 设置随机种子
    set_seed(args.seed)
    output_dir = Path(args.output_dir + ("_smoke" if args.smoke else ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    # 加载与模型配套的分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    # 如果没有专门的 PAD token（用于补气长度），就把结束符 EOS 同时用作补齐符号
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 检查是否符合qwen的chat template
    consistency = check_template_consistency(tokenizer)
    print("chat template 自检：", json.dumps(consistency, ensure_ascii=False), flush=True)
    if not consistency["consistent"]:
        print("提示：手工拼接格式与官方模板渲染的完整对话不完全一致。训练/推理仍然自洽，但请确认这是预期的。", flush=True)
    # 读取训练集和测试集
    train_rows = read_rows(Path(args.data_dir) / "train.jsonl")
    eval_rows = read_rows(Path(args.data_dir) / "validation.jsonl")
    # 冒烟测试时，只取少量数据
    if args.smoke:
        train_rows = [r for r in train_rows if int(r["should_show"]) == 1][:16] + [
            r for r in train_rows if int(r["should_show"]) == 0
        ][:16]
        eval_rows = eval_rows[:32]
    # 编码数据，并创建 Collator
    train_ds, train_stats = build_dataset(train_rows, tokenizer, args, "train")
    eval_ds, eval_stats = build_dataset(eval_rows, tokenizer, args, "validation")
    collator = GateCollator(tokenizer.pad_token_id)

    use_cuda = torch.cuda.is_available()  # 无 GPU 时（本地冒烟测试）退回 fp32，其余设置不变
    # 加载预训练模型
    model = from_pretrained_compat(
        AutoModelForCausalLM,
        args.model,
        torch.bfloat16 if use_cuda else torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=True, # 允许加载模型提供的自定义实现
    )
    model.config.use_cache = False
    # 给模型加入 LoRA
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha or (2 * args.lora_r),
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],   # 前四个是注意力模块中的投影层，后三个是前馈网络中的投影层。也就是同时在注意力和前馈部分加入 LoRA
        ),
    )
    hidden_size = int(model.config.hidden_size)
    model.add_module("gate_head", build_gate_head(hidden_size, args.gate_head_dropout))
    if args.resume_from_checkpoint:
        head_path = Path(args.resume_from_checkpoint) / "gate_head.pt"
        if head_path.exists():
            model.gate_head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
            print(f"已从 {head_path} 恢复判断头", flush=True)
    model.print_trainable_parameters()

    strategy = "steps" if args.smoke else "epoch"
    # 训练配置
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=5 if args.smoke else -1,
        per_device_train_batch_size=1 if args.smoke else args.batch_size,
        per_device_eval_batch_size=1 if args.smoke else args.eval_batch_size,
        gradient_accumulation_steps=1 if args.smoke else args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=use_cuda,
        tf32=True if use_cuda else None,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
        eval_strategy=strategy,
        save_strategy=strategy,
        eval_steps=5 if args.smoke else None,
        save_steps=5 if args.smoke else None,
        logging_steps=1 if args.smoke else 10,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=tensorboard_or_none(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        prediction_loss_only=True,
        label_names=["labels"],
        seed=args.seed,
        data_seed=args.seed,
    )
    neg = sum(int(row["should_show"]) == 0 for row in train_rows)
    pos = sum(int(row["should_show"]) == 1 for row in train_rows)
    trainer = GateTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds, # 训练集
        eval_dataset=eval_ds,   # 验证集
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[GateHeadCheckpointCallback()],
        # 自定义的参数
        generation_loss_weight=args.generation_loss_weight,
        gate_loss_weight=args.gate_loss_weight,
        label_smoothing=args.label_smoothing,
        # 让正负样本对判断头的总体贡献相当（数据约 76% 为正样本）。
        gate_pos_weight=args.gate_pos_weight if args.gate_pos_weight is not None else neg / max(1, pos),
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    eval_result = trainer.evaluate()

    labels, probabilities = validation_gate_probabilities(model, eval_ds, collator, args.eval_batch_size)
    # 遍历验证集上出现过的每个概率值作为候选阈值，按 balanced accuracy 选最优。
    candidates = [binary_metrics(labels, probabilities, value) for value in sorted({0.0, 1.0, *probabilities})]
    best_gate = max(candidates, key=lambda item: (item["balanced_accuracy"], item["f1"]))

    final_dir = output_dir / "final_adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save(model.gate_head.state_dict(), final_dir / "gate_head.pt")
    (final_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    gate_config = {
        "hidden_size": hidden_size,
        "gate_head_dropout": args.gate_head_dropout,
        "threshold": best_gate["threshold"],
        "threshold_metric": "balanced_accuracy",
        "threshold_selected_on": "validation (in-distribution)",
        "validation_metrics": best_gate,
        "gate_loss_weight": args.gate_loss_weight,
        "generation_loss_weight": args.generation_loss_weight,
        "prompt_version": PROMPT_VERSION,
    }
    (final_dir / "gate_config.json").write_text(json.dumps(gate_config, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = runtime_metadata(
        {
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "smoke": args.smoke,
            "data_dir": args.data_dir,
            "train_data": train_stats,
            "validation_data": eval_stats,
            "gate_pos_weight": trainer.gate_pos_weight,
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_result,
            "gate_validation": best_gate,
            "arguments": vars(args),
        }
    )
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def read_rows(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    for row in rows:
        row.setdefault("should_show", int(bool(row.get("related_searches"))))
    return rows


if __name__ == "__main__":
    main()

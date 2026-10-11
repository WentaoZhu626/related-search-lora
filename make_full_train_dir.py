"""OOD 实验结束后，生成“全量训练”用的数据目录（用于最终上线的模型）。

    python make_full_train_dir.py --split-dir <ood_split_v3> --output-dir <full_split_v3>

  * train.jsonl      = 原 train + test_id + test_ood + buffer（几乎用上全部数据）
  * validation.jsonl = 原 validation（保持留出，用来选门控阈值、看训练曲线）

之后用同样的超参数训练：
    python train_gate_lora.py --data-dir <full_split_v3> --output-dir <.../full_qwen3_4b_gate_lora>

注意：全量模型没有干净的测试集了，OOD / ID 结论只来自前一个实验的模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rs_common import dedupe_key, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source, target = Path(args.split_dir), Path(args.output_dir)
    validation = read_jsonl(source / "validation.jsonl")
    held_out = {dedupe_key(row["query"]) for row in validation}
    train: list[dict] = []
    for name in ["train", "test_id", "test_ood", "buffer"]:
        path = source / f"{name}.jsonl"
        if path.exists():
            train.extend(row for row in read_jsonl(path) if dedupe_key(row["query"]) not in held_out)
    write_jsonl(target / "train.jsonl", train)
    write_jsonl(target / "validation.jsonl", validation)
    print(f"train {len(train)} 条，validation {len(validation)} 条 -> {target}")


if __name__ == "__main__":
    main()

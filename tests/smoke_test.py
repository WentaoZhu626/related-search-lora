"""CPU 冒烟测试：不需要 GPU、不需要网络、不需要真实模型。

    cd related_search_ood_v3
    python tests/smoke_test.py                    # 用合成数据
    python tests/smoke_test.py --csv /path/to/cleaned_dataset.csv --sample 3000   # 用真实数据的子集

检查内容：
  1. 划分脚本：无泄漏、OOD 簇整体被留出；
  2. tokenization：labels 只覆盖答案、gate_position 指向 assistant 起始标记；
  3. 训练脚本 --smoke：损失有限、产出 final_adapter（含 gate_head.pt / gate_config.json / system_prompt.txt）；
  4. 推理：右 padding 的门控概率与逐条计算一致（batch 内 padding 不影响结果）；
  5. 评估脚本：能写出 summary.md / *_metrics.json；
  6. 体验脚本：能跑通。
玩具模型是随机初始化的，输出没有意义，这里只验证“代码是否正确连通”。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

POOL = "山水花鸟鱼虫风雨雷电红黄蓝绿茶酒糖盐猫狗牛羊书画琴棋衣帽鞋袜车船飞机桌椅灯窗门墙砖瓦木石金银铜铁玉珠宝钻"


def synthetic_csv(path: Path, themes: int = 10, per_theme: int = 150, seed: int = 0) -> None:
    rng = random.Random(seed)
    chars = list(POOL)
    rng.shuffle(chars)
    size = len(chars) // themes
    rows = []
    for t in range(themes):
        pool = chars[t * size : (t + 1) * size]
        for _ in range(per_theme):
            query = "".join(rng.choice(pool) for _ in range(rng.randint(2, 4)))
            related = []
            if rng.random() < 0.76:
                related = ["".join(rng.choice(pool) for _ in range(rng.randint(2, 4))) for _ in range(rng.randint(2, 5))]
            rows.append([query] + related + [""] * (11 - len(related)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query"] + [f"相关搜索词{i}" for i in range(1, 12)])
        writer.writerows(rows)


def sample_csv(source: Path, target: Path, n: int, seed: int = 0) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header, rows = next(reader), list(reader)
    random.Random(seed).shuffle(rows)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows[:n])


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    work = Path(args.work_dir or tempfile.mkdtemp(prefix="rs_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # ---- 数据 + 玩具模型
    csv_path = work / "data.csv"
    if args.csv:
        sample_csv(Path(args.csv), csv_path, args.sample)
    else:
        synthetic_csv(csv_path)
    toy = work / "toy_model"
    run([py, "tests/make_toy_model.py", "--output", str(toy)])

    # ---- 1. 划分
    split = work / "split"
    run(
        [py, "prepare_ood_split.py", "--input", str(csv_path), "--output-dir", str(split),
         "--embedding-model", "tfidf", "--clusters", "10", "--ood-clusters", "2",
         "--validation-size", "100", "--test-id-size", "150", "--max-rate-gap", "0.3"]
    )
    from rs_common import dedupe_key, read_jsonl

    parts = {name: read_jsonl(split / f"{name}.jsonl") for name in ["train", "validation", "test_id", "test_ood"]}
    keys = {name: {dedupe_key(r["query"]) for r in rows} for name, rows in parts.items()}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (keys[a] & keys[b]), f"query 泄漏: {a} ∩ {b}"
    ood_clusters = {r["cluster_id"] for r in parts["test_ood"]}
    assert ood_clusters.isdisjoint({r["cluster_id"] for r in parts["train"]}), "OOD 簇出现在训练集"
    print("[1] 划分 OK：", {k: len(v) for k, v in parts.items()}, "OOD 簇", sorted(ood_clusters))

    # ---- 2. tokenization
    from transformers import AutoTokenizer

    from rs_common import target_text
    from rs_prompt import SYSTEM_PROMPT, check_template_consistency, encode_example, prompt_text

    tokenizer = AutoTokenizer.from_pretrained(toy)
    assert check_template_consistency(tokenizer)["consistent"], "模板自检失败"
    positive = next(r for r in parts["train"] if r["should_show"])
    negative = next(r for r in parts["train"] if not r["should_show"])
    answer = target_text(positive["related_searches"])
    enc = encode_example(tokenizer, positive["query"], answer, 4096, SYSTEM_PROMPT)
    prompt_len = len(tokenizer(prompt_text(tokenizer, positive["query"]), add_special_tokens=False)["input_ids"])
    assert enc["gate_position"] == prompt_len - 1
    assert all(x == -100 for x in enc["labels"][:prompt_len]) and all(x != -100 for x in enc["labels"][prompt_len:])
    assert enc["input_ids"][-1] == tokenizer.eos_token_id
    assert tokenizer.decode(enc["input_ids"][prompt_len:-1]) == answer
    enc_neg = encode_example(tokenizer, negative["query"], None, 4096, SYSTEM_PROMPT)
    assert all(x == -100 for x in enc_neg["labels"])
    assert encode_example(tokenizer, positive["query"], answer, 10, SYSTEM_PROMPT) is None
    print("[2] tokenization OK；单条样本 token 数", len(enc["input_ids"]))

    # ---- 3. 训练（--smoke）
    out = work / "train_out"
    run(
        [py, "train_gate_lora.py", "--model", str(toy), "--data-dir", str(split), "--output-dir", str(out),
         "--smoke", "--max-length", "4096"]
    )
    adapter = work / "train_out_smoke" / "final_adapter"
    for name in ["adapter_model.safetensors", "adapter_config.json", "gate_head.pt", "gate_config.json", "system_prompt.txt"]:
        assert (adapter / name).exists(), f"缺少 {name}"
    metadata = json.loads((work / "train_out_smoke" / "run_metadata.json").read_text(encoding="utf-8"))
    loss = metadata["train_metrics"]["train_loss"]
    assert loss == loss and loss < 1e4, f"训练损失异常: {loss}"
    print("[3] 训练 OK，train_loss =", loss)

    # ---- 4. 推理一致性
    from rs_infer import gate_probabilities, generate_terms, load_model

    lm = load_model(adapter, str(toy), device="cpu")
    queries = [r["query"] for r in parts["test_id"][:12]]
    batched = gate_probabilities(lm, queries, batch_size=12)
    single = [gate_probabilities(lm, [q], batch_size=1)[0] for q in queries]
    assert max(abs(a - b) for a, b in zip(batched, single)) < 1e-4, "batch padding 影响了门控概率"
    raw, terms = generate_terms(lm, queries[:4], batch_size=4, max_new_tokens=12)
    raw_single = [generate_terms(lm, [q], batch_size=1, max_new_tokens=12)[0][0] for q in queries[:4]]
    assert raw == raw_single, "左 padding 批量生成与逐条生成不一致"
    print("[4] 推理一致性 OK；示例输出:", raw[:2])
    del lm

    # ---- 5. 评估
    evaluation = work / "eval"
    run(
        [py, "evaluate_gate.py", "--adapter", str(adapter), "--base-model", str(toy), "--data-dir", str(split),
         "--output-dir", str(evaluation), "--limit", "40", "--max-new-tokens", "16", "--bootstrap", "50",
         "--batch-size", "8", "--soft-embedding-model", "tfidf"]
    )
    for name in ["summary.md", "summary.json", "test_id_metrics.json", "test_ood_metrics.json",
                 "test_id_predictions.jsonl", "test_ood_predictions.jsonl"]:
        assert (evaluation / name).exists(), f"缺少 {name}"
    print("[5] 评估 OK")

    # ---- 6. 体验脚本
    run([py, "try_gate.py", "--adapter", str(adapter), "--base-model", str(toy),
         "--query", parts["test_id"][0]["query"], "--always-generate", "--max-new-tokens", "12"])
    print("\n全部冒烟测试通过。工作目录：", work)
    if not args.keep and not args.work_dir:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

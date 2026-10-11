"""用同一个模型评估 test_id（同分布）和 test_ood（未见过的语义簇），并给出对比。

流程：
  1. 对每个测试集算门控概率；只对“教师标注应展示”或“门控判为展示”的样本生成（其余样本不需要）；
  2. 汇总所有出现过的词，（可选）一次性算 embedding 做软匹配；
  3. 计算指标并写出：
        <output-dir>/<split>_predictions.jsonl   逐条预测
        <output-dir>/<split>_metrics.json        该测试集的全部指标
        <output-dir>/summary.json / summary.md   ID vs OOD 对比表

门控阈值默认取训练时在“ID 验证集”上选出的值（gate_config.json），不会在 OOD 测试集上重新选。
指标口径：
  gate                   二分类指标（是否应展示相关搜索），含 AUROC/AP/校准
  forced_generation      在教师标注应展示的样本上强制生成——衡量生成能力本身
  generation_when_shown  在“应展示且门控也判为展示”的样本上——真实上线时能看到的那部分
  end_to_end             全部样本：门控判为不展示则输出空；空集合对空集合算命中
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rs_common import normalize_term, read_jsonl, write_jsonl
from rs_metrics import (
    binary_metrics,
    bootstrap_intervals,
    calibration_metrics,
    generation_metrics,
    output_health_metrics,
    ranking_metrics,
    soft_match_scores,
)

DEFAULT_ROOT = "/root/autodl-tmp/related_research"
CORE_KEYS = [
    ("gate.roc_auc", "门控 AUROC"),
    ("gate.balanced_accuracy", "门控 balanced accuracy"),
    ("gate.recall", "门控 recall（该展示的展示了多少）"),
    ("gate.specificity", "门控 specificity（不该展示的拦住了多少）"),
    ("forced_generation.micro_f1", "强制生成 精确匹配 micro-F1"),
    ("forced_generation.soft_f1", "强制生成 软匹配 F1（embedding）"),
    ("end_to_end.exact_set_match", "端到端 exact set match"),
    ("data.gold_term_seen_rate", "标准答案词在训练集中出现过的比例"),
    ("data.predicted_term_seen_rate", "模型输出词在训练集中出现过的比例"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=f"{DEFAULT_ROOT}/outputs/ood_v3_qwen3_4b_gate_lora/final_adapter")
    parser.add_argument("--base-model", default=None, help="缺省则读取 adapter_config.json 里训练时的 base 路径")
    parser.add_argument("--data-dir", default=f"{DEFAULT_ROOT}/data/ood_split_v3")
    parser.add_argument("--output-dir", default=None, help="缺省为 <adapter 的上一级>/eval")
    parser.add_argument("--splits", nargs="+", default=["test_id", "test_ood"], help="可加上 validation")
    parser.add_argument("--threshold", type=float, default=None, help="缺省使用 gate_config.json 里的阈值")
    parser.add_argument("--batch-size", type=int, default=32, help="生成的 batch；门控使用 2 倍")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap 次数，0 表示不计算置信区间")
    parser.add_argument("--soft-embedding-model", default="Qwen/Qwen3-Embedding-0.6B", help="填 none 则跳过软匹配")
    parser.add_argument("--limit", type=int, default=None, help="调试用：每个测试集只取前 N 条")
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def dig(mapping: dict, dotted: str):
    value = mapping
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def train_term_set(data_dir: Path) -> set[str]:
    path = data_dir / "train.jsonl"
    if not path.exists():
        return set()
    return {normalize_term(t) for row in read_jsonl(path) for t in row.get("related_searches", [])}


def run_predictions(lm, rows: list[dict], args) -> None:
    """就地写入 gate_probability / gate_shown / raw_output / predicted_terms / final_terms。"""
    from rs_infer import gate_probabilities, generate_terms

    threshold = args.threshold if args.threshold is not None else lm.threshold
    queries = [row["query"] for row in rows]
    probabilities = gate_probabilities(lm, queries, batch_size=args.batch_size * 2)
    need = [i for i, row in enumerate(rows) if int(row["should_show"]) == 1 or probabilities[i] >= threshold]
    raw, terms = generate_terms(
        lm,
        [queries[i] for i in need],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
    )
    generated = {index: (raw[k], terms[k]) for k, index in enumerate(need)}
    for i, row in enumerate(rows):
        row["gate_probability"] = probabilities[i]
        row["gate_shown"] = int(probabilities[i] >= threshold)
        row["raw_output"], row["predicted_terms"] = generated.get(i, ("", []))
        row["final_terms"] = row["predicted_terms"] if row["gate_shown"] else []


def compute_metrics(rows: list[dict], threshold: float, seen_terms: set[str], soft_scores: dict | None, args) -> dict:
    labels = [int(r["should_show"]) for r in rows]
    probabilities = [r["gate_probability"] for r in rows]
    positives = [r for r in rows if int(r["should_show"]) == 1]
    shown_positives = [r for r in positives if r["gate_shown"]]
    lengths = [len(r["query"]) for r in rows]

    gold_terms = [normalize_term(t) for r in positives for t in r["related_searches"]]
    predicted_terms = [normalize_term(t) for r in positives for t in r["predicted_terms"]]
    data = {
        "samples": len(rows),
        "positive_rate": sum(labels) / len(labels) if labels else 0.0,
        "mean_query_chars": float(np.mean(lengths)) if lengths else 0.0,
        "gold_term_seen_rate": sum(t in seen_terms for t in gold_terms) / len(gold_terms) if gold_terms and seen_terms else None,
        "predicted_term_seen_rate": (
            sum(t in seen_terms for t in predicted_terms) / len(predicted_terms) if predicted_terms and seen_terms else None
        ),
    }
    if rows and "cluster_id" in rows[0]:
        data["clusters"] = sorted({int(r["cluster_id"]) for r in rows if r.get("cluster_id") is not None})

    gate = {
        **binary_metrics(labels, probabilities, threshold),
        **ranking_metrics(labels, probabilities),
        "at_threshold_0.5": binary_metrics(labels, probabilities, 0.5),
        "calibration": calibration_metrics(labels, probabilities),
        "positive_rate_predicted": float(np.mean([r["gate_shown"] for r in rows])) if rows else 0.0,
    }
    forced = generation_metrics(positives, "predicted_terms")
    when_shown = generation_metrics(shown_positives, "predicted_terms")
    end_to_end = generation_metrics(
        [{"related_searches": r["related_searches"] if int(r["should_show"]) else [], "final_terms": r["final_terms"]} for r in rows],
        "final_terms",
    )
    health_rows = [r for r in rows if r["gate_shown"] or int(r["should_show"])]
    metrics = {
        "data": data,
        "gate": gate,
        "forced_generation": {**forced, "health": output_health_metrics(positives, "predicted_terms", "raw_output")},
        "generation_when_shown": when_shown,
        "end_to_end": end_to_end,
        "generated_rows": len(health_rows),
    }
    if soft_scores is not None:
        values = [soft_scores[r["query"]] for r in positives]
        metrics["forced_generation"]["soft_f1"] = float(np.mean(values)) if values else 0.0
        shown_values = [soft_scores[r["query"]] for r in shown_positives]
        metrics["generation_when_shown"]["soft_f1"] = float(np.mean(shown_values)) if shown_values else 0.0
    if args.bootstrap > 0 and rows:
        metrics["bootstrap_95ci"] = bootstrap_intervals(
            labels,
            probabilities,
            threshold,
            positives,
            "predicted_terms",
            [soft_scores[r["query"]] for r in positives] if soft_scores is not None else None,
            repeats=args.bootstrap,
            seed=args.seed,
        )
    return metrics


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_summary(results: dict, threshold: float) -> str:
    names = list(results)
    lines = [
        "# 相关搜索 ID / OOD 评估汇总",
        "",
        f"门控阈值：{threshold:.4f}（在 ID 验证集上选定，未在测试集上调整）。",
        "",
        "| 指标 | " + " | ".join(names) + (" | OOD − ID |" if {"test_id", "test_ood"} <= set(names) else " |"),
        "|---|" + "---|" * len(names) + ("---|" if {"test_id", "test_ood"} <= set(names) else ""),
    ]
    extra = [
        ("data.samples", "样本数"),
        ("data.positive_rate", "教师标注应展示比例"),
        ("data.mean_query_chars", "query 平均字数"),
    ]
    for key, label in extra + CORE_KEYS:
        values = [dig(results[name], key) for name in names]
        row = f"| {label} | " + " | ".join(fmt(v) for v in values) + " |"
        if {"test_id", "test_ood"} <= set(names):
            a, b = dig(results["test_id"], key), dig(results["test_ood"], key)
            comparable = isinstance(a, (int, float)) and isinstance(b, (int, float)) and key != "data.samples"
            delta = f"{b - a:+.4f}" if comparable else "—"
            row += f" {delta} |"
        lines.append(row)
    lines += ["", "## 95% bootstrap 置信区间", ""]
    for name in names:
        interval = results[name].get("bootstrap_95ci")
        if not interval:
            continue
        lines.append(f"**{name}**")
        for key, value in interval.items():
            if isinstance(value, dict):
                lines.append(f"- {key}: [{fmt(value['low'])}, {fmt(value['high'])}]")
        lines.append("")
    lines += [
        "## 解读提示",
        "",
        "- OOD 相对 ID 的下降幅度要结合置信区间看：区间明显重叠时，差异不显著。",
        "- “标准答案词在训练集中出现过的比例”越低，说明该测试集越依赖模型对新词汇的泛化，精确匹配 F1 天然会更低，软匹配更有参考价值。",
        "- 结论只适用于“同一 Pinterest 风格分布内、训练时没见过的语义主题”，不代表真实 Nalook 线上 query 的表现。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    from rs_infer import load_model

    adapter = Path(args.adapter)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else adapter.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    lm = load_model(adapter, args.base_model)
    threshold = args.threshold if args.threshold is not None else lm.threshold
    print(f"使用门控阈值 {threshold:.4f}；system prompt 来自 {adapter / 'system_prompt.txt'}", flush=True)

    split_rows: dict[str, list[dict]] = {}
    for split in args.splits:
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        for row in rows:
            row.setdefault("should_show", int(bool(row.get("related_searches"))))
        if args.limit:
            rows = rows[: args.limit]
        print(f"[{split}] {len(rows)} 条，开始预测……", flush=True)
        run_predictions(lm, rows, args)
        split_rows[split] = rows

    del lm
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    soft: dict[str, dict] = {name: None for name in split_rows}
    if args.soft_embedding_model and args.soft_embedding_model.lower() != "none":
        from rs_embed import embed_texts

        unique = sorted(
            {
                normalize_term(t)
                for rows in split_rows.values()
                for r in rows
                if int(r["should_show"]) == 1
                for t in r["related_searches"] + r["predicted_terms"]
            }
        )
        print(f"软匹配：embedding {len(unique)} 个词", flush=True)
        vectors = dict(zip(unique, embed_texts(unique, args.soft_embedding_model)))
        lookup = lambda terms: np.stack([vectors[t] for t in terms])  # noqa: E731
        for name, rows in split_rows.items():
            positives = [r for r in rows if int(r["should_show"]) == 1]
            scores = soft_match_scores(positives, "predicted_terms", lookup)
            soft[name] = {r["query"]: s for r, s in zip(positives, scores)}

    seen_terms = train_term_set(data_dir)
    results: dict[str, dict] = {}
    for name, rows in split_rows.items():
        results[name] = compute_metrics(rows, threshold, seen_terms, soft[name], args)
        write_jsonl(output_dir / f"{name}_predictions.jsonl", rows)
        (output_dir / f"{name}_metrics.json").write_text(
            json.dumps(results[name], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (output_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = render_summary(results, threshold)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"结果已写入 {output_dir}", flush=True)


if __name__ == "__main__":
    main()

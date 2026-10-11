"""评估指标：门控（二分类）、生成、输出健康度、bootstrap 置信区间、可选的 embedding 软匹配。

门控和排序指标沿用旧版 evaluate_qwen_instruct_gate.py 的口径；新增：
  * 在“教师标注为应展示”的样本上强制生成，衡量生成能力本身（不受门控误判影响）；
  * embedding 软匹配（可选）：精确匹配 F1 只有 0.07 左右，区分度太低；
  * bootstrap 95% 置信区间：测试集只有几百到一两千条，差距是否显著要看区间。
"""

from __future__ import annotations

import numpy as np

from rs_common import normalize_term
from rs_quality import repeated_raw_terms, text_issues


# --------------------------------------------------------------------------- 门控


def binary_metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict:
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    boundaries = np.flatnonzero(np.diff(sorted_values)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(values)]))
    average = (starts + ends + 1) / 2.0
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat(average, ends - starts)
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = _average_ranks(scores)
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def ranking_metrics(labels: list[int], probabilities: list[float]) -> dict:
    y = np.asarray(labels)
    s = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    hits = y[order] == 1
    cumulative = np.cumsum(hits)
    precision_at_hit = cumulative[hits] / (np.flatnonzero(hits) + 1)
    return {
        "roc_auc": roc_auc(y, s),
        "average_precision": float(precision_at_hit.sum() / max(1, hits.sum())),
    }


def calibration_metrics(labels: list[int], probabilities: list[float], bins: int = 10) -> dict:
    if not labels:
        return {"brier_score": 0.0, "expected_calibration_error": 0.0, "bins": []}
    brier = sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels)
    details, ece = [], 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (y, p)
            for y, p in zip(labels, probabilities)
            if low <= p < high or (index == bins - 1 and p == 1.0)
        ]
        if not members:
            continue
        confidence = sum(p for _, p in members) / len(members)
        positive_rate = sum(y for y, _ in members) / len(members)
        ece += len(members) / len(labels) * abs(confidence - positive_rate)
        details.append(
            {"low": low, "high": high, "samples": len(members),
             "mean_probability": confidence, "positive_rate": positive_rate}
        )
    return {"brier_score": brier, "expected_calibration_error": ece, "bins": details}


# --------------------------------------------------------------------------- 生成


def _term_sets(row: dict, key: str) -> tuple[set[str], set[str]]:
    truth = {normalize_term(t) for t in row["related_searches"]}
    pred = {normalize_term(t) for t in row[key]}
    return truth, pred


def generation_metrics(rows: list[dict], key: str) -> dict:
    """精确集合匹配指标；``key`` 指定预测词列表所在字段。"""
    exact = true_total = pred_total = overlap_total = 0
    for row in rows:
        truth, pred = _term_sets(row, key)
        exact += truth == pred
        true_total += len(truth)
        pred_total += len(pred)
        overlap_total += len(truth & pred)
    precision = overlap_total / pred_total if pred_total else 0.0
    recall = overlap_total / true_total if true_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": len(rows),
        "exact_set_match": exact / len(rows) if rows else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "average_prediction_count": pred_total / len(rows) if rows else 0.0,
        "average_truth_count": true_total / len(rows) if rows else 0.0,
    }


def output_health_metrics(rows: list[dict], term_key: str, raw_key: str, query_key: str = "query") -> dict:
    predicted_terms = [str(t) for row in rows for t in row[term_key]]
    malformed = [t for t in predicted_terms if text_issues(t)]
    repeated_rows = sum(repeated_raw_terms(str(row[raw_key])) for row in rows)
    empty_rows = sum(not row[term_key] for row in rows)
    query_copy_rows = 0
    for row in rows:
        query = normalize_term(row[query_key])
        query_copy_rows += query in {normalize_term(t) for t in row[term_key]}
    n = len(rows)
    return {
        "samples": n,
        "predicted_terms": len(predicted_terms),
        "empty_output_rate": empty_rows / n if n else 0.0,
        "repeated_raw_output_rate": repeated_rows / n if n else 0.0,
        "malformed_term_rate": len(malformed) / len(predicted_terms) if predicted_terms else 0.0,
        "query_copy_rate": query_copy_rows / n if n else 0.0,
        "average_unique_term_count": (
            sum(len({normalize_term(t) for t in row[term_key]}) for row in rows) / n if n else 0.0
        ),
    }


def soft_match_scores(rows: list[dict], key: str, embed) -> list[float]:
    """每条样本一个软匹配 F1：预测词与标准答案词之间的最大余弦相似度（双向取调和平均）。

    ``embed(list[str]) -> ndarray(n, d)`` 需返回 L2 归一化向量。没有预测词的样本得 0。
    """
    unique = sorted({normalize_term(t) for row in rows for t in row["related_searches"] + row[key]})
    if not unique:
        return [0.0] * len(rows)
    vectors = dict(zip(unique, embed(unique)))
    scores: list[float] = []
    for row in rows:
        gold = [normalize_term(t) for t in row["related_searches"]]
        pred = [normalize_term(t) for t in row[key]]
        if not gold or not pred:
            scores.append(0.0)
            continue
        similarity = np.stack([vectors[p] for p in pred]) @ np.stack([vectors[g] for g in gold]).T
        precision = float(similarity.max(axis=1).mean())
        recall = float(similarity.max(axis=0).mean())
        scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return scores


# --------------------------------------------------------------------------- bootstrap


def bootstrap_intervals(
    labels: list[int],
    probabilities: list[float],
    threshold: float,
    generation_rows: list[dict],
    generation_key: str,
    soft_scores: list[float] | None,
    repeats: int = 1000,
    seed: int = 0,
) -> dict:
    """对若干核心指标做 bootstrap（按样本有放回重采样）给出 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    shown = p >= threshold
    truth_n, pred_n, overlap_n = [], [], []
    for row in generation_rows:
        truth, pred = _term_sets(row, generation_key)
        truth_n.append(len(truth))
        pred_n.append(len(pred))
        overlap_n.append(len(truth & pred))
    truth_n, pred_n, overlap_n = map(np.asarray, (truth_n, pred_n, overlap_n))
    soft = np.asarray(soft_scores) if soft_scores is not None else None

    def interval(values: list[float]) -> dict:
        arr = np.asarray([v for v in values if not np.isnan(v)])
        if not len(arr):
            return {"low": None, "high": None}
        return {"low": float(np.percentile(arr, 2.5)), "high": float(np.percentile(arr, 97.5))}

    auc_values, ba_values, f1_values, soft_values = [], [], [], []
    for _ in range(repeats):
        index = rng.integers(0, len(y), len(y))
        yy, pp, ss = y[index], p[index], shown[index]
        auc_values.append(roc_auc(yy, pp))
        pos, neg = yy == 1, yy == 0
        recall = (ss & pos).sum() / max(1, pos.sum())
        specificity = (~ss & neg).sum() / max(1, neg.sum())
        ba_values.append((recall + specificity) / 2)
        if len(generation_rows):
            gi = rng.integers(0, len(generation_rows), len(generation_rows))
            precision = overlap_n[gi].sum() / max(1, pred_n[gi].sum())
            recall_g = overlap_n[gi].sum() / max(1, truth_n[gi].sum())
            f1_values.append(2 * precision * recall_g / (precision + recall_g) if precision + recall_g else 0.0)
            if soft is not None:
                soft_values.append(float(soft[gi].mean()))
    result = {
        "gate_roc_auc": interval(auc_values),
        "gate_balanced_accuracy": interval(ba_values),
        "repeats": repeats,
    }
    if f1_values:
        result["forced_generation_micro_f1"] = interval(f1_values)
    if soft_values:
        result["forced_generation_soft_f1"] = interval(soft_values)
    return result

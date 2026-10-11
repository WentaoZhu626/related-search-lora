"""数据划分：语义簇留出（近 OOD）+ 分层随机（ID）。

输入：清洗后的 cleaned_dataset.csv（列：query, 相关搜索词1 … 相关搜索词N；相关词为空 => should_show=0）。

划分方式
--------
1. 用 embedding 给所有 query 聚类（球面 K-means，默认 40 簇）。
2. 整簇留出若干个簇作为 **test_ood**：这些主题完全不参与训练。默认自动选“离其他簇最远、
   规模适中、正样本率与整体接近”的簇；也可以先 ``--report-only`` 看 cluster_report.csv，
   再用 ``--ood-cluster-ids 3,17,25`` 手工指定。
3. 缓冲带：把训练候选里与 OOD query 最相似的一小部分（默认 3%）丢进 buffer.jsonl，
   不参与任何 split，避免“换个说法的同一主题”泄漏进训练集。
4. 剩下的 query 按 should_show 分层随机切成 train / validation / test_id。
   validation 只用于选门控阈值和看训练曲线；test_id 与 test_ood 都不参与任何调参。

这样“一个模型、两个测试集”：test_id 和 test_ood 来自同一个模型、同一份训练数据，
两者的指标差距只反映“见没见过这些主题”。

输出（--output-dir）
--------------------
train.jsonl / validation.jsonl / test_id.jsonl / test_ood.jsonl / buffer.jsonl
cluster_report.csv   每个簇的规模、正样本率、离群程度、代表 query、是否被选为 OOD
split_report.json    各 split 的规模、正样本率、query 长度、到训练集的最近邻相似度分布、
                     标准答案词在训练集里出现过的比例（用来证明 OOD 集确实更“远”）
query_embeddings.npy 向量缓存（换参数重跑时不用重新编码）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from rs_common import clean_query, dedupe_key, normalize_term, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="语义簇留出的 ID / OOD 数据划分")
    parser.add_argument("--input", required=True, help="cleaned_dataset.csv 路径")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B", help="HF 名称或本地目录")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--clusters", type=int, default=40)
    parser.add_argument("--kmeans-iterations", type=int, default=60)
    parser.add_argument("--kmeans-restarts", type=int, default=3)
    parser.add_argument("--ood-clusters", type=int, default=4, help="自动选择时要留出的簇数量")
    parser.add_argument("--ood-cluster-ids", default="", help="手工指定留出簇，如 3,17,25；指定后不再自动选择")
    parser.add_argument("--max-ood-fraction", type=float, default=0.15, help="自动选择时 OOD 占全部数据的上限")
    parser.add_argument("--max-rate-gap", type=float, default=0.08, help="自动选择时，簇正样本率与整体的最大差距")
    parser.add_argument("--min-cluster-ratio", type=float, default=0.5, help="自动选择时簇规模下限（相对平均簇规模）")
    parser.add_argument("--max-cluster-ratio", type=float, default=2.0, help="自动选择时簇规模上限（相对平均簇规模）")
    parser.add_argument("--buffer-fraction", type=float, default=0.03, help="训练候选中与 OOD 最相似、被丢弃的比例")
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-id-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--report-only", action="store_true", help="只输出 cluster_report.csv 供人工挑选，不写数据划分")
    return parser.parse_args()


# --------------------------------------------------------------------------- 数据读取


def read_dataset(path: Path) -> tuple[list[dict], dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        term_columns = sorted(
            (name for name in fieldnames if name != "query"),
            key=lambda name: int(re.findall(r"\d+", name)[-1]) if re.findall(r"\d+", name) else 0,
        )
        rows: list[dict] = []
        seen: set[str] = set()
        stats = {"raw_rows": 0, "empty_query": 0, "duplicate_query": 0}
        for record in reader:
            stats["raw_rows"] += 1
            query = clean_query(record.get("query"))
            if not query:
                stats["empty_query"] += 1
                continue
            key = dedupe_key(query)
            if key in seen:
                stats["duplicate_query"] += 1
                continue
            seen.add(key)
            related: list[str] = []
            related_seen: set[str] = set()
            for column in term_columns:
                term = clean_query(record.get(column))
                marker = normalize_term(term)
                if term and marker not in related_seen:
                    related_seen.add(marker)
                    related.append(term)
            rows.append({"query": query, "related_searches": related, "should_show": int(bool(related))})
    stats["term_columns"] = term_columns
    return rows, stats


# --------------------------------------------------------------------------- 向量与聚类


def load_or_compute_embeddings(rows: list[dict], args: argparse.Namespace, output_dir: Path) -> np.ndarray:
    from rs_embed import embed_texts

    queries = [row["query"] for row in rows]
    fingerprint = hashlib.sha1(("\n".join(queries) + "||" + args.embedding_model).encode("utf-8")).hexdigest()
    cache = output_dir / "query_embeddings.npy"
    meta = output_dir / "query_embeddings.json"
    if cache.exists() and meta.exists():
        if json.loads(meta.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
            print(f"使用缓存向量：{cache}", flush=True)
            return np.load(cache)
    embeddings = embed_texts(queries, args.embedding_model, batch_size=args.embedding_batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings)
    meta.write_text(
        json.dumps({"fingerprint": fingerprint, "model": args.embedding_model, "count": len(queries)}),
        encoding="utf-8",
    )
    return embeddings


def spherical_kmeans(
    points: np.ndarray, clusters: int, iterations: int, restarts: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    n = x.size(0)
    best_objective = -1.0
    best_labels = best_centers = None
    for restart in range(restarts):
        generator = torch.Generator().manual_seed(seed + restart)
        centers = x[torch.randperm(n, generator=generator)[:clusters].to(device)].clone()
        previous = None
        for _ in range(iterations):
            labels = (x @ centers.T).argmax(dim=1)
            updated = torch.zeros_like(centers).index_add_(0, labels, x)
            empty = torch.bincount(labels, minlength=clusters) == 0
            if bool(empty.any()):
                replacements = torch.randperm(n, generator=generator)[: int(empty.sum())].to(device)
                updated[empty] = x[replacements]
            centers = F.normalize(updated, p=2, dim=1)
            if previous is not None and bool((labels == previous).all()):
                break
            previous = labels
        similarity = x @ centers.T
        labels = similarity.argmax(dim=1)
        objective = float(similarity.max(dim=1).values.sum().item())
        print(f"kmeans restart {restart + 1}/{restarts}: objective={objective:.2f}", flush=True)
        if objective > best_objective:
            best_objective, best_labels, best_centers = objective, labels.cpu().numpy(), centers.cpu().numpy()
    return best_labels, best_centers


def cluster_table(
    rows: list[dict], labels: np.ndarray, centers: np.ndarray, embeddings: np.ndarray
) -> list[dict]:
    center_similarity = centers @ centers.T
    np.fill_diagonal(center_similarity, -1.0)
    global_rate = float(np.mean([row["should_show"] for row in rows]))
    table: list[dict] = []
    for cluster_id in range(len(centers)):
        members = np.flatnonzero(labels == cluster_id)
        if not len(members):
            continue
        rate = float(np.mean([rows[i]["should_show"] for i in members]))
        nearest = members[np.argsort(-(embeddings[members] @ centers[cluster_id]))[:12]]
        table.append(
            {
                "cluster_id": cluster_id,
                "size": int(len(members)),
                "positive_rate": rate,
                "rate_gap": rate - global_rate,
                "isolation": float(1.0 - center_similarity[cluster_id].max()),
                "mean_query_length": float(np.mean([len(rows[i]["query"]) for i in members])),
                "sample_queries": " / ".join(rows[i]["query"] for i in nearest),
            }
        )
    return table


def choose_ood_clusters(table: list[dict], total: int, args: argparse.Namespace) -> list[int]:
    if args.ood_cluster_ids.strip():
        chosen = [int(part) for part in args.ood_cluster_ids.split(",") if part.strip()]
        known = {entry["cluster_id"] for entry in table}
        unknown = [cluster for cluster in chosen if cluster not in known]
        if unknown:
            raise ValueError(f"不存在的簇编号：{unknown}")
        return chosen
    average = total / len(table)
    eligible = [
        entry
        for entry in table
        if args.min_cluster_ratio * average <= entry["size"] <= args.max_cluster_ratio * average
        and abs(entry["rate_gap"]) <= args.max_rate_gap
    ]
    if len(eligible) < args.ood_clusters:
        print(
            f"警告：满足规模/正样本率条件的簇只有 {len(eligible)} 个，放宽为按离群程度直接排序。",
            flush=True,
        )
        eligible = list(table)
    chosen: list[int] = []
    size = 0
    for entry in sorted(eligible, key=lambda item: -item["isolation"]):
        if len(chosen) >= args.ood_clusters:
            break
        if (size + entry["size"]) / total > args.max_ood_fraction:
            continue
        chosen.append(entry["cluster_id"])
        size += entry["size"]
    if not chosen:
        raise ValueError("没有选出任何 OOD 簇，请调大 --max-ood-fraction 或用 --ood-cluster-ids 手工指定")
    return chosen


def write_cluster_report(path: Path, table: list[dict], chosen: list[int]) -> None:
    chosen_set = set(chosen)
    fields = [
        "cluster_id", "selected_as_ood", "size", "positive_rate", "rate_gap",
        "isolation", "mean_query_length", "sample_queries",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in sorted(table, key=lambda item: -item["isolation"]):
            writer.writerow({**entry, "selected_as_ood": int(entry["cluster_id"] in chosen_set)})


# --------------------------------------------------------------------------- 统计


def max_similarity(query_vectors: np.ndarray, reference: np.ndarray, chunk: int = 1024) -> np.ndarray:
    result = np.empty(len(query_vectors), dtype=np.float32)
    for start in range(0, len(query_vectors), chunk):
        result[start : start + chunk] = (query_vectors[start : start + chunk] @ reference.T).max(axis=1)
    return result


def summarize(rows: list[dict]) -> dict:
    positive = sum(row["should_show"] for row in rows)
    lengths = [len(row["query"]) for row in rows]
    counts = Counter(len(row["related_searches"]) for row in rows)
    return {
        "samples": len(rows),
        "positive": positive,
        "negative": len(rows) - positive,
        "positive_rate": positive / len(rows) if rows else 0.0,
        "mean_query_length": float(np.mean(lengths)) if lengths else 0.0,
        "related_count_distribution": {str(key): counts[key] for key in sorted(counts)},
    }


def quantiles(values: np.ndarray) -> dict:
    return {f"p{p}": float(np.percentile(values, p)) for p in (5, 25, 50, 75, 95)} | {"mean": float(values.mean())}


def stratified_take(pools: dict[int, list[int]], size: int, rng: random.Random) -> list[int]:
    total = len(pools[0]) + len(pools[1])
    positives = min(len(pools[1]), round(size * len(pools[1]) / total))
    negatives = min(len(pools[0]), size - positives)
    picked = [pools[1].pop() for _ in range(positives)] + [pools[0].pop() for _ in range(negatives)]
    rng.shuffle(picked)
    return picked


# --------------------------------------------------------------------------- 主流程


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, read_stats = read_dataset(Path(args.input))
    print(f"读取 {len(rows)} 条 query：{read_stats}", flush=True)
    embeddings = load_or_compute_embeddings(rows, args, output_dir)
    labels, centers = spherical_kmeans(
        embeddings, args.clusters, args.kmeans_iterations, args.kmeans_restarts, args.seed
    )
    table = cluster_table(rows, labels, centers, embeddings)
    chosen = choose_ood_clusters(table, len(rows), args)
    write_cluster_report(output_dir / "cluster_report.csv", table, chosen)
    print(f"选中的 OOD 簇：{chosen}（详见 cluster_report.csv）", flush=True)
    if args.report_only:
        print("--report-only：已写出 cluster_report.csv，不生成数据划分。", flush=True)
        return

    for index, row in enumerate(rows):
        row["cluster_id"] = int(labels[index])

    ood_set = set(chosen)
    ood_index = [i for i in range(len(rows)) if int(labels[i]) in ood_set]
    pool_index = [i for i in range(len(rows)) if int(labels[i]) not in ood_set]

    # 缓冲带：训练候选里与 OOD 最相似的一小部分。
    pool_similarity = max_similarity(embeddings[pool_index], embeddings[ood_index])
    buffer_count = int(round(args.buffer_fraction * len(pool_index)))
    order = np.argsort(-pool_similarity)
    buffer_positions = set(order[:buffer_count].tolist())
    buffer_cutoff = float(pool_similarity[order[buffer_count - 1]]) if buffer_count else None
    buffer_index = [pool_index[p] for p in sorted(buffer_positions)]
    id_pool = [pool_index[p] for p in range(len(pool_index)) if p not in buffer_positions]

    # ID 内部按 should_show 分层随机切分。
    rng = random.Random(args.seed)
    pools = {0: [], 1: []}
    for i in id_pool:
        pools[rows[i]["should_show"]].append(i)
    for values in pools.values():
        rng.shuffle(values)
    if args.validation_size + args.test_id_size >= len(id_pool):
        raise ValueError("validation-size + test-id-size 需要小于 ID 数据量")
    test_id_index = stratified_take(pools, args.test_id_size, rng)
    validation_index = stratified_take(pools, args.validation_size, rng)
    train_index = pools[0] + pools[1]
    rng.shuffle(train_index)

    splits = {
        "train": train_index,
        "validation": validation_index,
        "test_id": test_id_index,
        "test_ood": ood_index,
        "buffer": buffer_index,
    }
    # 泄漏检查：任何 query 只能出现在一个 split。
    owner: dict[str, str] = {}
    for name, indices in splits.items():
        for i in indices:
            key = dedupe_key(rows[i]["query"])
            if key in owner:
                raise AssertionError(f"query 同时出现在 {owner[key]} 和 {name}: {rows[i]['query']}")
            owner[key] = name

    for name, indices in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", [rows[i] for i in indices])

    # 报告：到训练集的最近邻相似度 + 标准答案词的“见过率”。
    train_vectors = embeddings[train_index]
    train_terms = {normalize_term(t) for i in train_index for t in rows[i]["related_searches"]}
    report_splits: dict = {}
    for name in ("train", "validation", "test_id", "test_ood", "buffer"):
        indices = splits[name]
        entry = summarize([rows[i] for i in indices])
        if name != "train" and indices:
            entry["nearest_train_similarity"] = quantiles(max_similarity(embeddings[indices], train_vectors))
            terms = [normalize_term(t) for i in indices for t in rows[i]["related_searches"]]
            entry["gold_term_seen_in_train_rate"] = (
                sum(t in train_terms for t in terms) / len(terms) if terms else None
            )
        report_splits[name] = entry

    report = {
        "seed": args.seed,
        "input": str(Path(args.input)),
        "read_stats": read_stats,
        "embedding_model": args.embedding_model,
        "clusters": args.clusters,
        "ood_cluster_ids": chosen,
        "ood_selection": "manual" if args.ood_cluster_ids.strip() else "auto(isolation)",
        "buffer": {"fraction": args.buffer_fraction, "count": len(buffer_index), "similarity_cutoff": buffer_cutoff},
        "overall": summarize(rows),
        "splits": report_splits,
        "note": (
            "nearest_train_similarity 越低说明越“远离”训练集；test_ood 应明显低于 test_id/validation。"
            "gold_term_seen_in_train_rate 是测试集标准答案词在训练集答案里出现过的比例。"
        ),
    }
    (output_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

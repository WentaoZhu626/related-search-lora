"""query embedding：数据划分（聚类）和评估（软匹配）共用。

默认模型 Qwen/Qwen3-Embedding-0.6B（safetensors 权重，不会触发 torch<2.6 的 .bin 加载限制），
和旧 analyze_embedding_distribution.py 使用同一种“最后一个非 padding token”池化。
"""

from __future__ import annotations

import numpy as np

from rs_common import from_pretrained_compat


def _tfidf_embeddings(texts: list[str], dim: int = 256) -> np.ndarray:
    """仅用于调试/无法下载模型时的兜底：字符 n-gram TF-IDF + SVD。语义质量远不如真正的 embedding。"""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    matrix = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=2, sublinear_tf=True).fit_transform(texts)
    dim = min(dim, matrix.shape[1] - 1)
    reduced = TruncatedSVD(n_components=dim, random_state=0).fit_transform(matrix).astype(np.float32)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    return reduced / np.maximum(norms, 1e-8)


def embed_texts(
    texts: list[str],
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 128,
    max_length: int = 64,
    device: str | None = None,
    pooling: str | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """返回 L2 归一化的 float32 矩阵，形状 (len(texts), dim)，顺序与输入一致。"""
    if model_name == "tfidf":
        return _tfidf_embeddings(texts)

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pooling = pooling or ("last" if "qwen" in model_name.lower() else "cls")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 明确使用右侧 padding + 长度索引，避免依赖 tokenizer 默认的 padding 方向。
    tokenizer.padding_side = "right"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = from_pretrained_compat(
        AutoModel, model_name, dtype, use_safetensors=True, trust_remote_code=True
    ).to(device)
    model.eval()

    # 按长度排序以减少 padding，最后还原顺序。
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    chunks: list[np.ndarray] = []
    for start in range(0, len(order), batch_size):
        batch_texts = [texts[index] for index in order[start : start + batch_size]]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            if pooling == "last":
                lengths = batch["attention_mask"].sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
            elif pooling == "mean":
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                pooled = hidden[:, 0]
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        chunks.append(pooled.cpu().numpy())
        if verbose and (start // batch_size) % 20 == 0:
            print(f"embedded {min(start + batch_size, len(order))}/{len(order)}", flush=True)

    sorted_embeddings = np.concatenate(chunks, axis=0)
    restored = np.empty_like(sorted_embeddings)
    restored[np.asarray(order)] = sorted_embeddings
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return restored

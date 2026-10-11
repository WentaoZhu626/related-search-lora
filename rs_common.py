"""公共工具：读写 jsonl、文本归一化、随机种子、运行环境信息。

本目录（related_search_ood_v3）是一套独立于旧 model_training 的新代码，
所有模块都以 ``rs_`` 前缀命名，避免和旧目录里的 common.py 等同名文件冲突。
"""

from __future__ import annotations

import json
import os
import platform
import random
import unicodedata
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_query(value: object) -> str:
    """只折叠空白，不改动 query 的任何字符（清洗数据的原则：query 保持原样）。"""
    return " ".join(str(value or "").split())


def dedupe_key(value: object) -> str:
    """跨 split 去重/泄漏检查用的 key：NFKC + 折叠空白 + casefold。"""
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()


def normalize_term(text: object) -> str:
    """指标计算时比较相关词用的归一化（与旧版 common.normalize_term 一致）。"""
    return unicodedata.normalize("NFKC", str(text)).strip().casefold()


def target_text(terms: list[str]) -> str:
    return " | ".join(term.strip() for term in terms if term and term.strip())


def parse_terms(text: str, max_terms: int | None = None) -> list[str]:
    """把模型输出解析成相关词列表（去空、去重、保持顺序）。"""
    text = text.strip()
    if text.startswith("["):
        try:
            value = json.loads(text)
            if isinstance(value, list):
                pieces = [str(item) for item in value]
                return _dedupe_terms(pieces, max_terms)
        except json.JSONDecodeError:
            pass
    pieces = text.replace("｜", "|").replace("；", "|").replace(";", "|").split("|")
    return _dedupe_terms(pieces, max_terms)


def _dedupe_terms(pieces: list[str], max_terms: int | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        piece = piece.strip().strip("[]\"'").strip()
        marker = normalize_term(piece)
        if piece and marker not in seen:
            seen.add(marker)
            result.append(piece)
        if max_terms is not None and len(result) >= max_terms:
            break
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def from_pretrained_compat(cls, path: str, dtype, **kwargs):
    """兼容不同 transformers 版本：新版用 ``dtype=``，旧版用 ``torch_dtype=``。"""
    try:
        return cls.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError as error:
        if "dtype" not in str(error):
            raise
        return cls.from_pretrained(path, torch_dtype=dtype, **kwargs)


def runtime_metadata(extra: dict | None = None) -> dict:
    import torch

    result: dict = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cwd": os.getcwd(),
    }
    for name in ("transformers", "peft"):
        try:
            module = __import__(name)
            result[name] = module.__version__
        except ModuleNotFoundError:
            result[name] = None
    if extra:
        result.update(extra)
    return result

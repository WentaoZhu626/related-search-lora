"""输出健康度检查用的文本规则（从旧版 quality_rules.py 精简而来，行为保持一致）。"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

HAN_RE = re.compile(r"[㐀-䶿一-鿿]")
LATIN_RE = re.compile(r"[A-Za-z]")
KANA_RE = re.compile(r"[぀-ヿㇰ-ㇿ]")
HANGUL_RE = re.compile(r"[가-힯]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
ARABIC_RE = re.compile(r"[؀-ۿ]")
THAI_RE = re.compile(r"[฀-๿]")
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
LONG_DIGIT_RE = re.compile(r"\d{10,}")
REPEATED_TOKEN_RE = re.compile(
    r"^\s*(\w+)\s*(?:[|,，;；。.]\s*\1\s*)+[。.,，;；!?！？]*$",
    re.IGNORECASE,
)
SHORT_TRAILING_PUNCT_RE = re.compile(r"^[\w㐀-鿿]{1,2}[,.，。;；:：!?！？]+$")


def normalize_text(text: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).strip().split()).casefold()


def visible_length(text: str) -> int:
    return sum(not char.isspace() for char in text)


def script_counts(text: str) -> Counter:
    counts: Counter = Counter()
    patterns = {
        "han": HAN_RE,
        "latin": LATIN_RE,
        "kana": KANA_RE,
        "hangul": HANGUL_RE,
        "cyrillic": CYRILLIC_RE,
        "arabic": ARABIC_RE,
        "thai": THAI_RE,
    }
    for name, pattern in patterns.items():
        counts[name] = len(pattern.findall(text))
    return counts


def text_issues(text: object) -> list[str]:
    """返回一个相关词的格式问题列表；空列表表示没发现问题。"""
    raw = str(text)
    normalized = normalize_text(raw)
    issues: list[str] = []
    if not normalized:
        return ["empty_text"]
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in raw):
        issues.append("control_character")
    if "�" in raw or "<unk>" in raw.casefold():
        issues.append("decoding_artifact")
    if URL_RE.search(raw):
        issues.append("url")
    if HTML_RE.search(raw):
        issues.append("html")
    if LONG_DIGIT_RE.search(normalized):
        issues.append("long_digit_run")
    if not any(char.isalnum() or HAN_RE.match(char) for char in normalized):
        issues.append("punctuation_only")
    if SHORT_TRAILING_PUNCT_RE.fullmatch(normalized):
        issues.append("short_text_with_trailing_punctuation")
    compact = "".join(char for char in normalized if char.isalnum() or HAN_RE.match(char))
    if len(compact) >= 3 and len(set(compact)) == 1:
        issues.append("single_character_repetition")
    if REPEATED_TOKEN_RE.fullmatch(normalized):
        issues.append("repeated_token_sequence")
    if visible_length(normalized) > 80:
        issues.append("very_long_text")
    scripts = script_counts(normalized)
    if sum(count > 0 for count in scripts.values()) >= 3:
        issues.append("three_or_more_scripts")
    return issues


def repeated_raw_terms(raw_prediction: str) -> bool:
    """模型原始输出里是否有重复的词。"""
    pieces = []
    for piece in re.split(r"[|,，;；]", raw_prediction):
        piece = normalize_text(piece.strip().strip("[]\"'。.,，;；!?！？"))
        if piece:
            pieces.append(piece)
    return len(pieces) != len(set(pieces))

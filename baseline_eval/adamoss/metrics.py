"""Scoring helpers: token F1, Wilson intervals and paired significance tests."""

from __future__ import annotations

import math
import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation + "“”‘’")


def normalize_answer(text: str) -> str:
    """Lowercase and strip punctuation, articles and extra whitespace (SQuAD style)."""
    text = text.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, gold: str) -> float:
    """Best token F1 across gold alternatives, which the dataset separates with ' / '."""
    return max(_token_f1(prediction, alternative) for alternative in gold.split(" / "))


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval, which behaves well for small n and extreme proportions."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the counts of discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n
    return min(1.0, 2 * tail)

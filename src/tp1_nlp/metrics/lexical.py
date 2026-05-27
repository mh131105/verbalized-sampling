from __future__ import annotations

import itertools
import re
from collections import Counter


TOKEN_RE = re.compile(r"\b[\wÀ-ÿ]+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def mean_pairwise_jaccard(texts: list[str]) -> float | None:
    token_sets = [set(tokenize(text)) for text in texts if text.strip()]
    if len(token_sets) < 2:
        return None
    scores: list[float] = []
    for left, right in itertools.combinations(token_sets, 2):
        union = left | right
        scores.append(len(left & right) / len(union) if union else 0.0)
    return sum(scores) / len(scores) if scores else None


def distinct_n(texts: list[str], n: int) -> float | None:
    ngrams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = tokenize(text)
        ngrams.extend(zip(*(tokens[i:] for i in range(n))))
    if not ngrams:
        return None
    return len(set(ngrams)) / len(ngrams)


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(zip(*(tokens[i:] for i in range(n))))


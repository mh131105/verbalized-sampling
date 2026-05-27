from __future__ import annotations

import math

from tp1_nlp.metrics.lexical import ngram_counts, tokenize


def self_bleu(texts: list[str], max_order: int = 4) -> float | None:
    clean_texts = [text for text in texts if text.strip()]
    if len(clean_texts) < 2:
        return None
    scores = []
    for index, candidate in enumerate(clean_texts):
        references = clean_texts[:index] + clean_texts[index + 1 :]
        scores.append(_bleu(candidate, references, max_order=max_order))
    return sum(scores) / len(scores) if scores else None


def _bleu(candidate: str, references: list[str], max_order: int) -> float:
    candidate_tokens = tokenize(candidate)
    reference_tokens = [tokenize(reference) for reference in references]
    if not candidate_tokens:
        return 0.0

    precisions: list[float] = []
    for order in range(1, max_order + 1):
        candidate_counts = ngram_counts(candidate_tokens, order)
        if not candidate_counts:
            precisions.append(1.0)
            continue
        max_ref_counts = {}
        for ref in reference_tokens:
            ref_counts = ngram_counts(ref, order)
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)
        overlap = sum(
            min(count, max_ref_counts.get(ngram, 0))
            for ngram, count in candidate_counts.items()
        )
        total = sum(candidate_counts.values())
        precisions.append((overlap + 1) / (total + 1))

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    ref_lens = [len(ref) for ref in reference_tokens]
    closest_ref_len = min(ref_lens, key=lambda length: abs(length - len(candidate_tokens)))
    brevity_penalty = (
        1.0
        if len(candidate_tokens) > closest_ref_len
        else math.exp(1 - closest_ref_len / max(len(candidate_tokens), 1))
    )
    return brevity_penalty * geo_mean


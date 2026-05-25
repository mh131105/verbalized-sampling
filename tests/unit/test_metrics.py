from __future__ import annotations

from tp1_nlp.metrics.bleu import self_bleu
from tp1_nlp.metrics.lexical import distinct_n, mean_pairwise_jaccard


def test_lexical_metrics_are_defined_for_multiple_texts() -> None:
    texts = [
        "Claro, posso revisar o texto ate amanha.",
        "Sem problema, reviso o material e retorno cedo.",
        "Posso verificar o texto e mandar comentarios pela manha.",
    ]

    assert mean_pairwise_jaccard(texts) is not None
    assert distinct_n(texts, 1) is not None
    assert distinct_n(texts, 2) is not None
    assert self_bleu(texts) is not None


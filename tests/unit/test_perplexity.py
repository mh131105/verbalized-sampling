from __future__ import annotations

import types

import numpy as np

from tp1_nlp.metrics import perplexity
from tp1_nlp.metrics.perplexity import PerplexityEvaluator


def test_portulan_quantized_model_uses_llama_cpp_backend() -> None:
    evaluator = PerplexityEvaluator(perplexity.PORTULAN_GERVASIO_GGUF_REPO)

    assert isinstance(evaluator._backend, perplexity._LlamaCppPerplexityBackend)


def test_default_hf_model_uses_transformers_backend() -> None:
    evaluator = PerplexityEvaluator("pierreguillou/gpt2-small-portuguese")

    assert isinstance(evaluator._backend, perplexity._TransformersPerplexityBackend)


def test_llama_cpp_backend_scores_with_shifted_token_logits(monkeypatch) -> None:
    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.scores = np.zeros((8, 3), dtype=np.float64)

        def tokenize(
            self, text: bytes, *, add_bos: bool = True, special: bool = False
        ) -> list[int]:
            return [0, 1, 2]

        def reset(self) -> None:
            return None

        def eval(self, tokens: list[int]) -> None:
            self.scores[0] = [0.0, 2.0, 0.0]
            self.scores[1] = [0.0, 0.0, 2.0]

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=lambda *, repo_id, filename: "/tmp/fake-model.gguf"
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "llama_cpp",
        types.SimpleNamespace(Llama=FakeLlama),
    )

    evaluator = PerplexityEvaluator(perplexity.PORTULAN_GERVASIO_GGUF_REPO)
    score = evaluator.score("texto curto")

    assert 1.0 < score < 2.0


def test_gguf_path_prefers_lm_studio_local_file(monkeypatch, tmp_path) -> None:
    model_path = (
        tmp_path
        / ".lmstudio"
        / "models"
        / perplexity.PORTULAN_GERVASIO_GGUF_REPO
        / perplexity.PORTULAN_GERVASIO_GGUF_FILE
    )
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf")

    monkeypatch.setattr(perplexity.Path, "home", lambda: tmp_path)

    assert perplexity._resolve_gguf_model_path(
        perplexity.PORTULAN_GERVASIO_GGUF_REPO
    ) == model_path

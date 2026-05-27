from __future__ import annotations

import math
from functools import cached_property


class PerplexityUnavailable(RuntimeError):
    pass


class PerplexityEvaluator:
    def __init__(self, model_name: str, max_length: int = 256) -> None:
        self.model_name = model_name
        self.max_length = max_length

    @cached_property
    def _pipeline(self):  # type: ignore[no-untyped-def]
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise PerplexityUnavailable(
                "Install optional dependencies transformers and torch to compute perplexity"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.eval()
        return torch, tokenizer, model

    def score(self, text: str) -> float:
        if not text.strip():
            raise PerplexityUnavailable("Cannot compute perplexity for empty text")
        torch, tokenizer, model = self._pipeline
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            output = model(**encoded, labels=encoded["input_ids"])
        return float(math.exp(output.loss.item()))


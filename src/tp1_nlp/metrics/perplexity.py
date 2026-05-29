from __future__ import annotations

import math
from functools import cached_property
from pathlib import Path


PORTULAN_GERVASIO_GGUF_REPO = "PORTULAN/gervasio-8b-portuguese-ptpt-decoder-quantized-4bit"
PORTULAN_GERVASIO_GGUF_FILE = "gervasio-8b-portuguese-ptpt-decoder-Q4_K.gguf"


class PerplexityUnavailable(RuntimeError):
    pass


class PerplexityEvaluator:
    def __init__(self, model_name: str, max_length: int = 256) -> None:
        self.model_name = model_name
        self.max_length = max_length

    @cached_property
    def _backend(self):  # type: ignore[no-untyped-def]
        if _is_portulan_gguf_model(self.model_name):
            return _LlamaCppPerplexityBackend(self.model_name, self.max_length)
        return _TransformersPerplexityBackend(self.model_name, self.max_length)

    def score(self, text: str) -> float:
        if not text.strip():
            raise PerplexityUnavailable("Cannot compute perplexity for empty text")
        return self._backend.score(text)


class _TransformersPerplexityBackend:
    def __init__(self, model_name: str, max_length: int) -> None:
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
        model = _load_causal_lm(AutoModelForCausalLM, self.model_name)
        if _can_move_to_mps(torch, model):
            model = model.to("mps")
        model.eval()
        return torch, tokenizer, model

    def score(self, text: str) -> float:
        torch, tokenizer, model = self._pipeline
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded, labels=encoded["input_ids"])
        return float(math.exp(output.loss.item()))


class _LlamaCppPerplexityBackend:
    def __init__(self, model_name: str, max_length: int) -> None:
        self.model_name = model_name
        self.max_length = max_length

    @cached_property
    def _llama(self):  # type: ignore[no-untyped-def]
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise PerplexityUnavailable(
                "Install llama-cpp-python to compute GGUF perplexity"
            ) from exc

        model_path = _resolve_gguf_model_path(self.model_name)
        return Llama(
            model_path=str(model_path),
            logits_all=True,
            n_ctx=self.max_length + 16,
            n_batch=min(self.max_length + 16, 512),
            n_gpu_layers=-1,
            verbose=False,
        )

    def score(self, text: str) -> float:
        try:
            import numpy as np
        except ImportError as exc:
            raise PerplexityUnavailable(
                "Install numpy to compute GGUF perplexity"
            ) from exc

        llama = self._llama
        tokens = llama.tokenize(text.encode("utf-8"), add_bos=True, special=False)
        tokens = tokens[: self.max_length]
        if len(tokens) < 2:
            raise PerplexityUnavailable("Cannot compute perplexity for fewer than two tokens")

        llama.reset()
        llama.eval(tokens)

        logits = np.asarray(llama.scores[: len(tokens) - 1, :], dtype=np.float64)
        targets = np.asarray(tokens[1:], dtype=np.int64)
        max_logits = np.max(logits, axis=1, keepdims=True)
        logsumexp = max_logits[:, 0] + np.log(np.exp(logits - max_logits).sum(axis=1))
        target_logits = logits[np.arange(len(targets)), targets]
        mean_negative_log_likelihood = -float(np.mean(target_logits - logsumexp))
        return float(math.exp(mean_negative_log_likelihood))


def _is_portulan_gguf_model(model_name: str) -> bool:
    return model_name == PORTULAN_GERVASIO_GGUF_REPO


def _resolve_gguf_model_path(model_name: str) -> Path:
    local_path = _lm_studio_gguf_path(model_name)
    if local_path.exists():
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PerplexityUnavailable(
            "Install huggingface_hub or download the GGUF model with LM Studio"
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=model_name,
            filename=PORTULAN_GERVASIO_GGUF_FILE,
        )
    )


def _lm_studio_gguf_path(model_name: str) -> Path:
    return Path.home() / ".lmstudio" / "models" / model_name / PORTULAN_GERVASIO_GGUF_FILE


def _load_causal_lm(auto_model: object, model_name: str):  # type: ignore[no-untyped-def]
    try:
        return auto_model.from_pretrained(model_name, dtype="auto")  # type: ignore[attr-defined]
    except TypeError:
        try:
            return auto_model.from_pretrained(model_name, torch_dtype="auto")  # type: ignore[attr-defined]
        except TypeError:
            return auto_model.from_pretrained(model_name)  # type: ignore[attr-defined]


def _can_move_to_mps(torch: object, model: object) -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None or not mps.is_available():
        return False
    if getattr(model, "is_quantized", False):
        return False
    if getattr(model, "hf_device_map", None):
        return False
    return True

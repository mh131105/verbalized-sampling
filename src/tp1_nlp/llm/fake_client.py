from __future__ import annotations

import hashlib
import json
import random
import re
import time

from tp1_nlp.domain.models import ChatMessage, GenerationParams, GenerationResult, ModelConfig
from tp1_nlp.llm.base import BaseLLMClient


class FakeLLMClient(BaseLLMClient):
    """Deterministic client for tests, dry-runs, and artifact generation."""

    TONES = [
        "formal",
        "cordial",
        "objetivo",
        "empatico",
        "direto",
        "curto",
        "informal",
        "positivo",
    ]

    STARTERS = [
        "Claro",
        "Sem problema",
        "Obrigado pelo aviso",
        "Entendi",
        "Pode deixar",
        "Agradeco o contato",
        "Combinado",
        "Vou verificar",
    ]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.variant = int(hashlib.sha256(config.model_id.encode()).hexdigest(), 16) % 7

    def generate(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> GenerationResult:
        started = time.perf_counter()
        prompt = messages[-1]["content"]
        rng = random.Random(self._seed(prompt, params))
        n_candidates = params.n_candidates or self._extract_n_candidates(prompt) or 1
        wants_json = '"responses"' in prompt or "JSON" in prompt.upper()

        if wants_json:
            text = self._json_response(prompt, n_candidates, rng)
        else:
            text = self._baseline_response(prompt, rng)

        latency_ms = int((time.perf_counter() - started) * 1000)
        return GenerationResult(
            text=text,
            raw_response={
                "provider": "fake",
                "model": self.config.model_name,
                "messages": messages,
                "params": params.to_dict(),
                "text": text,
            },
            model_name=self.config.model_name,
            provider="fake",
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(text.split()),
                "total_tokens": len(prompt.split()) + len(text.split()),
            },
            latency_ms=latency_ms,
            finish_reason="stop",
        )

    def _seed(self, prompt: str, params: GenerationParams) -> int:
        base = f"{self.config.model_id}|{params.seed}|{prompt}"
        return int(hashlib.sha256(base.encode("utf-8")).hexdigest(), 16) % (2**32)

    def _extract_n_candidates(self, prompt: str) -> int | None:
        match = re.search(r"Gere\s+(\d+)", prompt, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _extract_input(self, prompt: str) -> str:
        marker = "Mensagem recebida:"
        if marker in prompt:
            return prompt.split(marker, 1)[1].strip().splitlines()[0].strip()
        return prompt.strip().splitlines()[-1]

    def _extract_expected_tones(self, prompt: str) -> list[str]:
        match = re.search(r"Tons esperados:\s*(.+)", prompt)
        if not match:
            return self.TONES
        tones = [item.strip() for item in match.group(1).split(",") if item.strip()]
        return tones or self.TONES

    def _baseline_response(self, prompt: str, rng: random.Random) -> str:
        message = self._extract_input(prompt)
        starter = rng.choice(self.STARTERS)
        suffixes = [
            "vou cuidar disso e retorno assim que tiver uma atualizacao.",
            "consigo te responder de forma objetiva: esta combinado.",
            "obrigado por avisar; vou me organizar com base nisso.",
            "posso seguir dessa forma e te aviso se surgir algum impedimento.",
        ]
        return f"{starter}, {rng.choice(suffixes)}"

    def _json_response(self, prompt: str, n_candidates: int, rng: random.Random) -> str:
        message = self._extract_input(prompt)
        expected = self._extract_expected_tones(prompt)
        tail = "cauda" in prompt.lower() or "menor que" in prompt.lower()
        responses = []
        for index in range(n_candidates):
            tone = (expected + self.TONES)[(index + self.variant) % len(expected + self.TONES)]
            probability = round(
                rng.uniform(0.05, 0.19) if tail else rng.uniform(0.18, 0.42),
                2,
            )
            text = self._alternative_text(message, tone, index, rng)
            responses.append({"tone": tone, "text": text, "probability": probability})
        return json.dumps({"responses": responses}, ensure_ascii=False)

    def _alternative_text(
        self, message: str, tone: str, index: int, rng: random.Random
    ) -> str:
        fragments = {
            "formal": "Agradeco a mensagem. Posso verificar e retorno com uma posicao clara.",
            "cordial": "Claro, obrigado por avisar. Vou acompanhar isso com atencao.",
            "objetivo": "Combinado. Vou tratar disso e te retorno.",
            "empatico": "Entendo a situacao. Vou fazer o possivel para ajudar da melhor forma.",
            "direto": "Certo, vou verificar e responder assim que possivel.",
            "curto": "Perfeito, combinado.",
            "informal": "Fechado, vou ver isso e te falo.",
            "positivo": "Otimo, fico a disposicao e sigo com isso.",
        }
        base = fragments.get(tone, "Obrigado pela mensagem. Vou verificar e retorno.")
        if index % 2 == 0:
            return base
        return f"{base} Sobre '{message[:42]}', vou manter o contexto em mente."


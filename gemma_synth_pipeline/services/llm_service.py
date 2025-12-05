from __future__ import annotations

import logging
from typing import Optional

from gemma_synth_pipeline import config


class LmStudioClient:
    """Thin wrapper around the OpenAI-compatible LMStudio endpoint."""

    def __init__(
        self,
        model: str = config.MODEL_NAME,
        base_url: str = config.LMSTUDIO_BASE_URL,
        api_key: str = config.LMSTUDIO_API_KEY,
        temperature: float = config.TEMPERATURE,
        max_tokens: int = config.MAX_TOKENS,
        seed: Optional[int] = config.SEED,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("openai package is required to talk to LMStudio.") from exc
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self._log = logging.getLogger(__name__)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Send a chat completion request and return raw text content."""
        eff_seed = seed if seed is not None else self.seed
        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": self.max_tokens,
            }
            if eff_seed is not None:
                kwargs["seed"] = eff_seed
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self._log.error("LMStudio request failed: %s", exc)
            raise
        choice = resp.choices[0].message.content or ""
        if not choice.strip():
            self._log.warning("Empty completion received from LMStudio.")
        return choice

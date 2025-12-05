from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class NoiseConfig:
    """
    Tunable heuristics for lightweight lexical noise.

    The defaults favor tiny edits that keep sentences readable while giving
    the downstream model slightly perturbed phrasing to learn from.
    """

    word_swap_probability: float = 0.5
    filler_probability: float = 0.3
    parenthetical_probability: float = 0.25
    jitter_probability: float = 0.25


class TextNoiser:
    """
    Deterministic(ish) text perturbations used to add light noise to ad copy.

    We keep the edits simple (word swaps, filler phrases, jittered punctuation)
    so the resulting copy still passes downstream format checks but gives Qwen a
    wider phrasing surface to learn from.
    """

    WORD_SUBSTITUTIONS: Dict[str, Sequence[str]] = {
        "clients": ["clients and families", "clients, partners, and friends"],
        "portfolio": ["portfolio mix", "portfolio setup"],
        "strategy": ["strategy blueprint", "strategy track"],
        "returns": ["returns profile", "results", "return stream"],
        "risk": ["risk exposure", "risk posture"],
        "market": ["market backdrop", "market setting"],
        "growth": ["expansion", "growth trajectory"],
        "cash": ["idle cash", "cash cushion"],
        "income": ["income stream", "cashflow"],
        "investors": ["investors and stakeholders", "investors we support"],
    }

    FILLER_PREFIXES: Sequence[str] = (
        "Candidly,",
        "Truthfully,",
        "In practical terms,",
        "For context,",
        "Said plainly,",
    )

    PARENTHETICALS: Sequence[str] = (
        "see recent desk notes",
        "per last quarter's memo",
        "based on our live dashboards",
        "per supervisor review",
        "as outlined in training docs",
    )

    def __init__(self, seed: int | None = None, config: NoiseConfig | None = None) -> None:
        self._rand = random.Random(seed)
        self._config = config or NoiseConfig()

    def add_noise(self, text: str, intensity: float = 0.2) -> str:
        """
        Apply low-stakes text perturbations.

        Intensity ∈ [0, 1] roughly maps to the probability that a non-empty line
        receives at least one edit.
        """
        if not text or intensity <= 0:
            return text

        lines = text.splitlines()
        mutated: List[str] = []
        for line in lines:
            if not line.strip():
                mutated.append(line)
                continue
            if self._rand.random() > intensity:
                mutated.append(line)
                continue
            mutated.append(self._mutate_line(line))
        return "\n".join(mutated)

    # --- internals -----------------------------------------------------

    def _mutate_line(self, line: str) -> str:
        updated = line
        if self._rand.random() < self._config.word_swap_probability:
            updated = self._swap_word(updated)
        if self._rand.random() < self._config.filler_probability:
            updated = self._prepend_filler(updated)
        if self._rand.random() < self._config.parenthetical_probability:
            updated = self._append_parenthetical(updated)
        if self._rand.random() < self._config.jitter_probability:
            updated = self._jitter_punctuation(updated)
        return updated

    def _swap_word(self, text: str) -> str:
        candidates = [
            word for word in self.WORD_SUBSTITUTIONS.keys() if re.search(self._word_pattern(word), text, flags=re.IGNORECASE)
        ]
        if not candidates:
            return text
        token = self._rand.choice(candidates)
        replacement = self._rand.choice(list(self.WORD_SUBSTITUTIONS[token]))

        def _replacer(match: re.Match[str]) -> str:
            return self._match_case(replacement, match.group(0))

        return re.sub(self._word_pattern(token), _replacer, text, count=1, flags=re.IGNORECASE)

    def _prepend_filler(self, text: str) -> str:
        filler = self._rand.choice(self.FILLER_PREFIXES)
        if text.lstrip().lower().startswith(filler.lower()):
            return text
        prefix = "" if text.startswith((" ", "\t")) else " "
        return f"{filler}{prefix}{text}"

    def _append_parenthetical(self, text: str) -> str:
        parenthetical = self._rand.choice(self.PARENTHETICALS)
        if parenthetical.lower() in text.lower():
            return text
        suffix = "" if text.endswith((".", "!", "?")) else "."
        return f"{text.rstrip()}{suffix} ({parenthetical})."

    def _jitter_punctuation(self, text: str) -> str:
        replacements = {
            ".": "...",
            ",": " —",
        }
        spots = [idx for idx, char in enumerate(text) if char in replacements]
        if not spots:
            return text
        idx = self._rand.choice(spots)
        char = text[idx]
        return f"{text[:idx]}{replacements[char]}{text[idx + 1 :]}"

    @staticmethod
    def _word_pattern(word: str) -> str:
        return rf"\b{re.escape(word)}\b"

    @staticmethod
    def _match_case(substitute: str, original: str) -> str:
        if original.isupper():
            return substitute.upper()
        if original[0].isupper():
            return substitute.capitalize()
        return substitute

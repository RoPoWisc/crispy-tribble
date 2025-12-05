from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_prompt(relative_path: str, strip: bool = False) -> str:
    """
    Load a prompt template from the prompts/ directory.

    Args:
        relative_path: Path relative to prompts/ (e.g., "trace_correction/base_system.txt").
        strip: When True, strip leading/trailing whitespace.
    """
    path = PROMPTS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8")
    return text.strip() if strip else text

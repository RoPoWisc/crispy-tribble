from __future__ import annotations

from pathlib import Path
from typing import Optional

from gemma_synth_pipeline import config


class RuleBook:
    """Load the compliance rulebook using docling to preserve structure."""

    def __init__(
        self,
        path: Path | str = config.RULES_DOCX,
        max_chars: Optional[int] = config.RULEBOOK_MAX_CHARS,
        text_override: Optional[Path] = config.RULES_TEXT_OVERRIDE,
    ):
        self.path = Path(path)
        self.max_chars = max_chars
        self.text_override = text_override

    def load_text(self) -> str:
        if self.text_override:
            override_path = Path(self.text_override)
            if not override_path.exists():
                raise FileNotFoundError(f"Rulebook override text not found: {override_path}")
            return override_path.read_text(encoding="utf-8")

        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("docling is required to read the rulebook.") from exc

        if not self.path.exists():
            raise FileNotFoundError(f"Rulebook not found: {self.path}")

        converter = DocumentConverter()
        result = converter.convert(str(self.path), raises_on_error=False)
        markdown_text = result.document.export_to_markdown()
        text = markdown_text.strip()
        if self.max_chars and len(text) > self.max_chars:
            return text[: self.max_chars]
        return text

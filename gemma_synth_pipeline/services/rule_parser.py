from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from gemma_synth_pipeline import config
from .rule_book import RuleBook


RULE_HEADINGS = {
    "general prohibitions",
    "testimonials and endorsements",
    "third-party ratings",
    "performance information generally",
    "general advertising guidelines",
    "advertising review guide",
}


@dataclass
class RuleSection:
    title: str
    bullets: List[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class RuleParseResult:
    sections: List[RuleSection]
    review_guide_raw: str = ""


class RuleParser:
    """Parse rulebook text (preferably docling-exported) into structured sections."""

    def __init__(self, text_path: Optional[Path] = config.RULES_TEXT_OVERRIDE) -> None:
        self.text_path = Path(text_path) if text_path else None

    def load_text(self) -> str:
        if self.text_path and self.text_path.exists():
            return self.text_path.read_text(encoding="utf-8")
        # fall back to RuleBook (docx via docling)
        return RuleBook().load_text()

    def parse(self) -> RuleParseResult:
        text = self.load_text()
        lines = [ln.strip() for ln in text.splitlines()]

        sections: list[RuleSection] = []
        current: Optional[RuleSection] = None
        review_guide_lines: list[str] = []

        def commit_section():
            nonlocal current
            if current:
                current.raw = current.raw.strip()
                if not current.bullets and current.raw:
                    # If no explicit bullets were captured, treat the raw block as one bullet.
                    current.bullets = [" ".join(current.raw.split())]
                sections.append(current)
            current = None

        for ln in lines:
            if not ln:
                if current:
                    current.raw += "\n"
                continue
            heading_match = re.match(r"^\*\*(.+?)\*\*$", ln)
            if heading_match:
                title = heading_match.group(1).strip()
                if title.lower() in RULE_HEADINGS:
                    commit_section()
                    current = RuleSection(title=title)
                    continue

            if current and current.title.lower() == "advertising review guide":
                review_guide_lines.append(ln)
                current.raw += ln + "\n"
                continue

            if current:
                current.raw += ln + "\n"
                if ln.startswith("- "):
                    bullet = ln[2:].strip()
                    if bullet:
                        current.bullets.append(bullet)

        commit_section()

        review_raw = "\n".join(review_guide_lines).strip()
        return RuleParseResult(sections=sections, review_guide_raw=review_raw)

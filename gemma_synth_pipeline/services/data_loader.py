from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from gemma_synth_pipeline import config


@dataclass
class LoadedText:
    path: Path
    method: str
    text: str


@dataclass
class AdCopySample:
    prefix: str
    review: LoadedText
    original: LoadedText

    @property
    def sample_id(self) -> str:
        return f"{self.prefix}:{self.original.path.name}"


class DatasetLoader:
    """Load ad copy + feedback pairs from extracted text outputs."""

    def __init__(
        self,
        base_dir: Path = config.TEXT_OUTPUT_ROOT,
        skip_prefixes: Iterable[str] = config.SKIP_PREFIXES,
        preferred_methods: Iterable[str] = config.PREFERRED_METHODS,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.skip_prefixes = set(skip_prefixes)
        self.preferred_methods = tuple(preferred_methods)

    def load(self, limit: Optional[int] = None) -> List[AdCopySample]:
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Text output directory not found: {self.base_dir}")

        samples: List[AdCopySample] = []
        for folder in sorted(p for p in self.base_dir.iterdir() if p.is_dir()):
            if folder.name in self.skip_prefixes:
                continue
            review = self._select_single(folder / "converted")
            if not review:
                continue
            for original in self._select_many(folder / "org_files"):
                samples.append(AdCopySample(prefix=folder.name, review=review, original=original))
                if limit and len(samples) >= limit:
                    return samples
        return samples

    def _select_single(self, folder: Path) -> Optional[LoadedText]:
        """Pick one text file from the folder, preferring docling then unstructured."""
        if not folder.exists():
            return None
        grouped = self._group_by_stem(folder)
        # For reviews, we just take the first stem available.
        for texts in grouped.values():
            chosen = self._choose_preferred(texts)
            if chosen:
                return chosen
        return None

    def _select_many(self, folder: Path) -> List[LoadedText]:
        if not folder.exists():
            return []
        grouped = self._group_by_stem(folder)
        results: List[LoadedText] = []
        for texts in grouped.values():
            chosen = self._choose_preferred(texts)
            if chosen:
                results.append(chosen)
        return results

    def _group_by_stem(self, folder: Path) -> dict[str, dict[str, Path]]:
        grouped: dict[str, dict[str, Path]] = {}
        for path in folder.glob("*__*.txt"):
            stem, method_part = path.name.rsplit("__", 1)
            method = method_part.replace(".txt", "")
            grouped.setdefault(stem, {})[method] = path
        return grouped

    def _choose_preferred(self, candidates: dict[str, Path]) -> Optional[LoadedText]:
        for method in self.preferred_methods:
            path = candidates.get(method)
            if not path:
                continue
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return LoadedText(path=path, method=method, text=text)
        # Fall back to any non-empty method for debugging.
        for method, path in sorted(candidates.items()):
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return LoadedText(path=path, method=method, text=text)
        return None

    @staticmethod
    def describe(samples: List[AdCopySample]) -> str:
        meta = {
            "count": len(samples),
            "prefixes": sorted({s.prefix for s in samples}),
            "methods": sorted({s.original.method for s in samples} | {s.review.method for s in samples}),
        }
        return json.dumps(meta, indent=2)


def load_seeds(limit: Optional[int] = None) -> List[AdCopySample]:
    """Convenience wrapper for callers that only need a small seed sample."""
    loader = DatasetLoader()
    return loader.load(limit=limit)


@dataclass
class SanitizedSample:
    prefix: str
    path: Path
    text: str
    method: str

    @property
    def sample_id(self) -> str:
        return f"{self.prefix}:{self.path.name}"


def load_sanitized_texts(
    root: Path = config.TEXT_OUTPUT_SANITIZED,
    limit: Optional[int] = None,
) -> List[SanitizedSample]:
    """Return sanitized ad-copy entries, preferring original org_files over reviews."""
    sanitized_root = Path(root)
    if not sanitized_root.exists():
        raise FileNotFoundError(f"Sanitized directory not found: {sanitized_root}")

    results: List[SanitizedSample] = []
    for prefix_dir in sorted(p for p in sanitized_root.iterdir() if p.is_dir()):
        org_dir = prefix_dir / "org_files"
        if not org_dir.exists():
            logging.warning("Skipping prefix %s because no sanitized org_files directory exists.", prefix_dir.name)
            continue

        for path in sorted(org_dir.glob("*__*.txt")):
            method = path.name.split("__", 1)[-1].replace(".txt", "")
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            if is_probable_review(text):
                logging.warning("Skipping %s (looks like review memo)", path)
                continue
            results.append(SanitizedSample(prefix=prefix_dir.name, path=path, text=text, method=method))
            if limit and len(results) >= limit:
                return results
    return results


REVIEW_GREETING_PATTERN = re.compile(r"^(hi|hello|dear)\s*\[redacted:name\]", re.IGNORECASE)
REVIEW_PHRASES = (
    "we have reviewed your",
    "we reviewed your",
    "here are our recommendations",
    "the rest of the booklet is compliant",
)


def is_probable_review(text: str) -> bool:
    """Heuristic: detect reviewer memos so they don't sneak into the ad-copy pipeline."""
    if not text:
        return False
    normalized = text.strip()
    head = normalized[:400].lower()
    if REVIEW_GREETING_PATTERN.match(normalized):
        return True
    if any(phrase in head for phrase in REVIEW_PHRASES):
        return True
    return False

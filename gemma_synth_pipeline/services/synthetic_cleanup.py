from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, List


@dataclass
class CleanupReport:
    """Structured stats so downstream steps can log/alert consistently."""

    input_path: Path
    output_path: Path
    total: int = 0
    kept: int = 0
    skipped_bad_json: int = 0
    skipped_empty_text: int = 0
    skipped_payload_json: int = 0
    skipped_missing_required: int = 0
    missing_key_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "total": self.total,
            "kept": self.kept,
            "skipped_bad_json": self.skipped_bad_json,
            "skipped_empty_text": self.skipped_empty_text,
            "skipped_payload_json": self.skipped_payload_json,
            "skipped_missing_required": self.skipped_missing_required,
            "missing_key_counts": self.missing_key_counts,
        }


class SyntheticPredictionCleaner:
    """Filters Gemini batch prediction rows down to valid preference triplets."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path | None = None,
        required_keys: Iterable[str] = ("broken", "chosen", "rejected"),
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else self._default_output_path()
        self.required_keys: List[str] = [*required_keys]

    def _default_output_path(self) -> Path:
        parent = self.input_path.parent
        stem = self.input_path.name
        suffix = ".jsonl"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        return parent / f"{stem}-clean{suffix}"

    def clean(self) -> CleanupReport:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Synthetic prediction file not found: {self.input_path}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")

        report = CleanupReport(input_path=self.input_path, output_path=self.output_path)

        with self.input_path.open("r", encoding="utf-8") as src, temp_path.open("w", encoding="utf-8") as dst:
            for line_number, raw_line in enumerate(src, start=1):
                report.total += 1
                try:
                    envelope = json.loads(raw_line)
                except json.JSONDecodeError:
                    report.skipped_bad_json += 1
                    continue

                text = self._extract_candidate_text(envelope)
                if not text:
                    report.skipped_empty_text += 1
                    continue

                payload = self._parse_candidate_payload(text)
                if payload is None:
                    report.skipped_payload_json += 1
                    continue

                missing = [key for key in self.required_keys if not payload.get(key)]
                if missing:
                    report.skipped_missing_required += 1
                    for key in missing:
                        report.missing_key_counts[key] = report.missing_key_counts.get(key, 0) + 1
                    continue

                record = {
                    "broken": payload["broken"],
                    "chosen": payload["chosen"],
                    "rejected": payload["rejected"],
                    "prompt": envelope.get("request"),
                    "meta": {
                        "source_file": str(self.input_path),
                        "source_line": line_number,
                        "processed_time": envelope.get("processed_time"),
                        "status": envelope.get("status"),
                        "response_id": envelope.get("response", {}).get("responseId"),
                        "model_version": envelope.get("response", {}).get("modelVersion"),
                    },
                }
                json.dump(record, dst, ensure_ascii=True)
                dst.write("\n")
                report.kept += 1

        temp_path.replace(self.output_path)
        return report

    @staticmethod
    def _extract_candidate_text(envelope: dict) -> str:
        try:
            parts = envelope["response"]["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            return ""
        if not isinstance(parts, list):
            return ""
        chunks: List[str] = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    @staticmethod
    def _parse_candidate_payload(text: str) -> dict | None:
        """Extract the inner JSON object (approach/score payload) from candidate text."""
        sanitized = SyntheticPredictionCleaner._sanitize_payload_text(text)
        if not sanitized:
            return None

        start_idx = sanitized.find("{")
        end_idx = sanitized.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return None

        inner = sanitized[start_idx : end_idx + 1]
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _sanitize_payload_text(text: str) -> str:
        """Remove markdown fences or other wrappers so json.loads succeeds."""
        if not text:
            return ""
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        # Remove leading ```json / ``` (case-insensitive on the hint token).
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped, count=1, flags=re.IGNORECASE)
        # Remove trailing ``` fence (optionally surrounded by whitespace/newlines).
        stripped = re.sub(r"\s*```$", "", stripped, count=1)
        return stripped.strip()

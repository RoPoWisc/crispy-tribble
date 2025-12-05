#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Iterable, Set, Tuple, List, Hashable


BAD_PATTERN_DEFAULT = "Hi [REDACTED:NAME]"

# Patterns of LLM artifact commentary we want to strip from prompts
PROMPT_COMMENTARY_PREFIXES = [
    "Here is the rewritten marketing copy.",
    "Here is the re-written marketing copy.",
    "This version incorporates the specific compliance recommendations",
    "This version incorporates the specific changes requested by the compliance review",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge DPO output JSONL files, filtering bad ad copies."
    )

    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("output/old"),
        help="Directory containing older DPO / breaker_fixer JSONL files.",
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("output/synthetic"),
        help="Directory containing recent synthetic DPO / breaker_fixer JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/merged/merged_dpo.jsonl"),
        help="Path to write merged JSONL.",
    )
    parser.add_argument(
        "--exclude-pattern",
        type=str,
        default=BAD_PATTERN_DEFAULT,
        help=(
            "Substring that marks bad ad copies (e.g. compliance letter as input). "
            "Any record whose prompt/base_text/broken/chosen/rejected contains this "
            "pattern will be dropped."
        ),
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.jsonl",
        help="Glob pattern to match JSONL files under old/new dirs.",
    )
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="",
        help=(
            "If set, only JSONL files whose basename starts with this prefix "
            "will be merged (e.g. 'output_procedural_dpo')."
        ),
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping malformed lines."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSON in %s line %d", path, lineno)
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                logging.warning("Non-dict JSON in %s line %d; skipping", path, lineno)


def extract_prompt_text(prompt_obj) -> str:
    if isinstance(prompt_obj, str):
        return prompt_obj
    if isinstance(prompt_obj, dict):
        texts = []
        for c in prompt_obj.get("contents", []):
            for part in c.get("parts", []):
                t = part.get("text")
                if isinstance(t, str):
                    texts.append(t)
        return "\n".join(texts)
    return ""


def clean_prompt_text(text: str) -> str:
    """
    Remove LLM artifact commentary lines from prompt text, while keeping
    the underlying ad copy / document intact.
    """
    if not text:
        return text

    prefixes = [p.lower() for p in PROMPT_COMMENTARY_PREFIXES]
    cleaned_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.lstrip()
        lower = stripped.lower()
        if any(lower.startswith(p) for p in prefixes):
            # Drop this commentary line entirely
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_prompt_object(prompt_obj: Any) -> Any:
    """
    Apply commentary stripping to the prompt, whether it's a plain string
    or a Vertex-style {contents: [...]} dict.
    """
    if isinstance(prompt_obj, str):
        return clean_prompt_text(prompt_obj)

    if isinstance(prompt_obj, dict):
        new_prompt = json.loads(json.dumps(prompt_obj))  # deep copy
        for c in new_prompt.get("contents", []):
            for part in c.get("parts", []):
                t = part.get("text")
                if isinstance(t, str):
                    part["text"] = clean_prompt_text(t)
        return new_prompt

    # Unknown structure, leave as-is
    return prompt_obj


def is_bad_record(record: dict, pattern: str) -> bool:
    # check prompt (string or structured)
    prompt_text = extract_prompt_text(record.get("prompt"))
    if pattern in prompt_text:
        return True

    # check other flat string fields
    for field in ["base_text", "broken", "chosen", "rejected"]:
        value = record.get(field)
        if isinstance(value, str) and pattern in value:
            return True

    return False


def _make_hashable(x: Any) -> Hashable:
    """
    Convert a value into something hashable for use in a dedupe key.

    - None, str, int, float, bool are used as-is.
    - Lists/dicts/other objects are JSON-dumped (sorted keys) so they can be hashed.
    """
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    # Fallback: stable string representation
    try:
        return json.dumps(x, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(x)


def make_dedupe_key(record: Dict[str, Any]) -> Tuple[Hashable, ...]:
    """
    Build a key for deduplication.

    We use (sample_id, prompt, chosen, rejected) but make each component hashable.
    """
    return tuple(
        _make_hashable(record.get(field))
        for field in ("sample_id", "prompt", "chosen", "rejected")
    )


def collect_files(root: Path, glob: str, prefix: str) -> List[Path]:
    if not root.exists():
        return []
    all_files = sorted(root.rglob(glob))
    if not prefix:
        return all_files
    filtered = [p for p in all_files if p.name.startswith(prefix)]
    return filtered


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    old_files = collect_files(args.old_dir, args.glob, args.filename_prefix)
    new_files = collect_files(args.new_dir, args.glob, args.filename_prefix)

    logging.info("Found %d JSONL files in %s", len(old_files), args.old_dir)
    logging.info("Found %d JSONL files in %s", len(new_files), args.new_dir)

    all_files: List[Path] = []
    all_files.extend(old_files)
    all_files.extend(new_files)

    if not all_files:
        logging.warning(
            "No JSONL files matching glob '%s' and prefix '%s' under %s or %s",
            args.glob,
            args.filename_prefix,
            args.old_dir,
            args.new_dir,
        )
        return

    dedupe_keys: Set[Tuple[Hashable, ...]] = set()
    kept: List[Dict[str, Any]] = []
    dropped_bad = 0
    dropped_dup = 0

    for path in all_files:
        logging.info("Reading %s", path)
        for rec in iter_jsonl(path):
            # 1) Strip LLM artifact commentary from prompt
            if "prompt" in rec:
                rec["prompt"] = clean_prompt_object(rec["prompt"])

            # 2) Filter obviously bad inputs
            if is_bad_record(rec, args.exclude_pattern):
                dropped_bad += 1
                continue

            # 3) Deduplicate
            key = make_dedupe_key(rec)
            if key in dedupe_keys:
                dropped_dup += 1
                continue

            dedupe_keys.add(key)
            kept.append(rec)

    if not kept:
        logging.warning("No records kept after filtering; nothing to write.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f_out:
        for rec in kept:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logging.info("Kept %d records", len(kept))
    logging.info("Dropped %d records due to bad pattern", dropped_bad)
    logging.info("Dropped %d duplicate records", dropped_dup)
    logging.info("Wrote merged file to %s", args.output)


if __name__ == "__main__":
    main()
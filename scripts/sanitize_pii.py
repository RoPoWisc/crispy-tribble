from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Ensure repo root is on sys.path for module imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gemma_synth_pipeline.services.llm_service import LmStudioClient


def load_prompt(template_path: Path) -> tuple[str, str]:
    """Load the sanitizer prompt template and split into system/user parts."""
    text = template_path.read_text(encoding="utf-8")
    # Prompt file is written with "System:" then "User:" sections.
    if "User:" not in text:
        raise ValueError("Prompt template missing 'User:' section.")
    system, user = text.split("User:", 1)
    system = system.replace("System:", "", 1).strip()
    user = user.strip()
    return system, user


def sanitize_text(client: LmStudioClient, system: str, user_template: str, text: str) -> str:
    user = user_template.replace("{{document_text}}", text)
    try:
        return client.generate(system_prompt=system, user_prompt=user, temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "context length" in msg or "tokens to keep" in msg:
            logging.warning("Skipping file: input too long for model context (%s)", msg)
            return None  # signal to caller to skip
        raise


def iter_txt_files(root: Path):
    for path in sorted(root.rglob("*.txt")):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Redact PII in text files via LMStudio.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("output/text/data/ad_copy_reviews"),
        help="Root directory containing source txt files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/sanitized_text"),
        help="Root directory to write redacted txt files (mirrors input structure).",
    )
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=Path("prompts/pii_sanitizer_prompt.txt"),
        help="Path to the PII sanitizer prompt template.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    system_prompt, user_template = load_prompt(args.prompt_template)
    client = LmStudioClient(seed=None, temperature=0.0)

    src_files = list(iter_txt_files(args.input_root))
    if args.limit:
        src_files = src_files[: args.limit]
    logging.info("Found %d txt files under %s", len(src_files), args.input_root)

    for path in src_files:
        rel = path.relative_to(args.input_root)
        out_path = args.output_root / rel
        if out_path.exists():
            logging.info("Skipping existing %s", out_path)
            continue
        print(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8")
        redacted = sanitize_text(client, system_prompt, user_template, text)
        if redacted is None:
            logging.info("Skipped %s (context too long)", rel)
            continue
        out_path.write_text(redacted, encoding="utf-8")
        logging.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

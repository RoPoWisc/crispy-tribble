from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gemma_synth_pipeline.services.synthetic_cleanup import SyntheticPredictionCleaner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter synthetic Gemini predictions down to valid preference triplets."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the raw predictions jsonl file produced by run_breaker_fixer.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the cleaned jsonl file. Defaults to <input>-clean.jsonl in the same directory.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cleaner = SyntheticPredictionCleaner(input_path=args.input, output_path=args.output)
    report = cleaner.clean()
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()

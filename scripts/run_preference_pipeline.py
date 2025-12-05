#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gemma_synth_pipeline import config as project_config
from gemma_synth_pipeline.services.pipeline_flow import (
    PreferencePipeline,
    PreferencePipelineConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ingestion → breaker → trace-correction flow with one command."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Root of the stanford_preference repository.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter used for every stage (default: current interpreter).",
    )
    parser.add_argument(
        "--sanitize-output",
        type=Path,
        default=project_config.TEXT_OUTPUT_SANITIZED,
        help="Directory where sanitized text assets are written.",
    )
    parser.add_argument(
        "--sanitize-limit",
        type=int,
        default=None,
        help="Optional limit passed to run_full_pipeline.py --sanitize-limit.",
    )
    parser.add_argument(
        "--full-args",
        nargs="*",
        default=[],
        help="Additional args forwarded to run_full_pipeline.py (append after this flag).",
    )
    parser.add_argument(
        "--breaker-output",
        type=Path,
        default=project_config.BREAKER_FIXER_OUTPUT,
        help="Path for breaker/fixer JSONL output.",
    )
    parser.add_argument(
        "--breaker-args",
        nargs="*",
        default=[],
        help="Additional args forwarded to run_breaker_fixer.py.",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=Path("output/experiments/output_procedural_dpo.jsonl"),
        help="Destination for the DPO-ready file emitted by run_trace_correction.py.",
    )
    parser.add_argument(
        "--trace-args",
        nargs="*",
        default=[],
        help="Additional args forwarded to run_trace_correction.py.",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.2,
        help="Lexical noise intensity forwarded to run_trace_correction.py.",
    )
    parser.add_argument(
        "--noise-targets",
        nargs="+",
        default=["prompt"],
        choices=["prompt", "broken", "chosen", "rejected"],
        help="Fields to perturb when injecting noise (default: prompt/ad copy).",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=13,
        help="Seed for deterministic noise injection.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip run_full_pipeline.py (assumes sanitized data already exists).",
    )
    parser.add_argument(
        "--skip-breaker",
        action="store_true",
        help="Skip run_breaker_fixer.py (assumes breaker output already exists).",
    )
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Skip run_trace_correction.py.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level for the orchestrator (default: INFO).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    config = PreferencePipelineConfig(
        repo_root=args.repo_root.resolve(),
        python_bin=args.python_bin,
        sanitize_output=args.sanitize_output,
        sanitize_limit=args.sanitize_limit,
        full_pipeline_args=tuple(args.full_args),
        breaker_output=args.breaker_output,
        breaker_args=tuple(args.breaker_args),
        trace_output=args.trace_output,
        trace_args=tuple(args.trace_args),
        noise_level=args.noise_level,
        noise_targets=tuple(args.noise_targets),
        noise_seed=args.noise_seed,
        skip_full_pipeline=args.skip_full,
        skip_breaker=args.skip_breaker,
        skip_trace=args.skip_trace,
    )

    pipeline = PreferencePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()

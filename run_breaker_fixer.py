from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Sequence, Dict, Any

from gemma_synth_pipeline import config
from gemma_synth_pipeline.services import (
    build_rule_defs,
    has_reasoning_format,
    load_sanitized_texts,
    sample_rules,
)
from gemma_synth_pipeline.services.data_loader import SanitizedSample, is_probable_review
from gemma_synth_pipeline.services.gemini_processor import GeminiProcessor
from gemma_synth_pipeline.services.rule_parser import RuleParser
from prompts import load_prompt

# ----------------------------------------------------------------------
# PROMPT LOADING
# ----------------------------------------------------------------------


BATCH_FIXER_SYSTEM_PROMPT = load_prompt(
    "breaker_fixer/batch_system_instruction.txt", strip=True
)
STAGE2_SYSTEM_PROMPT = load_prompt(
    "breaker_fixer/stage2_system_instruction.txt", strip=True
)
STAGE2_PROMPT_TEMPLATE = load_prompt(
    "breaker_fixer/stage2_prompt_template.txt", strip=False
)

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Breaker-Fixer DPO data generator (Stage 1 + Stage 2 both via Gemini batch)."
    )

    # Input / sampling
    parser.add_argument(
        "--sanitized-root",
        type=Path,
        default=config.TEXT_OUTPUT_SANITIZED,
        help="Root directory for sanitized text inputs.",
    )
    parser.add_argument(
        "--sanitized-limit",
        type=int,
        default=None,
        help="Limit number of sanitized inputs (for smoke tests).",
    )
    parser.add_argument(
        "--rules-limit",
        type=int,
        default=3,
        help="Limit number of rules used per run.",
    )

    # Variant counts
    parser.add_argument(
        "--atomic-per-rule",
        type=int,
        default=1,
        help="Atomic variants per rule per seed.",
    )
    parser.add_argument(
        "--near-miss-per-rule",
        type=int,
        default=0,
        help="Near-miss variants per rule per seed.",
    )
    parser.add_argument(
        "--compound-pairs",
        type=int,
        default=0,
        help="Number of rule pairs to sample per seed.",
    )
    parser.add_argument(
        "--compound-triples",
        type=int,
        default=0,
        help="Number of rule triples to sample per seed.",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=config.BREAKER_FIXER_OUTPUT,
        help="Output JSONL path.",
    )

    # Quality / validation
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="(Unused in batch Stage 2) Kept for backward compatibility.",
    )
    parser.add_argument(
        "--strict-format",
        action="store_true",
        help="Require <think>/Critique structure for teacher/hallucinator outputs.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="(Unused in batch Stage 2) Kept for backward compatibility.",
    )

    # Batch fixer controls (Stage 1)
    parser.add_argument(
        "--batch-fixer",
        dest="batch_fixer",
        action="store_true",
        default=True,
        help="Run batch fixer via Gemini before breaking.",
    )
    parser.add_argument(
        "--no-batch-fixer",
        dest="batch_fixer",
        action="store_false",
        help="Skip the batch fixer stage and use sanitized text as-is.",
    )

    # cache for batch fixer results
    parser.add_argument(
        "--batch-fixer-cache",
        type=Path,
        help=(
            "Optional JSONL cache file for batch fixer outputs. "
            "If present, Stage 1 reuses cached fixed text and only "
            "calls Gemini for uncached samples."
        ),
    )
    parser.add_argument(
        "--fixed-ad-raw",
        type=Path,
        default=Path("output/cache/fixed_ad_copy_raw.jsonl"),
        help=(
            "Optional raw Vertex batch output JSONL for fixed ad copies. "
            "If this file exists, Stage 1 will prepopulate baselines from it "
            "before calling Gemini."
        ),
    )

    # Stage 2 cache
    parser.add_argument(
        "--stage2-cache",
        type=Path,
        default=Path("output/cache/breaker_fixer.jsonl"),
        help=(
            "Optional JSONL cache for Stage-2 breaker/fixer outputs. "
            "If present, Stage 2 reuses cached records instead of calling Gemini. "
            "Example: cache/breaker_fixer.jsonl"
        ),
    )


    # Vertex / Gemini config
    parser.add_argument(
        "--key-file-path",
        type=str,
        default="./key.json",
        help="Vertex service account key file.",
    )
    parser.add_argument(
        "--vertex-project-id",
        type=str,
        default=config.VERTEX_PROJECT_ID,
        help="Vertex project ID.",
    )
    parser.add_argument(
        "--vertex-location",
        type=str,
        default=config.VERTEX_LOCATION,
        help="Vertex location (e.g. 'global').",
    )
    parser.add_argument(
        "--staging-bucket",
        type=str,
        default=config.VERTEX_STAGING_BUCKET,
        help="GCS bucket for Gemini batch jobs.",
    )
    parser.add_argument(
        "--batch-fixer-model",
        type=str,
        default=config.MODEL_FAST,
        help="Gemini model used for the batch fixer stage.",
    )
    parser.add_argument(
        "--batch-system-instruction",
        type=str,
        default=BATCH_FIXER_SYSTEM_PROMPT,
        help="System instruction for the batch fixer job.",
    )

    # Stage 2 batch config
    parser.add_argument(
        "--stage2-model",
        type=str,
        default=config.MODEL_SMART,
        help="Gemini model used for Stage 2 (breaker/teacher/hallucinator in one shot).",
    )
    parser.add_argument(
        "--stage2-system-instruction",
        type=str,
        default=STAGE2_SYSTEM_PROMPT,
        help="System instruction for the Stage 2 batch job.",
    )

    parser.add_argument(
        "--batch-poll-interval",
        type=int,
        default=30,
        help="Seconds between Vertex batch job polls (used for both Stage 1 and Stage 2).",
    )

    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=3072,  # match MAX_PROMPT_LENGTH for fine tuning qwen model
        help=(
            "Approximate max token length for the baseline (fixed) ad copy. "
            "If the fixed ad copy for a sample exceeds this, we skip "
            "breaker-fixer synthesis for that sample because the resulting "
            "DPO pair would be filtered out by the fine-tuning length limits."
        ),
    )

    parser.add_argument(
        "--batch-fixer-countdown",
        type=int,
        default=30,
        help=(
            "Seconds to wait (countdown) before launching Gemini batch fixer; "
            "set 0 to disable the countdown."
        ),
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Stage 1 cache helpers
# ----------------------------------------------------------------------


def load_batch_cache(cache_path: Path) -> dict[str, str]:
    """
    Load a JSONL cache of batch fixer outputs.

    Each line is expected to be:
      {"sample_id": "...", "fixed": "..."}
    """
    fixed_map: dict[str, str] = {}
    if not cache_path.exists():
        return fixed_map

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("sample_id")
            fixed = obj.get("fixed")
            if isinstance(sid, str) and isinstance(fixed, str) and fixed.strip():
                fixed_map[sid] = fixed
    return fixed_map


def save_batch_cache(cache_path: Path, fixed_map: dict[str, str]) -> None:
    """
    Save batch fixer outputs to JSONL cache.

    Overwrites the file with current mapping.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for sid, fixed in fixed_map.items():
            obj = {"sample_id": sid, "fixed": fixed}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_stage2_cache(cache_path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL cache of Stage-2 breaker/fixer outputs.

    Each line is expected to be a dict compatible with run_stage2_batch output:
      {
        ... job fields ...,
        "broken": "...",
        "chosen": "...",
        "rejected": "..."
      }
    """
    records: List[Dict[str, Any]] = []
    if not cache_path.exists():
        return records

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # minimum sanity check
            if not isinstance(obj, dict):
                continue
            if not obj.get("sample_id"):
                continue
            records.append(obj)

    logging.info(
        "Loaded %d Stage-2 records from cache %s",
        len(records),
        cache_path,
    )
    return records


def load_stage2_raw(
    raw_path: Path,
    jobs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Load Stage-2 breaker/fixer outputs from a raw Vertex batch predictions JSONL file.

    Each line is expected to look like:
      {
        "request": {...},
        "status": "...",
        "response": {
          "candidates": [
            {
              "content": {
                "parts": [
                  {"text": "{ \\"broken\\": ..., \\"chosen\\": ..., \\"rejected\\": ... }"}
                ]
              }
            }
          ],
          ...
        },
        "processed_time": ...
      }

    We align lines with `jobs` by index.
    """
    records: List[Dict[str, Any]] = []
    if not raw_path.exists():
        return records

    with raw_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        return records

    n = min(len(lines), len(jobs))
    if len(lines) != len(jobs):
        logging.warning(
            "Stage-2 raw file line count (%d) does not match job count (%d); "
            "using min=%d aligned pairs.",
            len(lines),
            len(jobs),
            n,
        )

    for idx in range(n):
        line = lines[idx]
        job = jobs[idx]

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logging.warning("Failed to parse Stage-2 raw JSON at line %d", idx)
            continue

        resp = obj.get("response") or {}
        try:
            candidates = resp["candidates"]
            if not candidates:
                continue
            parts = candidates[0]["content"]["parts"]
            texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
            resp_text = "\n".join(t for t in texts if t)
        except (KeyError, IndexError, TypeError):
            logging.warning("Malformed Stage-2 raw response at line %d", idx)
            continue

        if not resp_text.strip():
            continue

        try:
            data = json.loads(resp_text)
        except json.JSONDecodeError:
            logging.warning("Failed to parse Stage-2 inner JSON at line %d", idx)
            continue

        broken = (data.get("broken") or "").strip()
        chosen = (data.get("chosen") or "").strip()
        rejected = (data.get("rejected") or "").strip()

        if not broken or not chosen or not rejected:
            logging.warning(
                "Missing broken/chosen/rejected in Stage-2 raw response at line %d", idx
            )
            continue

        records.append(
            {
                **job,
                "broken": broken,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    logging.info(
        "Loaded %d Stage-2 records from raw Vertex batch file %s",
        len(records),
        raw_path,
    )
    return records


def save_stage2_cache(cache_path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Save Stage-2 breaker/fixer outputs to JSONL cache.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info("Saved %d Stage-2 records to cache %s", len(records), cache_path)


def load_fixed_ad_copy_raw(
    raw_path: Path,
    samples: List[SanitizedSample],
) -> dict[str, str]:
    """
    Load fixed ad copy from a raw Vertex batch output file.

    Supports two shapes:
      1) Vertex-style batch output:
         {
           "request": {...},
           "response": {
             "candidates": [
               {"content": {"parts": [{"text": "..."}]}}
             ],
             ...
           },
           ...
         }

      2) Simple cache:
         {"sample_id": "...", "fixed": "..."}

    For Vertex-style rows with no explicit sample_id we rely on
    positional alignment with `samples`.
    """
    fixed_map: dict[str, str] = {}
    if not raw_path.exists():
        return fixed_map

    with raw_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        sid = obj.get("sample_id")
        fixed_text: str | None = None

        # Shape 2: simple {sample_id, fixed}
        if sid is not None and isinstance(obj.get("fixed"), str):
            fixed_text = obj["fixed"]

        # Shape 1: Vertex batch prediction
        if fixed_text is None and "response" in obj:
            try:
                candidates = obj["response"]["candidates"]
                if not candidates:
                    continue
                parts = candidates[0]["content"]["parts"]
                texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
                fixed_text = "\n".join(t for t in texts if t)
            except (KeyError, IndexError, TypeError):
                fixed_text = None

        # Positional fallback for sample_id if not present in JSON
        if sid is None and idx < len(samples):
            sid = samples[idx].sample_id

        if (
            isinstance(sid, str)
            and isinstance(fixed_text, str)
            and fixed_text.strip()
        ):
            fixed_map[sid] = fixed_text.strip()

    logging.info(
        "Loaded %d fixed baselines from raw file %s (lines=%d)",
        len(fixed_map),
        raw_path,
        len(lines),
    )
    return fixed_map


def run_batch_wait(args: argparse.Namespace):
    if args.batch_fixer:
        if args.batch_fixer_countdown and args.batch_fixer_countdown > 0:
            total = args.batch_fixer_countdown
            logging.info(
                "Batch fixer will start in %d seconds for %d uncached samples. "
                "Press Ctrl+C to abort if this is unintended.",
                total,
            )
            for remaining in range(total, 0, -1):
                logging.info("... starting batch fixer in %d seconds", remaining)
                time.sleep(1)

# ----------------------------------------------------------------------
# Stage 1: Batch fixer (Gemini Batch) – optional
# ----------------------------------------------------------------------


def run_batch_fixer(
    samples: List[SanitizedSample],
    args: argparse.Namespace,
) -> dict[str, str]:
    """
    Fix every sanitized sample through a Vertex Gemini batch job and return
    a mapping: sample_id -> fixed text.

    This is the 'Stage 1' batch step. Stage 2 runs on the fixed baselines.
    """
    run_batch_wait(args)

    processor = GeminiProcessor(
        key_file_path=args.key_file_path,
        project_id=args.vertex_project_id,
        location=args.vertex_location,
        model_name=args.batch_fixer_model,
        staging_bucket=args.staging_bucket,
        system_instruction=args.batch_system_instruction,
    )

    prompts: List[str] = [s.text for s in samples]

    logging.info("Submitting %d sanitized prompts to Gemini batch fixer...", len(prompts))
    responses = processor.run_batch_job(prompts, poll_interval=args.batch_poll_interval)

    fixed_map: dict[str, str] = {}
    for sample, fixed in zip(samples, responses):
        if fixed and fixed.strip():
            fixed_map[sample.sample_id] = fixed.strip()

    logging.info("Retrieved %d fixed copies from Gemini batch", len(fixed_map))
    return fixed_map


# ----------------------------------------------------------------------
# Stage 2: Batch breaker / teacher / hallucinator pipeline
# ----------------------------------------------------------------------


def build_stage2_jobs(
    samples: List[SanitizedSample],
    rules: Sequence,
    args: argparse.Namespace,
    base_texts: Dict[str, str],
    batch_results: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Enumerate all requested variants as 'jobs' to send through Stage 2 batch."""
    jobs: List[Dict[str, Any]] = []

    for sample in samples:
        base_text = base_texts.get(sample.sample_id, sample.text)
        if not base_text or not base_text.strip():
            logging.warning("Skipping sample %s because base text is empty.", sample.sample_id)
            continue

        # NOTE: We skip generating breaker/fixer variants for very long fixed ad copies
        # because they would later be filtered out by the fine-tuning pipeline's
        # MAX_PROMPT_LENGTH token limit (see MAX_PROMPT_LENGTH in fine_tuning collab).
        # This avoids wasting synthesis and review budget on examples that cannot
        # be used for DPO training.
        if args.max_prompt_tokens and args.max_prompt_tokens > 0:
            # Rough proxy for model tokens: whitespace-separated tokens.
            approx_tokens = len(base_text.split())
            if approx_tokens > args.max_prompt_tokens:
                logging.info(
                    "Skipping sample %s for breaker-fixer synthesis: baseline length "
                    "%d tokens exceeds max_prompt_tokens=%d.",
                    sample.sample_id,
                    approx_tokens,
                    args.max_prompt_tokens,
                )
                continue

        baseline_source = "batch_fixer" if sample.sample_id in batch_results else "sanitized"

        # --- Atomic variants ---
        for rule in rules:
            for idx in range(args.atomic_per_rule):
                loudness = "obvious" if idx % 2 else "subtle"
                jobs.append(
                    {
                        "sample_id": sample.sample_id,
                        "prefix": sample.prefix,
                        "sanitized_method": sample.method,
                        "sanitized_path": str(sample.path),
                        "rule_ids": [rule.id],
                        "rule_titles": [rule.title],
                        "variant_type": "atomic",
                        "loudness": loudness,
                        "base_text": base_text,
                        "baseline_source": baseline_source,
                    }
                )

        # --- Near-miss variants ---
        if args.near_miss_per_rule > 0:
            for rule in rules:
                for _ in range(args.near_miss_per_rule):
                    jobs.append(
                        {
                            "sample_id": sample.sample_id,
                            "prefix": sample.prefix,
                            "sanitized_method": sample.method,
                            "sanitized_path": str(sample.path),
                            "rule_ids": [rule.id],
                            "rule_titles": [rule.title],
                            "variant_type": "near_miss",
                            "loudness": "subtle",
                            "base_text": base_text,
                            "baseline_source": baseline_source,
                        }
                    )

        # --- Compound pairs ---
        if args.compound_pairs > 0 and len(rules) > 1:
            pairs = list(itertools.combinations(rules, 2))
            random.shuffle(pairs)
            for pair in pairs[: args.compound_pairs]:
                jobs.append(
                    {
                        "sample_id": sample.sample_id,
                        "prefix": sample.prefix,
                        "sanitized_method": sample.method,
                        "sanitized_path": str(sample.path),
                        "rule_ids": [r.id for r in pair],
                        "rule_titles": [r.title for r in pair],
                        "variant_type": "compound_pair",
                        "loudness": "obvious",
                        "base_text": base_text,
                        "baseline_source": baseline_source,
                    }
                )

        # --- Compound triples ---
        if args.compound_triples > 0 and len(rules) > 2:
            triples = list(itertools.combinations(rules, 3))
            random.shuffle(triples)
            for triple in triples[: args.compound_triples]:
                jobs.append(
                    {
                        "sample_id": sample.sample_id,
                        "prefix": sample.prefix,
                        "sanitized_method": sample.method,
                        "sanitized_path": str(sample.path),
                        "rule_ids": [r.id for r in triple],
                        "rule_titles": [r.title for r in triple],
                        "variant_type": "compound_triple",
                        "loudness": "obvious",
                        "base_text": base_text,
                        "baseline_source": baseline_source,
                    }
                )

    logging.info("Prepared %d Stage-2 jobs.", len(jobs))
    return jobs


def build_stage2_prompt(job: Dict[str, Any], rule_objs: Sequence) -> str:
    """Construct a single Stage-2 prompt that asks the model to do breaker + teacher + hallucinator."""

    def get_rule_text(rule) -> str:
        return getattr(rule, "text", getattr(rule, "description", ""))

    rule_block_parts = []
    for r in rule_objs:
        rule_block_parts.append(
            f"Rule {r.id} - {r.title}:\n{get_rule_text(r)}"
        )
    rule_block = "\n\n".join(rule_block_parts)

    return STAGE2_PROMPT_TEMPLATE.format(
        base_text=job["base_text"],
        rule_block=rule_block,
        variant_type=job["variant_type"],
        loudness=job["loudness"],
    )


def run_stage2_batch(
    jobs: List[Dict[str, Any]],
    rules: Sequence,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Run all Stage-2 jobs via a single Gemini batch job."""
    run_batch_wait(args)
    processor = GeminiProcessor(
        key_file_path=args.key_file_path,
        project_id=args.vertex_project_id,
        location=args.vertex_location,
        model_name=args.stage2_model,
        staging_bucket=args.staging_bucket,
        system_instruction=args.stage2_system_instruction,
    )

    # Build rule map for quick lookup
    rule_map = {r.id: r for r in rules}

    prompts: List[str] = []
    for job in jobs:
        rule_objs = [rule_map[rid] for rid in job["rule_ids"] if rid in rule_map]
        prompt = build_stage2_prompt(job, rule_objs)
        prompts.append(prompt)

    logging.info("Submitting %d Stage-2 prompts to Gemini batch...", len(prompts))
    responses = processor.run_batch_job(prompts, poll_interval=args.batch_poll_interval)

    outputs: List[Dict[str, Any]] = []
    for job, resp in zip(jobs, responses):
        if not resp or not resp.strip():
            logging.warning("Empty Stage-2 response for sample %s", job["sample_id"])
            continue

        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            logging.warning("Failed to parse Stage-2 JSON for sample %s", job["sample_id"])
            continue

        broken = data.get("broken", "") or ""
        chosen = data.get("chosen", "") or ""
        rejected = data.get("rejected", "") or ""

        if not broken.strip() or not chosen.strip() or not rejected.strip():
            logging.warning("Missing fields in Stage-2 JSON for sample %s", job["sample_id"])
            continue

        outputs.append(
            {
                **job,
                "broken": broken.strip(),
                "chosen": chosen.strip(),
                "rejected": rejected.strip(),
            }
        )

    logging.info("Retrieved %d Stage-2 records from batch", len(outputs))
    return outputs


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    random.seed(config.SEED)
    args = parse_args()

    # 1) Load sanitized samples (already privacy-safe text)
    samples = load_sanitized_texts(root=args.sanitized_root, limit=args.sanitized_limit)
    if not samples:
        logging.warning("No sanitized samples found at %s", args.sanitized_root)
        return

    filtered_samples: List[SanitizedSample] = []
    for sample in samples:
        if "org_files" not in Path(sample.path).parts:
            logging.warning("Skipping %s (not sourced from org_files)", sample.sample_id)
            continue
        if is_probable_review(sample.text):
            logging.warning("Skipping %s (content still looks like reviewer memo)", sample.sample_id)
            continue
        filtered_samples.append(sample)

    samples = filtered_samples
    if not samples:
        logging.warning("All sanitized samples were filtered out; aborting.")
        return

    logging.info("Loaded %d sanitized samples from %s", len(samples), args.sanitized_root)

    # 2) Load rulebook and build rule definitions
    parser = RuleParser()
    parsed = parser.parse()
    rules = sample_rules(build_rule_defs(parsed), limit=args.rules_limit)

    logging.info("Loaded %d rules (limit=%d).", len(rules), args.rules_limit)

    # 3) Stage 1: optional batch fixer (with cache)
    base_texts: dict[str, str] = {s.sample_id: s.text for s in samples}
    batch_results: dict[str, str] = {}

    if args.batch_fixer:
        # 3a) Preload from raw fixed-ad batch output (if available)
        if args.fixed_ad_raw is not None and args.fixed_ad_raw.exists():
            logging.info(
                "Preloading fixed baselines from raw fixed ad copy file %s",
                args.fixed_ad_raw,
            )
            raw_map = load_fixed_ad_copy_raw(args.fixed_ad_raw, samples)
            if raw_map:
                logging.info(
                    "Using %d fixed baselines from raw file before calling Gemini.",
                    len(raw_map),
                )
                batch_results.update(raw_map)

        # 3b) Load simple {sample_id, fixed} cache if provided
        if args.batch_fixer_cache is not None:
            cache_map = load_batch_cache(args.batch_fixer_cache)
            if cache_map:
                logging.info(
                    "Loaded %d batch fixer entries from cache %s",
                    len(cache_map),
                    args.batch_fixer_cache,
                )
                # Do not overwrite raw results if both exist
                for sid, txt in cache_map.items():
                    batch_results.setdefault(sid, txt)

        # 3c) Determine which samples still need to be fixed via Gemini
        # NOTE: We skip running the ad copy fixer on very long sanitized texts,
        # because the resulting examples would later be filtered out by the
        # fine-tuning pipeline's MAX_PROMPT_LENGTH token limit (see
        # MAX_PROMPT_LENGTH in fine_tuning-4.py). This avoids wasting
        # synthesis/review budget on ad copies that cannot be used for DPO.
        missing_samples: list[SanitizedSample] = []
        for s in samples:
            if s.sample_id in batch_results:
                continue
            if args.max_prompt_tokens and args.max_prompt_tokens > 0:
                approx_tokens = len(s.text.split())
                if approx_tokens > args.max_prompt_tokens:
                    logging.info(
                        "Skipping sample %s in batch fixer: sanitized length %d tokens "
                        "exceeds max_prompt_tokens=%d; would be filtered in fine-tuning.",
                        s.sample_id,
                        approx_tokens,
                        args.max_prompt_tokens,
                    )
                    continue
            missing_samples.append(s)

        if missing_samples:
            logging.info(
                "Running batch fixer stage via Gemini on %d uncached samples...",
                len(missing_samples),
            )
            new_results = run_batch_fixer(missing_samples, args)
            batch_results.update(new_results)

            # 3d) Save updated simple cache if enabled
            if args.batch_fixer_cache is not None:
                save_batch_cache(args.batch_fixer_cache, batch_results)
                logging.info(
                    "Saved %d total batch fixer entries to cache %s",
                    len(batch_results),
                    args.batch_fixer_cache,
                )
        else:
            logging.info(
                "All %d samples already covered by raw+cache; skipping Gemini batch fixer.",
                len(samples),
            )

        # Stage 1 baseline: sanitized text overridden by any fixed copies
        base_texts.update(batch_results)
    else:
        logging.info("Skipping batch fixer; using sanitized text as baseline only.")

    # 4) Stage 2: build jobs and run via batch (or cache/raw)
    jobs = build_stage2_jobs(samples, rules, args, base_texts, batch_results)
    if not jobs:
        logging.warning("No Stage-2 jobs to process.")
        return

    stage2_records: List[Dict[str, Any]] = []

    if args.stage2_cache is not None and args.stage2_cache.exists():
        logging.info(
            "Loading Stage-2 breaker/fixer records from cache %s "
            "(skipping Gemini batch if cache is usable).",
            args.stage2_cache,
        )
        # First try direct cache format (already job + broken/chosen/rejected)
        stage2_records = load_stage2_cache(args.stage2_cache)

        if not stage2_records:
            logging.warning(
                "Stage-2 cache %s is empty or not in direct format; "
                "attempting to parse as raw Vertex batch predictions.",
                args.stage2_cache,
            )
            stage2_records = load_stage2_raw(args.stage2_cache, jobs)

        if not stage2_records:
            logging.warning(
                "Stage-2 cache %s could not be parsed; falling back to Gemini batch.",
                args.stage2_cache,
            )
            stage2_records = run_stage2_batch(jobs, rules, args)
            if args.stage2_cache is not None:
                save_stage2_cache(args.stage2_cache, stage2_records)
    else:
        stage2_records = run_stage2_batch(jobs, rules, args)
        if args.stage2_cache is not None:
            save_stage2_cache(args.stage2_cache, stage2_records)

    # 5) Write JSONL output
    sample_map = {s.sample_id: s for s in samples}
    rule_map = {r.id: r for r in rules}

    records: List[dict] = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f_out:
        for rec in stage2_records:
            sample = sample_map.get(rec["sample_id"])
            if not sample:
                continue

            # reconstruct rule objects for titles
            rec_rules = [rule_map[rid] for rid in rec["rule_ids"] if rid in rule_map]

            # Optional strict-format check
            if args.strict_format:
                if not has_reasoning_format(rec["chosen"]) or not has_reasoning_format(rec["rejected"]):
                    logging.warning(
                        "Skipping record for sample %s due to strict-format failure.", rec["sample_id"]
                    )
                    continue

            record = write_record(
                fh=f_out,
                sample=sample,
                rules=rec_rules,
                variant_type=rec["variant_type"],
                loudness=rec["loudness"],
                base_text=rec["base_text"],
                baseline_source=rec["baseline_source"],
                broken=rec["broken"],
                chosen=rec["chosen"],
                rejected=rec["rejected"],
            )
            records.append(record)

    logging.info("Wrote %d records to %s", len(records), out_path)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def write_record(
    fh,
    sample: SanitizedSample,
    rules: Sequence,
    variant_type: str,
    loudness: str,
    base_text: str,
    baseline_source: str,
    broken: str,
    chosen: str,
    rejected: str,
):
    rule_ids = [r.id for r in rules]
    record = {
        "sample_id": sample.sample_id,
        "prefix": sample.prefix,
        "sanitized_method": sample.method,
        "sanitized_path": str(sample.path),
        "rule_ids": rule_ids,
        "rule_titles": [r.title for r in rules],
        "variant_type": variant_type,
        "loudness": loudness,
        "base_text": base_text,              # fully compliant baseline (after human-feedback / batch fixer)
        "baseline_source": baseline_source,  # "sanitized" or "batch_fixer"
        "prompt": f"Review the following document for rule violations:\n\n{broken}",
        "broken": broken,
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "sanitized_method": sample.method,
            "sanitized_path": str(sample.path),
        },
    }
    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


if __name__ == "__main__":
    main()
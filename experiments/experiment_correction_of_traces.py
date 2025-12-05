from __future__ import annotations

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import re
from typing import Any, Dict, List
from gemma_synth_pipeline import config
from gemma_synth_pipeline.services.gemini_processor import GeminiProcessor
from prompts import load_prompt


BASE_SYSTEM_PROMPT = load_prompt("trace_correction/base_system.txt", strip=True)
RUBRIC_TEXT = load_prompt("trace_correction/rubric.txt", strip=False)
APPROACHES: Dict[str, str] = {
    "mirrored_voice": load_prompt("trace_correction/mirrored_voice.txt", strip=False),
    "procedural": load_prompt("trace_correction/procedural.txt", strip=False),
    "phrase_spotter": load_prompt("trace_correction/phrase_spotter.txt", strip=False),
}


FIXED_COPY_PATTERN = re.compile(r"<fixed_copy>([\s\S]*?)</fixed_copy>", re.IGNORECASE)


def extract_fixed_copy(chosen_value: Any) -> str:
    """
    Support either the legacy string format or dict-based records that already
    separate <think>, Critique, and fixed_copy fields.
    """
    if not chosen_value:
        return ""
    if isinstance(chosen_value, dict):
        fixed = chosen_value.get("fixed_copy")
        if isinstance(fixed, str):
            return fixed
        # Some datasets may embed a single text blob instead.
        possible_text = chosen_value.get("text") or chosen_value.get("content")
        if isinstance(possible_text, str):
            chosen_value = possible_text
        else:
            return ""
    if not isinstance(chosen_value, str):
        return ""
    match = FIXED_COPY_PATTERN.search(chosen_value)
    return match.group(1) if match else ""


def force_original_fixed_copy(revised_chosen: str, original_fixed: str) -> str:
    if not original_fixed:
        return revised_chosen
    if "<fixed_copy>" not in revised_chosen.lower():
        return f"{revised_chosen.rstrip()}\n\n<fixed_copy>{original_fixed}</fixed_copy>"

    return re.sub(
        r"<fixed_copy>[\s\S]*?</fixed_copy>",
        f"<fixed_copy>{original_fixed}</fixed_copy>",
        revised_chosen,
        flags=re.IGNORECASE,
    )


def load_record(path: Path, index: int) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as src:
        rows = [line.strip() for line in src if line.strip()]
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Index {index} out of range for file with {len(rows)} rows")
    return json.loads(rows[index])

def build_prompt(record: Dict[str, Any], approach_name: str, approach_text: str) -> str:
    record_text = json.dumps(record, ensure_ascii=False, indent=2)
    return f"""
You will evaluate ONE synthetic preference record for SEC compliance review.

{RUBRIC_TEXT.strip()}

APPROACH NAME: {approach_name}
{approach_text.strip()}

Global constraints:
- <think> must be structured exactly as specified for this approach and should be concise:
  avoid long explanations or repeating the entire document.
- <critique> must be 2–6 sentences, friendly, and easy to follow.
  It should NOT contain greetings, sign-offs, or names, and should NOT mention specific rule numbers.
  It should briefly explain what went wrong, name the issue category (e.g., “promissory language”), and
  suggest how to fix it.

Instructions:
- Only modify the TEACHER's 'chosen' answer.
- Preserve <fixed_copy> exactly as provided (do not change a single character inside it).
- Rewrite the <think> block and <critique> to follow the approach style and the constraints above.
- You MUST wrap the reasoning in a single <think>...</think> block,
  and the advisor-facing explanation in a single <critique>...</critique> block.
  Do NOT prefix the critique with the word "Critique:" and do NOT add greetings or signatures.
- The final 'revised_chosen' MUST have exactly this shape, in this order:
  <think>...</think>\\n<critique>...</critique>\\n<fixed_copy>...</fixed_copy>
- Return JSON with keys:
  {{
    "approach": "{approach_name}",
    "think_score": int,
    "critique_score": int,
    "fixed_copy_score": int,
    "keep_as_gold": bool,
    "fix_difficulty": "low" | "medium" | "high",
    "improvement_note": "short string",
    "revised_chosen": "<think>...</think>\\n<critique>...</critique>\\n<fixed_copy>...</fixed_copy>"
  }}

Record to process:
{record_text}
"""


def invoke_gemini(
    processor: GeminiProcessor,
    prompt: str,
    temperature: float,
    candidate_count: int,
) -> List[Dict[str, Any]]:
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    response = processor.client.models.generate_content(
        model=processor.model_name,
        contents=contents,
        config={
            "temperature": temperature,
            "candidate_count": candidate_count,
            "response_mime_type": "application/json",
            "system_instruction": {"parts": [{"text": BASE_SYSTEM_PROMPT}]},
        },
    )

    candidates: List[Dict[str, Any]] = []
    for idx, cand in enumerate(getattr(response, "candidates", []) or []):
        text_parts = []
        parts = getattr(getattr(cand, "content", None), "parts", []) or []
        for part in parts:
            txt = getattr(part, "text", None)
            if txt:
                text_parts.append(txt)
        raw_text = "\n".join(text_parts).strip()
        parsed = None
        error = None
        if raw_text:
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                error = str(exc)
        else:
            error = "No text returned"
        candidates.append(
            {
                "candidate_index": idx,
                "raw_text": raw_text,
                "parsed": parsed,
                "error": error,
                "finish_reason": getattr(cand, "finish_reason", None),
            }
        )
    return candidates


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    record = load_record(args.input, args.index)
    original_chosen = record.get("chosen", "") or ""
    original_fixed = extract_fixed_copy(original_chosen)
    processor = GeminiProcessor(
        key_file_path=args.key_file,
        project_id=args.project_id,
        location=args.location,
        model_name=args.model_name,
        staging_bucket=args.staging_bucket,
    )

    approach_results = []
    for name, description in APPROACHES.items():
        prompt = build_prompt(record, name, description)
        candidates = invoke_gemini(processor, prompt, args.temperature, args.candidate_count)
        for cand in candidates:
            parsed = cand.get("parsed")
            if not parsed:
                continue
            revised = parsed.get("revised_chosen", "")
            if not revised:
                continue
            patched = force_original_fixed_copy(revised, original_fixed)
            parsed["revised_chosen"] = patched
        result = {
            "record_index": args.index,
            "record_meta": record.get("meta"),
            "approach": name,
            "prompt": prompt.strip(),
            "candidates": candidates,
        }
        approach_results.append(result)
        print(f"\n=== {name.upper()} ===")
        for cand in candidates:
            print(f"\n--- Candidate {cand['candidate_index'] + 1} ---")
            if cand["error"]:
                print(f"[parse error] {cand['error']}")
                print(cand["raw_text"])
            else:
                pretty = json.dumps(cand["parsed"], ensure_ascii=False, indent=2)
                print(pretty)

    output_payload = {
        "input_file": str(args.input),
        "record_index": args.index,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "candidate_count": args.candidate_count,
        "results": approach_results,
    }
    return output_payload


def write_output(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as dst:
        dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"\nSaved experiment run to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemini Flash trace-correction experiments on a single DPO record."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/output.jsonl"),
        help="JSONL file containing breaker/fixer style records.",
    )
    parser.add_argument("--index", type=int, default=0, help="Zero-based index of the record to test.")
    parser.add_argument("--key-file", type=str, default="key.json", help="Service account key file.")
    parser.add_argument(
        "--project-id",
        type=str,
        default=config.VERTEX_PROJECT_ID,
        help="Vertex AI project id.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=config.VERTEX_LOCATION,
        help="Vertex AI location (e.g., global).",
    )
    parser.add_argument(
        "--staging-bucket",
        type=str,
        default=config.VERTEX_STAGING_BUCKET,
        help="GCS bucket (required by GeminiProcessor initialization).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=config.MODEL_FAST,
        help="Gemini model to use (defaults to gemini-2.5-flash).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature passed to Gemini.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=2,
        help="Number of candidates to request from Gemini per approach.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/experiments/correction_of_traces.jsonl"),
        help="Destination JSONL file to append experiment results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_payload = run_experiment(cli_args)
    write_output(cli_args.output, run_payload)

from __future__ import annotations

import argparse
import json
import re
import time
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Hashable

from gemma_synth_pipeline import config
from gemma_synth_pipeline.services.gemini_processor import GeminiProcessor
from experiments.experiment_correction_of_traces import (
    APPROACHES,
    BASE_SYSTEM_PROMPT,
    build_prompt,
    extract_fixed_copy,
    force_original_fixed_copy,
)
from gemma_synth_pipeline.services.noise import TextNoiser

PROCEDURAL_APPROACH = "procedural"


# ---------------------------------------------
# Concise, structured trace-correction approaches
# with better + weaker traces
# ---------------------------------------------

# These prompt variants all share the same logical output schema:
#   {
#     "approach": "...",
#     "think_score": int,
#     "critique_score": int,
#     "fixed_copy_score": int,
#     "keep_as_gold": bool,
#     "fix_difficulty": "easy" | "medium" | "hard" | null,
#     "improvement_note": str,
#
#     # main (better) trace:
#     "think": str,
#     "critique": str,
#     "fixed_copy": str,
#
#     # weaker (rejected) trace (optional but preferred):
#     "rejected_think": str,
#     "rejected_critique": str,
#     "rejected_fixed_copy": str,
#
#     # optional; if absent we will assemble it from think/critique/fixed_copy
#     "revised_chosen": str
#   }

APPROACHES.update(
    {
        "pipeline_short": """
You are an experienced lawyer in a compliance team that reviews marketing and client communications.
Think in a quiet, structured, internal way – like real notes in a review file – not like an AI explaining itself.

You will be given a JSON record with:
- "broken": original problematic copy
- "chosen": a previous improved version (may be verbose)
- "rejected": an alternative version to ignore
- "meta": context for humans only

Your task is to produce:
1) A SHORT, clearly structured, high-quality trace and final rewrite (the BETTER version).
2) A weaker but still plausible alternative trace (the REJECTED version) with similar structure and length.

GENERAL STYLE RULES
- Be concise and concrete.
- Sound like a real compliance lawyer.
- No motivational fluff or generic AI disclaimers.
- No boilerplate intros like "Based on the compliance review provided" or "Here is the rewritten marketing copy".
- Do not repeat the original text verbatim unless necessary.

HARD LENGTH RULES (if you exceed them, the answer is low quality)
- THINK: ABSOLUTE MAX 80 words. Target 40–70 words.
- CRITIQUE: ABSOLUTE MAX 140 words. Target 80–120 words.
- FIXED_COPY: 80–220 words.

====================
BETTER (CHOSEN) TRACE
====================

1) THINK (internal, not shown to the end user)
   - 1–3 numbered steps.
   - ABSOLUTE MAX 80 WORDS.
   - Focus only on:
     - the goal of the copy,
     - 2–3 key compliance risks (e.g., unsubstantiated performance, testimonials, one-sided benefits),
     - 1–2 main edits you will make.
   - Avoid rambling; no long narration about your process.

2) CRITIQUE (diagnostic of the ORIGINAL broken copy)
   - 3–5 bullet points.
   - ABSOLUTE MAX 140 WORDS.
   - Each bullet must:
     - quote or paraphrase one concrete phrase from the broken copy, and
     - tie it to a rule concept:
       - untrue or misleading statements or omissions
       - unsubstantiated material claims
       - testimonials/endorsements issues
       - performance presentation / "fair and balanced" concerns
       - one-sided benefit language or absolutes ("always", "guaranteed", "best")

3) FIXED_COPY (final user-facing text)
   - 1–3 short paragraphs, 80–220 words.
   - Directly usable as final copy.
   - No explanation of what you changed.

========================
WEAKER (REJECTED) TRACE
========================
You must also produce a weaker but still realistic alternative trace that a junior or rushed lawyer might produce.

The weaker version should:
- Use the SAME overall structure and roughly similar lengths,
- But be clearly inferior in at least one of these ways:
  - misses one or two issues that the better critique catches, OR
  - uses softer, more hedge-y or vague language ("might", "could", "policy issue") instead of clear rule references, OR
  - mis-classifies one issue under a generic "compliance concern" label instead of a specific rule, OR
  - leaves the fixed copy slightly under-fixed (e.g., keeps a borderline promotional phrase) while still broadly compliant.

Across different records, vary which weakness pattern you use:
- For some records, mainly weaken THINK.
- For others, mainly weaken CRITIQUE.
- For others, keep THINK/CRITIQUE similar but make FIXED_COPY slightly weaker or less crisp.
Do NOT always use the same weakness pattern.

Weaker THINK:
- same length band as the better THINK,
- more generic and hedge-y (e.g., "check for any compliance concerns" instead of naming exact rules),
- may omit one relevant risk area.

Weaker CRITIQUE:
- same number of bullets or one fewer,
- still references the text but:
  - uses vague labels like "policy" or "compliance expectations",
  - misses or mislabels 1–2 important issues.

Weaker FIXED_COPY:
- may reuse the better fixed copy OR provide a slightly less polished version (e.g., weaker disclosures),
- but must not contain blatantly non-compliant language.

OUTPUT FORMAT
Return ONLY JSON (no markdown) with keys:

Common metadata:
- "approach": "pipeline_short"
- "think_score": 0–10
- "critique_score": 0–10
- "fixed_copy_score": 0–10
- "keep_as_gold": boolean
- "fix_difficulty": "easy" | "medium" | "hard" | null
- "improvement_note": one short sentence

Main (better) trace fields:
- "think": THINK text only (no XML tags)
- "critique": CRITIQUE text only (no XML tags)
- "fixed_copy": FIXED_COPY text only (no XML tags)

Weaker (rejected) trace fields:
- "rejected_think": weaker THINK text only
- "rejected_critique": weaker CRITIQUE text only
- "rejected_fixed_copy": weaker or identical FIXED_COPY text only
""",
        "checklist_short": """
You are an experienced compliance lawyer editing marketing copy.
Think like you are writing structured internal review notes for another lawyer.

You will be given a JSON record with:
- "broken": original problematic copy
- "chosen": a previous improved rewrite
- "rejected": an alternative version to ignore
- "meta": context for humans only

Your job is to produce:
1) A SHORT checklist-style reasoning trace and concise final copy (BETTER version).
2) A weaker but plausible checklist-style trace (REJECTED version).

STYLE RULES
- Prefer headings and short bullet lists.
- Be concrete and rule-based.
- Sound like real internal compliance commentary.
- Avoid boilerplate like "Based on the compliance review provided".

HARD LENGTH RULES
- THINK: ABSOLUTE MAX 80 WORDS.
- CRITIQUE: ABSOLUTE MAX 140 WORDS.
- FIXED_COPY: 70–210 WORDS.

====================
BETTER (CHOSEN) TRACE
====================

1) THINK (checklist)
   - Use mini-headings with bullets.
   - ABSOLUTE MAX 80 WORDS. Target 40–70.
   - Example structure:
     Goal:
     - one bullet
     Compliance risks:
     - 2–4 bullets for specific risks (untrue/misleading, substantiation, testimonials/endorsements, performance, one-sided benefits)
     Rewrite plan:
     - 1–2 bullets for the main changes you will make.

2) CRITIQUE
   - 3–5 bullets, 80–140 words.
   - Focus on specific phrases that create compliance risk and link them to rule concepts.

3) FIXED_COPY
   - 1–3 short paragraphs, 70–210 words.
   - Clear, compliant, and suitable for a client-facing communication.

========================
WEAKER (REJECTED) TRACE
========================
Produce a second, weaker checklist-style trace:

- THINK:
  - same headings, similar length,
  - more generic bullets (e.g., “review performance claims” instead of naming time-period requirement),
  - may skip one important risk category.
- CRITIQUE:
  - same or fewer bullets (2–4),
  - uses softer language and fewer explicit references to specific phrases/rules,
  - may lump issues together under “policy concerns” instead of being precise.
- FIXED_COPY:
  - reuse the better fixed copy OR a slightly less crisp version,
  - no obviously non-compliant language.

Across different records, vary where the weakness appears:
- Sometimes mainly in THINK,
- sometimes mainly in CRITIQUE,
- sometimes mainly in FIXED_COPY.
Do not use the exact same weakening pattern every time.

OUTPUT FORMAT
Return ONLY JSON with keys:
- "approach": "checklist_short"
- "think_score", "critique_score", "fixed_copy_score"
- "keep_as_gold", "fix_difficulty", "improvement_note"
- "think", "critique", "fixed_copy"
- "rejected_think", "rejected_critique", "rejected_fixed_copy"
""",
        "minimal_think": """
You are a compliance lawyer optimizing for minimal visible thinking and strong final copy.
Your internal notes should be brief, practical, and to the point – like a margin note, not a memo.

You will be given a JSON record with broken / chosen / rejected / meta fields.

Your job is to:
- think briefly,
- write a short critique,
- and focus effort on a clean, usable FIXED_COPY.
You must also produce a weaker but plausible alternative trace.

HARD LENGTH RULES
- THINK: ABSOLUTE MAX 50 WORDS. Target 25–40 words.
- CRITIQUE: ABSOLUTE MAX 120 WORDS. Target 70–110 words.
- FIXED_COPY: 70–210 words.

====================
BETTER (CHOSEN) TRACE
====================

1) THINK
   - 1–2 sentences only.
   - Summarize what you will do to fix the copy.
   - Mention at most 2 rule areas (e.g., performance and testimonials).

2) CRITIQUE
   - 3–4 bullets, 70–110 words.
   - Each bullet describes one concrete issue in the original broken copy and ties it to a rule concept.

3) FIXED_COPY
   - 1–3 paragraphs, 70–210 words.
   - Plain language, compliant, no explanation.

========================
WEAKER (REJECTED) TRACE
========================

- THINK:
  - 1–2 sentences, similar length, but more generic and hedge-y,
  - e.g., “review for general compliance issues” without naming rule areas.
- CRITIQUE:
  - 2–4 bullets, similar length,
  - still flags some issues but:
    - may miss one major issue, or
    - talk about “tone” instead of the specific rule concept.
- FIXED_COPY:
  - reuse or slightly weaken the better fixed copy,
  - avoid obviously non-compliant language.

Across records, sometimes weaken only THINK, sometimes only CRITIQUE, sometimes only FIXED_COPY.
Do not always weaken the same component.

OUTPUT FORMAT
Return ONLY JSON with:
- "approach": "minimal_think"
- numeric scores and flags as before
- "think", "critique", "fixed_copy"
- "rejected_think", "rejected_critique", "rejected_fixed_copy"
""",
        "policy_first": """
You are a compliance specialist editing marketing copy to follow internal policy.
Write as if documenting a policy-focused review for another senior lawyer.

You will receive a JSON record with broken / chosen / rejected / meta.

Your job is to:
- identify policy-relevant constraints,
- give a short policy-focused critique,
- produce a concise policy-safe rewrite,
- and a weaker but still plausible policy-style review.

HARD LENGTH RULES
- THINK: ABSOLUTE MAX 90 WORDS. Target 50–80 words.
- CRITIQUE: ABSOLUTE MAX 150 WORDS. Target 90–130 words.
- FIXED_COPY: 80–220 words.

====================
BETTER (CHOSEN) TRACE
====================

1) THINK (policy-first)
   - 2–4 sentences.
   - Cover:
     - what the copy is trying to claim,
     - which policy themes are relevant (risk, guarantees, performance, testimonials),
     - how you will adjust the copy.

2) CRITIQUE
   - 3–5 bullets, 90–130 words.
   - Each bullet links specific language to policy concerns (e.g., unsubstantiated performance, unfair balance of risk/benefit).

3) FIXED_COPY
   - 1–3 paragraphs, 80–220 words.
   - Incorporates adjustments but reads naturally.

========================
WEAKER (REJECTED) TRACE
========================

- THINK:
  - similar length,
  - focuses more on general “policy alignment” language and less on concrete themes,
  - may omit one policy theme that the better THINK covers.
- CRITIQUE:
  - 2–4 bullets,
  - uses high-level phrases like “may not align with policy” instead of naming exact issues,
  - may mix risk/benefit and performance concerns into one bullet.
- FIXED_COPY:
  - reuse or slightly weaker wording (e.g., less explicit risk disclosure),
  - but not blatantly non-compliant.

Across records, alternate which part is weaker (THINK, CRITIQUE, or FIXED_COPY) so patterns are not obvious.

OUTPUT FORMAT
Return ONLY JSON with:
- "approach": "policy_first"
- scores/flags
- "think", "critique", "fixed_copy"
- "rejected_think", "rejected_critique", "rejected_fixed_copy"
""",
        "client_memo": """
You are drafting a short internal client memo as a compliance lawyer.
Your tone is that of a practical senior associate writing to a partner.

You will receive a JSON record with broken / chosen / rejected / meta.

You will produce:
- a concise internal memo-style trace (BETTER version),
- and a weaker but plausible memo (REJECTED version).

HARD LENGTH RULES
- THINK: ABSOLUTE MAX 80 WORDS. Target 40–70 words.
- CRITIQUE: ABSOLUTE MAX 150 WORDS. Target 90–130 words.
- FIXED_COPY: 80–220 words.

====================
BETTER (CHOSEN) TRACE
====================

1) THINK
   - 1–3 sentences (40–70 words).
   - State what you are reviewing and at a high level what you will check.

2) CRITIQUE (memo format)
   - Use headings: Background / Issues / Recommendations.
   - Background: 1–2 bullets summarizing what the copy claims.
   - Issues: 2–4 bullets, each tied to a specific rule concern.
   - Recommendations: 2–4 bullets with concrete edits.

3) FIXED_COPY
   - 1–3 paragraphs, 80–220 words.
   - Final client-ready copy with no meta commentary.

========================
WEAKER (REJECTED) TRACE
========================

- THINK:
  - still memo-like, but more generic and hedge-y,
  - may not specify exactly which rules will be checked.
- CRITIQUE:
  - same headings, fewer or vaguer bullets,
  - Background: similar,
  - Issues: 1–3 bullets that merge multiple issues or label them as “compliance concerns”,
  - Recommendations: more high-level suggestions (“add disclosure”) without precise wording.
- FIXED_COPY:
  - reuse or slightly weaker version of the better fixed copy.

Vary the weakness across records: sometimes THINK is weaker, sometimes CRITIQUE, sometimes FIXED_COPY.

OUTPUT FORMAT
Return ONLY JSON with:
- "approach": "client_memo"
- scores/flags
- "think", "critique", "fixed_copy"
- "rejected_think", "rejected_critique", "rejected_fixed_copy"
""",
        "investor_impression": """
You are a lawyer focused on what impression a reasonable investor would take from the copy.
Your notes should read like a short internal risk memo, not marketing copy.

You will receive a JSON record with broken / chosen / rejected / meta.

You will produce:
- a better investor-impression-focused trace,
- and a weaker but plausible alternative.

HARD LENGTH RULES
- THINK: ABSOLUTE MAX 80 WORDS. Target 40–70 words.
- CRITIQUE: ABSOLUTE MAX 150 WORDS. Target 90–130 words.
- FIXED_COPY: 80–220 words.

====================
BETTER (CHOSEN) TRACE
====================

1) THINK
   - 1–3 bullets, 40–70 words.
   - Describe:
     - what impression a reasonable investor would likely take today,
     - what impressions you need to avoid,
     - in one bullet, how you will adjust the copy.

2) CRITIQUE
   - Use headings like:
     Investor impression:
     - ...
     Missing or weak disclosures:
     - ...
     Balance of benefits and risks:
     - ...
   - 3–5 bullets total, 90–130 words.

3) FIXED_COPY
   - 1–3 paragraphs, 80–220 words.
   - Adjust claims and add context so a reasonable investor is not misled.

========================
WEAKER (REJECTED) TRACE
========================

- THINK:
  - same style but more generic (“tone may be a bit positive”),
  - may not clearly separate current vs desired investor impression.
- CRITIQUE:
  - similar headings, but:
    - fewer or vaguer bullets,
    - may miss one important missing disclosure or risk statement.
- FIXED_COPY:
  - reuse or slightly weaker version of the better fixed copy.

Across records, vary whether THINK, CRITIQUE, or FIXED_COPY is notably weaker so the pattern is not trivial.

OUTPUT FORMAT
Return ONLY JSON with:
- "approach": "investor_impression"
- scores/flags
- "think", "critique", "fixed_copy"
- "rejected_think", "rejected_critique", "rejected_fixed_copy"
""",
        "disclosure_focus": """
You are a compliance lawyer concentrating on disclosure obligations.
Your internal notes should be short and structured, like a disclosure checklist.

You will receive a JSON record with broken / chosen / rejected / meta.

You will produce:
- a better disclosure-focused trace,
- and a weaker but plausible disclosure analysis.

HARD LENGTH RULES
- THINK: ABSOLUTE MAX 80 WORDS. Target 40–70 words.
- CRITIQUE: ABSOLUTE MAX 160 WORDS. Target 90–140 words.
- FIXED_COPY: 80–220 words.

====================
BETTER (CHOSEN) TRACE
====================

1) THINK
   - 1–3 bullets, 40–70 words.
   - Outline main claims and which disclosures or contextual statements are needed.

2) CRITIQUE
   - Headings:
     Key claims and implications:
     - ...
     Required or advisable disclosures:
     - ...
     Language to soften or remove:
     - ...
   - 3–5 bullets, 90–140 words.

3) FIXED_COPY
   - 1–3 paragraphs, 80–220 words.
   - Bakes the needed disclosures and softening into the copy.

========================
WEAKER (REJECTED) TRACE
========================

- THINK:
  - similar length,
  - more generic language about “adding disclosures” without specifying which.
- CRITIQUE:
  - same headings, but:
    - fewer bullets or bullets that combine multiple issues,
    - references “disclosures” in general without tying them to specific claims.
- FIXED_COPY:
  - reuse or slightly weaker text.

Across records, alternate whether THINK, CRITIQUE, or FIXED_COPY is weaker so there is no single obvious pattern.

OUTPUT FORMAT
Return ONLY JSON with:
- "approach": "disclosure_focus"
- scores/flags
- "think", "critique", "fixed_copy"
- "rejected_think", "rejected_critique", "rejected_fixed_copy"
""",
    }
)

# First-pass multi-approach set
MULTI_DPO_APPROACHES: List[str] = [
    "pipeline_short",
    "checklist_short",
    "minimal_think",
    "policy_first",
    "client_memo",
    "investor_impression",
    "disclosure_focus",
]

# ---------------------------------------------
# Second-pass offshoot approaches (compression + shorthand)
# ---------------------------------------------

SECOND_PASS_SUFFIX = """

=========================
SECOND-PASS INSTRUCTIONS
=========================

You are performing a SECOND PASS on an already improved but possibly verbose trace.

Your goals in this second pass:

1) THINK (internal reasoning)
   - Compress aggressively: 1–3 sentences (about 25–60 words).
   - You MAY use brief internal shorthand that real compliance teams use, but ONLY in THINK, e.g.:
     - "perf/no period" (performance without time period),
     - "disc weak" (disclosure weak),
     - "bal risk/benefit" (balance of risk and benefit),
     - "test/endorse" (testimonial / endorsement issues),
     - "inv. impression" (investor impression).
   - Do NOT explain the task, prompts, system, or tags.

2) CRITIQUE (for internal but client-friendly readers)
   - NO shorthand. Write in clear plain English for lawyers / business partners.
   - Keep 3–5 bullets, concise but specific.
   - Name concrete issues and briefly suggest HOW to improve (e.g., "add period + risk context", "soften absolute").

3) FIXED_COPY (client-ready ad copy)
   - NO shorthand and NO LLM commentary like "Here is the rewritten marketing copy" or "This version incorporates...".
   - 1–3 short paragraphs (roughly 80–200 words).
   - Preserve the business goal, fix compliance issues, and keep tone professional.

4) SCORING AND GOLD FLAG
   - Only set keep_as_gold = true if THINK/CRITIQUE/FIXED_COPY are all at least a 5 on a 0–10 scale.
   - If anything is borderline, keep_as_gold = false and explain briefly in improvement_note.
"""

MULTI_DPO_APPROACHES_SECOND_PASS: List[str] = [
    "pipeline_short_compress",
    "checklist_short_compress",
    "minimal_think_compress",
    "policy_first_compress",
    "client_memo_compress",
    "investor_impression_compress",
    "disclosure_focus_compress",
]

APPROACHES.update(
    {
        "pipeline_short_compress": APPROACHES["pipeline_short"] + SECOND_PASS_SUFFIX,
        "checklist_short_compress": APPROACHES["checklist_short"] + SECOND_PASS_SUFFIX,
        "minimal_think_compress": APPROACHES["minimal_think"] + SECOND_PASS_SUFFIX,
        "policy_first_compress": APPROACHES["policy_first"] + SECOND_PASS_SUFFIX,
        "client_memo_compress": APPROACHES["client_memo"] + SECOND_PASS_SUFFIX,
        "investor_impression_compress": APPROACHES["investor_impression"] + SECOND_PASS_SUFFIX,
        "disclosure_focus_compress": APPROACHES["disclosure_focus"] + SECOND_PASS_SUFFIX,
    }
)

DEFAULT_BATCH_WAIT_SECONDS = 30
BASE_DIR = Path(__file__).resolve().parent

# First-pass cache (existing behavior)
THINKING_CORRECTION_CACHE_PATH = BASE_DIR / "output" / "thinking_correction_output1.jsonl"
# Second-pass cache (for compressed / shorthand runs)
THINKING_CORRECTION_CACHE_PATH_SECOND = BASE_DIR / "output" / "thinking_correction_output2.jsonl"

# original requests used to generate thinking corrections
INPUT_THINKING_PATH = BASE_DIR / "output" / "input_thinking.jsonl"

# debug log just for problematic records
TRACE_DEBUG_LOG_PATH = BASE_DIR / "output" / "trace_correction_debug.jsonl"


LLM_COMMENTARY_PREFIXES = [
    "Here is the rewritten marketing copy.",
    "Here is the revised marketing copy.",
    "Here is the re-written marketing copy.",
    "This version incorporates the specific compliance recommendations",
    "The following reflects the requested compliance edits",
]


def strip_llm_commentary_lines(text: str) -> str:
    """
    Remove obvious LLM meta-lines from fixed_copy. We only drop lines
    that look like 'Here is the rewritten marketing copy.' etc.,
    leaving the substantive ad copy intact.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    prefixes = [p.lower() for p in LLM_COMMENTARY_PREFIXES]
    kept: List[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        lower = stripped.lower()
        if any(lower.startswith(p) for p in prefixes):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def append_debug_issue(payload: Dict[str, Any]) -> None:
    """
    Append a single debug record as JSONL to TRACE_DEBUG_LOG_PATH.

    This is ONLY used for:
      - parse failures
      - missing original <fixed_copy>
    """
    TRACE_DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _safe_snippet(value: Any, limit: int = 2000) -> str:
    """
    Safely turn any value into a short, human-readable snippet for logging.
    """
    try:
        if isinstance(value, str):
            return value[:limit]
        try:
            return json.dumps(value, ensure_ascii=False)[:limit]
        except (TypeError, ValueError):
            return repr(value)[:limit]
    except Exception:
        return "<unprintable value>"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- Helpers for IDs & meta keys ----------

def _meta_key_from_record(record: Dict[str, Any]) -> Optional[str]:
    """
    Build a stable ID from record.meta.{source_file,source_line}.
    """
    meta = record.get("meta")
    if not isinstance(meta, dict):
        return None
    source_file = meta.get("source_file")
    source_line = meta.get("source_line")
    if not isinstance(source_file, str) or source_line is None:
        return None
    return f"{source_file}:{source_line}"


# ---------- Robust JSON + XML-ish parsing helpers ----------

def _extract_candidate_json(raw_text: str) -> Optional[str]:
    """
    Try to extract a JSON object from the raw model text.

    Handles:
      - Optional ```json ... ``` fences
      - Language tag on first line inside the fence
      - Extra junk before/after by using json.JSONDecoder.raw_decode
    """
    text = (raw_text or "").strip()
    if not text:
        return None

    # Remove optional Markdown code fences, e.g. ```json ... ```
    if text.startswith("```"):
        fence_end = text.find("```", 3)
        if fence_end != -1:
            inner = text[3:fence_end]
            first_newline = inner.find("\n")
            if first_newline != -1:
                maybe_lang = inner[:first_newline].strip().lower()
                if maybe_lang in {"json", "jsonc", "javascript"}:
                    inner = inner[first_newline + 1 :]
            text = inner.strip()

    if not text:
        return None

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end_pos = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        return json.dumps(obj, ensure_ascii=False)

    return None


def _extract_tag_block(text: str, tag: str) -> Optional[str]:
    """
    Extract the full <tag>...</tag> block from text, if present.
    Returns the entire block including tags, or None.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return text[m.start():m.end()].strip()


def _parse_xmlish_candidate(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Fallback parser when JSON is missing or invalid.

    If we see <think>...</think> and <fixed_copy>...</fixed_copy>,
    we construct a minimal parsed dict with a 'revised_chosen' field.
    """
    if not raw_text:
        return None

    text = raw_text.strip()
    if "<think>" not in text.lower() or "</fixed_copy>" not in text.lower():
        return None

    think_block = _extract_tag_block(text, "think")
    critique_block = _extract_tag_block(text, "critique")
    fixed_block = _extract_tag_block(text, "fixed_copy")

    if not think_block or not fixed_block:
        return None

    if not critique_block:
        critique_block = "<critique></critique>"

    revised_chosen = "\n".join(
        part for part in [think_block.strip(), critique_block.strip(), fixed_block.strip()] if part.strip()
    )

    return {
        "approach": "procedural_fallback",
        "think_score": None,
        "critique_score": None,
        "fixed_copy_score": None,
        "keep_as_gold": False,
        "fix_difficulty": None,
        "improvement_note": "",
        "revised_chosen": revised_chosen,
    }


def parse_candidate(raw_text: str) -> Dict[str, Any] | None:
    """
    Parse a model response into a structured dict.

    Priority:
      1) Robust JSON extraction (handles code fences and extra junk).
         We support two JSON shapes:
           a) { ..., "revised_chosen": "<think>...<critique>...<fixed_copy>..." }
           b) { ..., "think": "...", "critique": "...", "fixed_copy": "..." }
              --> we assemble revised_chosen from sections.
      2) Fallback: parse raw <think>/<critique>/<fixed_copy> tags.
    """
    if not raw_text:
        print("[WARN] Empty raw text for candidate; returning None.")
        return None

    json_text = _extract_candidate_json(raw_text)
    if json_text:
        try:
            obj = json.loads(json_text)

            if isinstance(obj, dict) and any(k in obj for k in ("think", "critique", "fixed_copy")):
                think = obj.get("think") or ""
                critique = obj.get("critique") or ""
                fixed = obj.get("fixed_copy") or ""

                if "<think>" not in think.lower():
                    think = f"<think>\n{think.strip()}\n</think>"
                if "<critique>" not in critique.lower():
                    critique = f"<critique>\n{critique.strip()}\n</critique>"
                if "<fixed_copy>" not in fixed.lower():
                    fixed = f"<fixed_copy>\n{fixed.strip()}\n</fixed_copy>"

                revised = "\n\n".join(part for part in (think, critique, fixed) if part.strip())
                obj["revised_chosen"] = revised
                return obj

            return obj
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON decode failed, attempting XML-ish fallback: {e}")

    xmlish = _parse_xmlish_candidate(raw_text)
    if xmlish is not None:
        return xmlish

    print("[WARN] No JSON or XML-ish structure found for candidate; returning None.")
    return None


# ---------- Loading original fixed_copy from input_thinking.jsonl ----------

def _extract_request_text_from_input_row(row: Dict[str, Any]) -> Optional[str]:
    """
    Given one line from input_thinking.jsonl, reconstruct the big user prompt text.
    """
    request = row.get("request")
    if not isinstance(request, dict):
        return None
    contents = request.get("contents") or []
    chunks: List[str] = []
    for msg in contents:
        if not isinstance(msg, dict):
            continue
        parts = msg.get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            t = part.get("text")
            if isinstance(t, str) and t:
                chunks.append(t)
    if not chunks:
        return None
    return "\n".join(chunks)


def _parse_record_from_request_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Inside the big instruction text, find the 'Record to process:' JSON block
    and parse it into a dict (broken/chosen/rejected/prompt/meta/...).
    """
    if not text:
        return None
    marker = "Record to process:"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    snippet = text[idx + len(marker) :]
    json_text = _extract_candidate_json(snippet)
    if not json_text:
        return None
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return None


def load_original_fixed_map_from_input_thinking(path: Path) -> Dict[str, str]:
    """
    Build a mapping: meta_key -> original <fixed_copy> from the
    'Record to process' inside input_thinking.jsonl.
    """
    mapping: Dict[str, str] = {}
    if not path.exists():
        print(f"[INFO] input_thinking file not found at {path}; skipping extra fixed_copy recovery.")
        return mapping

    rows = read_jsonl(path)
    for row in rows:
        text = _extract_request_text_from_input_row(row)
        if not text:
            continue
        rec = _parse_record_from_request_text(text)
        if not isinstance(rec, dict):
            continue

        meta = rec.get("meta")
        if not isinstance(meta, dict):
            continue
        source_file = meta.get("source_file")
        source_line = meta.get("source_line")
        if not isinstance(source_file, str) or source_line is None:
            continue
        key = f"{source_file}:{source_line}"

        chosen = rec.get("chosen")
        if not isinstance(chosen, str):
            continue
        orig_fixed = extract_fixed_copy(chosen)
        if isinstance(orig_fixed, str) and orig_fixed.strip():
            mapping[key] = orig_fixed

    print(f"[INFO] Loaded {len(mapping)} original <fixed_copy> entries from {path}")
    return mapping


# ---------- Prompt builder ----------

def build_prompts(
    records: List[Dict[str, Any]],
    original_fixed_overrides: Optional[Dict[str, str]] = None,
    is_second_pass: bool = False,
) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Build one prompt per record, and capture the ORIGINAL fixed_copy so we can
    force it back into the revised chosen later.

    This version optionally mixes multiple prompt approaches:
      - First pass: MULTI_DPO_APPROACHES.
      - Second pass: MULTI_DPO_APPROACHES_SECOND_PASS (compressed / shorthand).
    """
    prompts: List[str] = []
    metadata: List[Dict[str, Any]] = []
    overrides = original_fixed_overrides or {}

    for idx, record in enumerate(records):
        if is_second_pass and MULTI_DPO_APPROACHES_SECOND_PASS:
            approach_name = random.choice(MULTI_DPO_APPROACHES_SECOND_PASS)
        elif MULTI_DPO_APPROACHES:
            approach_name = random.choice(MULTI_DPO_APPROACHES)
        else:
            approach_name = PROCEDURAL_APPROACH

        approach_text = APPROACHES[approach_name]
        prompt = build_prompt(record, approach_name, approach_text)
        prompts.append(prompt)

        key = _meta_key_from_record(record)

        original_fixed: str = ""
        if key and key in overrides:
            original_fixed = overrides[key]
        else:
            raw_chosen = record.get("chosen", "")
            chosen_text = raw_chosen if isinstance(raw_chosen, str) else ""
            original_fixed = extract_fixed_copy(chosen_text)

        metadata.append(
            {
                "index": idx,
                "record": record,
                "original_fixed": original_fixed,
                "meta_key": key,
                "approach": approach_name,
            }
        )

    return prompts, metadata


# ---------- Cached response helpers ----------

def _extract_text_from_cached_row(row: Dict[str, Any]) -> str | None:
    """Get the response text from a cached batch output row if available."""
    if not isinstance(row, dict):
        return None

    response_text = row.get("response_text")
    if isinstance(response_text, str):
        return response_text

    def _from_blob(blob: Any) -> str | None:
        if not isinstance(blob, dict):
            return None
        candidates = blob.get("candidates") or []
        if not candidates:
            return None
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts") or []
        if not parts:
            return None
        chunks: List[str] = []
        for part in parts:
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text:
                chunks.append(part_text)
        if not chunks:
            return None
        return "\n".join(chunks)

    for blob_source in ("response", "raw"):
        extracted = _from_blob(row.get(blob_source))
        if extracted:
            return extracted

    return None


def load_cached_responses(path: Path, expected_count: int) -> List[str] | None:
    if not path.exists():
        return None

    rows = read_jsonl(path)
    if len(rows) != expected_count:
        print(
            f"[WARN] Cached response count ({len(rows)}) does not match prompts ({expected_count}); ignoring cache."
        )
        return None

    responses: List[str] = []
    for idx, row in enumerate(rows):
        text = _extract_text_from_cached_row(row)
        if not isinstance(text, str):
            print(f"[WARN] Cache entry {idx} missing extractable text; ignoring cache.")
            return None
        responses.append(text)

    print(f"[INFO] Loaded {len(responses)} cached responses from {path}")
    return responses


def cache_responses(path: Path, responses: List[str]) -> None:
    payload = [{"index": idx, "response_text": text} for idx, text in enumerate(responses)]
    write_jsonl(path, payload)
    print(f"[INFO] Cached {len(responses)} responses to {path}")


def maybe_wait_before_batch(wait_seconds: int, num_prompts: int) -> None:
    if wait_seconds <= 0:
        return

    print(
        f"[INFO] Batch job for {num_prompts} prompts will start in "
        f"{wait_seconds} seconds. Press Ctrl+C to cancel."
    )
    for remaining in range(wait_seconds, 0, -1):
        print(f"   Starting in {remaining:2d}s...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 30, end="\r")
    print("[INFO] Starting batch job now.")


# ---------- DPO + trace helpers ----------

def _has_well_formed_tags_in_order(text: str) -> bool:
    """
    Return True iff text contains:
      <think>...</think><critique>...</critique><fixed_copy>...</fixed_copy>
    in that exact order.
    """
    if not isinstance(text, str):
        return False
    try:
        ti = text.index("<think>")
        te = text.index("</think>")
        ci = text.index("<critique>")
        ce = text.index("</critique>")
        fi = text.index("<fixed_copy>")
        fe = text.index("</fixed_copy>")
    except ValueError:
        return False
    return ti < te < ci < ce < fi < fe


def _extract_section(text: str, tag: str) -> Optional[str]:
    """
    Extract inner text from a single-tag section like <tag>...</tag>.
    Returns the inner content without the tags, or None if missing.
    """
    if not isinstance(text, str):
        return None
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def split_sections(text: str) -> Dict[str, Optional[str]]:
    """
    Split a chosen/revised_chosen string into its <think>, <critique>,
    and <fixed_copy> components.
    """
    return {
        "think": _extract_section(text, "think"),
        "critique": _extract_section(text, "critique"),
        "fixed_copy": _extract_section(text, "fixed_copy"),
    }


def _word_count(s: Optional[str]) -> int:
    if not isinstance(s, str):
        return 0
    return len(s.split())


def _truncate_words(text: str, max_words: int) -> str:
    """Return at most max_words words from text."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _add_hedging(text: str) -> str:
    """
    Make the tone slightly weaker / more tentative by sprinkling in
    hedging phrases. Intentionally simple and heuristic.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    replacements = {
        "violates": "may violate",
        "violate": "may violate",
        "breaches": "might breach",
        "breach": "might breach",
        "is misleading": "could be misleading",
        "misleading": "potentially misleading",
        "unsubstantiated": "possibly unsubstantiated",
        "problematic": "potentially problematic",
        "non-compliant": "may be non-compliant",
        "inaccurate": "may not be fully accurate",
        "unbalanced": "may not be fully balanced",
    }
    out = text
    for src, tgt in replacements.items():
        out = re.sub(r"\b" + re.escape(src) + r"\b", tgt, out, flags=re.IGNORECASE)
    return out


def _split_bullets_and_other(text: str) -> tuple[List[str], List[str]]:
    """Split a critique into bullet lines vs non-bullet lines."""
    if not isinstance(text, str) or not text.strip():
        return [], []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    bullet_lines: List[str] = []
    other_lines: List[str] = []
    for ln in lines:
        if ln.lstrip().startswith(("-", "•", "*")):
            bullet_lines.append(ln)
        else:
            other_lines.append(ln)
    return bullet_lines, other_lines


def _reject_variant_hedged_short(think: str, critique: str) -> tuple[str, str]:
    """
    Variant A:
    - THINK: shorter + hedged.
    - CRITIQUE: about half the bullets, hedged.
    """
    weak_think = _add_hedging(_truncate_words(think, max_words=50))

    bullets, _ = _split_bullets_and_other(critique)
    if not bullets:
        weak_crit = _add_hedging(_truncate_words(critique, max_words=80))
        return weak_think, weak_crit

    keep_count = max(2, len(bullets) // 2)
    kept = bullets[:keep_count]
    hedged_kept = [_add_hedging(b) for b in kept]
    weak_critique = "\n".join(hedged_kept)
    return weak_think.strip(), weak_critique.strip()


def _reject_variant_missing_issue(think: str, critique: str) -> tuple[str, str]:
    """
    Variant B:
    - THINK: fairly normal but slightly shorter.
    - CRITIQUE: keeps most structure, but deliberately drops one bullet
      (usually the longest -> often the most detailed issue).
    """
    weak_think = _truncate_words(think, max_words=70)

    bullets, _ = _split_bullets_and_other(critique)
    if not bullets:
        weak_crit = _truncate_words(critique, max_words=100)
        return weak_think.strip(), weak_crit.strip()

    if len(bullets) <= 2:
        hedged = [_add_hedging(b) for b in bullets]
        return weak_think.strip(), "\n".join(hedged).strip()

    lengths = [len(b.split()) for b in bullets]
    longest_idx = max(range(len(bullets)), key=lambda i: lengths[i])
    kept = [b for i, b in enumerate(bullets) if i != longest_idx]

    weak_critique = "\n".join(kept)
    return weak_think.strip(), weak_critique.strip()


def _reject_variant_vague_policy(think: str, critique: str) -> tuple[str, str]:
    """
    Variant C:
    - THINK: similar length, but more generic and hedge-y.
    - CRITIQUE: keeps bullet count, but replaces specific rule references
      with vague "policy/compliance" language.
    """
    weak_think = _add_hedging(_truncate_words(think, max_words=80))

    bullets, _ = _split_bullets_and_other(critique)
    if not bullets:
        generic = _add_hedging(_truncate_words(critique, max_words=110))
        return weak_think.strip(), generic.strip()

    def _vague_line(ln: str) -> str:
        out = ln

        # Remove explicit rule cites like "(4.2)" etc.
        out = re.sub(r"\(\d+\.\d+\)", " (policy requirement) ", out)

        replacements = {
            "performance": "results",
            "testimonial": "feedback",
            "testimonials": "feedback",
            "endorsement": "third-party comment",
            "endorsements": "third-party comments",
            "unsubstantiated": "may not fully meet policy expectations",
            "fair and balanced": "balanced enough",
            "marketing rule": "internal policy",
            "SEC": "regulatory expectations",
        }
        for src, tgt in replacements.items():
            out = re.sub(r"\b" + re.escape(src) + r"\b", tgt, out, flags=re.IGNORECASE)

        out = _add_hedging(out)
        return out

    vague_bullets = [_vague_line(b) for b in bullets]
    weak_critique = "\n".join(vague_bullets)
    return weak_think.strip(), weak_critique.strip()


def synthesize_rejected_from_chosen(chosen_text: str) -> str:
    """
    Build a weaker but still structured trace to use as the DPO 'rejected' side
    if Gemini didn't provide explicit rejected_* fields.

    We keep the same fixed_copy content and only weaken THINK/CRITIQUE so that
    some pairs focus purely on reasoning quality.
    """
    if not isinstance(chosen_text, str) or not chosen_text.strip():
        return ""

    sections = split_sections(chosen_text)
    think = sections.get("think") or ""
    critique = sections.get("critique") or ""
    fixed = sections.get("fixed_copy") or ""

    if not think and not critique:
        return ""

    variants = [
        _reject_variant_hedged_short,
        _reject_variant_missing_issue,
        _reject_variant_vague_policy,
    ]
    variant_fn = random.choice(variants)
    weak_think, weak_critique = variant_fn(think, critique)

    parts: List[str] = []
    parts.append("<think>\n" + (weak_think.strip() or "") + "\n</think>")
    parts.append("<critique>\n" + (weak_critique.strip() or "") + "\n</critique>")
    parts.append("<fixed_copy>\n" + (fixed or "").strip() + "\n</fixed_copy>")

    return "\n".join(parts)


def build_tagged_trace(
    think: Optional[str],
    critique: Optional[str],
    fixed_copy: Optional[str],
) -> str:
    """
    Assemble a <think>/<critique>/<fixed_copy> string from raw sections.
    Empty sections become empty tags, but all three tags are always present.
    """
    def _norm(s: Optional[str]) -> str:
        return (s or "").strip()

    t = _norm(think)
    c = _norm(critique)
    f = _norm(fixed_copy)

    parts: List[str] = []
    parts.append("<think>\n" + t + "\n</think>")
    parts.append("<critique>\n" + c + "\n</critique>")
    parts.append("<fixed_copy>\n" + f + "\n</fixed_copy>")
    return "\n".join(parts)


# Soft length bounds to prefer shorter, task-oriented traces.
MAX_THINK_WORDS = 90
MAX_CRITIQUE_WORDS = 160
MAX_FIXED_COPY_WORDS = 230
MIN_FIXED_COPY_WORDS = 70


def is_procedural_and_concise(text: str) -> bool:
    """
    Enforce:
      - all three tags present in correct order
      - each section within reasonable length bounds
    """
    if not isinstance(text, str):
        return False

    if not _has_well_formed_tags_in_order(text):
        return False

    sections = split_sections(text)
    think = sections.get("think")
    critique = sections.get("critique")
    fixed = sections.get("fixed_copy")

    if not think or not critique or not fixed:
        return False

    think_words = _word_count(think)
    crit_words = _word_count(critique)
    fixed_words = _word_count(fixed)

    if think_words == 0 or think_words > MAX_THINK_WORDS:
        return False
    if crit_words == 0 or crit_words > MAX_CRITIQUE_WORDS:
        return False
    if fixed_words < MIN_FIXED_COPY_WORDS or fixed_words > MAX_FIXED_COPY_WORDS:
        return False

    return True


def is_dpo_candidate(record: Dict[str, Any]) -> bool:
    """
    Apply the DPO selection criteria:

      1) trace_correction.keep_as_gold == True
      2) think_score >= 5, critique_score >= 5, fixed_copy_score >= 5
      3) chosen has well-formed tags in order:
         <think>...</think><critique>...</critique><fixed_copy>...</fixed_copy>
      4) chosen is reasonably concise and procedural (length-bounded sections)
    """
    tc = record.get("trace_correction") or {}
    if not tc.get("keep_as_gold"):
        return False

    if (
        tc.get("think_score", 0) < 5
        or tc.get("critique_score", 0) < 5
        or tc.get("fixed_copy_score", 0) < 5
    ):
        return False

    chosen = record.get("chosen")
    if not isinstance(chosen, str):
        return False

    if not is_procedural_and_concise(chosen):
        return False

    return True

def ensure_broken_field(rec: Dict[str, Any]) -> None:
    """Ensure rec['broken'] exists by reconstructing it from 'prompt' when needed.

    This keeps new DPO records compatible with the older schema expected by
    fine-tuning scripts, without requiring any new LLM calls.
    """
    # If 'broken' is already present and non-empty, leave it alone.
    if "broken" in rec and isinstance(rec["broken"], str) and rec["broken"].strip():
        return

    prompt = rec.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return

    marker = "Review the following document for rule violations:"
    # Expected: "<marker>\n\n<bad ad copy>"
    if prompt.startswith(marker):
        parts = prompt.split("\n\n", 1)
        if len(parts) == 2:
            rec["broken"] = parts[1]
            return
        lines_ = prompt.splitlines()
        if lines_ and lines_[0].startswith(marker):
            rec["broken"] = "\n".join(lines_[1:]).lstrip()
            return

    # Fallback: if the format is different, store the full prompt
    rec["broken"] = prompt


def _apply_noise_to_record(
    record: Dict[str, Any],
    noiser: TextNoiser,
    intensity: float,
    targets: Sequence[str],
    seed: Optional[int] = None,
) -> None:
    """
    Apply lexical noise to the configured fields and track provenance metadata.
    """
    mutated_fields: List[str] = []
    for field in targets:
        text = record.get(field)
        if not isinstance(text, str) or not text.strip():
            continue

        if field == "prompt":
            prefix, sep, payload = text.partition("\n\n")
            if sep:
                mutated_payload = noiser.add_noise(payload, intensity)
                mutated = f"{prefix}{sep}{mutated_payload}"
            else:
                mutated = noiser.add_noise(text, intensity)
        else:
            mutated = noiser.add_noise(text, intensity)

        if mutated != text:
            record[field] = mutated
            mutated_fields.append(field)

    if not mutated_fields:
        return

    meta = record.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        record["meta"] = meta

    noise_info = {
        "level": intensity,
        "targets": mutated_fields,
    }
    if seed is not None:
        noise_info["seed"] = seed
    meta["noise"] = noise_info


# ---------- Main correction pipeline ----------

def _record_identity(rec: Dict[str, Any]) -> tuple[str, str]:
    """
    Stable identity for an example across passes.

    We key off fields that are the same in the original input and
    in the DPO rows created in the first pass. 'prompt' + 'broken'
    are usually stable and specific enough for this dataset.

    If you ever change the input schema, update this function.
    """
    prompt = rec.get("prompt") or ""
    broken = rec.get("broken") or ""
    return (prompt, broken)

def correct_records(args: argparse.Namespace) -> None:
    # Detect pass mode based on whether DPO file already exists.
    is_second_pass = args.dpo_output.exists()
    mode = "SECOND PASS (compressed traces, remaining records only)" if is_second_pass else "FIRST PASS"
    print(f"[INFO] Trace correction mode: {mode}")

    input_records = read_jsonl(args.input)
    if args.limit:
        input_records = input_records[: args.limit]
    if not input_records:
        print("No records to process.")
        return

    # Backfill missing 'broken' field from 'prompt' so downstream tools
    # (including fine-tuning scripts) can rely on a stable schema.
    for rec in input_records:
        ensure_broken_field(rec)

    # If this is the second pass, exclude records that already produced
    # good DPO entries in the existing args.dpo_output file.
    if is_second_pass:
        prev_rows = read_jsonl(args.dpo_output)

        # Make sure old rows also have 'broken' populated
        for rec in prev_rows:
            ensure_broken_field(rec)

        # Build a set of content-identities for "already good" rows
        already_good_ids = {_record_identity(row) for row in prev_rows}
        print(
            f"[INFO] Second pass: loaded {len(prev_rows)} first-pass DPO records "
            f"from {args.dpo_output}"
        )

        filtered: List[Dict[str, Any]] = []
        for rec in input_records:
            rid = _record_identity(rec)
            if rid in already_good_ids:
                # This record already yielded a DPO row in pass 1 -> skip
                continue
            filtered.append(rec)

        dropped = len(input_records) - len(filtered)
        print(
            f"[INFO] Second pass: excluding {dropped} records that already passed "
            f"first DPO filter; {len(filtered)} records remain."
        )
        input_records = filtered

        if not input_records:
            print("[INFO] Second pass: nothing left to process after exclusion.")
            return

    noiser: Optional[TextNoiser] = None
    if args.noise_level > 0:
        noiser = TextNoiser(seed=args.noise_seed)

    # Load original fixed_copy overrides from input_thinking.jsonl, if present.
    original_fixed_overrides = load_original_fixed_map_from_input_thinking(INPUT_THINKING_PATH)

    prompts, payload_meta = build_prompts(
        input_records,
        original_fixed_overrides,
        is_second_pass=is_second_pass,
    )
    # INSERT_YOUR_CODE
    # Pipe prompts to a file for inspection/debugging
    with open("generated_prompts.txt", "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt)
            f.write("\n" + "="*80 + "\n")

    # Pick cache based on pass
    cache_path = (
        THINKING_CORRECTION_CACHE_PATH_SECOND if is_second_pass else THINKING_CORRECTION_CACHE_PATH
    )

    responses = load_cached_responses(cache_path, len(payload_meta))
    if responses is None:
        maybe_wait_before_batch(args.batch_wait_seconds, len(payload_meta))
        processor = GeminiProcessor(
            key_file_path=args.key_file,
            project_id=args.project_id,
            location=args.location,
            model_name=args.model_name,
            staging_bucket=args.staging_bucket,
            system_instruction=BASE_SYSTEM_PROMPT,
        )

        responses = processor.run_batch_job(prompts, poll_interval=args.poll_interval)
        cache_responses(cache_path, responses)

    if len(responses) != len(payload_meta):
        raise RuntimeError("Mismatch between responses and prompts.")

    dpo_rows: List[Dict[str, Any]] = []

    for meta, raw_text in zip(payload_meta, responses):
        record = meta["record"]
        original_fixed = meta["original_fixed"]

        parsed = parse_candidate(raw_text)
        if not parsed:
            print(f"[WARN] Failed to parse response for record {meta['index']}; skipping.")
            append_debug_issue(
                {
                    "index": meta["index"],
                    "meta_key": meta.get("meta_key"),
                    "issue": "parse_failed",
                    "raw_text_snippet": _safe_snippet(raw_text),
                }
            )
            continue

        revised = parsed.get("revised_chosen", "") or ""
        if not isinstance(revised, str) or not revised.strip():
            print(f"[WARN] Empty or non-string revised_chosen for record {meta['index']}; skipping.")
            continue

        model_fixed = extract_fixed_copy(revised)

        if not original_fixed or not isinstance(original_fixed, str) or not original_fixed.strip():
            print(
                f"[WARN] No original <fixed_copy> for record {meta['index']}; "
                f"skipping record."
            )
            append_debug_issue(
                {
                    "index": meta["index"],
                    "meta_key": meta.get("meta_key"),
                    "issue": "missing_original_fixed_copy",
                    "chosen_type": type(record.get("chosen")).__name__,
                    "chosen_snippet": _safe_snippet(record.get("chosen")),
                }
            )
            continue

        # Strip LLM commentary and keep only substantive ad copy.
        original_fixed_clean = strip_llm_commentary_lines(original_fixed)

        # Force the (cleaned) original <fixed_copy> back into the revised_chosen string.
        preserved = force_original_fixed_copy(revised, original_fixed_clean)
        parsed["revised_chosen"] = preserved

        new_record = dict(record)
        new_record["chosen"] = preserved

        # --- synthesize a structured 'rejected' trace with varied weakness patterns ---

        new_record["original_rejected"] = record.get("rejected")

        rejected_think = parsed.get("rejected_think")
        rejected_critique = parsed.get("rejected_critique")
        rejected_fixed_copy = parsed.get("rejected_fixed_copy")

        rejected_text = ""

        # 1) Prefer Gemini’s weaker trace if provided.
        if (
            isinstance(rejected_think, str)
            or isinstance(rejected_critique, str)
            or isinstance(rejected_fixed_copy, str)
        ):
            mode_choice = random.choice(["same_fixed", "use_rejected_fixed"])
            if mode_choice == "same_fixed":
                rejected_text = build_tagged_trace(
                    rejected_think,
                    rejected_critique,
                    original_fixed_clean,
                )
            else:
                fixed_for_rej = (
                    rejected_fixed_copy
                    if isinstance(rejected_fixed_copy, str) and rejected_fixed_copy.strip()
                    else original_fixed_clean
                )
                rejected_text = build_tagged_trace(
                    rejected_think,
                    rejected_critique,
                    fixed_for_rej,
                )
        else:
            # 2) Fall back to Python-side weakening from the chosen trace.
            rejected_text = synthesize_rejected_from_chosen(preserved)

        if not rejected_text:
            rejected_text = record.get("rejected") or ""

        new_record["rejected"] = rejected_text
        # --- END NEW ---

        if noiser:
            _apply_noise_to_record(
                new_record,
                noiser,
                args.noise_level,
                args.noise_targets,
                seed=args.noise_seed,
            )

        approach_used = parsed.get("approach") or meta.get("approach") or PROCEDURAL_APPROACH

        new_record["trace_correction"] = {
            "approach": approach_used,
            "think_score": parsed.get("think_score"),
            "critique_score": parsed.get("critique_score"),
            "fixed_copy_score": parsed.get("fixed_copy_score"),
            "keep_as_gold": parsed.get("keep_as_gold"),
            "fix_difficulty": parsed.get("fix_difficulty"),
            "improvement_note": parsed.get("improvement_note"),
            "model_name": args.model_name,
            "model_altered_fixed_copy": bool(
                model_fixed
                and original_fixed_clean
                and isinstance(model_fixed, str)
                and model_fixed.strip() != original_fixed_clean.strip()
            ),
        }

        if is_dpo_candidate(new_record):
            dpo_record = dict(new_record)
            dpo_record["dataset"] = "dpo"
            dpo_rows.append(dpo_record)

    if dpo_rows:
        # First pass: overwrite; second pass: append to existing file.
        append_mode = args.dpo_output.exists()
        write_jsonl(args.dpo_output, dpo_rows, append=append_mode)

        if append_mode:
            print(
                f"Wrote {len(dpo_rows)} additional DPO-ready records (second pass) "
                f"to {args.dpo_output}"
            )
        else:
            print(
                f"Wrote {len(dpo_rows)} combined DPO-ready records (first pass) "
                f"to {args.dpo_output}"
            )

        rows_by_approach: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in dpo_rows:
            tc = row.get("trace_correction") or {}
            approach = tc.get("approach") or PROCEDURAL_APPROACH
            rows_by_approach[approach].append(row)

        for approach, rows in rows_by_approach.items():
            suffix = approach.replace(" ", "_")
            per_approach_path = args.dpo_output.with_name(
                f"{args.dpo_output.stem}.{suffix}{args.dpo_output.suffix}"
            )
            write_jsonl(per_approach_path, rows, append=False)
            print(f"  - {len(rows)} records for approach '{approach}' -> {per_approach_path}")
    else:
        print("[INFO] No records met DPO filter criteria – DPO file not written.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch rewrite breaker/fixer traces using Gemini Flash procedural approaches with DPO-ready pairs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/synthetic/breaker_fixer.jsonl"),
        help="JSONL file containing breaker/fixer records.",
    )
    parser.add_argument(
        "--dpo-output",
        type=Path,
        default=Path("output/experiments/output_procedural_dpo.jsonl"),
        help="Destination JSONL for DPO-ready subset (gold + scores>=5 + well-formed tags).",
    )
    parser.add_argument(
        "--key-file",
        type=str,
        default="key.json",
        help="Service account JSON for Vertex Gemini access.",
    )
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
        help="Vertex AI region.",
    )
    parser.add_argument(
        "--staging-bucket",
        type=str,
        default=config.VERTEX_STAGING_BUCKET,
        help="GCS bucket used for Gemini batch jobs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=config.MODEL_FAST,
        help="Gemini model id (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between batch status polls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records.",
    )
    parser.add_argument(
        "--batch-wait-seconds",
        type=int,
        default=DEFAULT_BATCH_WAIT_SECONDS,
        help="Countdown before submitting a new batch job (set 0 to skip).",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="Probability (0-1) that each line of ad copy receives a tiny lexical perturbation.",
    )
    parser.add_argument(
        "--noise-targets",
        nargs="+",
        default=["prompt"],
        choices=["prompt", "broken", "chosen", "rejected"],
        help="Fields to apply lexical noise to when --noise-level > 0.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Random seed for noise generation (kept for determinism).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    correct_records(cli_args)
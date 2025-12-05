from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from gemma_synth_pipeline import config
from .data_loader import AdCopySample
from .llm_service import LmStudioClient
from .rule_parser import RuleParseResult, RuleParser, RuleSection


@dataclass
class RuleDef:
    id: str
    title: str
    text: str


def build_rule_defs(parsed: RuleParseResult) -> List[RuleDef]:
    """Flatten parsed sections into numbered rules."""
    rules: List[RuleDef] = []
    sec_counter = 1
    for section in parsed.sections:
        if section.title.lower() == "advertising review guide":
            continue  # examples only, not a primary rule bucket
        bullet_counter = 1
        for bullet in section.bullets:
            rid = f"{sec_counter}.{bullet_counter}"
            text = f"{section.title}: {bullet}"
            rules.append(RuleDef(id=rid, title=section.title, text=text))
            bullet_counter += 1
        sec_counter += 1
    return rules


class BreakerAgent:
    """Rewrite copy to violate a specific rule."""

    def __init__(self, llm: Optional[LmStudioClient] = None):
        self.llm = llm or LmStudioClient()

    def generate(self, seed_text: str, rules: Sequence[RuleDef], loudness: str = "subtle") -> str:
        loudness_note = (
            "Introduce a subtle violation; keep 95% identical."
            if loudness == "subtle"
            else "Introduce an obvious violation; keep ~80% identical."
        )
        rules_text = "\n".join([f"- {r.id} ({r.title}): {r.text}" for r in rules])

        system = (
            "You are a Red-Team Data Generator for a compliance software test suite. "
            "You are NOT providing financial advice. You are generating strictly hypothetical "
            "examples to test a text classifier. It is critical that you follow the instructions "
            "to introduce specific errors for the test case."
        )

        user = (
            f"Rules to violate:\n{rules_text}\n"
            f"Loudness: {loudness_note}\n"
            "Requirements:\n"
            "1. Insert at least one explicit change that clearly violates ONLY the rules above.\n"
            "2. Do not add new sections or change tone.\n"
            "3. Do NOT violate rules outside the list above.\n\n"
            f"Original copy:\n{seed_text}\n\n"
            "Output: the rewritten copy only (no explanations)."
        )

        return self.llm.generate(system, user)

    def generate_near_miss(self, seed_text: str, rules: Sequence[RuleDef]) -> str:
        rules_text = "\n".join([f"- {r.id} ({r.title}): {r.text}" for r in rules])
        system = (
            "You rewrite marketing copy to appear borderline but remain compliant. "
            "Stay close to the rules and avoid actual violations."
        )
        user = (
            f"Rules to skirt but not violate:\n{rules_text}\n"
            "Constraints: Keep tone/structure; hint at the risky area but stay compliant.\n\n"
            f"Original copy:\n{seed_text}\n\n"
            "Output: the rewritten copy only."
        )
        return self.llm.generate(system, user)


class FixerAgent:
    """Normalize copy into a clean, compliant baseline before breaking it."""

    def __init__(self, llm: Optional[LmStudioClient] = None):
        self.llm = llm or LmStudioClient()

    def generate(self, seed_text: str) -> str:
        system = "You are a compliance editor who rewrites text to remove all potential violations."
        user = (
            "Rewrite the following text to be fully compliant and neutral while preserving intent and tone.\n\n"
            f"{seed_text}\n\n"
            "Output only the rewritten text."
        )
        return self.llm.generate(system, user)


class ValidatorAgent:
    """Ask the model to confirm whether the intended rules are clearly violated (or not)."""

    def __init__(self, llm: Optional[LmStudioClient] = None):
        self.llm = llm or LmStudioClient()

    def validate(self, text: str, rules: Sequence[RuleDef], expect_violation: bool = True) -> bool:
        rule_list = ", ".join([f"{r.id}: {r.title}" for r in rules])

        system = "You are a QA Compliance Auditor. Think carefully before answering."

        user = (
            f"Document:\n{text}\n\n"
            f"Target Rules: {rule_list}\n\n"
            "Task: Determine if the document violates ANY of the Target Rules.\n"
            "Respond in this format:\n"
            "<analysis> [Check the text against each rule step-by-step] </analysis>\n"
            "Verdict: [YES if it violates at least one target rule, NO if it does not]"
        )

        resp = self.llm.generate(system, user)
        if not resp:
            return False

        upper = resp.upper()
        # We only care about the final Verdict line
        is_violation = "VERDICT: YES" in upper

        if expect_violation:
            # We expect this doc to violate the rule(s)
            return is_violation
        else:
            # We expect this doc to be compliant w.r.t the rule(s)
            return not is_violation


class TeacherAgent:
    """Generate reasoning trace and critique for a broken doc."""

    def __init__(
        self,
        llm: Optional[LmStudioClient] = None,
        review_examples: Optional[str] = None,
        profile: str = "minimal",
    ):
        self.llm = llm or LmStudioClient()
        self.review_examples = review_examples
        self.profile = profile

    def _build_minimal_prompt(self, broken_doc: str, rules: Sequence[RuleDef]) -> tuple[str, str]:
        rules_text = "\n".join([f"- {r.id} ({r.title}): {r.text}" for r in rules])
        examples = f"\nReference examples:\n{self.review_examples}\n" if self.review_examples else ""

        system = (
            "You are an expert financial compliance auditor. "
            "You strictly follow the provided rules and explain violations clearly."
        )

        user = (
            f"Rules to evaluate:\n{rules_text}\n"
            f"{examples}\n"
            "Document to review:\n"
            f"{broken_doc}\n\n"
            "Respond with EXACTLY this structure:\n"
            "<think>\n"
            "1) Quote the problematic text.\n"
            "2) Link it to the specific rule(s) above.\n"
            "3) Explain why it violates the rule(s).\n"
            "</thought>\n"
            "Critique: [one concise paragraph explaining the issue to a marketing writer]\n"
            "<fixed_copy>\n"
            "[The corrected, fully compliant version of the document]\n"
            "</fixed_copy>"
        )

        return system, user

    def _build_contrastive_prompt(self, broken_doc: str, rules: Sequence[RuleDef]) -> tuple[str, str]:
        """Include simple in-context examples contrasting weak vs. strong critiques."""
        rules_text = "\n".join([f"- {r.id} ({r.title}): {r.text}" for r in rules])
        tutorial = (
            "Example 1 - Promising returns (good response):\n"
            "<think>\n"
            "Quoted text: \"We guarantee 20% annual performance.\"\n"
            "Rule link: - Advertising Claims - avoid guarantees.\n"
            "Explanation: Promotional copy cannot imply certain gains.\n"
            "</thought>\n"
            "Critique: The copy guarantees a 20% return, which violates rules against promising performance.\n"
            "<fixed_copy>Historical returns vary. We aim to balance growth and risk and cannot promise future performance.</fixed_copy>\n"
            "\n"
            "Example 2 - Missing disclosure (good response):\n"
            "<think>\n"
            "Quoted text: \"Call now to lock in this exclusive strategy.\"\n"
            "Rule link: - Disclosures - highlight risks and fees.\n"
            "Explanation: Marketing omits costs and risks.\n"
            "</thought>\n"
            "Critique: The copy pushes urgency without the required disclosure of fees and investment risks.\n"
            "<fixed_copy>Contact us to discuss whether this diversified strategy fits your goals, including its risks, fees, and liquidity considerations.</fixed_copy>\n"
        )

        examples = f"\nReference examples:\n{self.review_examples}\n" if self.review_examples else ""
        system = (
            "You are an expert compliance reviewer who studies contrastive examples before auditing a document. "
            "Reuse the demonstrated structure faithfully."
        )
        user = (
            f"Rules to evaluate:\n{rules_text}\n"
            f"{examples}\n"
            f"Reference walkthroughs:\n{tutorial}\n"
            "Document to review:\n"
            f"{broken_doc}\n\n"
            "Respond with EXACTLY this structure:\n"
            "<think>\n"
            "1) Quote the problematic text.\n"
            "2) Link it to the specific rule(s) above.\n"
            "3) Explain why it violates the rule(s).\n"
            "</thought>\n"
            "Critique: [one concise paragraph explaining the issue to a marketing writer]\n"
            "<fixed_copy>\n"
            "[The corrected, fully compliant version of the document]\n"
            "</fixed_copy>"
        )
        return system, user

    def generate(self, broken_doc: str, rules: Sequence[RuleDef]) -> str:
        if self.profile == "minimal":
            system, user = self._build_minimal_prompt(broken_doc, rules)
        elif self.profile == "contrastive_icl":
            system, user = self._build_contrastive_prompt(broken_doc, rules)
        else:
            raise ValueError(f"Unknown Teacher profile: {self.profile}")

        return self.llm.generate(system, user)

    def generate_fix_for_seed(self, original_text: str, rules: Sequence[RuleDef]) -> str:
        """Use Teacher-style reasoning to fix an original (possibly non-compliant) seed."""
        rules_text = "\n".join([f"- {r.id} ({r.title}): {r.text}" for r in rules])
        system = (
            "You are an expert financial compliance editor. "
            "You rewrite documents to be fully compliant with all given rules, "
            "preserving the original intent as much as possible."
        )
        user = (
            f"Rules to enforce:\n{rules_text}\n\n"
            f"Original document:\n{original_text}\n\n"
            "Task: Rewrite the document to be fully compliant with all the rules listed above. "
            "Output only the corrected, fully compliant version of the document."
        )
        return self.llm.generate(system, user)

class HallucinatorAgent:
    """Generate incorrect or misattributed critique for DPO loser.

    Design:
      - Content is deliberately flawed (misses or misapplies rules).
      - Output format is always:

          <think>
          ...
          </thought>
          Critique: ...

      - We also wrap generation in a small retry loop to enforce structure.
    """

    def __init__(self, llm: Optional[LmStudioClient] = None, profile: str = "smart_wrong"):
        # Use a non-deterministic seed to avoid identical rejected traces.
        self.llm = llm or LmStudioClient(seed=None)
        # Profile controls *content* style, not format.
        #   - "naive": vague, surface-level, no rule IDs.
        #   - "smart_wrong": confident, blames the wrong rule ID.
        self.profile = profile

    def generate(
        self,
        broken_doc: str,
        target_rules: Sequence[RuleDef],
        all_rules: Sequence[RuleDef],
    ) -> str:
        """Generate a DPO 'loser' response with consistent structure."""
        def _once() -> str:
            if self.profile == "naive":
                system, user = self._build_naive_prompt(broken_doc, target_rules, all_rules)
            elif self.profile == "smart_wrong":
                system, user = self._build_smart_wrong_prompt(broken_doc, target_rules, all_rules)
            else:
                raise ValueError(f"Unknown Hallucinator profile: {self.profile}")
            return self.llm.generate(system, user)

        # Use the shared retry helper + structural checker.
        return generate_with_retry(
            _once,
            max_retries=3,
            format_checker=has_reasoning_format,
        ) or ""

    # ------------------------------------------------------------------
    # NAIVE PROFILE: vague, shallow, no numeric rule IDs
    # ------------------------------------------------------------------

    def _build_naive_prompt(
        self,
        broken_doc: str,
        target_rules: Sequence[RuleDef],
        all_rules: Sequence[RuleDef],
    ) -> tuple[str, str]:
        target_ids = {r.id for r in target_rules}

        # Pick some other rule family to vaguely allude to (no numeric ID used).
        alt_rules = [r for r in all_rules if r.id not in target_ids]
        wrong_rule = random.choice(alt_rules) if alt_rules else target_rules[0]

        system = (
            "You are a junior, distracted compliance reviewer. "
            "You often miss the main issue or focus on something minor. "
            "You are generating training data for another model. "
            "Your analysis should sound unsure, superficial, or slightly off-topic."
        )

        user = (
            f"Document:\n{broken_doc}\n\n"
            "Sensitive rule IDs (for context ONLY, NEVER write these codes in your answer):\n"
            f"  - {', '.join(sorted(target_ids))}\n\n"
            "HARD CONSTRAINTS:\n"
            "1. Do NOT write any rule codes that look like '1.1', '2.3', etc.\n"
            "2. Do NOT mention any specific rule ID numbers at all.\n"
            "3. You may only refer vaguely to 'our guidelines' or to a rule by its topic.\n"
            "4. Do NOT say that the document clearly has a serious violation.\n\n"
            "Content Task:\n"
            f"- Either say the document looks mostly fine and nitpick superficial wording or tone, OR\n"
            f"- Vaguely suggest it MIGHT relate to our general guidelines about: {wrong_rule.title} "
            "(but without giving any rule number).\n"
            "- Do NOT propose a corrected or rewritten version of the copy.\n"
            "- Do NOT say that the document clearly violates the main rule that was actually broken.\n\n"
            "OUTPUT FORMAT (VERY IMPORTANT):\n"
            "You MUST respond in EXACTLY this structure, and nothing else:\n"
            "<think>\n"
            "[2–4 short sentences of shaky or superficial reasoning. "
            "Do NOT include any numeric rule IDs.]\n"
            "</thought>\n"
            "Critique: [One short paragraph (2–4 sentences) with vague or trivial feedback. "
            "Do NOT include any numeric rule IDs.]\n\n"
            "VALID EXAMPLE (for some other document):\n"
            "<think>\n"
            "At a quick read, this copy seems mostly fine. The language is a bit strong in a few places, "
            "but it's typical marketing language. I don't immediately see any major problems.\n"
            "</thought>\n"
            "Critique: Overall, the document probably meets our general guidelines, though you could consider "
            "softening one or two phrases to sound less promotional. Adding a very general reminder about risks "
            "might help, but it may not be strictly necessary.\n\n"
            "Now follow the SAME STRUCTURE for the document above.\n"
        )

        return system, user

    # ------------------------------------------------------------------
    # SMART_WRONG PROFILE: confident, misapplies a single wrong rule
    # ------------------------------------------------------------------

    def _build_smart_wrong_prompt(
        self,
        broken_doc: str,
        target_rules: Sequence[RuleDef],
        all_rules: Sequence[RuleDef],
    ) -> tuple[str, str]:
        target_ids = {r.id for r in target_rules}

        # Choose a different rule to blame (this one must be explicitly named).
        alt_rules = [r for r in all_rules if r.id not in target_ids]
        wrong_rule = random.choice(alt_rules) if alt_rules else target_rules[0]

        system = (
            "You are a confident compliance reviewer who sounds very sure of yourself, "
            "but you routinely misapply the rules. Your critique should sound polished and assertive, "
            "yet be conceptually wrong about which rule applies. "
            "You are generating training data for another model."
        )

        user = (
            f"Document:\n{broken_doc}\n\n"
            "Sensitive rule IDs (for context ONLY, NEVER mention these codes in your answer):\n"
            f"  - {', '.join(sorted(target_ids))}\n\n"
            "Content Task:\n"
            f"1. Pick ONE sentence or short phrase from the document that seems important.\n"
            f"2. Incorrectly claim that it violates rule {wrong_rule.id} ({wrong_rule.title}) ONLY.\n"
            "3. You MUST NOT reference any other numbered rule code besides that one.\n"
            "4. Provide a detailed explanation that sounds plausible and confident, but is logically wrong or off-topic.\n"
            "5. NEVER mention or hint at any of the sensitive rule IDs above.\n"
            "6. Do NOT provide a corrected or rewritten version of the copy.\n\n"
            "OUTPUT FORMAT (VERY IMPORTANT):\n"
            "You MUST respond in EXACTLY this structure, and nothing else:\n"
            "<think>\n"
            "1) Quote a sentence or short phrase from the document.\n"
            f"2) State that it is a problem under rule {wrong_rule.id} ({wrong_rule.title}) ONLY.\n"
            "3) Give a confident but incorrect explanation of why it supposedly violates that rule.\n"
            "</thought>\n"
            "Critique: [One concise paragraph (2–4 sentences) summarizing your wrong assessment. "
            f"Do NOT mention any other rule numbers besides {wrong_rule.id}.]\n\n"
            "VALID EXAMPLE (for some other document):\n"
            "<think>\n"
            "1) The phrase \"Our strategy ensures consistent success in any market condition\" stands out.\n"
            "2) This appears to violate rule 4.4 (Performance Information Generally) because it discusses "
            "ongoing success.\n"
            "3) Under rule 4.4, any broad statement about doing well over time can be seen as problematic, "
            "even if it does not mention performance numbers directly.\n"
            "</thought>\n"
            "Critique: This sentence raises concerns under rule 4.4, since it frames the strategy as broadly "
            "successful regardless of conditions. The issue is that the language sounds too confident about "
            "outcomes, which falls squarely under that rule. To align with 4.4, the copy should avoid presenting "
            "the approach as inherently effective in all environments.\n\n"
            "Now follow the SAME STRUCTURE, but apply it to the document above using rule "
            f"{wrong_rule.id} ({wrong_rule.title}) as the supposed basis for your concerns.\n"
        )

        return system, user

def sample_rules(rules: List[RuleDef], limit: Optional[int] = None) -> List[RuleDef]:
    if limit:
        return rules[:limit]
    return rules


def has_reasoning_format(text: str) -> bool:
    """Basic structural guardrail for reasoning outputs."""
    if not text:
        return False
    lowered = text.lower()
    return "<think>" in lowered and "</thought>" in lowered and "critique" in lowered


def generate_with_retry(
    agent_func,
    *,
    max_retries: int = 3,
    format_checker=None,
    semantic_validator=None,
    **kwargs,
):
    """
    Generic wrapper to retry generation if:
      - the format is wrong (e.g., missing <think> tags), or
      - semantic validation fails.

    Parameters
    ----------
    agent_func : callable
        Function to call, e.g. teacher_agent.generate or hallucinator_agent.generate.
    max_retries : int
        Maximum number of attempts before giving up.
    format_checker : callable | None
        Function that takes the raw text and returns True/False (e.g., has_reasoning_format).
    semantic_validator : callable | None
        Function that takes the raw text and returns True/False (e.g., custom rule checks).
    kwargs : dict
        Passed directly to agent_func.

    Returns
    -------
    str | None
        The generated string if successful, or None after exhausting retries.
    """
    for _ in range(max_retries):
        result = agent_func(**kwargs)
        if not result:
            continue

        # 1) Structural format check (tags, etc.)
        if format_checker is not None and not format_checker(result):
            continue

        # 2) Optional semantic validation (extra custom check)
        if semantic_validator is not None and not semantic_validator(result):
            continue

        return result

    return None  # Failed after retries

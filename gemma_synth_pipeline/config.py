from __future__ import annotations

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
RULES_DOCX = ROOT_DIR / "data" / "rules" / "rules.docx"
# Prefer the pre-extracted rulebook text when present to avoid re-parsing docx.
RULES_TEXT_OVERRIDE: Path | None = ROOT_DIR / "output" / "rules" / "rules__docling.txt"
TEXT_OUTPUT_ROOT = ROOT_DIR / "output" / "text" / "data" / "ad_copy_reviews"
TEXT_OUTPUT_SANITIZED = ROOT_DIR / "output" / "sanitized_text"
SYNTH_OUTPUT = ROOT_DIR / "output" / "synthetic" / "gemma3_adcopy.jsonl"
BREAKER_FIXER_OUTPUT = ROOT_DIR / "output" / "synthetic" / "breaker_fixer.jsonl"
EXPERIMENT_CONFIG_PATH = ROOT_DIR / "experiments.yaml"
EXPERIMENT_OUTPUT_DIR = ROOT_DIR / "output" / "experiments"


# Data selection
SKIP_PREFIXES = {"_1", "_9", "_12", "_14", "_15", "_20"}
PREFERRED_METHODS = ("docling", "unstructured")

# Rulebook handling
RULEBOOK_MAX_CHARS = 4000

# LLM / LMStudio config
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
MODEL_NAME = "gemma-3-12b-it"
TEMPERATURE = 0.35
MAX_TOKENS = 4096
SEED = 1234

# Vertex AI / Gemini config
# Model IDs – used by batch + online
MODEL_FAST = "gemini-2.5-flash"      # Breaker / Validator / Hallucinator
MODEL_SMART = "gemini-3-pro-preview"          # Teacher / Fixer

# Central Vertex deployment details so services do not hardcode IDs elsewhere.
VERTEX_PROJECT_ID = "cs329h-ria-compliance"
VERTEX_LOCATION = "global"
VERTEX_STAGING_BUCKET = "ria-compliance-bucket"

# Frozen agent defaults
FROZEN_TEACHER_PROFILE = "minimal"
FROZEN_HALLUCINATOR_PROFILE = "smart_wrong"

# Pipeline behavior
DEFAULT_SAMPLE_LIMIT = None

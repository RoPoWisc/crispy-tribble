from __future__ import annotations

import sys
import argparse
from pathlib import Path
import re
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import yaml
from typing import Any, Dict, List

from gemma_synth_pipeline.services.data_loader import DatasetLoader
from gemma_synth_pipeline.services.rule_parser import RuleParser
from gemma_synth_pipeline.services.breaker_fixer import (
    BreakerAgent,
    TeacherAgent,
    HallucinatorAgent,
    FixerAgent,
    RuleDef,
    build_rule_defs,
)
from gemma_synth_pipeline import config
from gemma_synth_pipeline.services.gemini_processor import GeminiProcessor


class Experimenter:
    def __init__(self, config_path: str = config.EXPERIMENT_CONFIG_PATH):
        with open(config_path, "r") as f:
            self.experiment_config = yaml.safe_load(f)

        self.output_dir = Path(self.experiment_config["defaults"].get("output_dir", "output/experiments"))
        self.output_dir.mkdir(exist_ok=True)
        self.clean_predictions_path = Path(
            self.experiment_config["defaults"].get("clean_predictions_path", "")
        )

        print("Loading data and rules...")
        self.loader = DatasetLoader(base_dir=config.TEXT_OUTPUT_SANITIZED)
        self.seeds = self.loader.load()

        rule_parser = RuleParser()
        parsed_rules = rule_parser.parse()
        self.all_rules = build_rule_defs(parsed_rules)
        self.clean_predictions = self._load_clean_predictions()
        print(f"Loaded {len(self.seeds)} seeds and {len(self.all_rules)} rules.")
        if self.clean_predictions_path:
            print(f"Loaded {len(self.clean_predictions)} cleaned synthetic samples from {self.clean_predictions_path}.")
        print("Initialization complete.")

    def run_all(self, selected_ids: set[str] | None = None):
        print("Starting all experiments...")
        for exp_config in self.experiment_config.get("experiments", []):
            exp_id = exp_config.get("id", "unknown_experiment")
            if selected_ids and exp_id not in selected_ids:
                continue
            exp_type = exp_config.get("type", "unknown")
            print(f"--- Running Experiment: {exp_id} (type: {exp_type}) ---")

            if exp_type == "teacher":
                self._run_teacher_experiment(exp_config)
            elif exp_type == "hallucinator":
                self._run_hallucinator_experiment(exp_config)
            elif exp_type == "fixer":
                self._run_fixer_experiment(exp_config)
            elif exp_type in {"compound_triple", "atomic"} or exp_config.get("variant_type") or exp_config.get("loudness"):
                self._run_mini_dataset_experiment(exp_config)
            else:
                print(f"Unknown experiment type: {exp_type}")
        print("All experiments finished.")

    def _get_gemini_processor(self, model_name: str) -> GeminiProcessor:
        return GeminiProcessor(
            key_file_path="key.json",
            project_id=config.VERTEX_PROJECT_ID,
            location=config.VERTEX_LOCATION,
            model_name=model_name,
            staging_bucket=config.VERTEX_STAGING_BUCKET
        )

    def _run_teacher_experiment(self, exp_cfg: Dict[str, Any]):
        dataset_size = exp_cfg.get("dataset_size", self.experiment_config["defaults"].get("dataset_size", 3))
        rules_sample_size = self.experiment_config["defaults"].get("rules_sample_size", 3)

        breaker_model = exp_cfg.get("breaker_model", config.MODEL_FAST)
        teacher_model = exp_cfg.get("teacher_model", config.MODEL_SMART)

        breaker_llm = self._get_gemini_processor(breaker_model)
        teacher_llm = self._get_gemini_processor(teacher_model)

        breaker = BreakerAgent(llm=breaker_llm)
        
        teacher_profile = exp_cfg.get("teacher_profile", "minimal")
        teacher = TeacherAgent(llm=teacher_llm, profile=teacher_profile)

        records = []
        sample_seeds = self.seeds[:dataset_size]
        sample_rules = self.all_rules[:rules_sample_size]

        for seed in sample_seeds:
            for rule in sample_rules:
                print(f"  Processing seed '{seed.sample_id}' with rule '{rule.id}'")
                broken_doc = breaker.generate(seed.original.text, [rule])
                if not broken_doc:
                    print("    Breaker failed to generate a document.")
                    continue

                teacher_output = teacher.generate(broken_doc, [rule])
                if not teacher_output:
                    print("    Teacher failed to generate output.")
                    continue
                
                record = {
                    "exp_id": exp_cfg["id"],
                    "seed_id": seed.sample_id,
                    "rule_id": rule.id,
                    "broken_doc": broken_doc,
                    "teacher_profile": teacher_profile,
                    "teacher_output": teacher_output,
                    "metadata": {
                        "teacher_model": teacher_model,
                    },
                }
                records.append(record)

        self._write_records(exp_cfg["id"], records)

    def _run_hallucinator_experiment(self, exp_cfg: Dict[str, Any]):
        dataset_size = exp_cfg.get("dataset_size", self.experiment_config["defaults"].get("dataset_size", 3))
        rules_sample_size = self.experiment_config["defaults"].get("rules_sample_size", 3)

        breaker_model = exp_cfg.get("breaker_model", config.MODEL_FAST)
        teacher_model = exp_cfg.get("teacher_model", config.MODEL_SMART)

        breaker_llm = self._get_gemini_processor(breaker_model)
        teacher_llm = self._get_gemini_processor(teacher_model)
        hallucinator_llm = self._get_gemini_processor(breaker_model) # Using fast model for hallucinator

        breaker = BreakerAgent(llm=breaker_llm)
        teacher_profile = exp_cfg.get("teacher_profile", "minimal")
        teacher = TeacherAgent(llm=teacher_llm, profile=teacher_profile)

        hallucinator_profile = exp_cfg.get("hallucinator_profile", "naive")
        hallucinator = HallucinatorAgent(llm=hallucinator_llm, profile=hallucinator_profile)

        records = []
        sample_seeds = self.seeds[:dataset_size]
        sample_rules = self.all_rules[:rules_sample_size]

        for seed in sample_seeds:
            for rule in sample_rules:
                print(f"  Processing seed '{seed.sample_id}' with rule '{rule.id}'")
                broken_doc = breaker.generate(seed.original.text, [rule])
                if not broken_doc:
                    continue

                teacher_output = teacher.generate(broken_doc, [rule])
                hallucinator_output = hallucinator.generate(broken_doc, [rule], self.all_rules)

                record = {
                    "exp_id": exp_cfg["id"],
                    "seed_id": seed.sample_id,
                    "rule_id": rule.id,
                    "broken_doc": broken_doc,
                    "teacher_profile": teacher_profile,
                    "teacher_output": teacher_output,
                    "hallucinator_profile": hallucinator_profile,
                    "hallucinator_output": hallucinator_output,
                    "metadata": {
                        "teacher_model": teacher_model,
                        "hallucinator_model": breaker_model,
                    },
                }
                records.append(record)
        
        self._write_records(exp_cfg["id"], records)


    def _run_fixer_experiment(self, exp_cfg: Dict[str, Any]):
        num_seeds = exp_cfg.get("num_seeds", 3)
        sample_seeds = self.seeds[:num_seeds]

        fixer_model = exp_cfg.get("model_fixer", config.MODEL_SMART)
        teacher_model = exp_cfg.get("model_teacher", config.MODEL_SMART)

        fixer_llm = self._get_gemini_processor(fixer_model)
        teacher_llm = self._get_gemini_processor(teacher_model)

        fixer = FixerAgent(llm=fixer_llm)
        teacher_as_fixer_profile = exp_cfg.get("teacher_as_fixer_profile", "minimal")
        teacher = TeacherAgent(llm=teacher_llm, profile=teacher_as_fixer_profile)

        records = []
        for seed in sample_seeds:
            print(f"  Processing seed '{seed.sample_id}'")
            
            # Run the standard FixerAgent
            fixed_copy_fixer = fixer.generate(seed.original.text)
            
            # Run the Teacher-as-Fixer
            fixed_copy_teacher = teacher.generate_fix_for_seed(seed.original.text, self.all_rules)
            
            record = {
                "exp_id": exp_cfg["id"],
                "seed_id": seed.sample_id,
                "original_doc": seed.original.text,
                "fixer_output": fixed_copy_fixer,
                "teacher_as_fixer_output": fixed_copy_teacher,
                "metadata": {
                    "fixer_model": fixer_model,
                    "teacher_as_fixer_model": teacher_model,
                },
            }
            records.append(record)
            
        self._write_records(exp_cfg["id"], records)

    def _write_records(self, exp_id: str, records: List[Dict[str, Any]]):
        output_file = self.output_dir / f"{exp_id}.jsonl"
        with open(output_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(records)} records to {output_file}")

    def _load_clean_predictions(self) -> List[Dict[str, Any]]:
        if not self.clean_predictions_path:
            return []
        if not self.clean_predictions_path.exists():
            print(f"Clean predictions file not found: {self.clean_predictions_path}")
            return []
        results: List[Dict[str, Any]] = []
        with self.clean_predictions_path.open("r", encoding="utf-8") as src:
            for line in src:
                record = json.loads(line)
                prompt_text = self._extract_prompt_text(record.get("prompt"))
                meta = self._parse_prompt_metadata(prompt_text)
                record.update(meta)
                results.append(record)
        return results

    @staticmethod
    def _extract_prompt_text(prompt_obj: Any) -> str:
        try:
            return prompt_obj["contents"][0]["parts"][0]["text"]
        except (TypeError, KeyError, IndexError):
            return ""

    @staticmethod
    def _parse_prompt_metadata(prompt_text: str) -> Dict[str, Any]:
        variant_type = None
        loudness = None
        rule_count = 0
        if prompt_text:
            variant_match = re.search(r"variant_type:\s*([a-z_]+)", prompt_text, flags=re.IGNORECASE)
            loudness_match = re.search(r"loudness:\s*([a-z_]+)", prompt_text, flags=re.IGNORECASE)
            if variant_match:
                variant_type = variant_match.group(1).strip().lower()
            if loudness_match:
                loudness = loudness_match.group(1).strip().lower()
            rules_marker = "Rules (you may only reference"
            if rules_marker in prompt_text:
                try:
                    rules_section = prompt_text.split(rules_marker, 1)[1]
                    rule_count = sum(1 for line in rules_section.splitlines() if line.strip().startswith("Rule"))
                except IndexError:
                    rule_count = 0
        return {"variant_type": variant_type, "loudness": loudness, "rule_count": rule_count}

    def _run_mini_dataset_experiment(self, exp_cfg: Dict[str, Any]) -> None:
        if not self.clean_predictions:
            print("No cleaned synthetic predictions available; skipping mini dataset experiment.")
            return

        dataset_size = exp_cfg.get("dataset_size", self.experiment_config["defaults"].get("dataset_size", 100))
        target_variant = exp_cfg.get("variant_type") or exp_cfg.get("type")
        target_variant = target_variant.lower() if isinstance(target_variant, str) else None
        target_loudness = exp_cfg.get("loudness")
        target_loudness = target_loudness.lower() if isinstance(target_loudness, str) else None

        filtered = self.clean_predictions
        if target_variant:
            filtered = [row for row in filtered if row.get("variant_type") == target_variant]
        if target_loudness:
            filtered = [row for row in filtered if row.get("loudness") == target_loudness]

        if not filtered:
            print(f"No samples matched filters for experiment {exp_cfg.get('id')}; falling back to full dataset.")
            filtered = self.clean_predictions

        selected = filtered[:dataset_size]
        if len(selected) < dataset_size:
            print(
                f"Only {len(selected)} samples available for experiment {exp_cfg.get('id')} "
                f"(requested {dataset_size})."
            )
        records: List[Dict[str, Any]] = []
        for row in selected:
            records.append(
                {
                    "broken": row["broken"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                    "variant_type": row.get("variant_type"),
                    "loudness": row.get("loudness"),
                    "rule_count": row.get("rule_count"),
                    "source_prompt": row.get("prompt"),
                    "source_meta": row.get("meta"),
                }
            )

        self._write_records(exp_cfg["id"], records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configured compliance experiments.")
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Optional experiment IDs to run (default: run all).",
    )
    args = parser.parse_args()
    experimenter = Experimenter()
    selected = set(args.ids) if args.ids else None
    experimenter.run_all(selected_ids=selected)

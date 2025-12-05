from __future__ import annotations

import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from gemma_synth_pipeline import config


@dataclass
class PipelineStage:
    """
    Small value object that encapsulates an individual CLI invocation.
    """

    name: str
    command: Sequence[str]

    def run(self, workdir: Path) -> None:
        printable = shlex.join(str(arg) for arg in self.command)
        logging.info("Running stage '%s': %s", self.name, printable)
        subprocess.run([str(arg) for arg in self.command], cwd=workdir, check=True)


@dataclass
class PreferencePipelineConfig:
    """
    Consumer-facing knobs for the end-to-end extraction → breaker → trace flow.
    """

    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    python_bin: str = "python"
    sanitize_output: Path = config.TEXT_OUTPUT_SANITIZED
    sanitize_limit: int | None = None
    full_pipeline_args: Sequence[str] = field(default_factory=tuple)
    breaker_output: Path = config.BREAKER_FIXER_OUTPUT
    breaker_args: Sequence[str] = field(default_factory=tuple)
    trace_output: Path = Path("output/experiments/output_procedural_dpo.jsonl")
    trace_args: Sequence[str] = field(default_factory=tuple)
    noise_level: float = 0.15
    noise_targets: Sequence[str] = field(default_factory=lambda: ("prompt",))
    noise_seed: int = 0
    skip_full_pipeline: bool = False
    skip_breaker: bool = False
    skip_trace: bool = False


class PreferencePipeline:
    """
    Minimal orchestrator that keeps each stage testable and traceable.
    """

    def __init__(self, config: PreferencePipelineConfig) -> None:
        self.config = config

    def build_stages(self) -> List[PipelineStage]:
        stages: List[PipelineStage] = []
        repo_root = self.config.repo_root

        if not self.config.skip_full_pipeline:
            cmd = [
                self.config.python_bin,
                str(repo_root / "run_full_pipeline.py"),
                "--sanitize",
                "--sanitize-output",
                str(self.config.sanitize_output),
            ]
            if self.config.sanitize_limit is not None:
                cmd += ["--sanitize-limit", str(self.config.sanitize_limit)]
            cmd += list(self.config.full_pipeline_args)
            stages.append(PipelineStage("extract_and_sanitize", cmd))

        if not self.config.skip_breaker:
            cmd = [
                self.config.python_bin,
                str(repo_root / "run_breaker_fixer.py"),
                "--sanitized-root",
                str(self.config.sanitize_output),
                "--output",
                str(self.config.breaker_output),
            ]
            cmd += list(self.config.breaker_args)
            stages.append(PipelineStage("breaker_fixer", cmd))

        if not self.config.skip_trace:
            noise_targets = list(self.config.noise_targets) or ["prompt"]
            cmd = [
                self.config.python_bin,
                str(repo_root / "run_trace_correction.py"),
                "--input",
                str(self.config.breaker_output),
                "--dpo-output",
                str(self.config.trace_output),
                "--noise-level",
                str(self.config.noise_level),
                "--noise-seed",
                str(self.config.noise_seed),
                "--noise-targets",
                *noise_targets,
            ]
            cmd += list(self.config.trace_args)
            stages.append(PipelineStage("trace_correction", cmd))

        return stages

    def run(self) -> None:
        repo_root = self.config.repo_root
        for stage in self.build_stages():
            stage.run(repo_root)

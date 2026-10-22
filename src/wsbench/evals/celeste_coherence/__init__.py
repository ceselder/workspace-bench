"""Coherence: is a lens's prose readable text that conveys a claim (never: is it true)? Every bullet
at the hallucination family's read sites is graded 0-3; headline = share graded 0-1."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from .judge import run
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="celeste_coherence",
        title="Coherence",
        group="celeste-evals",
        bank=Path("evals") / "celeste_coherence" / "items.json",
        judge=JudgeConfig(model="claude-sonnet-5", prompt_version=PROMPT_VERSION),
        metric="incoherent_rate",
        higher_is_better=False,
        run=run,
        calls_per_arm="≈ 0.3k (20 bullets per call)",
        sources="",
    )
)

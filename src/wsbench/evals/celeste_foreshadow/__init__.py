"""Foreshadowing: while Qwen3.6-27B is still reading a user message, can a lens surface what it
will later think about a detail buried in the message, which its own continuations do not say?"""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from .judge import run
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="celeste_foreshadow",
        title="Foreshadowed thoughts",
        group="celeste-evals",
        bank=Path("evals") / "celeste_foreshadow" / "items.json",
        judge=JudgeConfig(model="claude-sonnet-5", prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=run,
        calls_per_arm="≈ 1k per readout sample",
        sources="",
    )
)

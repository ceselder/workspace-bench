"""Hinted CoT (Celeste supplement): does the lens see that a reasoning model's answer was caused
by a hint it never mentions, against items where it would have answered that way anyway."""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from . import judge
from .prompts import PROMPT_VERSION

JUDGE_MODEL = "claude-sonnet-5"  # the family's judge of record (Daria Ivanova's two-stage prompts)

SPEC = register(
    EvalSpec(
        name="celeste_hint_cot",
        title="Hinted CoT (unverbalized hint following)",
        group="celeste-evals",
        bank=Path("evals/celeste_hint_cot/items.json"),
        judge=JudgeConfig(model=JUDGE_MODEL, prompt_version=PROMPT_VERSION, reasoning=None),
        metric="auroc",
        higher_is_better=True,
        run=judge.run,
        calls_per_arm="≈ 30k stage 1 + the stage-1 survivors in stage 2 (4 readouts x 19 sites)",
        sources="CommonsenseQA",
    )
)

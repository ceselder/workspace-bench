"""The Celeste supplement families (encoded answer, lookahead, withheld answer): the shared bank
judge prompt asked once per read cell, an item passing on any cell; banks ship the captured
Qwen3.6-27B token ids and read sites, and every item was leak-checked against real continuations."""

from pathlib import Path

from wsbench.basic.prompts import PROMPT_VERSION
from wsbench.celeste.judge import run_family
from wsbench.judge_config import JudgeConfig
from wsbench.registry import REPO_ROOT, EvalSpec

GROUP = "celeste"
JUDGE_MODEL = "claude-sonnet-5"


def celeste_family(
    name: str, title: str, *, calls_per_arm: str, forbidden: bool = False
) -> EvalSpec:
    """``forbidden``: the item's written output (the encoded answer) is sent as the bank judge's
    forbidden-answer line, so a readout that only names it does not pass."""
    bank = Path("evals") / name / "items.json"
    return EvalSpec(
        name=name,
        title=title,
        group=GROUP,
        bank=bank,
        judge=JudgeConfig(model=JUDGE_MODEL, prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=lambda args: run_family(args, name=name, bank=REPO_ROOT / bank, forbidden=forbidden),
        calls_per_arm=calls_per_arm,
        sources="",
    )

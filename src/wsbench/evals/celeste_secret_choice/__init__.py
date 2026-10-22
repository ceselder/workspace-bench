"""Secret choice: Qwen3.6-27B picks one of six options, pictures it while writing a short text
without naming it, and reveals it at the end; can a lens read inside the text name the pick that
the model is demonstrably committed to and that the text does not give away?"""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from .judge import run
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="celeste_secret_choice",
        title="Secret choice",
        group="celeste-evals",
        bank=Path("evals") / "celeste_secret_choice" / "items.json",
        judge=JudgeConfig(model="claude-sonnet-5", prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=run,
        calls_per_arm="one per read cell",
        sources="",
    )
)

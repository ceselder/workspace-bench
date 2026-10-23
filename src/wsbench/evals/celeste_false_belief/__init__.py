"""False belief (ELK-style): Qwen3.6-27B consistently gets an arithmetic problem wrong; read at the
end of the prompt, before a forced preamble, does the lens name the model's own wrong answer (what
it believes) rather than the true one (what a reader of the transcript would compute)?"""

from pathlib import Path

from wsbench.judge_config import JudgeConfig
from wsbench.registry import EvalSpec, register

from .judge import SCORER, run
from .prompts import PROMPT_VERSION

SPEC = register(
    EvalSpec(
        name="celeste_false_belief",
        title="False belief (the model's own wrong answer)",
        group="celeste-evals",
        bank=Path("evals") / "celeste_false_belief" / "items.json",
        judge=JudgeConfig(prompt_version=PROMPT_VERSION),
        metric="pass_rate",
        higher_is_better=True,
        run=run,
        calls_per_arm="0 (regex)",
        sources="",
        scorer=SCORER,
    )
)

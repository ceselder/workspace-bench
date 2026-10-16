"""The prompts this family sends: the bank judge shared by the single-token basic families, plus
its forbidden-answer line (the model's written encoded output)."""

from wsbench.basic.prompts import FORBIDDEN_TEMPLATE, PROMPT_VERSION, SCHEMA, SYSTEM, render_user
from wsbench.basic.prompts import PROMPTS as SHARED
from wsbench.celeste.judge import FAMILY_RULE_CELESTE

PROMPTS: dict[str, str] = {
    **SHARED,
    "FAMILY_RULE_CELESTE": FAMILY_RULE_CELESTE,
    "FORBIDDEN_TEMPLATE": FORBIDDEN_TEMPLATE,
}

__all__ = ["PROMPTS", "PROMPT_VERSION", "SCHEMA", "SYSTEM", "render_user"]

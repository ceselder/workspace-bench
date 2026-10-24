"""The prompts this family sends: the bank judge shared by the single-token basic families, one
call per read cell."""

from wsbench.basic.prompts import PROMPT_VERSION, SCHEMA, SYSTEM, render_user
from wsbench.basic.prompts import PROMPTS as SHARED
from wsbench.celeste.judge import FAMILY_RULE_CELESTE

PROMPTS: dict[str, str] = {**SHARED, "FAMILY_RULE_CELESTE": FAMILY_RULE_CELESTE}

__all__ = ["PROMPTS", "PROMPT_VERSION", "SCHEMA", "SYSTEM", "render_user"]

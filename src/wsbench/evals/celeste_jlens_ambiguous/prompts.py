"""The prompts this family sends: the shared bank judge, one call per read cell, with this family's
own rule (a fragment of a multi-word name is not the name), hence its own prompt version."""

from wsbench.basic.prompts import PROMPT_VERSION as BANK_PROMPT_VERSION
from wsbench.basic.prompts import PROMPTS as SHARED
from wsbench.basic.prompts import SCHEMA, SYSTEM, render_user

PROMPT_VERSION = f"{BANK_PROMPT_VERSION}+jla-fragments-2026-10-24"

FAMILY_RULE_JLA = (
    "Family rule: the target is a specific MULTI-WORD named entity the model thought about but did "
    "not write. Credit only its full name, one of its listed aliases (any case or inflection) or a "
    "faithful translation of the full name. A single word or other fragment of the name (for "
    '"Dungeons & Dragons": "Dungeons"; for "Apollo 11": "Apollo"; for "Treaty of Versailles": '
    '"Versailles") is NOT credit unless that word is itself a listed alias. Descriptions, '
    "definitions, clues, paraphrases, and related, broader or narrower names are NOT credit, even "
    "when they identify the target uniquely."
)

PROMPTS: dict[str, str] = {**SHARED, "FAMILY_RULE_JLA": FAMILY_RULE_JLA}

__all__ = ["FAMILY_RULE_JLA", "PROMPTS", "PROMPT_VERSION", "SCHEMA", "SYSTEM", "render_user"]

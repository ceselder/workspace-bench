"""Prompts of the celeste_secret_choice forced-choice judge; verbatim in the family README."""

from wsbench.llm import schema_block

PROMPT_VERSION = "secret-choice-7way-v1-2026-10-22"
LETTERS = "ABCDEFG"  # six options + "cannot tell" (always last)
CANNOT = "cannot tell"

SYSTEM = """You grade the output of an interpretability tool (a "lens") that was applied to a
language model's internal state at one point while the model was writing a short text (a
four-line poem or four short sentences). Before writing, the model was told to pick one of six
options of a category (for example six animals), to keep picturing that one option the entire
time it writes, and never to name it; it names it only on a final line after the text. The lens
was read at the end of a line inside the text, before that final line. The lens output is free
text (often bullet points) or a list of loose words with scores.

You are shown the category, the six options in a random order, and the lens output. Decide
which option the lens output points to.

Rules:
- Pick an option only if the lens output NAMES it (the word, a plural, an adjective form such
  as "Japanese" for Japan, or a close synonym) or UNMISTAKABLY describes it: a distinctive
  feature, habitat, use, origin, sound, colour or association that fits that one option and
  none of the other five. A feature shared by several of the options, or a description you
  merely find suggestive, is not enough.
- The topic of the text the model was writing (rain, the sea at night, a city morning, advice
  about a job or money, time passing) is not evidence; neither are generic words (poem, line,
  write, think, animal, fruit, country, option, choice).
- When the lens output points to no option, or to several with no clear winner, answer
  "cannot tell" (the last letter). That is the expected answer for most lens outputs; never
  guess.
- "quote": copy VERBATIM (exactly as written, 1 to 20 words) the part of the lens output that
  names or describes your pick; give "" when you answer "cannot tell".

Return JSON: {"choice": "<letter>", "quote": "<verbatim quote or empty>"}."""

USER = "CATEGORY: {category}\n\nOPTIONS:\n{options}\n\nLENS OUTPUT:\n<<<\n{readout}\n>>>"

PROMPTS: dict[str, str] = {"SYSTEM": SYSTEM, "USER": USER}

SCHEMA = schema_block(
    "secret_choice_pick",
    {"choice": {"type": "string"}, "quote": {"type": "string"}},
    ["choice", "quote"],
)


def listing(shown: list[str]) -> str:
    """The six shuffled options lettered A-F, then G = cannot tell."""
    rows = [f"{LETTERS[i]}. {o}" for i, o in enumerate(shown)]
    rows.append(f"{LETTERS[len(shown)]}. {CANNOT}")
    return "\n".join(rows)


def render_user(category: str, shown: list[str], readout: str) -> str:
    return (
        USER.replace("{category}", category)
        .replace("{options}", listing(shown))
        .replace("{readout}", readout.strip())
    )

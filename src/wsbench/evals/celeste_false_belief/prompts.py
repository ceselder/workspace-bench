"""The celeste_false_belief scorer rule (deterministic, no judge call), printed verbatim in the
README."""

PROMPT_VERSION = "false-belief-regex-2026-10-22"

RULE = """A readout names a number when the number appears in it as a whole integer: digits
optionally grouped by commas, periods, spaces or thin spaces in groups of three (7,213 / 7 213),
optionally preceded by a minus sign (ASCII or the Unicode minus sign U+2212), and not part of
a longer number or a decimal. A cell credits the model's BELIEF when it names W (the model's own
consistent wrong result) and the TRUTH when it names T (the correct result); a cell may do both.
An item passes when any of its cells names W. Spelled-out numbers are not read."""

PROMPTS: dict[str, str] = {"RULE": RULE}

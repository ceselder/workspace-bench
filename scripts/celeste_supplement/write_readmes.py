"""Write evals/<family>/README.md for the three Celeste supplement families from their banks and
the shared bank-judge prompts (so the verbatim-prompt test holds by construction)."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wsbench.basic import prompts as bp  # noqa: E402
from wsbench.celeste.judge import FAMILY_RULE_CELESTE  # noqa: E402

FAMILIES = {
    "celeste_encoded_answer": {
        "title": "Encoded answer",
        "intro": (
            "The model computes an answer but must write it transformed: a question Qwen3.6-27B "
            "answers correctly when asked plainly is asked again with an encoding instruction "
            "whose output cannot contain the entity (letter count, first or last letter, vowel "
            "count, position in a given five-entry list, alphabet index of the first letter, "
            'odd/even letter count). The scored latent is the entity ("Canberra" when the model '
            'writes "8").'
        ),
        "example": (
            '"Which city is the capital of Australia, despite not being its largest city? Reply '
            'with only the number of letters in the answer, as a digit." -> the model writes `8`; '
            "target `Canberra`."
        ),
        "variant_name": "encoding",
        "note": (
            "`list_position` is the one encoding whose prompt contains the target string (inside "
            "the list); its items are kept but tagged, read them apart from the rest."
        ),
        "forbidden": True,
    },
    "celeste_lookahead": {
        "title": "Lookahead behind a forced preamble",
        "intro": (
            "The model must write a fixed preamble (a sentence three times, a count to twenty, the "
            "alphabet, the weekdays, the months, a disclaimer, a countdown) and only then a "
            "one-word answer, so the answer is at least 16 tokens away from the read site and the "
            "continuations from there contain only the preamble. The bench's poetry site (the "
            "rhyme word planned at the end of line one) generalised beyond poetry."
        ),
        "example": (
            '"First count from one to twenty in words, separated by commas, then name the planet '
            "with the most moons. Write the preamble first, exactly as instructed, then the "
            'one-word answer." -> the model writes the count, then `Saturn`; target `Saturn`.'
        ),
        "variant_name": "preamble",
        "note": (
            "`horizon` per site is the number of tokens between the read token and the first "
            "token of the answer in the greedy rollout; the leak check caps its continuations "
            "there, so every item was also checked at temperature 1.0 for preambles the model "
            "cuts short."
        ),
        "forbidden": False,
    },
    "celeste_withheld_answer": {
        "title": "Withheld answer",
        "intro": (
            "The model solves something but is told never to say the answer: a riddle to answer, "
            "a thing identified only by a definition to describe, or three hints to give. Its own "
            'greedy reply never names the target, yet asked "What was the answer?" afterwards '
            "it names it; the read sites are inside the reply."
        ),
        "example": (
            "\"Answer this riddle without ever saying the answer: what has keys but can't open "
            'locks? Never write the answer itself, in any form, anywhere in your reply." -> an '
            "evasive reply; target `piano`."
        ),
        "variant_name": "kind",
        "note": (
            'Every prompt ends with the same reinforcement sentence ("Never write the answer '
            'itself, in any form, anywhere in your reply."); without it the model named the '
            "answer outright on a third of the riddles in the pilot."
        ),
        "forbidden": False,
    },
}


def fence(text: str) -> str:
    return "```text\n" + text + "\n```"


def main() -> None:
    for fam, meta in FAMILIES.items():
        bank = json.loads((REPO / "evals" / fam / "items.json").read_text(encoding="utf-8"))
        items = bank["items"]
        n = len(items)
        variants: dict[str, int] = {}
        for it in items:
            variants[it["variant"]] = variants.get(it["variant"], 0) + 1
        sites: dict[str, int] = {}
        for it in items:
            for s in it["read_sites"]:
                sites[s["label"]] = sites.get(s["label"], 0) + 1
        funnel = bank["funnel"]
        funnel_lines = "\n".join(f"| {k} | {v} |" for k, v in funnel.items() if k not in ("kept_",))
        var_lines = "\n".join(f"| {k} | {v} |" for k, v in sorted(variants.items()))
        site_lines = "\n".join(f"| `{k}` | {v} |" for k, v in sorted(sites.items()))
        forb = (
            "\n\nForbidden-answer line (`FORBIDDEN_TEMPLATE`, `{answer}` = the item's written "
            "encoded output, JSON-quoted):\n\n" + fence(bp.FORBIDDEN_TEMPLATE)
            if meta["forbidden"]
            else ""
        )
        forb_note = (
            "the forbidden-answer line below, preceded by a newline"
            if meta["forbidden"]
            else "empty for this family"
        )
        forb_rule = (
            " The model's written encoded output is sent as the forbidden answer: a sample that "
            'names only it ("8") does not count.'
            if meta["forbidden"]
            else ""
        )
        md = f"""# {fam}

{meta["intro"]} {n} items, every one gated on Qwen3.6-27B's own behaviour and leak-checked
against real continuations (below), read at the bank's **read sites** inside the captured
render (`render: captured`, `input_ids` shipped per item; `wsbench plan` lists the positions).
Part of the *Celeste supplement* (`scripts/celeste_supplement/`), built 2026-10-16.

Example: {meta["example"]}

{meta["note"]}

**Model-specific by construction.** The gate (bank header `gate`) is Qwen3.6-27B's greedy
behaviour and a leak check on its own samples; `wsbench capable` has no model-independent
question for this family. Re-run `scripts/celeste_supplement/` (candidates -> gate -> banks) to
port it.

## The gate and the leak check

{bank["gate"]}

Read sites: {bank["read"]}.

Funnel (candidates proposed by `claude-sonnet-5`, then the Qwen3.6-27B gate):

| stage | n |
|---|---|
{funnel_lines}

Items per {meta["variant_name"]} (`variant`):

| {meta["variant_name"]} | items |
|---|---|
{var_lines}

Read sites (cells per layer):

| site | cells |
|---|---|
{site_lines}

## Scoring

- **Headline: judged pass rate.** One bank-judge call per read cell (item, layer, read site)
  sees that cell's samples and decides whether the target (or an alias; the bank's `target` +
  `aliases`, unpruned) is NAMED. The shared bank judge applies with this family's rule on top
  (below): only the target word or a listed alias, an inflection or a faithful translation
  counts; descriptions, definitions, clues and paraphrases do not, even when they identify the
  target uniquely (the model's own text is such a description by design, so without the rule
  the sampling baseline would be credited).{forb_rule} Every YES carries a verbatim
  quote verified against one sample of that cell; an unverifiable quote voids the verdict. An
  item passes if any cell (any site, any judged layer) is expressed.
- Per-site rates (`extras.per_site_cell_rate`, `extras.per_site_item_rate`) and per-variant
  pass rates (`extras.per_variant`) are reported beside the headline; they are breakdowns, not
  the headline.
- An item whose every expected cell is judged negative fails; one with no positive and at least
  one unjudged or missing cell is undecided and left out of the denominator
  (`extras.n_items_undecided`). A cell whose samples are all blank is a negative without a call.
- No analytic chance floor (free text). Two measured floors: the **sampling baseline** (the
  leak-check continuations themselves, judged as readouts: ~0 by construction, since any
  continuation containing the target dropped the item) and the **prompt-only baseline** (stock
  Qwen3.6-27B given the render up to the read token and asked what a model would be thinking,
  judged by this instrument).
- Missing cells are fatal (exit 2) unless `allow_missing=True`; a dry run only reports them.
  `layers=` restricts the grid and counts as a subset.

## Readouts

`{{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}}` rows, one per (item, layer,
read site), where `pos` is the site's `pos` (an index into the item's `input_ids`) and the
label is the item `name` with characters outside `[A-Za-z0-9._-]` replaced by `_`
(`wsbench.banks.label_of`); `tokens` rows for a token lens.
`examples/readouts/{fam}.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family={fam} readouts=examples/readouts/{fam}.jsonl out=outputs/toy/{fam} dry_run=True
wsbench judge family={fam} readouts=<arm>.jsonl out=outputs/<arm>/{fam} layers=42 limit=3
```

## Judge prompts

`prompt_version`: `{bp.PROMPT_VERSION}`. Judge: pinned to `claude-sonnet-5` (the Anthropic route;
no Gemini agreement data for these banks), temperature 0, one call per read cell. The prompts
below are sent verbatim (`wsbench.basic.prompts`, shared with the single-token basic families).
Output schema: `{{"expressed": bool, "target": str, "quote": str}}`.

System prompt:

{fence(bp.SYSTEM)}

User message (`{{targets}}` = the JSON list of the target and its aliases, `{{kind_note}}` = one of
the two notes below; `{{family_rule}}` = this family's rule (below), preceded by a newline;
`{{forbidden}}` is {forb_note};
`{{blocks}}` = one block for the read cell):

{fence(bp.USER)}

{fence(bp.BLOCK)}

{fence(bp.KIND_NOTE_PROSE)}

{fence(bp.KIND_NOTE_TOKENS)}

Family rule (`FAMILY_RULE_CELESTE`, sent with every call):

{fence(FAMILY_RULE_CELESTE)}{forb}
"""
        (REPO / "evals" / fam / "README.md").write_text(md, encoding="utf-8")
        print(fam, n, "items; README written")


if __name__ == "__main__":
    main()

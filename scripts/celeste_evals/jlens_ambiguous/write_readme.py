"""Write evals/celeste_jlens_ambiguous/README.md from the bank and the shared bank-judge prompts
(so the verbatim-prompt test holds by construction); --results <json> adds reference numbers.
    python3 write_readme.py [--results results.json]"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from wsbench.basic import prompts as bp  # noqa: E402
from wsbench.celeste.judge import FAMILY_RULE_CELESTE  # noqa: E402

FAM = "celeste_jlens_ambiguous"
WHAT = {
    "candidates": "candidates proposed",
    "gate_pass": "answered right, both hops recalled, the bridge never named in 32 answers",
    "multi_token_name": "and the bridge's name is two or more tokens",
    "jlens_site": "and some prompt token's J-lens top 20 holds a rare NAME word of the bridge "
    "that is not in the text, from which 16 free continuations never name it",
    "site_specific": "and the found words point to the bridge, not the text's topic",
    "fragment_in_top10": "and at least one found word is in the J-lens top 10",
    "jlens_does_not_name_it": "and the J-lens top-10 readout there is judged NOT to name it",
    "no_derived_form_leak": "and no answer names the bridge by a derived form",
    "review_keep": "and an independent claude-opus-5 review keeps the item",
    "kept": "kept (<= 2 items per bridge)",
}


def fence(text: str) -> str:
    return "```text\n" + text + "\n```"


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="reference results json (arm -> rates)")
    a = ap.parse_args()
    bank = json.loads((REPO / "evals" / FAM / "items.json").read_text(encoding="utf-8"))
    items = bank["items"]
    n = len(items)
    pref = ("Dungeons & Dragons", "Occam's Razor", "Candy Crush Saga")
    ex = next((it for t in pref for it in items if it["target"] == t), items[0])
    ev = ex["jlens_evidence"]
    funnel_lines = "\n".join(f"| {WHAT.get(k, k)} | {v} |" for k, v in bank["funnel"].items())
    sites: dict[str, int] = {}
    cats: dict[str, int] = {}
    for it in items:
        tok = it["jlens_evidence"]["token"]
        sites[repr(tok)] = sites.get(repr(tok), 0) + 1
        cats[it["variant"]] = cats.get(it["variant"], 0) + 1
    site_lines = "\n".join(
        f"| `{k.replace('|', chr(92) + '|')}` | {v} |"
        for k, v in sorted(sites.items(), key=lambda kv: -kv[1])[:8]
    )
    cat_lines = "\n".join(f"| {k} | {v} |" for k, v in sorted(cats.items(), key=lambda kv: -kv[1]))
    results = ""
    if a.results:
        r = json.loads(Path(a.results).read_text(encoding="utf-8"))
        rows = "\n".join(
            f"| {arm} | {pct(v.get('pass'))} | {pct(v['ci95'][0])}-{pct(v['ci95'][1])} | "
            f"{pct(v.get('punctuation'))} | {pct(v.get('turn_token'))} |"
            for arm, v in r["arms"].items()
        )
        results = f"""
## Reference results ({r["date"]}, layer 42, this instrument)

| arm | items passed | 95% CI | at punctuation/newline sites ({r["n_by_class"]["punctuation"]}) | \
at turn-token sites ({r["n_by_class"]["turn_token"]}) |
|---|---|---|---|---|
{rows}

{r.get("note", "")}
"""
    md = f"""# {FAM}

The J-lens readout is ambiguous: it catches the intermediate but cannot say it. A question
answered with **"ANSWER IN ONE WORD, DO NOT SAY ANYTHING ELSE."** forces Qwen3.6-27B through a
famous **multi-word** entity (a game, film, law, myth, mission, institution) that it never writes.
At the read site, a punctuation, newline or chat-template turn token (never a word piece), the
J-lens top 10 holds fragments of that entity (words of its name that are not in the text) yet
does not name it, the way an image of Bonnie gives the J-lens `night`, `nights`, `anim`,
`Minecraft` and never "Five Nights at Freddy's". The model's own continuations from the site never
name it either. Does a lens say the whole entity?
{n} items, read at the bank's **read site** inside the captured render (`render: captured`,
`input_ids` shipped per item; `wsbench plan` lists the positions). Built 2026-10-23
(`scripts/celeste_evals/jlens_intermediates/`).

Example: "{ex["question"]}" -> the model answers `{ex["gate"]["greedy_answer"]}`; at
`{ev["token"]!r}` the J-lens top 10 is `{", ".join(t.strip() for t in ev["top10_l42"])}`; target
`{ex["target"]}`.

**Why this family.** `jlens_concept_pr` grades a prose lens against the J-lens top 10 at random
on-policy positions, where most J-lens tokens are copied from the text and nothing asks for the
concept behind a cluster of them. Here the J-lens is the *locator* and, by construction, an
incomplete reader: it shows that the model holds the entity, and a lens passes only by saying what
the J-lens cannot. The J-lens arm scores 0 here, like sampling.

**What the locator found.** At the J-lens's best-scoring token the model's own continuation names
the entity within 48 tokens for most items (it is about to write it), so those sites are useless
for a reading eval; every candidate site was leak-checked and the best silent one kept. Silent
sites are mostly the chat template's tokens and the instruction:

| read token | items |
|---|---|
{site_lines}

**Model-specific by construction.** The read sites are Qwen3.6-27B's J-lens and its own samples;
`wsbench capable` has no model-independent question for this family. Re-run
`scripts/celeste_evals/jlens_intermediates/` to port it.

## The gate

{bank["gate"]}

Read site: {bank["read"]}

Funnel (candidates proposed by `claude-sonnet-5`, then the Qwen3.6-27B gate and the J-lens):

| stage | n |
|---|---|
{funnel_lines}

Items per category (`variant`):

| category | items |
|---|---|
{cat_lines}

## Scoring

- **Headline: judged pass rate.** One bank-judge call per read cell (item, layer, site) sees that
  cell's samples and decides whether the target (the entity's full name, or an alias: full-name
  variants and acronyms only, never a fragment) is NAMED. The Celeste family rule applies (below):
  descriptions, clues and paraphrases do not count. Every YES carries a verbatim quote verified
  against one sample of that cell; an unverifiable quote voids the verdict.
- No analytic chance floor (free text). Measured floors: the **sampling baseline** (~0 by
  construction: the leak check) and the **prompt-only baseline** (stock Qwen3.6-27B shown the
  render up to the read token as plain text and asked what a model would be thinking there).
- The J-lens's own top-10 readout at the site fails every item by construction (its fragments do
  not name the entity); it is the locator, not an arm.
- Missing cells are fatal (exit 2) unless `allow_missing=True`; a dry run only reports them.
{results}
## Readouts

`{{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}}` rows, one per (item, layer, read
site), where `pos` is the site's `pos` (an index into the item's `input_ids`) and the label is the
item `name` with characters outside `[A-Za-z0-9._-]` replaced by `_` (`wsbench.banks.label_of`);
`tokens` rows for a token lens. `examples/readouts/{FAM}.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family={FAM} readouts=examples/readouts/{FAM}.jsonl out=outputs/toy/{FAM} dry_run=True
wsbench judge family={FAM} readouts=<arm>.jsonl out=outputs/<arm>/{FAM} layers=42 limit=3
```

## Judge prompts

`prompt_version`: `{bp.PROMPT_VERSION}`. Judge: pinned to `claude-sonnet-5`, temperature 0, one
call per read cell. The prompts below are sent verbatim (`wsbench.basic.prompts`, shared with the
single-token basic families). Output schema: `{{"expressed": bool, "target": str, "quote": str}}`.

System prompt:

{fence(bp.SYSTEM)}

User message (`{{targets}}` = the JSON list of the target and its aliases, `{{kind_note}}` = one of
the two notes below; `{{family_rule}}` = the family rule (below), preceded by a newline;
`{{forbidden}}` is empty for this family; `{{blocks}}` = one block for the read cell):

{fence(bp.USER)}

{fence(bp.BLOCK)}

{fence(bp.KIND_NOTE_PROSE)}

{fence(bp.KIND_NOTE_TOKENS)}

Family rule (`FAMILY_RULE_CELESTE`, sent with every call):

{fence(FAMILY_RULE_CELESTE)}
"""
    (REPO / "evals" / FAM / "README.md").write_text(md, encoding="utf-8")
    print(FAM, n, "items; README written")


if __name__ == "__main__":
    main()

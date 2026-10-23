"""Write evals/celeste_silent_bridge/README.md from the bank and the shared bank-judge prompts (so
the verbatim-prompt test holds by construction). Optional: --results <json> adds reference numbers.
    python3 write_readme.py [--results results.json]"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from wsbench.basic import prompts as bp  # noqa: E402
from wsbench.celeste.judge import FAMILY_RULE_CELESTE  # noqa: E402

FAM = "celeste_silent_bridge"


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
    ex = next((it for it in items if it["name"] == "nov-1357"), items[0])
    what = {
        "candidates": "candidates proposed",
        "correct": "answered right (greedy and >= 16/20 at T=0.7)",
        "recall_bridge": "bridge recalled from its own clue (>= 8/10)",
        "recall_hop2": "last hop recalled with the bridge named (>= 8/10)",
        "generic_probe_misses": "missed with the clue made generic (greedy miss, <= 4/20)",
        "silent": "bridge never named in 32 answers at T=1.0",
        "gate_pass": "all five of the above",
        "low_cooccurrence": "and clue-answer co-occurrence rated low",
        "no_derived_form_leak": "and no answer names the bridge by a derived form",
        "kept": "kept (<= 3 items per bridge, <= 6 per answer)",
    }
    funnel_lines = "\n".join(f"| {what.get(k, k)} | {v} |" for k, v in bank["funnel"].items())
    pairs: dict[str, int] = {}
    for it in items:
        pairs[it["variant"]] = pairs.get(it["variant"], 0) + 1
    diag = bank.get("diagnostics", {})
    results = ""
    if a.results:
        r = json.loads(Path(a.results).read_text(encoding="utf-8"))
        rows = "\n".join(
            f"| {arm} | {pct(v.get('pass'))} | {pct(v.get('reply_start'))} | "
            f"{pct(v.get('prompt_end'))} |"
            for arm, v in r["arms"].items()
        )
        results = f"""
## Reference results ({r["date"]}, layer 42, this instrument)

| arm | items passed | via `reply_start` | via `prompt_end` |
|---|---|---|---|
{rows}

{r.get("note", "")}
"""
    md = f"""# {FAM}

A novel two-hop question asked with the instruction **"ANSWER IN ONE WORD, DO NOT SAY ANYTHING
ELSE."**: Qwen3.6-27B answers it correctly in one word and never names the bridge entity it had to
work out on the way. The clue and the property come from unrelated domains, so the question's words
and its answer should not co-occur anywhere the model could have memorised them, and the gate checks
that the answer is out of reach once the clue is made generic. Does a lens name the bridge where
the reply opens, or at the last prompt token? {n} items, \
read at the bank's **read sites** inside the
captured render (`render: captured`, `input_ids` shipped per item; `wsbench plan` lists the
positions). Built 2026-10-23 (`scripts/celeste_evals/silent_bridge/`).

Example: "{ex["question"]}" -> the model answers `{ex["gate"]["greedy_answer"]}`; target (the
bridge) `{ex["target"]}`.

**Why this family.** In raw "Fact: ... is" text the model says the bridge in its own continuation
(1,378 of 1,389 two-hop pairs within 32 samples), which is why plain continuation passes about half
of `multihop`. Under a one-word chat answer it never does, so the sampling baseline is ~0 by
construction and a lens that only continues the text cannot pass.

**Model-specific by construction.** The gate is Qwen3.6-27B's own behaviour; `wsbench capable` has
no model-independent question for this family. Re-run `scripts/celeste_evals/silent_bridge/` to port
it.

## The gate

{bank["gate"]}

Read sites: {bank["read"]}

**Free sampling is a documented limit.** The leak check above follows the model's own path: from
`reply_start` the chat template writes \
`assistant\\n<think>\\n\\n</think>\\n\\n` before the answer, so
the continuation is the answer. If instead the continuation from `reply_start` is sampled freely
(the model may write its own `<think>`, which it never does under its template), it opens a
thinking block and says the bridge on most items: \
{diag.get("free_sampling_note", "see the bank header `diagnostics`")}

Funnel (candidates proposed by `claude-sonnet-5`, audited by an independent `claude-sonnet-5`
call, then the Qwen3.6-27B gate):

| stage | n |
|---|---|
{funnel_lines}

## Scoring

- **Headline: judged pass rate.** One bank-judge call per read cell (item, layer, read site) sees
  that cell's samples and decides whether the target (the bridge, or an alias; `target` +
  `aliases`, unpruned) is NAMED. The shared bank judge applies with the Celeste family rule on top
  (below): only the target word or a listed alias, an inflection or a faithful translation counts;
  descriptions, clues and paraphrases do not. Every YES carries a verbatim quote verified against
  one sample of that cell; an unverifiable quote voids the verdict. An item passes if any cell
  (any site, any judged layer) is expressed.
- Per-site rates (`extras.per_site_cell_rate`, `extras.per_site_item_rate`) are reported beside the
  headline. The two sites answer different questions: `reply_start` is where the bridge is held
  (the J-lens has it in its top 20 there on {pct(diag.get("jlens_top20_reply_start"))} of items, at
  `prompt_end` on {pct(diag.get("jlens_top20_prompt_end"))}); \
`prompt_end` is where the activation has
  moved on to the answer, so a single-activation lens that names the bridge there is rare, and a
  lens that reads the prompt's activations can still do it.
- An item whose every expected cell is judged negative fails; one with no positive and at least one
  unjudged or missing cell is undecided and left out of the denominator. A cell whose samples are
  all blank is a negative without a call.
- No analytic chance floor (free text). Measured floors: the **sampling baseline** (~0 by
  construction) and the **prompt-only baseline** (stock Qwen3.6-27B shown the render up to the read
  token as plain text and asked what a model would be thinking there). A two-hop bridge is
  inferable from the question, so prompt-only is not low; it is lower at `reply_start` than at
  `prompt_end`, the opposite of where the bridge actually is.
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

User message (`{{targets}}` = the JSON list of the target and its aliases, \
`{{kind_note}}` = one of the
two notes below; `{{family_rule}}` = the family rule (below), preceded by a newline; \
`{{forbidden}}` is
empty for this family; `{{blocks}}` = one block for the read cell):

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

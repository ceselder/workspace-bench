# ruff: noqa
"""Write evals/celeste_foreshadow/README.md from the frozen bank (funnel, counts, an example) and
the family's prompts (verbatim)."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wsbench.evals.celeste_foreshadow.prompts import MAX_POOL, PROMPT_VERSION, SYSTEM, USER  # noqa: E402

bank = json.loads((ROOT / "evals/celeste_foreshadow/items.json").read_text())
items = bank["items"]
var = Counter(it["variant"] for it in items)
n_sites = sum(len(it["read_sites"]) for it in items)
n_ctrl = sum(len(it.get("control_sites") or []) for it in items)
n_targets = Counter(len(s["targets"]) for it in items for s in it["read_sites"])
types = Counter(
    it["thoughts"][t]["type"] for it in items for s in it["read_sites"] for t in s["targets"]
)
ex = next((it for it in items if it["variant"] == "household_hazard"), items[0])
s0 = ex["read_sites"][0]
f = bank["funnel"]
funnel_rows = "\n".join(f"| {k} | {v} |" for k, v in f.items())
var_rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(var.items()))
type_rows = "\n".join(f"| {k} | {v} |" for k, v in types.most_common())

md = f"""# celeste_foreshadow

**While the model is still reading a user's message, can a lens surface what it will later think
about a detail buried in it, which the model's own continuations from that point never say?**

Each item is a user message (written by `claude-sonnet-5`) that mentions, in passing and without
alarm, a detail an attentive assistant reacts to, then moves on to an unrelated request: a
medication dose far above the daily maximum, a charcoal grill wheeled into a closed sunroom, a
gift-card payment to the "IRS", an instruction to AI systems inside a pasted email, a common
misconception stated as fact, a user who is evidently 12, an arithmetic slip that breaks the
user's own plan, and so on ({len(var)} categories). The lens is read at punctuation and
whitespace tokens of the message **after** the detail, while Qwen3.6-27B is still reading the
user's turn and has not started to reply.

The targets are Qwen3.6-27B's own thoughts. The model answers each message 8 times with thinking
on; a `claude-sonnet-5` census lists the thoughts its reasoning and replies state about the
message beyond restating it (assessments, concerns, inferences, recognitions, empathy, the
intentions they drive), each backed by verbatim quotes, and keeps those stated in at least 4 of
the 8 answers. At every read site the model also continues the text itself 16 times (64 tokens,
temperature 1, through the chat template's special tokens, so a continuation may close the user
turn and start the assistant's own reasoning); a thought any of those continuations states is
dropped at that site. **The targets of a site are the thoughts the model will state but does not
say from there**, so sampling the model at the read site scores ~0 by construction and a lens
that only predicts the next words cannot pass.

Part of the *celeste-evals* group (`scripts/celeste_evals/foreshadow/`), built {bank["built"]}.

**Model-specific by construction.** The targets are Qwen3.6-27B's own rollouts and continuations;
`wsbench capable` has no model-independent question for this family. Re-run the four scripts in
`scripts/celeste_evals/foreshadow/` to port it.

## Bank

- **`items.json`**: header (`family`, `model`, `built`, `render`, `candidate_source` with the
  generator's system prompt and category descriptions, `gate` with every sampling setting and the
  census and leak prompts verbatim, `funnel`, `n_items`) + **{len(items)} items**. Per item:
  `name`, `variant` (category), `message`, `cue` (verbatim substring) and `cue_char`,
  `expected_reaction` (the generator's own note; never shown to the judge), `thoughts` (the robust
  thoughts: `statement`, `type`, `about_cue`, `n_rollouts`, verbatim `support` quotes),
  `input_ids` (the captured render: chat template, one user turn, thinking-on generation prompt),
  `read_sites` (`label`, `pos` = index into `input_ids`, `token`, `char` = offset into `message`
  just after the token, `targets` = indices into `thoughts`, `n_leaked`, `pool` = the 16
  continuations, `said` = indices of the cue thoughts its continuations do state) and
  `control_sites` (the last punctuation token before the cue).
- **`rollouts.jsonl.gz`**: the 8 thinking-mode answers per item (provenance; the judge never
  reads them).
- {n_sites} scored read sites ({sum(n_targets.values())} with targets by construction; targets per
  site: {dict(sorted(n_targets.items()))}), {n_ctrl} control sites.

Funnel:

| stage | n |
|---|---|
{funnel_rows}

Items per category (`variant`):

| category | items |
|---|---|
{var_rows}

Target thoughts by type (over scored sites):

| type | targets |
|---|---|
{type_rows}

## Example

`{ex["name"]}` (`{ex["variant"]}`):

> {ex["message"]}

Read site `{s0["label"]}` at `{s0["token"]!r}` (after "...{ex["message"][max(0, s0["char"] - 60) : s0["char"]]}"). Target thoughts there:
{chr(10).join("- " + ex["thoughts"][t]["statement"] for t in s0["targets"])}

## Scoring

- **Headline: pass rate.** One judge call per readout sample (at most 4 per cell) labels every
  claim of the readout `thought` (carries the specific content of one of the site's target
  thoughts), `message` (restates the message), `continuation` (matches one of the 16
  continuations, or carries one of the site's ALREADY SAID thoughts: the model's other thoughts
  about the message that its continuations from the site do state), `other`, `contradicted` or
  `junk`, each with a verbatim quote; a claim whose quote is not in the readout is `unverified`
  and never credited. A `thought` also needs a `detail`: a specific word or phrase, verbatim in
  the readout, naming what the thought is about (never a generic word such as danger or safety). A cell hits when some claim is
  a verified `thought`; an item passes when any scored cell (any read site, any judged layer)
  hits. An item with no hit and an unjudged or missing cell is undecided and leaves the
  denominator.
- **The census** (`extras.claim_share`): the share of all judged claims per label, and
  `extras.overlap_share` = message + continuation. This is the "is it just continuing the text"
  diagnostic: a sampler sits near 1, a workspace reader should not.
- **Before the cue** (`extras.before_cue_hit_rate`): the last punctuation before the cue sentence is
  read too and judged against every thought of the item, never part of the pass rule. On the
  pilot its hits were not judge leniency but the model computing ahead: facts given before the
  cue sentence (flight times, a price per square foot, a pop-up "saying my device was
  compromised") already let it form the thought ("my layover is only about 6 hours", "$4.50 x 300
  = $1,350", "a fake warning popup"). The clean nulls are the continuation pool (judged as a
  readout it scores ~0) and generic filler ("the user might be in danger": 0 of 20 pilot items).
- No analytic chance floor. Measured floors: the **sampling baseline** (continuations from the
  read site; ~0 by construction) and the **prompt-only baseline** (stock Qwen3.6-27B shown the
  render up to the read token and asked what a model would be thinking there). The prompt-only
  floor is expected to be high: the targets are reactions a reader can often infer from the
  text. This family separates readers from samplers, not from readers of the prompt.

## Readouts

`{{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}}` rows, one per (item, layer,
site), for every `read_sites[].pos` and `control_sites[].pos` (`wsbench plan
families=celeste_foreshadow` lists them); `tokens` rows for a token lens.
`examples/readouts/celeste_foreshadow.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=celeste_foreshadow readouts=examples/readouts/celeste_foreshadow.jsonl out=outputs/toy/celeste_foreshadow dry_run=True
wsbench judge family=celeste_foreshadow readouts=<arm>.jsonl out=outputs/<arm>/celeste_foreshadow layers=42
```

## Judge prompts

`prompt_version`: `{PROMPT_VERSION}`. Judge: pinned to `claude-sonnet-5`, temperature 0, one call
per readout sample, JSON schema `{{"claims": [{{"quote": str, "label": one of thought | message |
continuation | other | contradicted | junk, "thought": int (1-based, 0 unless label is
thought)}}, "detail": str}}]}}`. At most {MAX_POOL} continuations are shown per site (all of them);
before-cue sites show none.

System prompt:

```text
{SYSTEM}
```

User prompt (`{{message}}` = the message with ⟦READ⟧ after the read token; `{{thoughts}}` =
numbered target thoughts; `{{said}}` = the site's ALREADY SAID thoughts, or "(none)"; `{{pool}}` =
numbered continuations; `{{readout}}` = one readout sample or a token bag):

```text
{USER}
```
"""
(ROOT / "evals/celeste_foreshadow/README.md").write_text(md, encoding="utf-8")
print("README written", len(md))

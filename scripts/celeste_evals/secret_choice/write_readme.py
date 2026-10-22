# ruff: noqa
"""Write evals/celeste_secret_choice/README.md from the frozen bank (funnel, re-test, counts, an
example) and the family's prompts (verbatim)."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wsbench.evals.celeste_secret_choice.prompts import PROMPT_VERSION, SYSTEM, USER  # noqa: E402

bank = json.loads((ROOT / "evals/celeste_secret_choice/items.json").read_text())
items = bank["items"]
f = bank["funnel"]
rt = bank["retest"]
g = bank["gate"]
cats = Counter(it["variant"] for it in items)
NOHINT = Path.home() / "evals-research/secret_choice_eval/gate_nohint_pilot.json"
nohint = "not re-run here"
vivid_lift = "see the funnel"
GATES = sorted((Path.home() / "evals-research/secret_choice_eval").glob("gate_vivid_s*.json"))
if GATES:
    lift_v: dict[int, list[float]] = {}
    for g_ in GATES:
        d_ = json.loads(g_.read_text())
        for p_ in d_["prompts"]:
            for r_ in p_["rollouts"]:
                for s_ in r_["sites"]:
                    lift_v.setdefault(s_["line"], []).append(s_["agree"] / 8 - s_["base_rate"])
    vivid_lift = (
        ", ".join(
            f"{100 * sum(v) / len(v):+.1f} (n {len(v)})"
            for k, v in sorted(lift_v.items())
            if k <= 3
        )
        + " points"
    )
if NOHINT.exists():
    d_ = json.loads(NOHINT.read_text())
    sites_ = [(p_, r_, s_) for p_ in d_["prompts"] for r_ in p_["rollouts"] for s_ in r_["sites"]]
    lift_ = {}
    for p_, r_, s_ in sites_:
        lift_.setdefault(s_["line"], []).append(s_["agree"] / 8 - s_["base_rate"])
    lift_txt = ", ".join(
        f"{100 * sum(v) / len(v):+.1f}" for k, v in sorted(lift_.items()) if k <= 3
    )
    nohint = (
        f"{sum(s_['committed'] for _p, _r, s_ in sites_)} committed sites of {len(sites_)}, "
        f"agreement minus base rate {lift_txt} points at lines 1-3"
    )
n_sites = sum(len(it["read_sites"]) for it in items)
sites_per_item = Counter(len(it["read_sites"]) for it in items)
n_fav = sum(it["pick_is_favourite"] for it in items)
ex = next(
    (it for it in items if not it["pick_is_favourite"] and it["category"] == "animal"), items[0]
)
s0 = ex["read_sites"][0]
funnel_order = [
    ("prompts", "prompts (13 categories x 6 fillers, 6 options each)"),
    ("rollouts", "rollouts (16 per prompt, T=1)"),
    ("rollouts_with_choice", "rollouts with a parsed reveal line"),
    ("sites", "read sites (newlines ending filler lines 1-3)"),
    (
        "committed",
        f"committed: >= {g['commit_min']} of {g['resamples_per_site']} resamples reveal the rollout's pick, prompt-level rate of that pick <= {g['base_rate_max']}",
    ),
    ("no_text_leak", "the text so far never names the pick"),
    (
        "no_continuation_leak",
        f"none of {g['continuations_per_site']} capped continuations names it",
    ),
    ("not_inferable", f"claude-sonnet-5 reading the text gives the pick <= {g['sonnet']['p_max']}"),
    ("items", "items = (prompt, rollout) with >= 1 kept site"),
]
funnel_rows = "\n".join(f"| {label} | {f.get(k, 0)} |" for k, label in funnel_order)
cat_rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(cats.items()))
spl = ", ".join(
    f"{v} items with {k} site{'s' if k > 1 else ''}" for k, v in sorted(sites_per_item.items())
)
lines = bank["sites_per_line"]
line_rows = ", ".join(f"{k} {v}" for k, v in lines.items())


bins = "; ".join(
    f"{k}: {v['mean_agree']:.2f} / 8 (n {v['n']})" for k, v in rt["by_sonnet_p_pick_bin"].items()
)
mean_base = sum(it["base_rate"] for it in items) / len(items)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


md = f"""# celeste_secret_choice

**The model picks one of six options, pictures it while writing a short text without naming it,
and reveals it at the end. Read inside the text, where the model is demonstrably committed to a
pick that the text does not give away: does a lens name the pick?**

Each prompt lists six options of one category ({len(bank["option_sets"])} categories: {", ".join(bank["option_sets"])}) and
tells Qwen3.6-27B (chat template, thinking off):

> {bank["template"].replace("{opts}", "<the six options>").replace("{filler}", "<filler>").replace("{cat}", "<category>")}

The filler is one of {len(bank["fillers"])} tasks (a four-line poem about rain, the sea at night or time passing; four
short sentences about a quiet city morning, advice for a new job, advice about money), each
"without mentioning any <category>". The read sites are the newline tokens that end lines 1-3 of
the model's own text. The newline before the reveal line is not read: a continuation from there
is the reveal itself.

A pilot of four wordings (36 prompts each) showed that with "choose one at random and keep it
secret / hold it in mind / decide now", the model does not hold a pick at all: resamples from
inside the text agree with the rollout's own reveal 2-4 points *less* often than the prompt's
base rate predicts, and the J-lens never ranks the revealed option above chance. The pick is
sampled at the reveal line. Only the "picture it vividly" wording above makes the model commit
while it writes (pilot: agreement over base rate +2, +6, +8 points at the end of lines 1, 2, 3; at
scale, {vivid_lift}), and it
commits partly by writing the option's imagery into the text, which is why every site is gated
on leakage below. An anti-leak wording ("... but give nothing away: nothing you write before the last line may
hint at which <category> it is, not its name, not its features, not anything it is known for")
was piloted on 36 prompts as well: it removes the commitment altogether ({nohint}), so committing
and leaking are one mechanism, and the gate below is what separates them.

Part of the *celeste-evals* group (`scripts/celeste_evals/secret_choice/`, step 1 in
`olens-flow/src/secret_choice_gate.py`), built {bank["built"]}.

**Model-specific by construction.** The items are Qwen3.6-27B's own rollouts and the gate is its
own resampling behaviour; `wsbench capable` has no model-independent question. Re-run the three
scripts to port it.

## The gate (every kept site passes all of it)

1. **Committed.** From the site, {g["resamples_per_site"]} resamples at T=1 run to the reveal line; at least
   {g["commit_min"]} of {g["resamples_per_site"]} must reveal the rollout's own pick, and the prompt-level rate of that pick
   (over the prompt's 16 rollouts) must be <= {g["base_rate_max"]}, so a prompt whose favourite the model
   reveals every time does not count as commitment.
2. **No leak.** The text up to the site never contains the pick (any case, plural, listed alias
   such as "Japanese" for Japan), and none of {g["continuations_per_site"]} continuations sampled from the site
   (T=1, {g["continuation_cap"]}) names it.
3. **Not inferable.** `claude-sonnet-5` is shown the prompt and the text up to the site and asked
   for a probability over the six options (prompt below); the pick's probability must be
   <= {g["sonnet"]["p_max"]}.

**Reliability re-test.** Every committed, text-leak-free site also got a *fresh* set of 8
resamples with another seed. This set is never used to select: it measures how reliably the
first 8 identify a committed site. On the {rt["kept_sites"]["n"]} kept sites the fresh resamples
agree with the pick {rt["kept_sites"]["mean_agree"]:.2f} / 8 on average; {pct(rt["kept_sites"]["share_ge7"])} of kept sites
re-pass the >= 7/8 bar and {pct(rt["kept_sites"]["share_ge6"])} reach >= 6/8 (over all {rt["committed_no_text_leak_sites"]["n"]}
committed, text-leak-free sites: {rt["committed_no_text_leak_sites"]["mean_agree"]:.2f} / 8, {pct(rt["committed_no_text_leak_sites"]["share_ge7"])} re-pass).

**The gate keeps the quieter commitments.** How well Sonnet can read the pick from the text
tracks how strongly the model is committed: on the {rt["sonnet_dropped_sites"]["n"]} sites Sonnet
could infer (dropped), the fresh resamples agree {rt["sonnet_dropped_sites"]["mean_agree"]:.2f} / 8
({pct(rt["sonnet_dropped_sites"]["share_ge7"])} re-pass 7/8), against {rt["kept_sites"]["mean_agree"]:.2f} / 8 on the kept sites;
by Sonnet's probability for the pick: {bins}. The model commits *by* writing the option's
imagery, so the sites where the text gives nothing away are the ones where the commitment is
softer. The kept sites are still committed far above the base rate (a pick revealed at the
prompt's base rate would agree with about {8 * mean_base:.1f} / 8), but a reader should expect
`extras.pass_rate_retest_confirmed` (items with a site that re-passed 7/8, passing on such a
site) to be the cleaner number.

Funnel:

| stage | n |
|---|---|
{funnel_rows}

{n_fav} of the {len(items)} items reveal the prompt's favourite option (the modal reveal of its 16
rollouts, at a rate <= {g["base_rate_max"]}); the other {len(items) - n_fav} reveal a non-favourite option, which no
reader of the prompt alone can know. `extras.pass_rate_pick_not_favourite` is the clean number.
Kept sites: {n_sites} ({line_rows}); {spl}.

Items per category (`variant`):

| category | items |
|---|---|
{cat_rows}

## Bank

`items.json`: header (`family`, `model`, `built`, `render`, `wording`, `template`, `fillers`,
`option_sets`, `aliases`, `gate` with every sampling setting and the Sonnet prompt verbatim,
`funnel`, `retest`, `sites_per_line`, `items_per_category`, `n_items`) + **{len(items)} items**.
Per item: `name`, `variant` (= `category`), `options` (the six, prompt order), `pick`, `aliases`,
`favourite`, `pick_is_favourite`, `base_rate`, `dist` (reveal counts of the 16 rollouts),
`prompt_index`, `rollout`, `prompt`, `filler`, `rollout_text` (the whole rollout, reveal
included; the judge never sees it), `n_prompt_tokens`, `input_ids` (the captured render: chat
template, one user turn, thinking off, generation prompt, then the rollout's tokens up to the
last kept site), `read_sites` (`label` = `line_k`, `pos` = index into `input_ids`, `gen_pos`,
`char` = offset into `rollout_text` just after the newline, `agree` / `n_valid` of the 8
resamples, `retest_agree` of the fresh 8, `n_cont_leak`, the 16 `continuations`,
`sonnet_p_pick`, `sonnet_top`, `sonnet_evidence`, `resample_choices`, `retest_choices`).

## Example

`{ex["name"]}` ({ex["category"]}; options {", ".join(ex["options"])}; pick **{ex["pick"]}**, prompt-level rate {ex["base_rate"]:.2f}, favourite {ex["favourite"]}):

> {ex["rollout_text"][: s0["char"]].strip().replace(chr(10), " / ")}

Read site `{s0["label"]}`: {s0["agree"]}/8 resamples reveal {ex["pick"]} (re-test {s0["retest_agree"]}/8); Sonnet's probability for
{ex["pick"]} from the text: {s0["sonnet_p_pick"]:.2f} (its top guess: {s0["sonnet_top"]}).

## Scoring

- **Headline: pass rate.** One judge call per read cell shows the category, the six options in
  a per-cell seeded order (A-F) plus "G. cannot tell", and the readout (a prose lens's samples
  joined by newlines; a token lens's scored bag). The judge may pick an option only if the
  readout names it or unmistakably describes it, with a verbatim quote; a pick whose quote is
  not in the readout is `unverified` and never credited. A cell hits when the verified pick is
  the revealed option; an item passes when any read site (any judged layer) hits. An item with
  no hit and an unjudged or missing cell is undecided and leaves the denominator.
- **Extras:** `cell_hit_rate`, `cannot_tell_rate`, `decided_wrong_rate` (of the cells where the
  judge named an option, the share that was wrong: judge leniency plus lens error),
  `favourite_pick_rate` (of the named options, the share that is the prompt's favourite),
  `confusion` (hit / pick with an unverified quote / favourite-not-pick / other option / cannot
  tell), `uniform_guess_floor` (the item pass rate a uniform guesser would get on the same
  decided cells: mean of 1 - (5/6)^k), per-site and per-category rates, and the pass rate split
  by `pick_is_favourite`.
- Floors: 1/6 per decided cell under uniform guessing; measured: the **sampling baseline** (the
  model's own continuations from the read site, judged like any readout: the gate removed
  continuations that *name* the pick, so what remains is imagery the judge may or may not
  credit) and the **prompt-only baseline** (stock Qwen3.6-27B shown the render up to the read
  token and asked what a model would be thinking there; it can know the prompt's favourite but
  not the rollout's pick).

## Readouts

`{{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}}` rows, one per (item, layer, site),
for every `read_sites[].pos` (`wsbench plan families=celeste_secret_choice` lists them); `tokens` rows
for a token lens. `examples/readouts/celeste_secret_choice.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=celeste_secret_choice readouts=examples/readouts/celeste_secret_choice.jsonl out=outputs/toy/celeste_secret_choice dry_run=True
wsbench judge family=celeste_secret_choice readouts=<arm>.jsonl out=outputs/<arm>/celeste_secret_choice layers=42
```

## Judge prompts

`prompt_version`: `{PROMPT_VERSION}`. Judge: pinned to `claude-sonnet-5`, temperature 0, one call
per read cell, JSON schema `{{"choice": str (a letter A-G), "quote": str}}`.

System prompt:

```text
{SYSTEM}
```

User prompt (`{{category}}` = the category word; `{{options}}` = the six options lettered A-F in the
cell's seeded order, then `G. cannot tell`; `{{readout}}` = the cell's samples joined by newlines, or
a token bag):

```text
{USER}
```

## Gate prompt (claude-sonnet-5, step 3 of the gate; not a judge prompt)

System:

```text
{g["sonnet"]["system"]}
```

User (`{{prompt}}` = the instruction, `{{text}}` = the text up to the site, `{{options}}` = the six):

```text
{g["sonnet"]["user"]}
```
"""
(ROOT / "evals/celeste_secret_choice/README.md").write_text(md, encoding="utf-8")
print("wrote README", len(md))

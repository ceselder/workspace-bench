# celeste_secret_choice

**The model picks one of six options, pictures it while writing a short text without naming it,
and reveals it at the end. Read inside the text, where the model is demonstrably committed to a
pick that the text does not give away: does a lens name the pick?**

Each prompt lists six options of one category (13 categories: animal, fruit, instrument, country, gemstone, tool, vegetable, flower, sport, vehicle, profession, metal, color) and
tells Qwen3.6-27B (chat template, thinking off):

> Pick one particular <category> from these six: <the six options>. Really think about that one <category>: picture it vividly in your mind and keep it there, focused on it, the entire time you write, without ever writing its name. <filler>. Then, on a new line, write 'My <category>:' followed by the <category> you have been thinking about.

The filler is one of 6 tasks (a four-line poem about rain, the sea at night or time passing; four
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
scale, +0.4 (n 3975), +2.6 (n 3975), +4.7 (n 3969) points), and it
commits partly by writing the option's imagery into the text, which is why every site is gated
on leakage below. An anti-leak wording ("... but give nothing away: nothing you write before the last line may
hint at which <category> it is, not its name, not its features, not anything it is known for")
was piloted on 36 prompts as well: it removes the commitment altogether (21 committed sites of 1374, agreement minus base rate -4.0, -3.7, -0.5 points at lines 1-3), so committing
and leaking are one mechanism, and the gate below is what separates them.

Part of the *celeste-evals* group (`scripts/celeste_evals/secret_choice/`, step 1 in
`olens-flow/src/secret_choice_gate.py`), built 2026-10-22.

**Model-specific by construction.** The items are Qwen3.6-27B's own rollouts and the gate is its
own resampling behaviour; `wsbench capable` has no model-independent question. Re-run the three
scripts to port it.

## The gate (every kept site passes all of it)

1. **Committed.** From the site, 8 resamples at T=1 run to the reveal line; at least
   7 of 8 must reveal the rollout's own pick, and the prompt-level rate of that pick
   (over the prompt's 16 rollouts) must be <= 0.5, so a prompt whose favourite the model
   reveals every time does not count as commitment.
2. **No leak.** The text up to the site never contains the pick (any case, plural, listed alias
   such as "Japanese" for Japan), and none of 16 continuations sampled from the site
   (T=1, 24 new tokens, stop at the first newline) names it.
3. **Not inferable.** `claude-sonnet-5` is shown the prompt and the text up to the site and asked
   for a probability over the six options (prompt below); the pick's probability must be
   <= 0.3.

**Reliability re-test.** Every committed, text-leak-free site also got a *fresh* set of 8
resamples with another seed. This set is never used to select: it measures how reliably the
first 8 identify a committed site. On the 161 kept sites the fresh resamples
agree with the pick 5.79 / 8 on average; 40% of kept sites
re-pass the >= 7/8 bar and 62% reach >= 6/8 (over all 744
committed, text-leak-free sites: 7.14 / 8, 77% re-pass).

**The gate keeps the quieter commitments.** How well Sonnet can read the pick from the text
tracks how strongly the model is committed: on the 556 sites Sonnet
could infer (dropped), the fresh resamples agree 7.52 / 8
(88% re-pass 7/8), against 5.79 / 8 on the kept sites;
by Sonnet's probability for the pick: 0.0-0.2: 5.54 / 8 (n 89); 0.2-0.4: 6.39 / 8 (n 117); 0.4-0.6: 7.35 / 8 (n 232); 0.6-0.8: 7.60 / 8 (n 159); 0.8-1.0: 7.98 / 8 (n 120). The model commits *by* writing the option's
imagery, so the sites where the text gives nothing away are the ones where the commitment is
softer. The kept sites are still committed far above the base rate (a pick revealed at the
prompt's base rate would agree with about 3.0 / 8), but a reader should expect
`extras.pass_rate_retest_confirmed` (items with a site that re-passed 7/8, passing on such a
site) to be the cleaner number.

Funnel:

| stage | n |
|---|---|
| prompts (13 categories x 6 fillers, 6 options each) | 312 |
| rollouts (16 per prompt, T=1) | 4992 |
| rollouts with a parsed reveal line | 4950 |
| read sites (newlines ending filler lines 1-3) | 11934 |
| committed: >= 7 of 8 resamples reveal the rollout's pick, prompt-level rate of that pick <= 0.5 | 786 |
| the text so far never names the pick | 744 |
| none of 16 capped continuations names it | 724 |
| claude-sonnet-5 reading the text gives the pick <= 0.3 | 161 |
| items = (prompt, rollout) with >= 1 kept site | 134 |

83 of the 134 items reveal the prompt's favourite option (the modal reveal of its 16
rollouts, at a rate <= 0.5); the other 51 reveal a non-favourite option, which no
reader of the prompt alone can know. `extras.pass_rate_pick_not_favourite` is the clean number.
Kept sites: 161 (line_1 48, line_2 52, line_3 61); 114 items with 1 site, 13 items with 2 sites, 7 items with 3 sites.

Items per category (`variant`):

| category | items |
|---|---|
| animal | 10 |
| color | 4 |
| country | 23 |
| flower | 1 |
| fruit | 11 |
| gemstone | 6 |
| instrument | 15 |
| metal | 5 |
| profession | 13 |
| sport | 10 |
| tool | 16 |
| vegetable | 7 |
| vehicle | 13 |

## Bank

`items.json`: header (`family`, `model`, `built`, `render`, `wording`, `template`, `fillers`,
`option_sets`, `aliases`, `gate` with every sampling setting and the Sonnet prompt verbatim,
`funnel`, `retest`, `sites_per_line`, `items_per_category`, `n_items`) + **134 items**.
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

`sc-animal-0039-r11` (animal; options moose, lizard, salmon, flamingo, otter, jaguar; pick **otter**, prompt-level rate 0.25, favourite moose):

> Embrace the learning curve with patience and grace. / Build strong team bonds by listening more than speaking.

Read site `line_2`: 8/8 resamples reveal otter (re-test 5/8); Sonnet's probability for
otter from the text: 0.25 (its top guess: otter).

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

`{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}` rows, one per (item, layer, site),
for every `read_sites[].pos` (`wsbench plan families=celeste_secret_choice` lists them); `tokens` rows
for a token lens. `examples/readouts/celeste_secret_choice.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=celeste_secret_choice readouts=examples/readouts/celeste_secret_choice.jsonl out=outputs/toy/celeste_secret_choice dry_run=True
wsbench judge family=celeste_secret_choice readouts=<arm>.jsonl out=outputs/<arm>/celeste_secret_choice layers=42
```

## Judge prompts

`prompt_version`: `secret-choice-7way-v1-2026-10-22`. Judge: pinned to `claude-sonnet-5`, temperature 0, one call
per read cell, JSON schema `{"choice": str (a letter A-G), "quote": str}`.

System prompt:

```text
You grade the output of an interpretability tool (a "lens") that was applied to a
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

Return JSON: {"choice": "<letter>", "quote": "<verbatim quote or empty>"}.
```

User prompt (`{category}` = the category word; `{options}` = the six options lettered A-F in the
cell's seeded order, then `G. cannot tell`; `{readout}` = the cell's samples joined by newlines, or
a token bag):

```text
CATEGORY: {category}

OPTIONS:
{options}

LENS OUTPUT:
<<<
{readout}
>>>
```

## Gate prompt (claude-sonnet-5, step 3 of the gate; not a judge prompt)

System:

```text
You are a careful reader taking part in a study of whether a language model gives
away a secret in its writing. The model received the INSTRUCTION below, which told it to pick one
of six options, to keep that option in mind while writing a short text, and never to name it. You
see the model's TEXT SO FAR (cut off at the end of a line; the final reveal line has not been
written). From the instruction and the text alone, which option is the writer thinking of?

Give a probability for each of the six options. The probabilities must sum to 1. Use the text's
imagery, word choices, sounds, settings and associations as evidence; a reference that fits one
option much better than the other five deserves most of the mass. When the text gives nothing
away, spread the mass evenly (about 0.17 each). Do not assume the model picked the first option
listed or the most common one; judge the text.

Return JSON only: {"probabilities": {"<option>": p, ... all six ...}, "evidence": "<one short
sentence naming the words in the text that moved you, or 'none'>"}
```

User (`{prompt}` = the instruction, `{text}` = the text up to the site, `{options}` = the six):

```text
INSTRUCTION:
<<<
{prompt}
>>>

TEXT SO FAR:
<<<
{text}
>>>

OPTIONS: {options}
```

## Results (2026-10-22, layer 42, judge claude-sonnet-5)

Arms: NLA after RL (`av_rl_adapters/iter_000400`), the single-activation O-lens (`abl_r8_rl2/step_0150`, 8 bullets), the KV-context
O-lens (`kvr_cont_rl1/step_0300`, prompt masked), the J-lens at layer 42 (top-10 tokens), the sampling baseline (8 of the model's own
continuations from the read site, 24 tokens, stop at newline, as bullets) and the prompt-only baseline. Item = (prompt, rollout), pass =
some read site names the pick with a verified quote.

| arm | pass rate (95% CI) | n | pick not favourite | pick = favourite | re-test-confirmed | cell hit | cannot tell | wrong when named | names favourite | uniform floor |
|---|---|---|---|---|---|---|---|---|---|---|
| NLA (after RL) | **20.9%** [14.2%, 27.6%] | 134 | 21.6% (n 51) | 20.5% (n 83) | 38.6% (n 57) | 17.4% | 73.9% | 33.3% | 50.0% | 5.2% |
| olens (1 activation) | **11.3%** [6.8%, 17.3%] | 133 | 15.7% (n 51) | 8.5% (n 82) | 21.4% (n 56) | 9.4% | 82.5% | 46.4% | 35.7% | 3.5% |
| KV olens | **20.9%** [14.2%, 27.6%] | 134 | 19.6% (n 51) | 21.7% (n 83) | 31.6% (n 57) | 19.3% | 73.9% | 26.2% | 59.5% | 5.1% |
| J-lens L42 | **6.0%** [2.2%, 10.4%] | 134 | 2.0% (n 51) | 8.4% (n 83) | 12.3% (n 57) | 5.0% | 88.8% | 55.6% | 61.1% | 2.2% |
| continuation baseline | **49.3%** [41.8%, 58.2%] | 134 | 58.8% (n 51) | 43.4% (n 83) | 63.2% (n 57) | 41.6% | 49.7% | 17.3% | 48.1% | 10.0% |
| prompt-only baseline | **10.4%** [5.2%, 15.7%] | 134 | 11.8% (n 51) | 9.6% (n 83) | 17.5% (n 57) | 9.3% | 83.9% | 42.3% | 38.5% | 3.2% |

**The sampling baseline wins.** The model's own next line names or unmistakably describes the pick (to the judge's standard) at 42% of
the read cells and in 49% of items: the gate removed continuations that *name* the pick, not ones that describe it ("iron rails" for a
tram). No lens reads the held pick better than sampling the model does; NLA and the KV O-lens (21%) sit above the prompt-only floor
(10%) and the uniform-guess floor (5%), the single-activation O-lens ties prompt-only, the J-lens ties the floor.

**Continuation-silent cells** (post hoc; the sampling baseline's own judge call did not credit the pick there): 94 of
161 cells, 68 items made only of such cells. A judge-based continuation gate (as celeste_foreshadow's
step 5) would keep roughly this subset and make the sampling floor 0 by construction:

| arm | cell hit on silent cells | item pass, silent cells only | item pass on all-silent items (n) |
|---|---|---|---|
| sc_nla_rl400 | 12.8% | 14.3% | 13.2% (68) |
| sc_abl_r8_rl2_0150 | 4.3% | 4.8% | 4.4% (68) |
| sc_kvr_cont_rl1_0300 | 13.8% | 15.5% | 13.2% (68) |
| sc_jlens_l42 | 2.1% | 2.4% | 2.9% (68) |
| sc_ctx_unsteered | 0.0% | 0.0% | 0.0% (68) |
| sc_prompt_only | 7.4% | 8.3% | 7.4% (68) |

**Does the model stick to its pick?** On the 161 kept sites the fresh re-test agrees with the reveal
5.79 / 8 on average against 2.99 / 8 expected at the picks' prompt-level base rates;
89.4% of sites re-test above their base rate. Distribution: 8/8 15.5%,
>= 7/8 39.8%, >= 6/8 62.1%, >= 5/8 78.3%, >= 4/8 88.2%.
27.6% of fresh continuations from a kept site end by revealing another
option (the switch rate); the sampling baseline's next line was judged to describe another option at 8.7%
of kept cells.

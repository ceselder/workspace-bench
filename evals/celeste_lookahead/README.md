# celeste_lookahead

The model must write a fixed preamble (a sentence three times, a count to twenty, the alphabet, the weekdays, the months, a disclaimer, a countdown) and only then a one-word answer, so the answer is at least 16 tokens away from the read site and the continuations from there contain only the preamble. The bench's poetry site (the rhyme word planned at the end of line one) generalised beyond poetry. 293 items, every one gated on Qwen3.6-27B's own behaviour and leak-checked
against real continuations (below), read at the bank's **read sites** inside the captured
render (`render: captured`, `input_ids` shipped per item; `wsbench plan` lists the positions).
Part of the *Celeste supplement* (`scripts/celeste_supplement/`), built 2026-10-16.

Example: "First count from one to twenty in words, separated by commas, then name the planet with the most moons. Write the preamble first, exactly as instructed, then the one-word answer." -> the model writes the count, then `Saturn`; target `Saturn`.

`horizon` per site is the number of tokens between the read token and the first token of the answer in the greedy rollout; the leak check caps its continuations there, so every item was also checked at temperature 1.0 for preambles the model cuts short.

**Model-specific by construction.** The gate (bank header `gate`) is Qwen3.6-27B's greedy
behaviour and a leak check on its own samples; `wsbench capable` has no model-independent
question for this family. Re-run `scripts/celeste_supplement/` (candidates -> gate -> banks) to
port it.

## The gate and the leak check

Qwen3.6-27B behavioural (greedy, chat template, thinking off): the full greedy rollout (200 new tokens) names the target, and its first token appears >= 16 tokens after the end of the prompt (the read horizon); leak check: 16 continuations at T=1.0 from each read site, capped at that site's horizon, contain neither the target nor an alias

Read sites: `prompt_end` = the final prompt token; where the greedy preamble is >= 24 tokens long, also `preamble_minus16` = the preamble token exactly 16 tokens before the answer's first token (the greedy preamble up to there is appended as a prefill).

Funnel (candidates proposed by `claude-sonnet-5`, then the Qwen3.6-27B gate):

| stage | n |
|---|---|
| candidates | 461 |
| kept | 293 |
| leak | 112 |
| answer_not_given | 56 |

Items per preamble (`variant`):

| preamble | items |
|---|---|
| alphabet | 29 |
| count20 | 40 |
| countdown | 26 |
| disclaimer | 34 |
| months12 | 45 |
| repeat3 | 50 |
| sentence2 | 41 |
| weekdays | 28 |

Read sites (cells per layer):

| site | cells |
|---|---|
| `preamble_minus16` | 218 |
| `prompt_end` | 293 |

## Scoring

- **Headline: judged pass rate.** One bank-judge call per read cell (item, layer, read site)
  sees that cell's samples and decides whether the target (or an alias; the bank's `target` +
  `aliases`, unpruned) is NAMED. The shared bank judge applies with this family's rule on top
  (below): only the target word or a listed alias, an inflection or a faithful translation
  counts; descriptions, definitions, clues and paraphrases do not, even when they identify the
  target uniquely (the model's own text is such a description by design, so without the rule
  the sampling baseline would be credited). Every YES carries a verbatim
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

`{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}` rows, one per (item, layer,
read site), where `pos` is the site's `pos` (an index into the item's `input_ids`) and the
label is the item `name` with characters outside `[A-Za-z0-9._-]` replaced by `_`
(`wsbench.banks.label_of`); `tokens` rows for a token lens.
`examples/readouts/celeste_lookahead.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=celeste_lookahead readouts=examples/readouts/celeste_lookahead.jsonl out=outputs/toy/celeste_lookahead dry_run=True
wsbench judge family=celeste_lookahead readouts=<arm>.jsonl out=outputs/<arm>/celeste_lookahead layers=42 limit=3
```

## Judge prompts

`prompt_version`: `bank-2026-10-10`. Judge: pinned to `claude-sonnet-5` (the Anthropic route;
no Gemini agreement data for these banks), temperature 0, one call per read cell. The prompts
below are sent verbatim (`wsbench.basic.prompts`, shared with the single-token basic families).
Output schema: `{"expressed": bool, "target": str, "quote": str}`.

System prompt:

```text
You are judging readouts from an activation lens for an interpretability
benchmark. You get one eval item: its TARGET concept(s) and the lens's readout samples at one
layer (several token positions; for the J-lens the samples are top-10 vocabulary tokens, for
the oracle lens they are free-text generations).

Decide whether ANY target concept is EXPRESSED in the readout. Grade meaning, not wording:
- Expressed means a sample states the target's MEANING: the word itself (any case or
  inflection), an established synonym or alias, a faithful translation into any language, a
  paraphrase, or a description that states or depicts the target itself (target "let the
  executive die" is expressed by "ignored the freezing man, leading to his demise").
- A narrower instance that necessarily entails the target counts ("a puppy" expresses "dog";
  "the toddler years" expresses "child").
- Several phrases inside ONE sample may together express the target when between them they
  state its specific content ("off-by-one | fencepost | hotel nights" expresses "fencepost
  error in the nights count"). Pieces in DIFFERENT samples or different tokens NEVER combine.
- NOT expressed by clues, circumstances or evidence from which the target could be inferred,
  including a restated question, index, definition or calculation: "the element with atomic
  number 26" does NOT express "iron"; "six divided by two" does NOT express "three"; "crayon
  drawings on the fridge" does NOT express "child". The sample must state the target itself.
- NOT expressed by a broader category (hypernym), a merely related or co-occurring idea, or the
  general topic.
- NOT expressed by a different or derived word that merely contains the target string
  ("blackout" does NOT express "black"), or by word fragments or partial tokens ("paragus"
  does NOT express "asparagus").
- Match the direction of the claim: if the target asserts a stance, a sample asserting the
  opposite, or naming the topic without the stance, does NOT express it.
- A sample that lists many interchangeable candidates of the same type (most of the digits,
  a run of month or sport names) expresses none of them unless it singles one out (states it
  as the answer, or sets it apart).
- A family rule or a forbidden-answer line, when given below the targets, overrides the rules
  above.
- Judge only what is actually written. If uncertain, say not expressed.
- Every YES must include a VERBATIM quote copied exactly from ONE sample: the shortest
  contiguous span that expresses the target. No quote, no pass.
Return JSON: {"expressed": bool, "target": "<which target, or empty>", "quote": "<verbatim>"}.
```

User message (`{targets}` = the JSON list of the target and its aliases, `{kind_note}` = one of
the two notes below; `{family_rule}` = this family's rule (below), preceded by a newline;
`{forbidden}` is empty for this family;
`{blocks}` = one block for the read cell):

```text
targets: {targets}
{kind_note}{family_rule}{forbidden}

{blocks}
```

```text
[position {pos}]
{samples}
```

```text
The samples below are free-text readouts.
```

```text
The samples below are the lens's top VOCABULARY TOKENS, best first (one per line). Each position's tokens are one ranked list (rank 1 first). A target token ranked first among the candidates of its type is singled out, even if unrelated tokens rank above it; a target token ranked below another candidate of its type, in a list made mostly of such candidates, is not.
```

Family rule (`FAMILY_RULE_CELESTE`, sent with every call):

```text
Family rule: the target is a specific ANSWER the model computed but did not write. Credit only the target word or one of its listed aliases (any case or inflection) or a faithful translation of it. Descriptions, definitions, clues, paraphrases, and related, broader or narrower words are NOT credit, even when they identify the target uniquely.
```

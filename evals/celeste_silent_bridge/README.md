# celeste_silent_bridge

A novel two-hop question asked with the instruction **"ANSWER IN ONE WORD, DO NOT SAY ANYTHING
ELSE."**: Qwen3.6-27B answers it correctly in one word and never names the bridge entity it had to
work out on the way. The clue and the property come from unrelated domains, so the question's words
and its answer should not co-occur anywhere the model could have memorised them, and the gate checks
that the answer is out of reach once the clue is made generic. Does a lens name the bridge where
the reply opens, or at the last prompt token? 111 items, read at the bank's **read sites** inside the
captured render (`render: captured`, `input_ids` shipped per item; `wsbench plan` lists the
positions). Built 2026-10-23 (`scripts/celeste_evals/silent_bridge/`).

Example: "Which composer was born in the home country of footballer Robert Lewandowski?" -> the model answers `Chopin`; target (the
bridge) `Poland`.

**Why this family.** In raw "Fact: ... is" text the model says the bridge in its own continuation
(1,378 of 1,389 two-hop pairs within 32 samples), which is why plain continuation passes about half
of `multihop`. Under a one-word chat answer it never does, so the sampling baseline is ~0 by
construction and a lens that only continues the text cannot pass.

**Model-specific by construction.** The gate is Qwen3.6-27B's own behaviour; `wsbench capable` has
no model-independent question for this family. Re-run `scripts/celeste_evals/silent_bridge/` to port
it.

## The gate

Qwen3.6-27B, chat template, thinking off, every question with the all-caps one-word instruction: (i) correct: greedy hit AND >= 16/20 at T=0.7; (ii) recall: the bridge from its own clue and the last hop with the bridge named, greedy hit AND >= 8/10 each; (iii) no shortcut: the question with the WHOLE clue replaced by a generic noun ('a certain composer') must miss (greedy miss AND <= 4/20); (iv) novelty: the auditor rates co-occurrence of clue and answer in ordinary text 'low'; (v) silence: 32 answers at T=1.0 never name the bridge or an alias, and no answer the model gave names it by a derived form (demonym, adjective, abbreviation: 'Norwegian Krone', 'Swiss'; claude-sonnet-5 check)

Read sites: `reply_start` = the <|im_start|> that opens the reply (the deployed path from it is template tokens, then the answer); `prompt_end` = the last prompt token (after the closed think block)

**Free sampling is a documented limit.** The leak check above follows the model's own path: from
`reply_start` the chat template writes `assistant\n<think>\n\n</think>\n\n` before the answer, so
the continuation is the answer. If instead the continuation from `reply_start` is sampled freely
(the model may write its own `<think>`, which it never does under its template), it opens a
thinking block and says the bridge on most items: 32 free continuations of 256 tokens from `reply_start` never name the bridge on 0% of items; from `prompt_end`, where the template has already closed thinking, on 96% (per item: `diagnostics.free_sampling_256_mentions_of_32`).

Funnel (candidates proposed by `claude-sonnet-5`, audited by an independent `claude-sonnet-5`
call, then the Qwen3.6-27B gate):

| stage | n |
|---|---|
| candidates proposed | 3577 |
| answered right (greedy and >= 16/20 at T=0.7) | 1640 |
| bridge recalled from its own clue (>= 8/10) | 2636 |
| last hop recalled with the bridge named (>= 8/10) | 2590 |
| missed with the clue made generic (greedy miss, <= 4/20) | 2202 |
| bridge never named in 32 answers at T=1.0 | 3315 |
| all five of the above | 457 |
| and clue-answer co-occurrence rated low | 146 |
| and no answer names the bridge by a derived form | 136 |
| kept (<= 3 items per bridge, <= 6 per answer) | 111 |

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
  (the J-lens has it in its top 20 there on 68% of items, at
  `prompt_end` on 13%); `prompt_end` is where the activation has
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

## Reference results (2026-10-23, layer 42, this instrument)

| arm | items passed | via `reply_start` | via `prompt_end` |
|---|---|---|---|
| olens | 91% | 86% | 53% |
| NLA | 91% | 84% | 46% |
| KV olens | 89% | 82% | 55% |
| steering vector | 89% | 88% | 61% |
| J-lens | 63% | 62% | 11% |
| prompt-only | 90% | 65% | 79% |
| plain continuation | 2% | 2% | 0% |

Arms, all reading the block-42 residual of Qwen3.6-27B: olens = single-activation oracle lens (8 samples, <= 160 tokens); KV olens = oracle lens that also reads the prompt's keys and values; NLA = natural-language autoencoder; steering vector = in-context steering-vector lens; J-lens = the Jacobian lens's top 10 tokens; prompt-only = stock Qwen3.6-27B shown the cut-off render as plain text (special tokens escaped) and asked what the model is thinking there; plain continuation = 8 continuations of up to 24 tokens from the read site. Two-hop bridges are inferable from the question, so the prompt-only floor is high (the bench's `multihop` floor is 0.91); it is lower at `reply_start` and higher at `prompt_end`, the reverse of the lenses.

## Readouts

`{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}` rows, one per (item, layer, read
site), where `pos` is the site's `pos` (an index into the item's `input_ids`) and the label is the
item `name` with characters outside `[A-Za-z0-9._-]` replaced by `_` (`wsbench.banks.label_of`);
`tokens` rows for a token lens. `examples/readouts/celeste_silent_bridge.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=celeste_silent_bridge readouts=examples/readouts/celeste_silent_bridge.jsonl out=outputs/toy/celeste_silent_bridge dry_run=True
wsbench judge family=celeste_silent_bridge readouts=<arm>.jsonl out=outputs/<arm>/celeste_silent_bridge layers=42 limit=3
```

## Judge prompts

`prompt_version`: `bank-2026-10-10`. Judge: pinned to `claude-sonnet-5`, temperature 0, one
call per read cell. The prompts below are sent verbatim (`wsbench.basic.prompts`, shared with the
single-token basic families). Output schema: `{"expressed": bool, "target": str, "quote": str}`.

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

User message (`{targets}` = the JSON list of the target and its aliases, `{kind_note}` = one of the
two notes below; `{family_rule}` = the family rule (below), preceded by a newline; `{forbidden}` is
empty for this family; `{blocks}` = one block for the read cell):

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

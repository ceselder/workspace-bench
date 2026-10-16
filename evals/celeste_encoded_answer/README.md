# celeste_encoded_answer

The model computes an answer but must write it transformed: a question Qwen3.6-27B answers correctly when asked plainly is asked again with an encoding instruction whose output cannot contain the entity (letter count, first or last letter, vowel count, position in a given five-entry list, alphabet index of the first letter, odd/even letter count). The scored latent is the entity ("Canberra" when the model writes "8"). 200 items, every one gated on Qwen3.6-27B's own behaviour and leak-checked
against real continuations (below), read at the bank's **read sites** inside the captured
render (`render: captured`, `input_ids` shipped per item; `wsbench plan` lists the positions).
Part of the *Celeste supplement* (`scripts/celeste_supplement/`), built 2026-10-16.

Example: "Which city is the capital of Australia, despite not being its largest city? Reply with only the number of letters in the answer, as a digit." -> the model writes `8`; target `Canberra`.

`list_position` is the one encoding whose prompt contains the target string (inside the list); its items are kept but tagged, read them apart from the rest.

**Model-specific by construction.** The gate (bank header `gate`) is Qwen3.6-27B's greedy
behaviour and a leak check on its own samples; `wsbench capable` has no model-independent
question for this family. Re-run `scripts/celeste_supplement/` (candidates -> gate -> banks) to
port it.

## The gate and the leak check

Qwen3.6-27B behavioural (greedy, chat template, thinking off): (i) asked the plain question it names the target (word-boundary match on target or alias); (ii) asked the encoded question its output is the correct encoding and never contains the target; (iii) leak check: 16 continuations at T=1.0 (32 new tokens) from EACH read site contain neither the target nor an alias (substring after folding)

Read sites: two sites in the captured render: `prompt_end` = the final prompt token (the assistant onset after the empty think block) and `answer_end` = the last token of the model's own greedy encoded output, appended as a prefill.

Funnel (candidates proposed by `claude-sonnet-5`, then the Qwen3.6-27B gate):

| stage | n |
|---|---|
| candidates | 462 |
| kept | 200 |
| encoded_output_wrong | 215 |
| plain_answer_wrong | 43 |
| leak | 4 |

Items per encoding (`variant`):

| encoding | items |
|---|---|
| first_letter | 46 |
| first_letter_index | 15 |
| last_letter | 33 |
| letter_count | 13 |
| letter_parity | 27 |
| list_position | 58 |
| vowel_count | 8 |

Read sites (cells per layer):

| site | cells |
|---|---|
| `answer_end` | 200 |
| `prompt_end` | 200 |

## Scoring

- **Headline: judged pass rate.** One bank-judge call per read cell (item, layer, read site)
  sees that cell's samples and decides whether the target (or an alias; the bank's `target` +
  `aliases`, unpruned) is NAMED. The shared bank judge applies with this family's rule on top
  (below): only the target word or a listed alias, an inflection or a faithful translation
  counts; descriptions, definitions, clues and paraphrases do not, even when they identify the
  target uniquely (the model's own text is such a description by design, so without the rule
  the sampling baseline would be credited). The model's written encoded output is sent as the forbidden answer: a sample that names only it ("8") does not count. Every YES carries a verbatim
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
`examples/readouts/celeste_encoded_answer.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=celeste_encoded_answer readouts=examples/readouts/celeste_encoded_answer.jsonl out=outputs/toy/celeste_encoded_answer dry_run=True
wsbench judge family=celeste_encoded_answer readouts=<arm>.jsonl out=outputs/<arm>/celeste_encoded_answer layers=42 limit=3
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
`{forbidden}` is the forbidden-answer line below, preceded by a newline;
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

Forbidden-answer line (`FORBIDDEN_TEMPLATE`, `{answer}` = the item's written encoded output, JSON-quoted):

```text
The input's final answer {answer} is NOT a target: a sample that names only {answer} does not count. A sample that names a target still counts, even beside {answer}.
```

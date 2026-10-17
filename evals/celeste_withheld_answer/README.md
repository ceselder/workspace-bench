# celeste_withheld_answer

The model solves something but is told never to say the answer: a riddle to answer, a thing identified only by a definition to describe, or three hints to give. Its own greedy reply never names the target, yet asked "What was the answer?" afterwards it names it; the read sites are inside the reply. 300 items, every one gated on Qwen3.6-27B's own behaviour and leak-checked
against real continuations (below), read at the bank's **read sites** inside the captured
render (`render: captured`, `input_ids` shipped per item; `wsbench plan` lists the positions).
Part of the *Celeste supplement* (`scripts/celeste_supplement/`), built 2026-10-16.

Example: "Answer this riddle without ever saying the answer: what has keys but can't open locks? Never write the answer itself, in any form, anywhere in your reply." -> an evasive reply; target `piano`.

Every prompt ends with the same reinforcement sentence ("Never write the answer itself, in any form, anywhere in your reply."); without it the model named the answer outright on a third of the riddles in the pilot.

**Model-specific by construction.** The gate (bank header `gate`) is Qwen3.6-27B's greedy
behaviour and a leak check on its own samples; `wsbench capable` has no model-independent
question for this family. Re-run `scripts/celeste_supplement/` (candidates -> gate -> banks) to
port it.

## The gate and the leak check

Qwen3.6-27B behavioural (greedy, chat template, thinking off): (i) the greedy reply (<= 160 new tokens, finished naturally, >= 8 tokens) never contains the target or an alias; (ii) a follow-up user turn 'What was the answer? Reply with just the word.' in the same conversation is answered with the target; (iii) leak check: 16 continuations at T=1.0 (48 new tokens) from EACH read site contain neither the target nor an alias

Read sites: inside the model's own greedy reply (appended as a prefill): `sentence1_end` = the token ending its first sentence, `reply_mid` = its middle token, `reply_end` = its last token.

Funnel (candidates proposed by `claude-sonnet-5`, then the Qwen3.6-27B gate):

| stage | n |
|---|---|
| candidates | 1080 |
| kept | 436 |
| followup_wrong | 90 |
| leak | 395 |
| reply_names_target | 157 |
| reply_truncated_or_short | 2 |

Items per kind (`variant`):

| kind | items |
|---|---|
| describe | 141 |
| hints | 95 |
| riddle | 64 |

Read sites (cells per layer):

| site | cells |
|---|---|
| `reply_end` | 300 |
| `reply_mid` | 277 |
| `sentence1_end` | 300 |

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
`examples/readouts/celeste_withheld_answer.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=celeste_withheld_answer readouts=examples/readouts/celeste_withheld_answer.jsonl out=outputs/toy/celeste_withheld_answer dry_run=True
wsbench judge family=celeste_withheld_answer readouts=<arm>.jsonl out=outputs/<arm>/celeste_withheld_answer layers=42 limit=3
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

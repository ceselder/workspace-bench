# celeste_jlens_intermediates

A question answered with **"ANSWER IN ONE WORD, DO NOT SAY ANYTHING ELSE."** forces Qwen3.6-27B
through a famous **multi-word** entity (a game, film, painting, law, landmark, event, institution)
that it never writes. The read site is a prompt token where the J-lens surfaces fragments of that
entity, words of its name or things unique to it that are not in the text (no text inversion),
and from which the model's own continuations never name it. Does a lens name the whole entity?
149 items, read at the bank's **read site** inside the captured render (`render: captured`,
`input_ids` shipped per item; `wsbench plan` lists the positions). Built 2026-10-23
(`scripts/celeste_evals/jlens_intermediates/`).

Example: "Think of the three 17th-century rules describing how planets trace ellipses around the sun with varying speed. What nationality was the astronomer who formulated them?" -> the model answers `German`; at
`'<|im_start|>'` the J-lens top 10 is `Kepler, Johannes, 定律, Johann, Rudolf, 奥地利, 德国的, 德国, Friedrich, قوانين`; target
`Kepler's Laws of Planetary Motion`.

**Why this family.** `jlens_concept_pr` grades a prose lens against the J-lens top 10 at random
on-policy positions, where most J-lens tokens are copied from the text and none is the concept
behind a cluster of them. Here the J-lens is only the *locator*: it picks a token where the model
demonstrably holds the entity, and the lens must say the entity's full name, which a top-10 list of
single tokens rarely can.

**What the locator found.** At the J-lens's best-scoring token the model's own continuation names
the entity within 48 tokens for most items (it is about to write it), so those sites are useless
for a reading eval; every candidate site was leak-checked and the best silent one kept. Silent
sites are mostly the chat template's tokens and the instruction:

| read token | items |
|---|---|
| `'<\|im_start\|>'` | 76 |
| `'\n\n'` | 7 |
| `' NOT'` | 6 |
| `'\n'` | 5 |
| `'<\|im_end\|>'` | 4 |
| `' League'` | 4 |
| `' a'` | 3 |
| `' the'` | 2 |

**Model-specific by construction.** The read sites are Qwen3.6-27B's J-lens and its own samples;
`wsbench capable` has no model-independent question for this family. Re-run
`scripts/celeste_evals/jlens_intermediates/` to port it.

## The gate

Qwen3.6-27B, chat, thinking off, all-caps one-word instruction: correct (greedy hit and >= 16/20 at T=0.7); the bridge recalled from its clue (asked 'ANSWER WITH THE NAME ONLY') and the last hop with the bridge named (>= 8/10 each); 32 answers at T=1.0 never name the bridge; no answer names it by a derived form (claude-sonnet-5 check). The generic-clue shortcut probe is recorded (gate.generic_probe_reached) but not required: the read site's J-lens evidence is what shows the model holds the bridge

Read site: `jlens_site` = the prompt token where the J-lens (neuronpedia n=1000, layer 42, top 20) ranks the most signature words of the bridge that are NOT in the text up to that token (one-token, rare words only, zipf < 4.5, and no inflection of a word in the text; at least one NAME word; a name word counts double), the found words judged specific to the bridge rather than the text's topic (claude-sonnet-5); 16 free continuations of 48 tokens from it never name the bridge

Funnel (candidates proposed by `claude-sonnet-5`, then the Qwen3.6-27B gate and the J-lens):

| stage | n |
|---|---|
| candidates proposed | 2231 |
| answered right, both hops recalled, the bridge never named in 32 answers | 828 |
| and the bridge's name is two or more tokens | 809 |
| and some prompt token's J-lens top 20 holds a rare NAME word of the bridge that is not in the text, from which 16 free continuations never name it | 201 |
| and the found words point to the bridge, not the text's topic | 156 |
| and no answer names the bridge by a derived form | 155 |
| kept (<= 2 items per bridge) | 149 |

Items per category (`variant`):

| category | items |
|---|---|
| institutions and organisations | 15 |
| historical events and missions | 15 |
| video games | 14 |
| companies and products | 14 |
| sports teams and tournaments | 13 |
| scientific theories and laws | 13 |
| fictional characters | 10 |
| mythological stories | 9 |
| films | 9 |
| landmarks and buildings | 8 |
| paintings and artworks | 8 |
| novels and book series | 7 |
| board and card games | 7 |
| anime and comics | 3 |
| internet phenomena and memes | 2 |
| TV series | 2 |

## Scoring

- **Headline: judged pass rate.** One bank-judge call per read cell (item, layer, site) sees that
  cell's samples and decides whether the target (the entity's full name, or an alias: full-name
  variants and acronyms only, never a fragment) is NAMED. The Celeste family rule applies (below):
  descriptions, clues and paraphrases do not count. Every YES carries a verbatim quote verified
  against one sample of that cell; an unverifiable quote voids the verdict.
- No analytic chance floor (free text). Measured floors: the **sampling baseline** (~0 by
  construction: the leak check) and the **prompt-only baseline** (stock Qwen3.6-27B shown the
  render up to the read token as plain text and asked what a model would be thinking there).
- The J-lens is the locator, so its own readout at the site is not a fair arm on this family.
- Missing cells are fatal (exit 2) unless `allow_missing=True`; a dry run only reports them.

## Reference results (2026-10-23, layer 42, this instrument)

| arm | items passed | 95% CI |
|---|---|---|
| olens | 74% | 67%-81% |
| KV olens | 65% | 58%-73% |
| NLA | 62% | 55%-70% |
| steering vector | 52% | 44%-60% |
| prompt-only | 41% | 33%-49% |
| J-lens (locator) | 25% | 18%-32% |
| plain continuation | 3% | 1%-6% |

Arms, all reading the block-42 residual of Qwen3.6-27B: olens = single-activation oracle lens (8 samples, <= 160 tokens); KV olens = oracle lens that also reads the prompt's keys and values; NLA = natural-language autoencoder; steering vector = in-context steering-vector lens; prompt-only = stock Qwen3.6-27B shown the cut-off render as plain text (special tokens escaped) and asked what the model is thinking there; J-lens = the Jacobian lens's top 10 tokens, which LOCATED the sites and is therefore not a fair arm (it passes only where the entity's everyday name is one token: Skyrim, WoW, 奥运会); plain continuation = 8 continuations of up to 24 tokens from the read site.

## Readouts

`{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}` rows, one per (item, layer, read
site), where `pos` is the site's `pos` (an index into the item's `input_ids`) and the label is the
item `name` with characters outside `[A-Za-z0-9._-]` replaced by `_` (`wsbench.banks.label_of`);
`tokens` rows for a token lens. `examples/readouts/celeste_jlens_intermediates.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=celeste_jlens_intermediates readouts=examples/readouts/celeste_jlens_intermediates.jsonl out=outputs/toy/celeste_jlens_intermediates dry_run=True
wsbench judge family=celeste_jlens_intermediates readouts=<arm>.jsonl out=outputs/<arm>/celeste_jlens_intermediates layers=42 limit=3
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

User message (`{targets}` = the JSON list of the target and its aliases, `{kind_note}` = one of
the two notes below; `{family_rule}` = the family rule (below), preceded by a newline;
`{forbidden}` is empty for this family; `{blocks}` = one block for the read cell):

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

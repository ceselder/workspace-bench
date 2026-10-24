# celeste_jlens_ambiguous

The J-lens readout is ambiguous: it catches the intermediate but cannot say it. A question
answered with **"ANSWER IN ONE WORD, DO NOT SAY ANYTHING ELSE."** forces Qwen3.6-27B through a
famous **multi-word** entity (a game, film, law, myth, mission, institution) that it never writes.
At the read site, a punctuation, newline or chat-template turn token (never a word piece), the
J-lens top 10 holds fragments of that entity (words of its name that are not in the text) yet
does not name it, the way an image of Bonnie gives the J-lens `night`, `nights`, `anim`,
`Minecraft` and never "Five Nights at Freddy's". The model's own continuations from the site never
name it either. Does a lens say the whole entity?
153 items, read at the bank's **read site** inside the captured render (`render: captured`,
`input_ids` shipped per item; `wsbench plan` lists the positions). Built 2026-10-23
(`scripts/celeste_evals/jlens_intermediates/`).

Example: "In a tabletop role-playing game where a game master guides players through fantasy adventures using polyhedral dice and character sheets, what year was it first published?" -> the model answers `1974`; at
`'\n'` the J-lens top 10 is `Dungeons, pioneered, RPG, Dungeon, 最早, 最早的, 开创, orig, 原创, origin`; target
`Dungeons & Dragons`.

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
| `'<\|im_start\|>'` | 101 |
| `'\n'` | 35 |
| `'\n\n'` | 9 |
| `'<\|im_end\|>'` | 4 |
| `','` | 3 |
| `"'"` | 1 |

**Model-specific by construction.** The read sites are Qwen3.6-27B's J-lens and its own samples;
`wsbench capable` has no model-independent question for this family. Re-run
`scripts/celeste_evals/jlens_intermediates/` to port it.

## The gate

Qwen3.6-27B, chat, thinking off, all-caps one-word instruction: correct (greedy hit and >= 16/20 at T=0.7); the bridge recalled from its clue (asked 'ANSWER WITH THE NAME ONLY') and the last hop with the bridge named (>= 8/10 each); 32 answers at T=1.0 never name the bridge; no answer names it by a derived form (claude-sonnet-5 check). The generic-clue shortcut probe is recorded (gate.generic_probe_reached) but not required: the read site's J-lens evidence is what shows the model holds the bridge

Read site: `jlens_site` = a BOUNDARY token (punctuation or newline, else the chat-template turn token <|im_start|> / <|im_end|>; never a word or mid-word piece; `site_class`) where the J-lens (neuronpedia n=1000, layer 42) catches the bridge without naming it: its top 20 holds a rare, one-token NAME word of the bridge that is not in the text so far (zipf < 4.5, no inflection of a text word), at least one found word is in its top 10, the found words are judged specific to the bridge rather than the text's topic (claude-sonnet-5), and its top-10 readout there is judged NOT to name the bridge by this family's judge (fragment rule) in two independent calls; 16 free continuations of 48 tokens from it never name the bridge; a punctuation/newline site is preferred over a turn token

Funnel (candidates proposed by `claude-sonnet-5`, then the Qwen3.6-27B gate and the J-lens):

| stage | n |
|---|---|
| candidates proposed | 5609 |
| answered right, both hops recalled, the bridge never named in 32 answers | 1914 |
| and the bridge's name is two or more tokens | 1881 |
| and some prompt token's J-lens top 20 holds a rare NAME word of the bridge that is not in the text, from which 16 free continuations never name it | 247 |
| and the found words point to the bridge, not the text's topic | 210 |
| and at least one found word is in the J-lens top 10 | 210 |
| and two independent judge calls: its top 10 does NOT name it | 175 |
| and no answer names the bridge by a derived form | 175 |
| and an independent claude-opus-5 review keeps the item | 175 |
| kept (<= 2 items per bridge) | 153 |

Items per category (`variant`):

| category | items |
|---|---|
| historical events and missions | 24 |
| sports teams and tournaments | 18 |
| institutions and organisations | 18 |
| landmarks and buildings | 16 |
| companies and products | 15 |
| video games | 10 |
| paintings and artworks | 9 |
| fictional characters | 8 |
| scientific theories and laws | 7 |
| novels and book series | 5 |
| board and card games | 5 |
| mythological stories | 5 |
| anime and comics | 5 |
| TV series | 3 |
| films | 3 |
| internet phenomena and memes | 1 |
| bands and musicians | 1 |

## Scoring

- **Headline: judged pass rate.** One bank-judge call per read cell (item, layer, site) sees that
  cell's samples and decides whether the target (the entity's full name, or an alias: full-name
  variants and acronyms only, never a fragment) is NAMED. This family's rule applies (below):
  fragments, descriptions, clues and paraphrases do not count. Every YES carries a verbatim quote
  verified against one sample of that cell; an unverifiable quote voids the verdict.
- No analytic chance floor (free text). Measured floors: the **sampling baseline** (~0 by
  construction: the leak check) and the **prompt-only baseline** (stock Qwen3.6-27B shown the
  render up to the read token as plain text and asked what a model would be thinking there).
- The J-lens's own top-10 readout at the site fails every item by construction (its fragments do
  not name the entity); it is the locator, not an arm.
- Missing cells are fatal (exit 2) unless `allow_missing=True`; a dry run only reports them.

## Reference results (2026-10-24, layer 42, this instrument)

| arm | items passed | 95% CI | at punctuation/newline sites (48) | at turn-token sites (105) |
|---|---|---|---|---|
| olens | 80% | 73%-86% | 50% | 93% |
| NLA | 80% | 74%-86% | 62% | 89% |
| KV olens | 71% | 64%-78% | 35% | 88% |
| steering vector | 65% | 58%-73% | 33% | 80% |
| prompt-only | 58% | 50%-66% | 56% | 59% |
| J-lens (top 10) | 0% | 0%-0% | 0% | 0% |
| plain continuation | 0% | 0%-0% | 0% | 0% |

Arms, all reading the block-42 residual of Qwen3.6-27B: olens = single-activation oracle lens (8 samples, <= 160 tokens); NLA = natural-language autoencoder; KV olens = oracle lens that also reads the prompt's keys and values; steering vector = in-context steering-vector lens; prompt-only = stock Qwen3.6-27B shown the cut-off render as plain text (special tokens escaped) and asked what the model is thinking there; J-lens = its top-10 tokens (0 by construction); plain continuation = 8 continuations of up to 24 tokens from the read site. At punctuation sites the lenses are weaker and the text-only reader is level with them; the lens advantage is at the turn token. **Run-to-run:** a second full run (fresh sampling seeds, recaptured activations, fresh judge calls) gives olens 78%, NLA 82%, KV olens 75%, steering vector 67%, prompt-only 65%, J-lens (top 10) 0%, plain continuation 1%; every arm within its 95% CI of run 1, same verdict on 72% (prompt-only, one sample) to 95-100% of items.

## Readouts

`{"id": "<item label>", "layer": L, "pos": P, "samples": [...]}` rows, one per (item, layer, read
site), where `pos` is the site's `pos` (an index into the item's `input_ids`) and the label is the
item `name` with characters outside `[A-Za-z0-9._-]` replaced by `_` (`wsbench.banks.label_of`);
`tokens` rows for a token lens. `examples/readouts/celeste_jlens_ambiguous.jsonl` is a toy file for `dry_run=True`.

```
wsbench judge family=celeste_jlens_ambiguous readouts=examples/readouts/celeste_jlens_ambiguous.jsonl out=outputs/toy/celeste_jlens_ambiguous dry_run=True
wsbench judge family=celeste_jlens_ambiguous readouts=<arm>.jsonl out=outputs/<arm>/celeste_jlens_ambiguous layers=42 limit=3
```

## Judge prompts

`prompt_version`: `bank-2026-10-10+jla-fragments-2026-10-24` (the shared bank judge, `bank-2026-10-10`, with this
family's own rule). Judge: pinned to `claude-sonnet-5`, temperature 0, one call per read cell.
The prompts below are sent verbatim (`wsbench.basic.prompts`, shared with the single-token basic
families, plus the family rule). Output schema:
`{"expressed": bool, "target": str, "quote": str}`.

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

Family rule (`FAMILY_RULE_JLA`, sent with every call). It replaces the Celeste rule because a
multi-word target makes fragments a live failure: a lenient judge credited a J-lens token like
`Dungeons` for "Dungeons & Dragons" on about one item in eight.

```text
Family rule: the target is a specific MULTI-WORD named entity the model thought about but did not write. Credit only its full name, one of its listed aliases (any case or inflection) or a faithful translation of the full name. A single word or other fragment of the name (for "Dungeons & Dragons": "Dungeons"; for "Apollo 11": "Apollo"; for "Treaty of Versailles": "Versailles") is NOT credit unless that word is itself a listed alias. Descriptions, definitions, clues, paraphrases, and related, broader or narrower names are NOT credit, even when they identify the target uniquely.
```

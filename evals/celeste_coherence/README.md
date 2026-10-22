# celeste_coherence

**Is a lens's prose readable text that conveys a claim?** Not whether the claim is true or
relevant: whether a reader can read it. RL'd verbalizers drift into repeated phrases, token
debris and script soup while staying "close" to their reward; this instrument counts that.

The cells are other families' cells: **50 items (seed 0) from each of 8 WorkspaceBench families**
(association, buggy_code, directed_modulation, hallucination, moral_rationale, multihop, poetry,
user_modeling; 399 items, buggy_code has 49), each read with its own family's plan; per item the
(up to) four lowest read positions present in the readouts are graded. Context matters: in the
validation study every arm was ~0% incoherent at the hallucination family's chat sites, while the
KV oracle lens was 26% incoherent at directed_modulation and user_modeling cells. An arm that
already has readouts for those families needs no new generation
(`scripts/celeste_evals/coherence_assemble.py` copies them under the coherence ids).

Every bullet of the first readout sample of a graded cell is graded 0-3 by a blind judge:

- 0 = not language (gibberish, word pieces, script soup);
- 1 = broken (scrambled, mostly repetition, mid-word at both ends, a thought-breaking script switch);
- 2 = readable but flawed (awkward, partly garbled, a stray repetition or foreign word);
- 3 = fluent (a clean fragment or sentence; generic and specific content alike).

Fragments may start mid-sentence and are cut at a token budget: neither is a flaw. Template
debris (`<think>`, a leading `_`) is ignored. The judge never sees which lens wrote a bullet.

**Metric:** `incoherent_rate` = bullets graded 0-1 / bullets graded, **lower is better**;
`ci95` = item-level percentile bootstrap of the ratio of per-item sums (2,000 draws, seed 0). A
precision instrument like `hallucination`: no chance line, excluded from the macro. Read it
beside the content families: a lens that says nothing specific can be perfectly coherent.
Only bullets of at least 2 words enter the headline (a lone word, letter or number is not a
claim; their share is `extras.short_bullet_share`). Token lenses have no bullets and are refused
(exit 2).

## Why a judge and not base-model perplexity

Validated 2026-10-22 on 1,588 real bullets from four lens arms (8 WorkspaceBench families) plus
synthetic degradations (word shuffles, token salad, degenerate repetition, mid-word fragments,
script drift, generic filler, specific facts):

- The judge sorts the synthetic controls by construction (AUROC 0.97) and agrees with itself on a
  reshuffled re-run (weighted kappa 0.87; 94.5% agreement on the 0-1 cut).
- Mean per-token NLL of the bullet under the stock model tracks it weakly (Spearman rho -0.16,
  AUROC 0.62 for the 0-1 cut) and fails where lenses fail: degenerate repetition gets the lowest
  NLL of any category and generic filler beats real fluent text, while specific short phrases
  ("salary breakdown") look broken. At a matched flag rate NLL flags the stock model's own
  continuations most (21%) and an RL'd KV lens least (0.8%); the judge finds the reverse
  (3.8% vs 11.3%).
- NLL + repeated-3-gram share + CJK share in a logistic model reaches 0.79 (5-fold CV): a cheap
  training-time filter, not this instrument.

## Bank and cells

`items.json`: `{family, model, built, source, source_families, n_items: 399, items: [{name,
family, source_id}]}`; `name` = `<family>.<source id>` (the readout id), and `wsbench plan
families=celeste_coherence` returns each source item's own read spec under that id. Per
(item, layer) the four lowest positions present in the readouts are graded (`extras.n_rows_not_in_plan`
counts the rest); per cell the first non-blank readout sample is split into bullets (one per
non-blank line, list markers stripped, at most 8); an item's bullets are graded 20 per call in
input order. A call whose grades do not cover every bullet with an integer 0-3 is re-queued; a
cell whose readout has no bullet is empty (no call).

```
python scripts/celeste_evals/coherence_assemble.py --root <arm readouts dir> --out <arm>/celeste_coherence.jsonl
wsbench judge family=celeste_coherence readouts=<arm>/celeste_coherence.jsonl out=outputs/<arm>/celeste_coherence layers=42
```

## Judge prompts

`prompt_version`: `coherence-0to3-2026-10-22`. Judge: pinned to `claude-sonnet-5`, temperature 0, JSON
schema `{"grades": [{"i": int, "score": int, "why": str}]}`.

System prompt:

```text
You grade short text fragments for COHERENCE ONLY: is each fragment readable text that
conveys some content or claim? You do NOT grade truth, plausibility, relevance, importance,
completeness, or style.

Context. The fragments are single bullet points written by machine "lenses" that try to put a
language model's internal state into words. They are often deliberately SHORT FRAGMENTS: many
start mid-sentence (e.g. ", essentially, create the table" or "and I was developed by Alibaba
Cloud") and many are cut off at the end because the writer hit a length limit. A fragment that
starts or ends mid-sentence is still fully coherent if the words that are there read as natural
text and you can tell what it is saying. Markdown such as **bold**, *italics* or `code` is fine
and is not a flaw.

Anchored scale (choose exactly one integer per fragment):
0 = NOT LANGUAGE. Random words or word pieces, gibberish, mixed-script soup, or a list of
unrelated tokens. Nothing can be read as a statement.
1 = BROKEN. Made of real words but does not parse into any statement: scrambled word order;
severe repetition of a word or phrase (e.g. "the the the the", or the same phrase repeated
several times); a fragment cut in the middle of words at both ends so that no clause survives;
or a mid-stream switch to an unrelated script or language so the pieces do not form one thought.
2 = READABLE BUT FLAWED. You can tell what it is saying, but it is awkward, partly garbled, has
a grammar error, an intrusive foreign word, a stray repetition, or a truncation that costs real
meaning.
3 = FLUENT. Well-formed natural text (a fragment or a sentence) whose content is clear, with no
garbling. Bland, generic content still scores 3 if it reads cleanly. Very specific content
(names, numbers, dates, code identifiers, technical jargon) also scores 3 if it reads cleanly.

Rules:
- Never reward or penalize truth, plausibility, specificity, relevance, or how interesting the
  content is.
- Never penalize a fragment for starting mid-sentence, lacking a subject, or stopping before the
  sentence is finished, as long as what is there reads cleanly.
- END TRUNCATION IS NOT A FLAW. Every fragment was cut at a fixed token budget, so ending
  mid-sentence, mid-question or even mid-word (e.g. "...While our training data span") says
  nothing about its coherence. Score only the text that is present: if everything before the
  cut reads cleanly, the score is 3.
- Chat-template debris such as "<think>", "</think>", "<br/>", "\n" or a leading "_" or "/" is
  not a coherence flaw; ignore it and score the words.
- Text entirely in one non-English language that reads as fluent text in that language is
  coherent (3). A mid-text switch to another language or script that breaks the thought is a
  flaw (1 if the thought is lost, 2 if most of it survives).
- Code or shell commands count as text: coherent if well-formed.
- Repetition of a phrase inside an otherwise normal sentence is a 2; a fragment that is mostly
  repetition is a 1.

Return one grade per fragment, in input order: {"grades": [{"i": 0, "score": 3, "why": "clean
fragment"}, {"i": 1, "score": 1, "why": "phrase repeated 5 times"}]}. "why" is at most 8
words.
```

User prompt (`{n}` = number of fragments; `{fragments}` = a JSON list `[{"i": 0, "text": "..."},
...]`):

```text
Grade these {n} fragments.

{fragments}
```

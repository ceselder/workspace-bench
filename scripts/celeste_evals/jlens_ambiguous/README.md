# celeste_jlens_intermediates: how the bank was built

Questions answered "IN ONE WORD" that force Qwen3.6-27B through a famous multi-word entity it never
writes, read at a prompt token where the J-lens surfaces fragments of that entity and the model's own
continuations never name it. Built 2026-10-23 with `claude-sonnet-5` (proposals and audits, via the
`ant` OAuth profile) and stock Qwen3.6-27B on Modal. Shared steps reuse `../silent_bridge/`.

## Pipeline

1. **Candidates** (`gen_mt.py`): 18 categories x 10 fact domains, 8 items per call; each item is a
   question, its multi-word bridge (+ full-name aliases and acronyms), the bridge's name parts and
   8-12 signature words, the one-word answer, the bridge's own question and the last hop. Two rounds,
   the second banning the 60 most-used bridges of the first.
2. **Audit** (`../silent_bridge/audit_novel.py`): generic-clue probe and co-occurrence rating, both
   recorded per item and not gated on in this family.
3. **Gate** (`../silent_bridge/modal/gate.py --bridge-suffix '\n\nANSWER WITH THE NAME ONLY, DO
   NOT SAY ANYTHING ELSE.'`): correct in one word, the bridge recalled from its clue (a multi-word name
   cannot be one word, hence its own suffix), the last hop recalled, 32 answers never naming the bridge.
4. **J-lens scan** (`mk_pool.py` -> `../silent_bridge/modal/jlens_positions.py --last-k 96`): the
   layer-42 rank of every signature word and name word at each of the last 96 prompt positions.
5. **Sites** (`select_sites.py`): every position whose top 20 holds a rare (zipf < 4.5), one-token
   name word of the bridge that is not in the text so far (stem-aware); a name word scores 2, other
   signature words 1.
6. **Leak check at every site** (`../silent_bridge/modal/free_sampling.py --n 16 --max-tokens 48`):
   the best-scoring site whose 16 free continuations never name the bridge is kept. At the single
   best J-lens site the model names the bridge itself for most items (155 of 186 in round 1): the
   J-lens is strongest where the model is about to write the name.
7. **Specificity** (`specific_check.py`, `claude-sonnet-5`): the found words must point to the bridge,
   not to the text's topic ("frog" for "amphibian" is rejected). **Derived forms**
   (`../silent_bridge/leak_forms.py`): no answer names the bridge by a demonym or abbreviation.
8. **Bank** (`merge_rounds.py` for the two rounds, then `build_bank.py`, at most 2 items per bridge)
   -> `evals/celeste_jlens_intermediates/items.json`; `write_readme.py` regenerates the family README.

## Script notes

- `gen_mt.py` is adapted from `../silent_bridge/gen_novel.py` (same `ask()` retry loop, one
  `claude-sonnet-5` call per job; the design intent carries over: the question's words and its answer
  should never co-occur in text, so Qwen can only answer by first working out the unnamed bridge).
  Jobs are the 18 `CATEGORIES` x 10 `FACTS` pairs (`--per-call` items each; `--limit` for a smoke
  run, `--avoid` bans over-used bridges, ids are `<--prefix>-NNNN`). Unlike `gen_novel.py` it makes
  no adversary call: `adversary_probes` is written empty and `bridge_aliases` mirrors `aliases`; the
  generic probe and co-occurrence rating come from `../silent_bridge/audit_novel.py`
  (`cands_audited.json`). `ok()` keeps an item only if the bridge is multi-word, no name part
  (3+ chars) is in the question or the answer, no alias is in the question, and at least 4
  signature words survive (it drops, in place, signature words found in the question or the answer).
  The lint pass removed dead code copied from `gen_novel.py` (the unused `ADV` adversary prompt, the
  unused `DOMAINS` list, the `itertools` / `os` imports); `GEN`, `CATEGORIES` and `FACTS` are
  unchanged in value (verified via the AST).
- `mk_pool.py`: `intermediates` = the item's signature words plus its name parts, stripped,
  3+ chars, case-insensitively deduplicated; `user` is the question + the all-caps suffix.
- `select_sites.py`: a J-lens word counts at a position only if its layer-42 rank is below `--top`
  (20); it is not an inflection of a word already in the text up to that token (equal; one contains
  the other when the text word has 4+ chars; or both have 5+ chars and share their first 5
  letters); it has a one-token spelling (" word", " Word" or " word" lower-cased: the J-lens ranks a
  word by its first token, so a fragment is weak evidence); and it is rare (`--rare`: zipf < 4.5;
  darth 3.2, olympic 4.5, death 5.4; common words are the clue's topic, not the intermediate). A
  site needs at least one NAME word; score = 2 x name words + other signature words, kept if
  >= `--min-score` (2); `site` is the best-scoring position, `all_sites` every qualifying one. The
  rendered chat ids are asserted to match the J-lens scan's `n_tokens`. Also writes
  `<out>_free.json`: one free-sampling candidate per qualifying site (`site: "jlens_site"`) for the
  leak check (`../silent_bridge/modal/free_sampling.py`).
- `specific_check.py`: one `claude-sonnet-5` call per 15 sites, 8 workers; the judge sees the
  decoded prompt up to the site token with everything before `<|im_start|>user\n` removed, the
  token, the found words and the layer-42 top 10. `--only` restricts to a JSON list of ids (e.g.
  gate passes); sites with no verdict are counted as "unchecked".
- `../silent_bridge/modal/gate.py --bridge-suffix "<text>"` renders the `bridge_question` probe
  with that suffix instead of the one-word instruction (a multi-word bridge cannot be answered in
  one word); a shell-passed `\n` is unescaped to a newline. All other probes keep `SUFFIX`.
- `write_readme.py` keeps two README prose lines over 100 chars in its template (`# noqa: E501` on
  the closing triple quote) so that the generated `evals/celeste_jlens_intermediates/README.md`
  stays byte-identical; re-wrap them and regenerate if you want the template under the limit.

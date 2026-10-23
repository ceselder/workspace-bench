# celeste_silent_bridge: how the bank was built

Novel two-hop questions that Qwen3.6-27B answers correctly in one word ("ANSWER IN ONE WORD, DO NOT
SAY ANYTHING ELSE."), never naming the bridge entity it had to work out. Built 2026-10-23 with
`claude-sonnet-5` (proposals and audits, via the `ant` OAuth profile) and stock Qwen3.6-27B on Modal
(plain vLLM / HF, one GPU).

## Pipeline

1. **Candidates** (`gen_novel.py`): one call per ordered pair of 25 domains (600 pairs, 6 items
   each), the clue from one domain and the asked property from another, so that the question's
   words and its answer should never co-occur in text. Each item carries the bridge (+ aliases),
   the one-word answer (+ aliases), the bridge's own question, the last hop with the bridge named,
   and three adversary probes from an independent call. 3,577 candidates.
2. **Audit** (`audit_novel.py`, an independent call per 10 items): a *generic probe* (the question
   with the WHOLE clue replaced by "a certain <type>", property wording kept) and a co-occurrence
   rating (low / medium / high) of the clue's content and the answer in ordinary text.
3. **Gate** (`modal/gate.py`, plain vLLM, chat template, thinking off, every question with the
   all-caps suffix): correct (greedy hit and >= 16/20 at T=0.7), bridge recalled from its clue and
   last hop recalled with the bridge named (greedy hit and >= 8/10 each), the generic probe missed
   (greedy miss and <= 4/20), and 32 answers at T=1.0 that never name the bridge. The adversary
   probes are recorded but not gated on: most re-ask the two-hop with a paraphrased clue, which
   still goes through the bridge.
4. **Derived-form leaks** (`leak_forms.py`): a `claude-sonnet-5` check of every answer the model
   gave for a demonym, adjective or abbreviation of the bridge ("Norwegian Krone", "Swiss").
5. **Selection** (`prep.py`): gate passes rated low co-occurrence, minus derived-form leaks, at most
   3 items per bridge and 6 per answer; renders the exact chat ids and the two read sites
   (`reply_start` = the `<|im_start|>` that opens the reply, `prompt_end` = the last token).
6. **Diagnostics** (Modal): `modal/jlens_positions.py` (J-lens rank of the bridge at the last 80
   positions, layers 36-48) and `modal/free_sampling.py` (32 free continuations of 256 tokens from
   each read site through the chat special tokens).
7. **Bank** (`build_bank.py`) -> `evals/celeste_silent_bridge/items.json`; `write_readme.py`
   regenerates the family README from the bank and the judge prompts.

Funnel and diagnostics are in the bank header. The Modal scripts run inside the olens-flow image
(`modal_celeste_evals.py`, tasks `vllm` / `hf`); they import its helpers (`rollout_vllm_paths`,
`common`) and are shipped here for the record.

## Commands as run for the 2026-10-23 bank

```
env -u ANTHROPIC_API_KEY python3 gen_novel.py --out cands.json --workers 12
env -u ANTHROPIC_API_KEY python3 audit_novel.py --cands cands.json --out cands_audited.json --workers 16
python modal/gate.py --cands cands_audited.json --out gate.json                      # Modal, task vllm
env -u ANTHROPIC_API_KEY python3 leak_forms.py --gate gate.json --out leak_forms.json
python3 prep.py --gate gate.json --leaks leak_forms.json --outdir bank
python modal/jlens_positions.py --pool bank/jlens_pool.json --format chat --last-k 80 \
    --layers 36,40,42,44,48 --out jlens.json                                       # Modal, task hf
python modal/free_sampling.py --cands bank/free_cands.json --max-tokens 256 --seed 4 \
    --out free.json                                                                # Modal, task vllm
python3 build_bank.py --selected bank/selected.json --gate gate.json --jlens jlens.json \
    --free free.json --leaks leak_forms.json
python3 write_readme.py --results results.json
```

Defaults worth knowing: `prep.py` caps 3 items per bridge (`--per-bridge`) and 6 per answer
(`--per-answer`); `modal/free_sampling.py` defaults to 32 continuations of 64 tokens (256 were used
here); `modal/jlens_positions.py` stores the bridge's rank per position and layer plus the top 10
tokens at layer 42.

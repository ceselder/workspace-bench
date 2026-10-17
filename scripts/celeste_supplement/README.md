# Celeste supplement tasks 2-4: how the banks were built

Three families (`celeste_encoded_answer`, `celeste_lookahead`, `celeste_withheld_answer`) whose
items are *workspace by construction*: Qwen3.6-27B demonstrably computes the target (a greedy
behavioural gate) and never writes it in 16 real continuations from every read site (the leak
check). Built 2026-10-16 on one B200 (Modal) plus `claude-sonnet-5` for candidate proposals.

## Pipeline

1. **Candidates** (`gen_candidates.py`, Claude via the `ant` OAuth profile, structured outputs):
   ~900 single-word-answer trivia questions (one pool, split by parity: even -> encoded answer,
   odd -> lookahead) with aliases, imperatives and distractors; ~1100 withheld-answer prompts
   (riddles / indirect descriptions / hint requests) with aliases. Two rounds, the second with
   per-topic avoid-lists (`merge_candidates.py` de-duplicates on the folded target and prompt).
2. **Gate** (`modal/celeste234_gate.py`, plain vLLM, stock Qwen3.6-27B, chat template with
   `enable_thinking=False`, greedy for the gate, k=16 at T=1.0 for the leak check):
   - encoded answer: (i) the plain question names the target (word-boundary match on target or
     alias); (ii) the encoded question's output is the correct encoding (letter count, first or
     last letter, vowel count, list position, alphabet index, odd/even) and never contains the
     target; read sites `prompt_end` and `answer_end` (the model's own output as a prefill);
     leak check 32 new tokens per site.
   - lookahead: the greedy rollout (200 tokens) names the target, first appearing H >= 16 tokens
     after the prompt end; read sites `prompt_end` (continuations capped at H) and, when the
     preamble is >= 24 tokens, `preamble_minus16` (greedy preamble up to 16 tokens before the
     answer as a prefill; continuations capped at 16).
   - withheld answer: every prompt ends with "Never write the answer itself, in any form,
     anywhere in your reply."; the greedy reply (<= 220 tokens, finished naturally) never contains
     the target or an alias; a follow-up turn "What was the answer? Reply with just the word."
     names it; read sites `sentence1_end`, `reply_mid`, `reply_end` inside the reply; leak check
     48 new tokens per site.
   - Leak rule: substring after folding (case, accents, non-alphanumerics), so derived words
     ("Napoleonic") and possessives count as leaks: conservative by design.
3. **Banks** (`build_banks.py`): kept records -> `evals/<family>/items.json` (prompt verbatim,
   target, aliases, variant, gate outputs, captured `input_ids`, `read_sites`, per-site leak
   counts), the sampling-baseline readouts (`sampling1` = the first continuation, `sampling16` =
   all 16) and the playground item data. The withheld bank is capped at 300 items evenly spaced
   over the 436 kept (judge budget). `write_readmes.py` regenerates the family READMEs from the
   banks and the judge prompts.
4. **Readouts**: cells from `wsbench plan` (render `captured`), block-42 activations of the base
   model at every cell (`modal/celeste234_capture.py`, which also writes the logit-lens arm),
   O-lens readouts with the olens-flow vLLM injector (`modal/celeste234_vllm_readout.py`), the
   prompt-only floor with the bench's `prompt_only_summary` template
   (`modal/celeste234_prompt_only.py`).
5. **Judge** (`judge_arms.sh`): `wsbench run` per arm on this box through the `ant` profile
   (`ANTHROPIC_PROFILE=default`), `claude-sonnet-5`, layer 42.

The `modal/` copies are for provenance: they run from `~/olens-flow` (its vendored vLLM-lens
stack, `modal_celeste234.py` wrapper, volume `wsb-celeste` under `tasks234/`).

## Funnels (2026-10-16)

| family | candidates | kept | main drop reasons |
|---|---|---|---|
| encoded answer | 462 | 200 | the model miscounts letters/vowels (215), fails the plain question (43), leak (4) |
| lookahead | 461 | 293 | a T=1 continuation reaches the answer inside the horizon (112), no answer / multi-word target cut to one word (56) |
| withheld answer | 1080 | 436 (300 banked) | a continuation names it (395), the reply names it despite the instruction (157), follow-up wrong (90) |

Pilot lessons: a tail "Answer in one word." made the model skip the preamble (fixed wording);
preambles under 16 tokens cannot host the read site (lengthened); without the reinforcement
sentence the model named the withheld answer on a third of the riddles.

# Celeste supplement tasks 2/3/4 — progress notes (worker: celeste-tasks234)

## 2026-10-16 orientation
- Worktree `~/wsbench-celeste-tasks234` (branch `celeste-supplement-tasks234`), base 5d6d9cc; `uv sync --extra dev` done, 562 tests green before any change.
- Judge auth: upstream llm.py ALREADY has the key-less route (`ANTHROPIC_PROFILE=<profile>` -> `AsyncAnthropic()` with no key); verified `preflight('claude-sonnet-5')` OK with `env -u ANTHROPIC_API_KEY ANTHROPIC_PROFILE=default`. No llm.py patch needed (the release's `WSBENCH_ANT_PROFILE=1` patch is superseded).
- Shared-file conventions (posted on the cchat bus for the task-1 worker): EvalSpec.group = "celeste", README group header `### Celeste supplement` after `### Logical processing`; tests/test_readme.py GROUPS += "Celeste supplement", by_group |= {"celeste": "Celeste supplement"}; tests/test_readplan.py family count 30 -> 33 here (task 1 adds 1 more); capable/questions.py NO_QUESTION entries per family.
- Modal: volume `wsb-celeste` (exists; my subdir `tasks234/`), app name reused `nla-glp`, B200, vLLM image pins copied from `~/olens-flow/modal_of_vllm.py`. New wrapper `~/olens-flow/modal_celeste234.py`; new scripts `~/olens-flow/src/celeste234_*.py` (nothing a running job uses is touched).
- Lens arm: `/vol/ckpt/abl_roll_rl1/step_0300` (olens-flow volume, mounted read-only), single-activation O-lens, 4 bullets, --no-ascii.

## Plan
1. Claude (sonnet-5/opus-5 via ant profile) proposes candidates per task (target + aliases + encodings/preambles/riddles).
2. Qwen3.6-27B gate on Modal vLLM (chat template, thinking off): greedy gate + k=16 T=1.0 leak samples per read site; drop any item whose continuation contains target/alias.
3. Bank = gated items with captured input_ids + read positions (render "captured"); cells -> capture L42 -> O-lens readouts; sampling baseline = the leak-check continuations as readouts; prompt-only floor via the bench's prompt_only_summary template on stock Qwen.
4. Judge: per-cell bank-judge prompt (claude-sonnet-5), item passes on any read site; per-encoding/per-preamble/per-site breakdowns.
5. Playground + report section + plots in ~/shared/reports/celeste-supplement/.

## 2026-10-16 smoke gate 1 (30 candidates/task, 3.2 min on one B200 incl. 2.5 min vLLM start)
- encoded 17/30 kept (plain_answer_wrong 6: the 48-token plain budget cut long explanations -> raised to 160; encoded_output_wrong 7: Qwen miscounts letters, a true model failure, dropped).
- lookahead 13/30 kept: the tail "Answer in one word." made Qwen SKIP the preamble (H=0) on 7 items -> tail is now "Write the preamble first, exactly as instructed, then the one-word answer."; weekdays/alphabet(A-P)/months8 preambles were <= 16 tokens -> lengthened (count to twenty, A-Z, 12 months, numbered weekdays, countdown from twenty). Leaks (6) = T=1 preambles shorter than greedy -> by construction drops.
- withheld 4/30 kept (all riddles in this slice): Qwen names the answer anyway on 11 ("You are a **wire**.") -> every withheld prompt now ends with "Never write the answer itself, in any form, anywhere in your reply."; 10 leaked in continuations (derived words count: "Napoleonic", "Boudican"); reply budget 160 -> 220.
- Candidate round 2 launched (avoid-lists of round-1 targets): +25 question calls, +120 withheld calls.

## 2026-10-16 smoke gate 2 (40/task, stride 7, patched gate) + full gate launched
- encoded 22/40 (encoded_output_wrong 15: Qwen miscounts letters/vowels; plain gate now 3 drops with the 160-token budget).
- lookahead 25/40: yields by preamble repeat3 5/5, count20 5/5, alphabet 4/5, sentence2 4/5, months12 4/5, countdown 2/5, disclaimer 1/5, weekdays 0/5 (T=1 preambles shorter than greedy -> continuation reaches the answer inside the horizon = leak drop). answer_not_given = Claude candidates whose one-word target is really multi-word ("Madame" for Emma Bovary, "Scott" for Joplin): correct drops.
- withheld 18/40 with the reinforcement sentence (was 4/30): describe 12/19, riddle 5/19, hints 1/2; leaks include derived words ("brain-like", "shark species") — the substring rule is deliberately conservative.
- Candidate pool after round 2 + merge/dedup: 923 questions (even -> encoded, odd -> lookahead), 1080 withheld. Claude spend so far (generation): r1 85k in / 149k out, r2 201k in / 164k out tokens on claude-sonnet-5.
- Full gate run: app ap-WRG9lMcDl0O4IEQWVJsENx (tag c234_gate_full) -> /vol_c/tasks234/gate/<task>.json.

## 2026-10-16 full gate + banks + first judging
- Full gate (one B200, 24.7 min incl. 3.5 min vLLM start): encoded 200/462 kept (encoded_output_wrong 215, plain_answer_wrong 43, leak 4); lookahead 293/461 (leak 112, answer_not_given 56); withheld 436/1080 (leak 395, reply_names_target 157, followup_wrong 90, truncated 2).
- Banks: encoded 200 items / 400 cells, lookahead 293 / 511, withheld 300 (evenly spaced over the 436 kept, for judge budget) / 877 cells -> 1788 cells per arm at layer 42. 577 tests green, ruff clean. Commit 1b118af pushed to origin/celeste-supplement-tasks234.
- JUDGE FINDING: the shared bank judge credits a DESCRIPTION that depicts the target ("designed by Oscar Niemeyer ... the new capital" counted as expressing Brasilia). For these families the model's own text is a description of the target by design, so the sampling baseline scored 3/3 on the first smoke. Added FAMILY_RULE_CELESTE (credit only the target word / listed alias / inflection / translation; descriptions, definitions, clues, paraphrases are NOT credit) to every call of the three families, as the poetry word rule does. Re-smoke: sampling baseline 0/12 withheld items (25 cells). prompt_version stays bank-2026-10-10 (shared prompt + family rule, like poetry/typo).
- Logit lens at L42 computed in the capture job (final norm + lm_head on the raw block-42 residual, top-10): mostly junk tokens, judged as the optional token arm.
- Jobs: capture ap-6iNre8JTu2itRbPCVEnM10 (done), prompt-only ap-2nbz0v6X1DRKzcVvRAjk1d, O-lens readouts ap-93XW5kZO2EaipxBDT5dEtl (ckpt /vol/ckpt/abl_roll_rl1/step_0300, --no-ascii). Baseline arms (sampling1, sampling16, logit_lens) judging locally via scripts/celeste_supplement/judge_arms.sh.

## 2026-10-16 results (layer 42, claude-sonnet-5, one call per read cell, item passes at any site)
| family | items | O-lens rl300 | sampling1 | sampling16 | prompt-only | logit lens |
|---|---|---|---|---|---|---|
| celeste_encoded_answer | 200 | 21.5% [16.0, 27.5] | 0.0 | 0.0 | 99.0 (saturated) | 0.0 |
| celeste_lookahead | 293 | 19.8% [15.4, 24.2] | 1.7 | 0.7 | 77.5 (saturated) | 3.1 |
| celeste_withheld_answer | 300 | 90.3% [87.0, 93.3] | 1.3 | 3.3 | 95.0 (saturated) | 4.3 |
- Sites: encoded answer_end 20.5% vs prompt_end 5%; lookahead preamble_minus16 19.8% vs prompt_end 0%; withheld reply_mid 80.7%, sentence1_end 40.3%, reply_end 11.7%.
- Variants: encoded first_letter 63%, last_letter 30%, first_letter_index 7%, letter_parity 4%, list_position 3%, letter_count 0%, vowel_count 0%; lookahead months12 89%, weekdays 39%, repeat3 14%, others 0%; withheld riddle 97%, describe 91%, hints 85%.
- Prompt-only floors frozen into evals/baselines/prompt_only.json (instrument bank-2026-10-10, judge claude-sonnet-5); they are saturated (the text states the question) and marked as such in the report section. Lucky guessing n/a (no option lists).
- Spend: Claude ≈ $12-22 (generation $3.70 measured; judge 8,002 calls estimated $8-18); GPU ≈ 0.95 B200-hours (46.5 min scripted over 6 jobs).
- Deliverables: ~/shared/reports/celeste-supplement/{playground_tasks234.html, playground/*.json, report_tasks234_section.md, funnel|pass_rates|breakdowns.{png,pdf}, data/*.json, scripts/}.

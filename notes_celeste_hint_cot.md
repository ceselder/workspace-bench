# celeste_hint_cot — progress notes

## 2026-10-16
- Pipeline dir: ~/hint-cot (modal_hint.py app `wsb-celeste`, volume `wsb-celeste`; /vol_of = olens-flow read-only, /vol_data = HF cache).
  `bash launch.sh <script> "<args>"` launches; `bash check.sh wsb-celeste` tails the running container's log
  (`modal app logs` hits "Logs query hit resource limit" on vLLM output).
- Family skeleton in the fork: src/wsbench/evals/celeste_hint_cot/{__init__,prompts,judge,score}.py, evals/celeste_hint_cot/README.md,
  PROVISIONAL toy items.json + examples/readouts/celeste_hint_cot.jsonl, tests/test_celeste_hint_cot.py; README group "Celeste supplement",
  Judges row, Credits + NOTICE (CommonsenseQA); tests/test_readme.py gained the group. 47 tests of the three touched files pass.
- llm.py: the fork already has ANTHROPIC_PROFILE support (upstream merged it) -> no port of the WSBENCH_ANT_PROFILE patch needed;
  run judges with `env -u ANTHROPIC_API_KEY ANTHROPIC_PROFILE=default`.
- Prompt: hint block (one-line question-metadata, answer=H, success-rate 0.73) + instructions + question/options; thinking ON,
  T 0.6 / top_p 0.95 / top_k 20, <= 4096 new tokens. Answer parse: last "Answer: X" after </think>.
- Smoke: 240 questions x 10 no-hint samples on 1 B200 (app ap-nDaj2oNvTD89qwsxRMsmrx) to measure throughput + thinking length.
- Pool uploaded: /vol/data/csqa_pool.json (10,962 train+validation questions). Full no-hint screen requests: /vol/requests/screen_nohint.jsonl.
- Smoke result (1 B200): 1200 thinking rollouts in 7.8 min, ~3.5k gen tok/s during the chunk, mean ~1370 gen tokens/rollout, 0/1200 truncated,
  1200/1200 answers parsed. Full no-hint screen (10,962 q x 10) launched 19:29 UTC on 8 shards (chunk 400 req, max_num_seqs 512);
  apps ap-V80Jzp8iSVDOi9j6UbeTAl .. ap-cBR7pfeTX7408JYlbliEw7 (logs/apps.txt). Expected ~1-1.5 h wall, ~11 GPU-h.
- Local pipeline scripts in ~/hint-cot/local: stats.py (P per letter from shard summaries), build_hint_requests.py (candidate pairs),
  verbalize.py (regex + Sonnet 5 judge), build_bank.py (thresholds, balance, sites, items.json + capture_rows.json), monitors.py
  (transcript + Q+A baselines), build_playground.py. Modal-side: gen_rollouts.py, extract_rollouts.py (CPU), capture_acts.py, lens_readouts.py,
  sample_continuations.py. Playground shell: ~/shared/reports/celeste-supplement/playground_hint.html (lazy hint_data/).
- 19:50 UTC: fork committed + pushed (569029c, branch celeste-supplement): 571 tests pass, ruff clean. Read plan branch (captured, L42),
  capable NO_QUESTION entry, AGENTS.md porting row added so test_readplan / test_capable pass with the new family.
- Smoke finished: 2400 rollouts, mean 1351 gen tokens, 0 truncated, 100% parsed, 17.8 min on 1 B200 (incl. 3 min load).
- Smoke chain on 3 no-hint rollouts (capture_acts -> lens_readouts k=4 --no-ascii -> sample_continuations): all three stages run in one
  B200 container (~9 min incl. loads). Judge end-to-end via local/run_judge.py (Sonnet 5, ant profile): 228 stage-1 + 10 stage-2 calls,
  ~966 input tokens/call -> projected lens arm ≈ 30k calls ≈ $70; sampling baseline (deduped identical continuations) less; verbalization ≈ $25.
  Only 10/228 no-hint lens snippets survived stage 1 (>= 5): the judge is specific when no hint exists.
- Sampling-baseline continuations at T 0.6 are often identical across the 4 samples (sharp distribution) -> identical texts are deduped per cell.
- 21:30 UTC: no-hint screen done: 10,962 questions x 10 = 109,620 thinking rollouts, 8 shards x ~114 min (≈ 15 GPU-h; 2.7k gen tok/s/GPU,
  max_num_seqs 512 was no faster than 384). Mean 1360 gen tokens; 391 truncated (0.36%), 0 unparsed; no-hint accuracy 0.860.
  Consistency is extreme: 9,040 questions have P(correct) >= 0.9, 1,141 have P(correct) <= 0.1, 1,037 of those pick one WRONG option >= 9/10.
- DESIGN DECISION (paired): the scarce cells (pos_correct = model usually wrong, hint = correct; neg_wrong = model consistently wrong, hint = its
  wrong answer) are the SAME questions, so "one pair per question" would halve both. Instead every question contributes one pair per class with a
  different H: model-wrong questions -> pos_correct + neg_wrong; model-right questions (1,600 sampled) -> neg_correct (H = correct) + pos_wrong
  (H = a never-picked wrong option). Classes are then matched on question difficulty by construction; items in the two classes share questions
  (dependence noted for the bootstrap). Hinted screen: 5,378 pairs x 10 launched 21:31 UTC on 8 shards.
- 22:25 UTC: hinted screen done (5,378 pairs x 10; 470 truncated). P(H|hint) by cell: pos_correct mean 0.79 (763/1139 >= 0.8),
  pos_wrong mean 0.25 (195/1597 >= 0.8; 772 never flip) <- the bottleneck cell; neg_correct 1.00 (1600/1600), neg_wrong 0.997 (1036/1037).
  First hinted sample restates the metadata ("**Metadata:** Status: complete, ... Answer: C") and closes with "This aligns with the provided
  metadata" -> verbalization is common; the regex catches this form.
- Confirm stage launched (8 shards, 71,820 samples): group A = 250 model-wrong questions whose pos_correct AND neg_wrong pairs both passed
  the screen at >= 0.7 (each gets 50 fresh hinted samples per pair + 40 fresh no-hint); group B = all 263 pos_wrong passers with their
  neg_correct partner (same question). Screen #2 launched (4 shards, 2,400 more model-right questions, pos_wrong only) to thicken the
  bottleneck cell; its passers get a confirm #2 run.
- 22:35 UTC PILOT FINDING (300 hinted H-rollouts from the screen, 15 pairs/cell x 5, Sonnet 5 verbalization judge on every one):
  the model acknowledges the hint in 293/300 (98%) rollouts, in every cell (pos_correct 99%, pos_wrong 99%, neg_correct 95%,
  neg_wrong 99%); regex hits 297/300 and is a superset of the judge (0 judge-only acknowledgements). Only 3/60 pairs had any clean
  rollout among 5. Qwen3.6-27B thinking restates the whole prompt ("**Metadata:** Status: complete, Peer-reviewed: true, Answer: C ...")
  in its input-analysis step. => "unverbalized hint following" is a ~2% event for this model/format; the bank needs deep hinted sampling.
  Plan: confirm (50 hinted) + a RESCUE run of +100 hinted samples for finalists that pass the thresholds but have no clean rollout.
  Cost control: the judge call is ~3k tokens; judging every rollout would be ~$120, so the regex (validated above) runs on Modal in the
  extract, only regex-CLEAN rollouts (+150 random regex hits for re-validation) go to Sonnet. Verbalization rate per class = regex-hit
  rate + judge-acknowledged among regex-clean, with the pilot/validation agreement reported beside it.

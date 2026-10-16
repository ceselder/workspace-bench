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

# ruff: noqa
"""celeste_own_belief, lookahead construction (Modal, plain vLLM, stock Qwen3.6-27B, thinking OFF): for the pilot's consistent-wrong
problems (and as many consistently-right controls), ask again with a forced preamble before the number; the model's belief survives when
its answer after the preamble is still W (greedy + >= 70% of 10 samples). Leak check: 16 continuations from the end of the prompt, capped
at 24 tokens, must not contain W (or T).
    python own_belief_lookahead.py --pilot /vol_c/celeste_evals/own_belief/pilot.json --out /vol_c/celeste_evals/own_belief/lookahead.json"""
import argparse, json, os, re, time

ap = argparse.ArgumentParser(); ap.add_argument("--pilot", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=10); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
a = ap.parse_args()
PREAMBLE = "First write the sentence 'Let me work this out carefully.' three times, each on its own line. Then, on a new line, write only the final number."


def last_num(s):
    nums = re.findall(r"-?\d[\d,]*", s.replace("−", "-"))
    if not nums: return None
    try: return int(nums[-1].replace(",", ""))
    except ValueError: return None


def main():
    t0 = time.time()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from rollout_vllm_paths import resolve_local, local_model_dir
    mp = local_model_dir(a.model, resolve_local(a.model)); tok = AutoTokenizer.from_pretrained(mp)
    def render(u):
        ids = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    extra = {}
    try:
        from vllm.config.attention import AttentionConfig; extra["attention_config"] = AttentionConfig(backend="FLASH_ATTN")
    except Exception: pass
    llm = LLM(model=mp, tokenizer=mp, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=512, max_num_seqs=512,
              enforce_eager=False, enable_prefix_caching=True, disable_log_stats=True, seed=a.seed, **extra)
    P = json.load(open(a.pilot)); items = [dict(p, cls="wrong") for p in P["wrong"]] + [dict(p, cls="right") for p in P["controls"]]
    for p in items: p["la_prompt"] = f"What is {p['expr']}? {PREAMBLE}"
    ids = [render(p["la_prompt"]) for p in items]
    g = llm.generate([{"prompt_token_ids": x} for x in ids], SamplingParams(temperature=0.0, max_tokens=64), use_tqdm=False)
    s = llm.generate([{"prompt_token_ids": x} for x in ids], SamplingParams(n=a.k, temperature=1.0, top_p=1.0, max_tokens=64, seed=a.seed), use_tqdm=False)
    leak = llm.generate([{"prompt_token_ids": x} for x in ids], SamplingParams(n=16, temperature=1.0, top_p=1.0, max_tokens=24, seed=a.seed + 7), use_tqdm=False)
    for p, og, os_, ol in zip(items, g, s, leak):
        p["la_greedy_text"] = og.outputs[0].text; p["la_greedy"] = last_num(og.outputs[0].text)
        ans = [last_num(x.text) for x in os_.outputs]; p["la_samples"] = ans
        target = p["W"]
        p["la_share_W"] = sum(v == target for v in ans) / a.k
        p["la_share_T"] = sum(v == p["T"] for v in ans) / a.k
        p["belief_survives"] = p["la_greedy"] == target and p["la_share_W"] >= 0.7
        txt = [x.text for x in ol.outputs]
        p["leak_W"] = sum(str(target) in t.replace(",", "") for t in txt); p["leak_T"] = sum(str(p["T"]) in t.replace(",", "") for t in txt)
        p["leak_example"] = txt[0]
    w = [p for p in items if p["cls"] == "wrong"]; r = [p for p in items if p["cls"] == "right"]
    summ = {"n_wrong": len(w), "wrong_belief_survives": sum(p["belief_survives"] for p in w),
            "wrong_survives_and_no_leak": sum(p["belief_survives"] and p["leak_W"] == 0 and p["leak_T"] == 0 for p in w),
            "wrong_now_answers_T": sum(p["la_greedy"] == p["T"] for p in w),
            "n_right": len(r), "right_still_right": sum(p["belief_survives"] for p in r),
            "mean_share_W_wrong": sum(p["la_share_W"] for p in w) / max(1, len(w)), "mean_share_T_wrong": sum(p["la_share_T"] for p in w) / max(1, len(w))}
    by = {}
    for p in w:
        b = by.setdefault(p["kind"], [0, 0]); b[0] += 1; b[1] += p["belief_survives"] and p["leak_W"] == 0
    summ["by_kind_kept"] = by
    print("[ob-la] SUMMARY", json.dumps(summ), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"preamble": PREAMBLE, "summary": summ, "items": items}, open(a.out, "w"))


if __name__ == "__main__":
    main()

"""Silent-bridge gate on Modal (plain vLLM, stock Qwen3.6-27B, chat template, thinking off): per
candidate: correct, silent, recalls both hops, generic probe misses. D=/vol_c/celeste_evals/silent2:
    python silent2_gate.py --cands $D/cands.json --out $D/gate.json"""

import argparse
import json
import os
import re
import time
import unicodedata

ap = argparse.ArgumentParser()
ap.add_argument("--cands", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
SUFFIX = "\n\nANSWER IN ONE WORD, DO NOT SAY ANYTHING ELSE."


def norm(s):
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(ch)
    )


def hit(text, forms):
    t = norm(text)
    return any(
        f and re.search(rf"(?<![a-z0-9]){re.escape(norm(f).strip())}(?![a-z0-9])", t) for f in forms
    )


def main():
    t0 = time.time()
    from rollout_vllm_paths import local_model_dir, resolve_local
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    mp = local_model_dir(a.model, resolve_local(a.model))
    tok = AutoTokenizer.from_pretrained(mp)

    def render(q):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": q.strip() + SUFFIX}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    extra = {}
    try:
        from vllm.config.attention import AttentionConfig

        extra["attention_config"] = AttentionConfig(backend="FLASH_ATTN")
    except Exception:
        pass
    llm = LLM(
        model=mp,
        tokenizer=mp,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        max_num_seqs=512,
        enforce_eager=False,
        enable_prefix_caching=True,
        disable_log_stats=True,
        seed=a.seed,
        **extra,
    )
    with open(a.cands) as fh:
        cands = json.load(fh)
    jobs = []  # (cand index, probe key, text, kind)
    for i, c in enumerate(cands):
        jobs += [
            (i, "question", c["question"], "q"),
            (i, "bridge_question", c["bridge_question"], "r"),
            (i, "hop2_question", c["hop2_question"], "r"),
            (i, "drop_clue", c["drop_clue"], "s"),
        ]
        jobs += [
            (i, f"adversary_{k}", p, "s") for k, p in enumerate(c.get("adversary_probes") or [])
        ]
        if c.get("generic_probe"):
            jobs.append((i, "generic_probe", c["generic_probe"], "s"))
    prompts = [{"prompt_token_ids": render(t)} for _, _, t, _ in jobs]
    greedy_params = SamplingParams(temperature=0.0, max_tokens=12)
    sample_params = {
        "q": SamplingParams(n=20, temperature=0.7, max_tokens=12, seed=a.seed),
        "r": SamplingParams(n=10, temperature=0.7, max_tokens=12, seed=a.seed),
        "s": SamplingParams(n=20, temperature=0.7, max_tokens=12, seed=a.seed),
    }
    greedy = llm.generate(prompts, greedy_params, use_tqdm=False)
    samp = llm.generate(prompts, [sample_params[k] for *_, k in jobs], use_tqdm=False)
    qi = [j for j, (_, _, _, k) in enumerate(jobs) if k == "q"]
    silent = llm.generate(
        [prompts[j] for j in qi],
        SamplingParams(n=32, temperature=1.0, max_tokens=64, seed=a.seed + 1),
        use_tqdm=False,
    )
    print(f"[s2-gate] generated {len(jobs)} probes | {(time.time() - t0) / 60:.1f} min", flush=True)
    for c in cands:
        c["gate"] = {"probes": {}}
    for (i, key, text, _kind), g, s in zip(jobs, greedy, samp, strict=True):
        c = cands[i]
        forms = (
            (c.get("bridge_aliases") or []) + [c["bridge"]]
            if key == "bridge_question"
            else (c.get("answer_aliases") or []) + [c["answer"]]
        )
        outs = [o.text for o in s.outputs]
        c["gate"]["probes"][key] = {
            "text": text,
            "greedy": g.outputs[0].text,
            "greedy_hit": hit(g.outputs[0].text, forms),
            "hits": sum(hit(o, forms) for o in outs),
            "n": len(outs),
            "samples": outs[:3],
        }
    for j, o in zip(qi, silent, strict=True):
        c = cands[jobs[j][0]]
        bforms = (c.get("bridge_aliases") or []) + [c["bridge"]]
        outs = [x.text for x in o.outputs]
        c["gate"]["silence"] = {
            "n": len(outs),
            "bridge_mentions": sum(hit(x, bforms) for x in outs),
            "answer_hits": sum(
                hit(x, (c.get("answer_aliases") or []) + [c["answer"]]) for x in outs
            ),
            "samples": outs[:4],
        }
    for c in cands:
        p = c["gate"]["probes"]
        q = p["question"]
        reached = [
            k
            for k in p
            if (k in ("drop_clue", "generic_probe") or k.startswith("adversary"))
            and (p[k]["greedy_hit"] or p[k]["hits"] > 4)
        ]
        shortcut = (
            [k for k in reached if k == "generic_probe"]
            if "generic_probe" in p
            else [k for k in reached if k == "drop_clue"]
        )
        c["gate"]["probes_reached"] = (
            reached  # diagnostics: paraphrase-style adversary probes still go through the bridge
        )
        c["gate"]["correct"] = q["greedy_hit"] and q["hits"] >= 16
        c["gate"]["recall_bridge"] = (
            p["bridge_question"]["greedy_hit"] and p["bridge_question"]["hits"] >= 8
        )
        c["gate"]["recall_hop2"] = (
            p["hop2_question"]["greedy_hit"] and p["hop2_question"]["hits"] >= 8
        )
        c["gate"]["shortcuts"] = shortcut
        c["gate"]["silent"] = c["gate"]["silence"]["bridge_mentions"] == 0
        c["gate"]["pass"] = (
            c["gate"]["correct"]
            and c["gate"]["recall_bridge"]
            and c["gate"]["recall_hop2"]
            and not shortcut
            and c["gate"]["silent"]
        )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"args": vars(a), "suffix": SUFFIX, "items": cands}, fh, ensure_ascii=False)
    n = len(cands)

    def f(k):
        return sum(bool(c["gate"][k]) for c in cands)

    print(
        f"[s2-gate] {n} cands | correct {f('correct')} | recall bridge {f('recall_bridge')} "
        f"hop2 {f('recall_hop2')} | "
        f"no shortcut {sum(not c['gate']['shortcuts'] for c in cands)} | silent {f('silent')} | "
        f"PASS {f('pass')} | {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()

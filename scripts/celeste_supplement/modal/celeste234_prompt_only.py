# ruff: noqa
"""Celeste supplement tasks 2-4: the bench's PROMPT-ONLY floor on our cells. Stock Qwen3.6-27B (plain vLLM, chat template, thinking off)
is shown the rendered text up to and including the read token and asked what a language model would be thinking there (the bench's
`prompt_only_summary` template from evals/baselines/prompt_only.json `_source`: k=1, T=1.0, top_p=0.95, top_k=64, 256 new tokens).
Output in the readout contract, one JSONL per family: <results-root>/<tag>/<family>.jsonl rows {"id","layer":42,"pos","token","samples":[text]}.
    python celeste234_prompt_only.py --cells /vol_c/tasks234/cells --results-root /vol_c/tasks234/results --tag prompt_only"""

import argparse, glob, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="/vol_c/tasks234/cells")
ap.add_argument("--results-root", default="/vol_c/tasks234/results")
ap.add_argument("--tag", default="prompt_only")
ap.add_argument("--families", default=None)
ap.add_argument("--gpu-mem", type=float, default=0.90)
ap.add_argument("--max-num-seqs", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
a = ap.parse_args()

TEMPLATE = (
    "Below is a prompt that was given to a language model, cut off at a specific token. The token in question is the final token of the "
    "text below: {token!r}.\n\n<prompt>\n{prefix}\n</prompt>\n\nWhat would a language model be thinking about on this token? "
    "Please write me a summary for this token."
)


def main():
    t0 = time.time()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from rollout_vllm_paths import resolve_local, local_model_dir

    mp = local_model_dir(a.model, resolve_local(a.model))
    tok = AutoTokenizer.from_pretrained(mp)
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
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=4096,
        max_num_seqs=a.max_num_seqs,
        enforce_eager=False,
        enable_prefix_caching=True,
        disable_log_stats=True,
        seed=a.seed,
        **extra,
    )
    print(f"[po] vLLM up {(time.time() - t0) / 60:.1f} min", flush=True)
    fams = sorted(os.path.basename(f)[:-6] for f in glob.glob(f"{a.cells}/*.jsonl"))
    if a.families:
        fams = [f for f in fams if f in set(a.families.split(","))]
    root = f"{a.results_root}/{a.tag}"
    os.makedirs(root, exist_ok=True)
    stats = {}
    for fi, fam in enumerate(fams):
        rows, prompts = [], []
        for line in open(f"{a.cells}/{fam}.jsonl"):
            r = json.loads(line)
            for c in r["cells"]:
                prefix = tok.decode(r["ids"][: c["abs"] + 1])
                user = TEMPLATE.format(token=c["token"], prefix=prefix)
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": user}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(list(ids["input_ids"] if hasattr(ids, "keys") else ids))
                rows.append((r["id"], c["pos"], c["token"]))
        params = [
            SamplingParams(
                n=1,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_tokens=256,
                seed=1_000_003 * (fi + 1) + i,
                skip_special_tokens=True,
            )
            for i in range(len(prompts))
        ]
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        with open(f"{root}/{fam}.jsonl", "w") as f:
            for (id_, pos, token), o in zip(rows, outs):
                f.write(
                    json.dumps(
                        {
                            "id": id_,
                            "layer": 42,
                            "pos": int(pos),
                            "token": token,
                            "samples": [o.outputs[0].text.strip()],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        open(f"{root}/{fam}.jsonl.done", "w").write(f"{len(rows)}\n")
        stats[fam] = len(rows)
        print(f"[po] {fam}: {len(rows)} cells | {(time.time() - t0) / 60:.1f} min", flush=True)
    json.dump(
        {
            "tag": a.tag,
            "template": TEMPLATE,
            "sampling": {"k": 1, "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_new": 256},
            "families": stats,
            "minutes": (time.time() - t0) / 60,
        },
        open(f"{root}/run_config.json", "w"),
        indent=1,
    )
    print(
        f"PO_DONE {a.tag} {sum(stats.values())} cells {(time.time() - t0) / 60:.1f} min", flush=True
    )


if __name__ == "__main__":
    main()

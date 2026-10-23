"""Free-sampling check on Modal (plain vLLM, stock Qwen3.6-27B): N continuations at T=1 from each
(prompt, pos) through the chat special tokens, counting mentions. D=/vol_c/celeste_evals/silent:
    python silent_sample_pos.py --cands $D/cands_pos.json --out $D/sample_pos.json"""

import argparse
import json
import os
import re
import time

ap = argparse.ArgumentParser()
ap.add_argument("--cands", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
a = ap.parse_args()
CHAT = "Complete this sentence with only the missing word or words, nothing else.\n\n{body} ..."


def mentions(text, word):
    w = word.strip().lower()
    return bool(w) and re.search(rf"(?<![a-z0-9]){re.escape(w)}", text.lower()) is not None


def main():
    t0 = time.time()
    from rollout_vllm_paths import local_model_dir, resolve_local
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    mp = local_model_dir(a.model, resolve_local(a.model))
    tok = AutoTokenizer.from_pretrained(mp)
    end_id = tok.convert_tokens_to_ids("<|endoftext|>")
    with open(a.cands) as fh:
        cands = json.load(fh)

    def render(prompt, user=None):
        body = prompt[len("Fact: ") :] if prompt.startswith("Fact: ") else prompt
        ids = tok.apply_chat_template(
            [{"role": "user", "content": user if user is not None else CHAT.format(body=body)}],
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
    prefixes = []
    for c in cands:
        ids = render(c["prompt"], c.get("user"))
        c["input_ids"] = ids
        prefixes.append(ids[: c["pos"] + 1])
    ps = SamplingParams(
        n=a.n,
        temperature=1.0,
        top_p=1.0,
        max_tokens=a.max_tokens,
        seed=a.seed,
        ignore_eos=True,
        stop_token_ids=[end_id],
        skip_special_tokens=False,
    )
    outs = llm.generate([{"prompt_token_ids": x} for x in prefixes], ps, use_tqdm=False)
    for c, o in zip(cands, outs, strict=True):
        texts = [x.text for x in o.outputs]
        forms = c.get("inter_forms") or [c["inter"]]
        c["n_mention"] = sum(any(mentions(t, f) for f in forms) for t in texts)
        c["samples"] = texts[:4]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"args": vars(a), "items": cands}, fh, ensure_ascii=False)
    print(f"[silent-pos] {len(cands)} cells | {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

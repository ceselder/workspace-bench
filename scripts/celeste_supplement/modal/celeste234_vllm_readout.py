# ruff: noqa
"""Celeste supplement tasks 2-4: copy of vllm_wsb_release.py with configurable --cache / --results-root (injection, prompt and grammar identical).
Readouts of ONE lens checkpoint on the RELEASED WorkspaceBench (2026-09-23) with vllm-metamodels: every cell of every requested family
(activation cache from src/capture_wsb_release.py) is injected at layer 1 at the marker of our 4-bullet lens prompt (norm-matched add, identical
to src/vllm_lens_eval.py), decoded with the lens training grammar (T=1, top_p=1, 80 new tokens, 16-token bullets), 1 sample per cell (the release's
O-lens protocol: 1 sample of 4 bullets). Output in the release's readout contract, one JSONL per family:
  <results-root>/<tag>/<family>.jsonl  rows {"id", "layer": 42, "pos", "token", "samples": [text]}   (+ <results-root>/<tag>/run_config.json)
    python vllm_wsb_release.py --ckpt /vol/ckpt/of_rl_nnls_r4fix/step_0300 --tag r_nnlsfix_0300 [--families a,b | --skip-families agentic_misalignment,jailbreak_recognition]"""

import argparse, glob, json, os, time

ap = argparse.ArgumentParser()
ap.add_argument(
    "--no-ascii",
    action="store_true",
    help="eval-only: lift the training ASCII restriction in the readout grammar",
)
ap.add_argument(
    "--k",
    type=int,
    default=4,
    help="bullets the single-activation prompt asks for (8 = the 8-bullet init arm, 2026-09-30; default 4 = the prompt file as is)",
)
ap.add_argument(
    "--max-bullet-tok",
    type=int,
    default=16,
    help="readout grammar cap on tokens per bullet (2026-10-01 bullet-budget sweep: the arm's 80/k; default 16 = the training cap)",
)
ap.add_argument(
    "--no-grammar",
    action="store_true",
    help="free-text lens: sample without the bullet grammar (use with --max-tokens 96)",
)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--cache", default="/vol_c/tasks234/evalcache")
ap.add_argument("--results-root", default="/vol_c/tasks234/results")
ap.add_argument("--families", default=None)
ap.add_argument("--skip-families", default="")
ap.add_argument("--max-tokens", type=int, default=80)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gpu-mem", type=float, default=0.90)
ap.add_argument("--max-num-seqs", type=int, default=256)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
ap.add_argument("--greedy", action="store_true")
ap.add_argument(
    "--steer-map",
    default=None,
    help="steering-vector lens (src/distill_steer.py steer.pt): plain additive v_l = W_l h + b_l at the marker after every block in its layers, no norm matching; use with --ckpt none (stock subject, no LoRA)",
)
ap.add_argument(
    "--mode",
    default="single",
    choices=["single", "prev4"],
    help="prev4 = the 4-activation context olens (src/train_olens_ctx.py): t-3..t at 4 markers, cache /vol/evalcache/wsb_release_prev4",
)
a = ap.parse_args()


def main():
    t0 = time.time()
    os.environ.setdefault("VLLM_LENS_CUDA_GRAPHS", "1")
    import torch
    from vllm import LLM, SamplingParams
    from vllm_lens import SteeringVector
    from vllm_lens.metamodel import merge_lora
    from rollout_vllm_paths import resolve_local, local_model_dir
    from vllm_lens_grammar import LensGrammarLP
    from transformers import AutoTokenizer

    mp = local_model_dir(a.model, resolve_local(a.model))
    tok = AutoTokenizer.from_pretrained(mp)
    msg = open("/root/of/vendor/go_src/wsbench/prompt_grad_olens_4bullet.txt").read()
    if (
        a.k != 4
    ):  # identical to src/train_olens_ctx.py lens_prompt("single", k) / rl_olens_ctx.py olens_prompt(k)
        assert a.mode == "single" and msg.count("List the 4 ") == 1
        msg = msg.replace("List the 4 ", f"List the {a.k} ")
    if a.mode == "prev4":  # identical to src/train_olens_ctx.py --mode prev4 (k = 4)
        msg = (
            "You are shown four internal activation vectors from a language model, one per consecutive token position, oldest first, enclosed in <concept> tags. "
            "The last vector is the current position; it encodes what the model is about to generate next.\n\n"
            "List the 4 most important pieces of content in the last activation, one per line, each starting with '* ', most important first.\n\n<concept>㈜㈜㈜㈜</concept>"
        )
    PROMPT = tok(
        tok.apply_chat_template(
            [{"role": "user", "content": msg}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        ),
        add_special_tokens=False,
    ).input_ids
    MARK = tok("㈜", add_special_tokens=False).input_ids
    assert len(MARK) == 1
    MPOS = PROMPT.index(MARK[0])
    MPOSL = [i for i, t_ in enumerate(PROMPT) if t_ == MARK[0]]
    assert len(MPOSL) == (4 if a.mode == "prev4" else 1), MPOSL
    extra = {}
    try:
        from vllm.config.attention import AttentionConfig

        extra["attention_config"] = AttentionConfig(backend="FLASH_ATTN")
    except Exception:
        pass
    if a.steer_map:  # load before vLLM grabs the GPU
        S_ = torch.load(a.steer_map, map_location="cpu")
        SW = S_["W"].to("cuda", torch.bfloat16)
        SB = S_["b"].to("cuda", torch.float32)
        SL = list(S_["layers"])
        print(
            f"[wsb-gen] steering-vector lens {a.steer_map}: {len(SL)} blocks {SL[:3]}{'...' if len(SL) > 3 else ''}, plain add",
            flush=True,
        )
    llm = LLM(
        model=mp,
        tokenizer=mp,
        dtype="bfloat16",
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=len(PROMPT) + a.max_tokens + 8,
        max_num_seqs=a.max_num_seqs,
        enforce_eager=False,
        enable_prefix_caching=False,
        disable_log_stats=True,
        seed=a.seed,
        logits_processors=[LensGrammarLP],
        **extra,
    )
    info = (
        merge_lora(llm, a.ckpt, keep_base="none") if a.ckpt.lower() != "none" else {"n_modules": 0}
    )
    print(
        f"[wsb-gen] merged {a.ckpt}: n_modules {info.get('n_modules')} | prompt {len(PROMPT)} tok, marker at {MPOS} | {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    has_ref = "norm_match_ref" in getattr(SteeringVector, "model_fields", {})

    def sv(v):
        kw = dict(
            activations=v.view(1, len(MPOSL), -1).float(),
            layer_indices=[1],
            position_indices=MPOSL,
            norm_match=True,
            scale=1.0,
        )
        if has_ref:
            kw["norm_match_ref"] = "residual_stream"
        return SteeringVector(**kw)

    def sv_steer(vl):
        return SteeringVector(
            activations=vl[:, None, :].float().clone(),
            layer_indices=SL,
            position_indices=MPOSL,
            norm_match=False,
            scale=1.0,
        )  # vl [L, d]

    def run(acts, seed_base):
        if a.steer_map:
            with torch.no_grad():
                VS = (
                    torch.einsum(
                        "bi,loi->blo", torch.stack(acts).to("cuda", torch.bfloat16), SW
                    ).float()
                    + SB
                ).cpu()
            mk = lambda i: sv_steer(VS[i])
        else:
            mk = lambda i: sv(acts[i])
        params = [
            SamplingParams(
                n=1,
                temperature=0.0 if a.greedy else 1.0,
                top_p=1.0,
                max_tokens=a.max_tokens,
                seed=seed_base + i,
                skip_special_tokens=True,
                extra_args={
                    "apply_steering_vectors": [mk(i)],
                    **(
                        {}
                        if a.no_grammar
                        else {
                            "lens_grammar": {
                                "max_bullet_tok": a.max_bullet_tok,
                                "ascii": not a.no_ascii,
                            }
                        }
                    ),
                },
            )
            for i in range(len(acts))
        ]
        outs = llm.generate([{"prompt_token_ids": PROMPT}] * len(acts), params, use_tqdm=False)
        return [o.outputs[0].text.strip() or "the" for o in outs]

    fams = sorted(os.path.basename(f)[:-3] for f in glob.glob(f"{a.cache}/*.pt"))
    if a.families:
        fams = [f for f in fams if f in set(a.families.split(","))]
    skip = set(x for x in a.skip_families.split(",") if x)
    fams = [f for f in fams if f not in skip]
    root = f"{a.results_root}/{a.tag}"
    os.makedirs(root, exist_ok=True)
    stats = {}
    for fi, fam in enumerate(fams):
        if os.path.exists(f"{root}/{fam}.jsonl.done"):
            print(f"[wsb-gen] {fam}: done already", flush=True)
            continue
        st = torch.load(f"{a.cache}/{fam}.pt")
        n = len(st["id"])
        t1 = time.time()
        out = open(f"{root}/{fam}.jsonl", "w")
        if a.mode == "prev4":
            st["acts"] = st["acts4"]
        for c0 in range(0, n, a.chunk):
            texts = run(
                [st["acts"][i] for i in range(c0, min(n, c0 + a.chunk))], 1_000_003 * (fi + 1) + c0
            )
            for j, t in enumerate(texts):
                i = c0 + j
                out.write(
                    json.dumps(
                        {
                            "id": st["id"][i],
                            "layer": 42,
                            "pos": int(st["pos"][i]),
                            "token": st["token"][i],
                            "samples": [t],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            out.flush()
        out.close()
        open(f"{root}/{fam}.jsonl.done", "w").write(f"{n}\n")
        stats[fam] = n
        print(
            f"[wsb-gen] {fam}: {n} cells | {(time.time() - t1) / 60:.1f} min | total {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )
    json.dump(
        {"tag": a.tag, "ckpt": a.ckpt, "families": stats, "minutes": (time.time() - t0) / 60},
        open(f"{root}/run_config.json", "w"),
        indent=1,
    )
    print(
        f"WSB_GEN_DONE {a.tag} {sum(stats.values())} cells {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()

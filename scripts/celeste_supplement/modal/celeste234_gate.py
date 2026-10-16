# ruff: noqa
"""Celeste supplement tasks 2-4: gate Claude's candidates on the subject model (stock Qwen3.6-27B, chat template, thinking off) with plain vLLM.
Per task: a greedy behavioural gate (the model computes the target) and a leak check (k samples at T=1.0 from every read site, capped at
the read horizon, must never contain the target or an alias). Output: <out>/<task>.json = every candidate with its gate outputs, read
sites (absolute positions in the captured token sequence), leak samples (= the sampling baseline) and a drop reason.
    python celeste234_gate.py --task all --cands-dir /vol_c/tasks234/candidates --out /vol_c/tasks234/gate [--limit 30]"""

import argparse, json, os, random, re, time, unicodedata

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="all", choices=["all", "encoded", "lookahead", "withheld"])
ap.add_argument("--cands-dir", default="/vol_c/tasks234/candidates")
ap.add_argument("--out", default="/vol_c/tasks234/gate")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gpu-mem", type=float, default=0.90)
ap.add_argument("--max-num-seqs", type=int, default=256)
ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
a = ap.parse_args()

FOLLOWUP = "What was the answer? Reply with just the word."
ENCODINGS = {
    "letter_count": "Reply with only the number of letters in the answer, as a digit.",
    "first_letter": "Reply with only the first letter of the answer.",
    "last_letter": "Reply with only the last letter of the answer.",
    "vowel_count": "Reply with only the number of vowels (a, e, i, o, u) in the answer, as a digit.",
    "list_position": "Here is a list: {lst}. Reply with only the number of the entry in this list that is the answer.",
    "first_letter_index": "Reply with only the position of the answer's first letter in the alphabet (A=1, B=2, ..., Z=26), as a number.",
    "letter_parity": "Reply with only the word 'odd' or the word 'even', according to whether the number of letters in the answer is odd or even.",
}
PREAMBLES = [
    (
        "repeat3",
        "First write the sentence 'I have thought about this and here is my answer:' three times, on separate lines, then {imp}.",
    ),
    ("count20", "First count from one to twenty in words, separated by commas, then {imp}."),
    (
        "alphabet",
        "First write the 26 letters of the alphabet from A to Z separated by spaces, then {imp}.",
    ),
    (
        "sentence2",
        "Begin your reply by writing the sentence 'Let me think about this carefully before I answer.' twice, then {imp}.",
    ),
    (
        "weekdays",
        "First list the seven days of the week in order, one per line, numbered 1 to 7, then {imp}.",
    ),
    (
        "months12",
        "First list the twelve months of the year in order, separated by commas, then {imp}.",
    ),
    (
        "disclaimer",
        "Start your reply with exactly this sentence: 'The following answer is given to the best of my knowledge and after careful consideration of the question.' Then {imp}.",
    ),
    (
        "countdown",
        "First count down from twenty to one in digits, separated by spaces, then {imp}.",
    ),
]
LOOKAHEAD_TAIL = " Write the preamble first, exactly as instructed, then the one-word answer."
MIN_HORIZON = 16
WITHHELD_TAIL = " Never write the answer itself, in any form, anywhere in your reply."
CANDS = {"encoded": "questions.json", "lookahead": "questions.json", "withheld": "withheld.json"}


def fold(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if c.isalnum())


def norm_words(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s)


def contains_any(text, forms):
    """Conservative (substring after folding): the leak rule. 'pianos', "Canberra's" count as leaks."""
    ft = fold(text)
    return any(fold(f) and fold(f) in ft for f in forms)


def names_any(text, forms):
    """Word-boundary match: the behavioural gate (the model named the target)."""
    nt = norm_words(text)
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(norm_words(f).strip())}(?![a-z0-9])", nt)
        for f in forms
        if norm_words(f).strip()
    )


def expected_encoding(enc, target, order):
    t = target
    if enc == "letter_count":
        return str(sum(ch.isalpha() for ch in t))
    if enc == "first_letter":
        return t[0].upper()
    if enc == "last_letter":
        return t[-1].upper()
    if enc == "vowel_count":
        return str(sum(ch in "aeiou" for ch in t.lower()))
    if enc == "list_position":
        return str(order.index(t) + 1)
    if enc == "first_letter_index":
        return str(ord(t[0].upper()) - 64)
    if enc == "letter_parity":
        return "odd" if sum(ch.isalpha() for ch in t) % 2 else "even"
    raise KeyError(enc)


def encoded_ok(enc, out, expected):
    if enc in ("letter_count", "vowel_count", "list_position", "first_letter_index"):
        m = re.findall(r"\d+", out)
        return bool(m) and m[0] == expected and len(m) == 1
    if enc in ("first_letter", "last_letter"):
        singles = re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", out)
        return (
            len(singles) == 1
            and singles[0].upper() == expected
            and len(re.findall(r"[A-Za-z]{2,}", out)) <= 4
        )
    if enc == "letter_parity":
        lo = out.lower()
        has_odd, has_even = "odd" in lo, "even" in lo
        return (has_odd != has_even) and (("odd" if has_odd else "even") == expected)
    raise KeyError(enc)


class Engine:
    def __init__(self):
        from vllm import LLM
        from transformers import AutoTokenizer
        from rollout_vllm_paths import resolve_local, local_model_dir

        mp = local_model_dir(a.model, resolve_local(a.model))
        self.tok = tok = AutoTokenizer.from_pretrained(mp)
        self.EOS = {
            tok.convert_tokens_to_ids("<|im_end|>"),
            tok.convert_tokens_to_ids("<|endoftext|>"),
        }
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        extra = {}
        try:
            from vllm.config.attention import AttentionConfig

            extra["attention_config"] = AttentionConfig(backend="FLASH_ATTN")
        except Exception:
            pass
        self.llm = LLM(
            model=mp,
            tokenizer=mp,
            dtype="bfloat16",
            gpu_memory_utilization=a.gpu_mem,
            max_model_len=2048,
            max_num_seqs=a.max_num_seqs,
            enforce_eager=False,
            enable_prefix_caching=True,
            disable_log_stats=True,
            seed=a.seed,
            **extra,
        )

    def render(self, user, prefill_ids=()):
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        return ids + list(prefill_ids)

    def strip_eos(self, ids):
        ids = list(ids)
        while ids and ids[-1] in self.EOS:
            ids.pop()
        return ids

    def gen(self, prompts, n, temperature, max_tokens, seed_base):
        """prompts: list of token-id lists (or list of (ids, max_tokens)) -> per prompt a list of n dicts {ids, text, finish}."""
        if not prompts:
            return []
        from vllm import SamplingParams

        norm = [(p, max_tokens) if isinstance(p[0], int) else p for p in prompts]
        params = [
            SamplingParams(
                n=n,
                temperature=temperature,
                top_p=1.0,
                max_tokens=mt,
                seed=seed_base + i,
                skip_special_tokens=True,
            )
            for i, (_p, mt) in enumerate(norm)
        ]
        outs = self.llm.generate(
            [{"prompt_token_ids": p} for p, _mt in norm], params, use_tqdm=False
        )
        return [
            [
                {"ids": self.strip_eos(o.token_ids), "text": o.text, "finish": o.finish_reason}
                for o in out.outputs
            ]
            for out in outs
        ]

    def first_containing(self, gen_ids, forms):
        """(start, end) token span of the first appearance of a form in the generated ids, else None: end = the first
        index whose decoded prefix contains a form; start = the largest index s with ids[s:end+1] still containing one."""
        for j in range(len(gen_ids)):
            if contains_any(self.tok.decode(gen_ids[: j + 1]), forms):
                for s in range(j, -1, -1):
                    if contains_any(self.tok.decode(gen_ids[s : j + 1]), forms):
                        return (s, j)
                return (0, j)
        return None


def leak_check(E, alive, labels, max_tokens_of, seed0):
    """k samples at T=1.0 from each read site, capped at the site's horizon; fills s['samples'], s['n_leak']."""
    for li, lab in enumerate(labels):
        sub = [(r, s) for r in alive for s in r["sites"] if s["label"] == lab]
        outs = E.gen(
            [(r["input_ids"][: s["abs"] + 1], max_tokens_of(s)) for r, s in sub],
            a.k,
            1.0,
            None,
            seed0 + li * 50_000,
        )
        for (r, s), o in zip(sub, outs):
            s["samples"] = [x["text"] for x in o]
            s["n_leak"] = sum(contains_any(x["text"], r["forms"]) for x in o)


def run_encoded(E, cands):
    encs = list(ENCODINGS)
    recs = []
    for i, c in enumerate(cands):
        enc = encs[i % len(encs)]
        forms = [c["target"], *c["aliases"]]
        order = None
        if enc == "list_position":
            order = [c["target"], *c["distractors"][:4]]
            random.Random(a.seed * 7919 + i).shuffle(order)
            instr = ENCODINGS[enc].format(
                lst=", ".join(f"{k + 1}. {w}" for k, w in enumerate(order))
            )
        else:
            instr = ENCODINGS[enc]
        recs.append(
            {
                **c,
                "encoding": enc,
                "list_order": order,
                "plain_prompt": c["question"],
                "prompt": f"{c['question']} {instr}",
                "expected": expected_encoding(enc, c["target"], order),
                "forms": forms,
            }
        )
    plain = E.gen([E.render(r["plain_prompt"]) for r in recs], 1, 0.0, 160, 1)
    for r, o in zip(recs, plain):
        r["plain_answer"] = o[0]["text"]
        r["gate_plain"] = names_any(o[0]["text"], r["forms"])
    enc_out = E.gen([E.render(r["prompt"]) for r in recs], 1, 0.0, 24, 2)
    for r, o in zip(recs, enc_out):
        r["encoded_output"] = o[0]["text"]
        r["encoded_ids"] = o[0]["ids"]
        r["gate_encoded"] = encoded_ok(
            r["encoding"], o[0]["text"], r["expected"]
        ) and not contains_any(o[0]["text"], r["forms"])
    alive = [r for r in recs if r["gate_plain"] and r["gate_encoded"]]
    for r in alive:  # A = final prompt token (assistant onset), B = last token of the model's own encoded output (prefill)
        base = E.render(r["prompt"])
        r["input_ids"] = base + r["encoded_ids"]
        r["n_prompt"] = len(base)
        r["sites"] = [{"label": "prompt_end", "abs": len(base) - 1, "horizon": 32}]
        if r["encoded_ids"]:
            r["sites"].append(
                {"label": "answer_end", "abs": len(r["input_ids"]) - 1, "horizon": 32}
            )
    leak_check(E, alive, ["prompt_end", "answer_end"], lambda s: s["horizon"], 100)
    for r in recs:
        if not r["gate_plain"]:
            r["drop"] = "plain_answer_wrong"
        elif not r["gate_encoded"]:
            r["drop"] = "encoded_output_wrong"
        elif any(s["n_leak"] for s in r["sites"]):
            r["drop"] = "leak"
        else:
            r["drop"] = None
    return recs


def run_lookahead(E, cands):
    recs = []
    for i, c in enumerate(cands):
        key, tpl = PREAMBLES[i % len(PREAMBLES)]
        recs.append(
            {
                **c,
                "preamble": key,
                "prompt": tpl.format(imp=c["imperative"]) + LOOKAHEAD_TAIL,
                "forms": [c["target"], *c["aliases"]],
            }
        )
    full = E.gen([E.render(r["prompt"]) for r in recs], 1, 0.0, 200, 3)
    for r, o in zip(recs, full):
        r["greedy_output"] = o[0]["text"]
        r["greedy_ids"] = o[0]["ids"]
        r["greedy_finish"] = o[0]["finish"]
        span = E.first_containing(o[0]["ids"], r["forms"])
        r["answer_start"] = span[0] if span else None
        r["horizon"] = span[0] if span else None
        r["gate_answer"] = span is not None
        r["answer_text"] = E.tok.decode(o[0]["ids"][span[0] : span[1] + 1]) if span else None
    alive = [r for r in recs if r["gate_answer"] and r["horizon"] >= MIN_HORIZON]
    for r in alive:
        base = E.render(r["prompt"])
        H = r["horizon"]
        pre = r["greedy_ids"][: H - MIN_HORIZON] if H >= MIN_HORIZON + 8 else []
        r["input_ids"] = base + pre
        r["n_prompt"] = len(base)
        r["sites"] = [{"label": "prompt_end", "abs": len(base) - 1, "horizon": H}]
        if pre:
            r["sites"].append(
                {
                    "label": "preamble_minus16",
                    "abs": len(r["input_ids"]) - 1,
                    "horizon": MIN_HORIZON,
                }
            )
    leak_check(E, alive, ["prompt_end", "preamble_minus16"], lambda s: s["horizon"], 200)
    for r in recs:
        if not r["gate_answer"]:
            r["drop"] = "answer_not_given"
        elif r["horizon"] < MIN_HORIZON:
            r["drop"] = "horizon_short"
        elif any(s["n_leak"] for s in r["sites"]):
            r["drop"] = "leak"
        else:
            r["drop"] = None
    return recs


def run_withheld(E, cands):
    recs = [
        {**c, "prompt": c["prompt"].rstrip() + WITHHELD_TAIL, "forms": [c["target"], *c["aliases"]]}
        for c in cands
    ]
    rep = E.gen([E.render(r["prompt"]) for r in recs], 1, 0.0, 220, 4)
    for r, o in zip(recs, rep):
        r["reply"] = o[0]["text"]
        r["reply_ids"] = o[0]["ids"]
        r["reply_finish"] = o[0]["finish"]
        r["gate_no_leak_reply"] = (
            not contains_any(o[0]["text"], r["forms"])
            and o[0]["finish"] == "stop"
            and len(o[0]["ids"]) >= 8
        )
    alive = [r for r in recs if r["gate_no_leak_reply"]]
    tail = E.tok(
        f"\n<|im_start|>user\n{FOLLOWUP}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )["input_ids"]
    fu = E.gen(
        [E.render(r["prompt"]) + r["reply_ids"] + [E.im_end] + tail for r in alive], 1, 0.0, 16, 5
    )
    for r, o in zip(alive, fu):
        r["followup_answer"] = o[0]["text"]
        r["gate_followup"] = names_any(o[0]["text"], r["forms"])
    alive = [r for r in alive if r["gate_followup"]]
    for r in alive:
        base = E.render(r["prompt"])
        ids = r["reply_ids"]
        r["input_ids"] = base + ids
        r["n_prompt"] = len(base)
        n = len(ids)
        first = next(
            (j for j in range(3, n - 1) if re.search(r"[.!?]\s*$|\n", E.tok.decode([ids[j]]))), None
        )
        sites = ([("sentence1_end", first)] if first is not None else []) + [
            ("reply_mid", n // 2),
            ("reply_end", n - 1),
        ]
        seen = set()
        r["sites"] = []
        for lab, j in sites:
            if j in seen:
                continue
            seen.add(j)
            r["sites"].append({"label": lab, "abs": len(base) + j, "horizon": 48})
    leak_check(E, alive, ["sentence1_end", "reply_mid", "reply_end"], lambda s: s["horizon"], 300)
    for r in recs:
        if not r["gate_no_leak_reply"]:
            r["drop"] = (
                "reply_names_target"
                if contains_any(r["reply"], r["forms"])
                else "reply_truncated_or_short"
            )
        elif not r.get("gate_followup"):
            r["drop"] = "followup_wrong"
        elif any(s["n_leak"] for s in r["sites"]):
            r["drop"] = "leak"
        else:
            r["drop"] = None
    return recs


RUN = {"encoded": run_encoded, "lookahead": run_lookahead, "withheld": run_withheld}


def main():
    t0 = time.time()
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    E = Engine()
    print(f"[gate] vLLM up {(time.time() - t0) / 60:.1f} min", flush=True)
    for task in list(RUN) if a.task == "all" else [a.task]:
        t1 = time.time()
        cands = json.load(open(f"{a.cands_dir}/{CANDS[task]}"))
        if task in ("encoded", "lookahead"):  # the shared question pool, split by parity
            cands = [c for i, c in enumerate(cands) if (i % 2 == 0) == (task == "encoded")]
        cands = cands[:: a.stride]
        if a.limit:
            cands = cands[: a.limit]
        print(f"[gate] {task}: {len(cands)} candidates", flush=True)
        recs = RUN[task](E, cands)
        kept = [r for r in recs if r["drop"] is None]
        funnel = {"candidates": len(recs), "kept": len(kept)}
        for r in recs:
            funnel[r["drop"] or "kept_"] = funnel.get(r["drop"] or "kept_", 0) + 1
        json.dump(
            {"task": task, "k": a.k, "model": a.model, "funnel": funnel, "records": recs},
            open(f"{a.out}/{task}.json", "w"),
            ensure_ascii=False,
        )
        print(
            f"[gate] {task} funnel {json.dumps(funnel)} | {(time.time() - t1) / 60:.1f} min",
            flush=True,
        )
    print(f"GATE_DONE {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

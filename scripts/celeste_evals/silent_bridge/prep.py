"""Select gate passes rated low co-occurrence (caps per bridge and per answer), render the chat
ids and write selected.json, cells/<family>.jsonl, jlens_pool.json and free_cands.json to --outdir.
    python3 prep.py --gate gate.json --outdir bank [--leaks leak_forms.json]"""

import argparse
import collections
import json
import os

from transformers import AutoTokenizer

FAM = "celeste_silent_bridge"
SUFFIX = "\n\nANSWER IN ONE WORD, DO NOT SAY ANYTHING ELSE."
ap = argparse.ArgumentParser()
ap.add_argument("--gate", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument(
    "--leaks",
    default=None,
    help="leak_forms.json: drop items whose answers name the bridge by a derived form",
)
ap.add_argument("--per-bridge", type=int, default=3)
ap.add_argument("--per-answer", type=int, default=6)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
IMS = tok.convert_tokens_to_ids("<|im_start|>")
with open(a.gate) as fh:
    items = json.load(fh)["items"]
leaks = set()
if a.leaks:
    with open(a.leaks) as fh:
        leaks = {k for k, v in json.load(fh).items() if v.get("leak")}


def target_of(c):
    forms = [
        f.strip()
        for f in (c.get("bridge_aliases") or [])
        if isinstance(f, str) and len(f.strip()) >= 3
    ]
    return min(forms, key=len) if forms else c["bridge"].strip()


keep, nb, na = [], collections.Counter(), collections.Counter()
for c in sorted(items, key=lambda c: c["id"]):
    g = c["gate"]
    if not (g["pass"] and c.get("cooccurrence") == "low") or c["id"] in leaks:
        continue
    t = target_of(c)
    ans = c["answer"].strip().lower()
    if nb[t.lower()] >= a.per_bridge or na[ans] >= a.per_answer:
        continue
    nb[t.lower()] += 1
    na[ans] += 1
    user = c["question"].strip() + SUFFIX
    ids = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    p_ims = max(i for i, x in enumerate(ids) if x == IMS)
    aliases = sorted(
        {
            f.strip()
            for f in [c["bridge"], *(c.get("bridge_aliases") or [])]
            if isinstance(f, str) and f.strip()
        }
        - {t},
        key=len,
    )
    keep.append(
        {
            **c,
            "target": t,
            "aliases": aliases,
            "user": user,
            "input_ids": ids,
            "read_sites": [
                {"label": "reply_start", "pos": p_ims, "token": "<|im_start|>"},
                {"label": "prompt_end", "pos": len(ids) - 1, "token": tok.decode([ids[-1]])},
            ],
        }
    )
os.makedirs(f"{a.outdir}/cells", exist_ok=True)
with open(f"{a.outdir}/selected.json", "w") as fh:
    json.dump(keep, fh, ensure_ascii=False)
with open(f"{a.outdir}/cells/{FAM}.jsonl", "w") as f:
    for c in keep:
        f.write(
            json.dumps(
                {
                    "family": FAM,
                    "id": c["id"],
                    "ids": c["input_ids"],
                    "prompt": c["user"],
                    "intermediates": [c["target"]],
                    "cells": [
                        {"pos": s["pos"], "abs": s["pos"], "token": s["token"]}
                        for s in c["read_sites"]
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
with open(f"{a.outdir}/jlens_pool.json", "w") as fh:
    json.dump(
        [
            {
                "name": c["id"],
                "prompt": c["question"],
                "user": c["user"],
                "intermediates": [c["target"]],
            }
            for c in keep
        ],
        fh,
        ensure_ascii=False,
    )
with open(f"{a.outdir}/free_cands.json", "w") as fh:
    json.dump(
        [
            {
                "name": c["id"],
                "prompt": c["question"],
                "user": c["user"],
                "inter": c["target"],
                "inter_forms": [c["target"], *c["aliases"]],
                "pos": s["pos"],
                "token": s["token"],
                "site": s["label"],
            }
            for c in keep
            for s in c["read_sites"]
        ],
        fh,
        ensure_ascii=False,
    )


def g(k):
    return sum(bool(c["gate"][k]) for c in items)


n_pass_low = sum(c["gate"]["pass"] and c.get("cooccurrence") == "low" for c in items)
n_leak_low = sum(
    c["gate"]["pass"] and c.get("cooccurrence") == "low" and c["id"] in leaks for c in items
)
print(
    f"{len(items)} gated | correct {g('correct')} | recall bridge {g('recall_bridge')} "
    f"hop2 {g('recall_hop2')} | generic probe misses "
    f"{sum(not c['gate']['shortcuts'] for c in items)} | silent {g('silent')} | pass {g('pass')} | "
    f"pass & low co-occurrence {n_pass_low} | derived-form leaks among those {n_leak_low} | "
    f"kept after caps {len(keep)}"
)

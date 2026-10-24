"""Per item, the read site where the J-lens (L42, top 20) surfaces the multi-word bridge: rare,
one-token signature words not yet in the text, one a NAME word; also <out>_free.json (leak check).
    python3 select_sites.py --gate gate.json --jlens jlens.json --out sites.json"""

import argparse
import json
import re

from transformers import AutoTokenizer
from wordfreq import zipf_frequency

ap = argparse.ArgumentParser()
ap.add_argument("--gate", required=True)
ap.add_argument("--jlens", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--min-score", type=int, default=2)
ap.add_argument(
    "--rare",
    type=float,
    default=4.5,
    help="zipf frequency below which a word counts as specific (darth 3.2, olympic 4.5; death 5.4)",
)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
SUFFIX = "\n\nANSWER IN ONE WORD, DO NOT SAY ANYTHING ELSE."
with open(a.gate) as fh:
    items = {c["id"]: c for c in json.load(fh)["items"]}
with open(a.jlens) as fh:
    J = {r["name"]: r for r in json.load(fh)["items"]}


def words(text):
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def inverted(w, prefix):
    """Text inversion, stem-aware: the word (or an inflection of it) is already in the text up to
    the read token."""
    w = w.lower()
    return any(
        x == w
        or (len(x) >= 4 and (x in w or w in x))
        or (len(x) >= 5 and len(w) >= 5 and x[:5] == w[:5])
        for x in prefix
    )


def single_token(w):
    """Only words with a one-token spelling count: the J-lens ranks a word by its FIRST token, so a
    fragment (" Fred") is weak evidence."""
    return any(
        len(tok(v, add_special_tokens=False)["input_ids"]) == 1
        for v in (" " + w, " " + w.capitalize(), " " + w.lower())
    )


out, n_eval = [], 0
for i, j in J.items():
    c = items[i]
    user = c["question"].strip() + SUFFIX
    ids = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    assert len(ids) == j["n_tokens"], (i, len(ids), j["n_tokens"])
    parts = {p.strip().lower() for p in c.get("name_parts") or [] if len(p.strip()) >= 3}
    best = None
    n_eval += 1
    all_sites = []
    for k, pos in enumerate(j["positions"]):
        prefix = words(tok.decode(ids[: pos + 1]))
        hits_name, hits_sig = [], []
        for w in j["ranks"]:
            rk = j["ranks"][w].get("42")
            if rk is None or rk[k] >= a.top or inverted(w, prefix) or not single_token(w):
                continue
            # common words are the clue's topic, not the intermediate
            if zipf_frequency(w.lower(), "en") >= a.rare:
                continue
            (hits_name if w.lower() in parts else hits_sig).append((w, rk[k]))
        score = 2 * len(hits_name) + len(hits_sig)
        if hits_name and score >= a.min_score:
            cand = {
                "pos": pos,
                "token": j["tokens"][k],
                "score": score,
                "name_hits": hits_name,
                "sig_hits": hits_sig,
                "top10_L42": j["top10_L42"][k],
            }
            all_sites.append(cand)
            if best is None or score > best["score"]:
                best = cand
    if best:
        out.append(
            {
                "id": i,
                "question": c["question"],
                "user": user,
                "bridge": c["bridge"],
                "aliases": c.get("aliases") or [],
                "name_parts": sorted(parts),
                "answer": c["answer"],
                "input_ids": ids,
                "site": best,
                "all_sites": all_sites,
            }
        )
with open(a.out, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
with open(a.out.replace(".json", "_free.json"), "w") as fh:
    json.dump(
        [
            {
                "name": o["id"],
                "prompt": o["question"],
                "user": o["user"],
                "inter": o["bridge"],
                "inter_forms": [o["bridge"], *o["aliases"]],
                "pos": st["pos"],
                "token": st["token"],
                "site": "jlens_site",
            }
            for o in out
            for st in o["all_sites"]
        ],
        fh,
        ensure_ascii=False,
    )
print(f"{len(out)} of {n_eval} items have a J-lens site (score >= {a.min_score})")

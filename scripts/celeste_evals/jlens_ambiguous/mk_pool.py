"""J-lens scan pool for the multi-word-bridge candidates: per item the exact user message (question
+ all-caps suffix) and, as words to rank, the bridge's signature words and distinctive name words.
    python3 mk_pool.py --cands cands_audited.json --out jlens_pool.json"""

import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--cands", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
SUFFIX = "\n\nANSWER IN ONE WORD, DO NOT SAY ANYTHING ELSE."
pool = []
with open(a.cands) as fh:
    cands = json.load(fh)
for c in cands:
    words = []
    for w in [*c["signature_words"], *(c.get("name_parts") or [])]:
        w = w.strip()
        if len(w) >= 3 and w.lower() not in {x.lower() for x in words}:
            words.append(w)
    pool.append(
        {
            "name": c["id"],
            "prompt": c["question"],
            "user": c["question"].strip() + SUFFIX,
            "intermediates": words,
        }
    )
with open(a.out, "w") as fh:
    json.dump(pool, fh, ensure_ascii=False)
print(
    len(pool),
    "items | words per item",
    sum(len(p["intermediates"]) for p in pool) / max(1, len(pool)),
)

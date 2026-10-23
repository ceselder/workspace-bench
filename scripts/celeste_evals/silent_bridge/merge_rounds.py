"""Merge two gated rounds before selection: round-2 items repeating a round-1 question or a kept
(bridge, answer) pair are dropped; leak checks are unioned. Round-1 ids sort first, so prep.py keeps
them under the per-bridge cap. usage: merge_rounds.py G1 G2 SELECTED1 L1 L2 OUT_GATE OUT_LEAKS"""

import json
import sys


def target(c: dict) -> str:
    forms = [f.strip() for f in c.get("bridge_aliases") or [] if isinstance(f, str)]
    forms = [f for f in forms if len(f) >= 3]
    return (min(forms, key=len) if forms else c["bridge"].strip()).lower()


def main() -> None:
    g1_path, g2_path, sel1_path, l1_path, l2_path, out_gate, out_leaks = sys.argv[1:8]
    with open(g1_path) as f:
        g1 = json.load(f)
    with open(g2_path) as f:
        g2 = json.load(f)
    with open(sel1_path) as f:
        sel1 = json.load(f)
    seen_q = {c["question"].strip().lower() for c in g1["items"]}
    seen_pairs = {(c["target"].lower(), c["answer"].strip().lower()) for c in sel1}
    keep = [
        c
        for c in g2["items"]
        if c["question"].strip().lower() not in seen_q
        and (target(c), c["answer"].strip().lower()) not in seen_pairs
    ]
    with open(out_gate, "w") as f:
        json.dump({**g1, "items": g1["items"] + keep}, f, ensure_ascii=False)
    leaks = {}
    for p in (l1_path, l2_path):
        with open(p) as f:
            leaks.update(json.load(f))
    with open(out_leaks, "w") as f:
        json.dump(leaks, f, ensure_ascii=False)
    print(f"round 2: kept {len(keep)} of {len(g2['items'])}; merged {len(g1['items']) + len(keep)}")


if __name__ == "__main__":
    main()

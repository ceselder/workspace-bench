"""Merge the rounds' gate, sites, leak, derived-form and specificity files into one set of
build_bank.py inputs; a later round's repeat of an earlier question is dropped (ids sort by round).
    python3 merge_rounds.py DIR OUT_DIR r2 r3   (round 1 = the unsuffixed files in DIR)"""

import json
import sys
from pathlib import Path

# (round-1 file, file stem of later rounds, merged name, kind); later rounds are <stem>_<rN>.json
FILES = [
    ("gate2.json", "gate", "gate.json", "items"),
    ("sites_final.json", "sites_final", "sites.json", "list"),
    ("free_final.json", "free_final", "free.json", "items"),
    ("leak_forms.json", "leak_forms", "leaks.json", "dict"),
    ("specific_final.json", "specific_final", "specific.json", "dict"),
]


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def key(x: dict) -> str:
    return x.get("id", x.get("name"))


def main() -> None:
    src, out, rounds = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    seen = {c["question"].strip().lower() for c in load(src / "gate2.json")["items"]}
    dropped: set[str] = set()
    for r in rounds:
        for c in load(src / f"gate_{r}.json")["items"]:
            q = c["question"].strip().lower()
            if q in seen:
                dropped.add(c["id"])
            seen.add(q)
    for first, stem, name, kind in FILES:
        merged = load(src / first)
        for r in rounds:
            nxt = load(src / f"{stem}_{r}.json")
            if kind == "items":
                merged = {
                    **merged,
                    "items": merged["items"] + [x for x in nxt["items"] if key(x) not in dropped],
                }
            elif kind == "list":
                merged = merged + [x for x in nxt if x["id"] not in dropped]
            else:
                merged = {**merged, **{k: v for k, v in nxt.items() if k not in dropped}}
        with open(out / name, "w") as f:
            json.dump(merged, f, ensure_ascii=False)
    print(f"merged round 1 + {', '.join(rounds)} into {out}; dropped {len(dropped)} repeats")


if __name__ == "__main__":
    main()

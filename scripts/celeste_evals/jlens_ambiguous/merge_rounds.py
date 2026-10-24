"""Merge two rounds' gate, sites, leak, derived-form and specificity files into one set of
build_bank.py inputs; round-2 repeats of a round-1 question are dropped (round-1 ids sort first).
    python3 merge_rounds.py R1_DIR R2_DIR OUT_DIR"""

import json
import sys
from pathlib import Path

# (round-1 name, round-2 name, merged name, kind)
FILES = [
    ("gate2.json", "gate_r2.json", "gate.json", "items"),
    ("sites_final.json", "sites_final_r2.json", "sites.json", "list"),
    ("free_final.json", "free_final_r2.json", "free.json", "items"),
    ("leak_forms.json", "leak_forms_r2.json", "leaks.json", "dict"),
    ("specific_final.json", "specific_final_r2.json", "specific.json", "dict"),
]


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    r1, r2, out = (Path(p) for p in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    seen = {c["question"].strip().lower() for c in load(r1 / "gate2.json")["items"]}
    dropped = {
        c["id"] for c in load(r2 / "gate_r2.json")["items"] if c["question"].strip().lower() in seen
    }
    for n1, n2, name, kind in FILES:
        a, b = load(r1 / n1), load(r2 / n2)
        if kind == "items":
            merged = {
                **a,
                "items": a["items"]
                + [x for x in b["items"] if x.get("id", x.get("name")) not in dropped],
            }
        elif kind == "list":
            merged = a + [x for x in b if x["id"] not in dropped]
        else:
            merged = {**a, **{k: v for k, v in b.items() if k not in dropped}}
        with open(out / name, "w") as f:
            json.dump(merged, f, ensure_ascii=False)
    print(f"merged into {out}; dropped {len(dropped)} round-2 repeats of round-1 questions")


if __name__ == "__main__":
    main()

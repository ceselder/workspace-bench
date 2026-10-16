"""Merge several candidate rounds (gen_candidates.py out dirs) into one, de-duplicated on the folded
target (questions: per pool; withheld: per kind) and on the folded prompt text."""

import argparse
import json
import unicodedata
from pathlib import Path


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if c.isalnum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fname, keyf in (
        ("questions.json", lambda it: fold(it["target"])),
        ("withheld.json", lambda it: fold(it["target"]) + "|" + it["kind"]),
    ):
        merged, seen_k, seen_p = [], set(), set()
        for r in a.rounds:
            p = Path(r) / fname
            if not p.exists():
                continue
            for it in json.loads(p.read_text()):
                k = keyf(it)
                pf = fold(it.get("question") or it.get("prompt"))
                if k in seen_k or pf in seen_p:
                    continue
                seen_k.add(k)
                seen_p.add(pf)
                merged.append(it)
        (out / fname).write_text(json.dumps(merged, indent=1, ensure_ascii=False))
        print(fname, len(merged))


if __name__ == "__main__":
    main()

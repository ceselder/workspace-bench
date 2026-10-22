"""Assemble celeste_coherence readouts from an arm's per-family readouts (no new generation).

    python scripts/celeste_evals/coherence_assemble.py --root <dir of <family>.jsonl> \
        --out <arm>/celeste_coherence.jsonl

Every bank item is another family's item: its rows are copied from ``<root>/<family>.jsonl`` (or
``.jsonl.gz``) with the id rewritten to the coherence item's label.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wsbench.banks import label_of, load_bank

BANK = Path(__file__).resolve().parents[2] / "evals" / "celeste_coherence" / "items.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    _h, items = load_bank(BANK)
    want: dict[str, dict[str, str]] = {}
    for it in items:
        want.setdefault(it["family"], {})[it["source_id"]] = label_of(it["name"])
    n, missing = 0, []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as out:
        for fam, ids in sorted(want.items()):
            src = a.root / f"{fam}.jsonl"
            if not src.exists() and (a.root / f"{fam}.jsonl.gz").exists():
                src = a.root / f"{fam}.jsonl.gz"
            if not src.exists():
                missing.append(fam)
                continue
            op = gzip.open if src.suffix == ".gz" else open
            with op(src, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("id") in ids:
                        out.write(json.dumps({**r, "id": ids[r["id"]]}, ensure_ascii=False) + "\n")
                        n += 1
    print(f"{n} rows -> {a.out}" + (f"; no readouts for {', '.join(missing)}" if missing else ""))


if __name__ == "__main__":
    main()

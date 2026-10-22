"""Continuation-gated re-scoring: each lens's pass rate on the items the sampling baseline fails.

    python scripts/celeste_evals/cont_gate.py --cont outputs/judged/continuation \
        --lens olens=outputs/judged/olens --lens jlens=outputs/judged/jlens [--json out.json]

Every directory holds ``<family>/results.json`` (``wsbench judge`` / ``wsbench run`` output). An
item passes if any of its rows passes (the rows' own ``pass`` flag; ``correct`` for
conjunctive_association); families whose rows carry no item-level pass flag are listed as not
gateable. No calls are made: the judged rows are re-read.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

PASS_FIELD = {"conjunctive_association": "correct"}


def item_passes(family: str, rows: list[dict]) -> dict[str, bool] | None:
    field = PASS_FIELD.get(family, "pass")
    out: dict[str, bool] = defaultdict(bool)
    for r in rows:
        if "id" not in r:
            continue
        if field not in r:
            return None
        out[r["id"]] = out[r["id"]] or bool(r[field])
    return dict(out) or None


def load(root: Path) -> dict[str, dict[str, bool] | None]:
    res = {}
    for p in sorted(root.glob("*/results.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        res[p.parent.name] = item_passes(p.parent.name, d.get("rows") or [])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cont", required=True, type=Path, help="judged results of the sampling baseline"
    )
    ap.add_argument("--lens", action="append", default=[], help="name=DIR, repeatable")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    cont = load(a.cont)
    lenses = {s.split("=", 1)[0]: load(Path(s.split("=", 1)[1])) for s in a.lens}
    names = list(lenses)
    table, skipped = {}, []
    for fam, cp in sorted(cont.items()):
        if cp is None or any(lenses[n].get(fam) is None for n in names):
            skipped.append(fam)
            continue
        ids = sorted(set(cp).intersection(*(lenses[n][fam] for n in names)))
        gated = [i for i in ids if not cp[i]]
        row = {
            "n": len(ids),
            "continuation": sum(cp[i] for i in ids) / len(ids),
            "n_gated": len(gated),
        }
        for n in names:
            lp = lenses[n][fam]
            row[n] = {
                "all": sum(lp[i] for i in ids) / len(ids),
                "gated": (sum(lp[i] for i in gated) / len(gated)) if gated else None,
            }
        table[fam] = row
    head = "| family | n | continuation passes | items it fails | " + " | ".join(
        f"{n} (all → gated)" for n in names
    )
    print(head + " |\n|" + "---|" * (4 + len(names)))
    for fam, r in table.items():
        cells = []
        for n in names:
            g = r[n]["gated"]
            cells.append(f"{r[n]['all']:.2f} → {'—' if g is None else f'{g:.2f}'}")
        print(
            f"| {fam} | {r['n']} | {r['continuation']:.2f} | {r['n_gated']} | "
            + " | ".join(cells)
            + " |"
        )
    if skipped:
        print(
            f"\nnot gateable (no item-level pass flag, or missing in an arm): {', '.join(skipped)}"
        )
    if a.json:
        a.json.write_text(json.dumps({"families": table, "not_gateable": skipped}, indent=1))


if __name__ == "__main__":
    main()

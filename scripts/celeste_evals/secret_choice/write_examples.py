# ruff: noqa
"""Write the toy examples/readouts/celeste_secret_choice.jsonl (three items of the frozen bank: one
readout naming the pick, one describing it, one off-topic) for the dry run."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
bank = json.loads((ROOT / "evals/celeste_secret_choice/items.json").read_text())
items = bank["items"][:3]
texts = [
    "* a {pick}, held in mind the whole time\n* grey rain on the pavement\n* the next line of the poem",
    "* something the writer keeps picturing without naming it\n* the rhythm of the lines\n* an ending that rhymes",
    "* quarterly revenue figures\n* a spreadsheet\n* nothing about the poem",
]
rows = []
for it, t in zip(items, texts):
    for s in it["read_sites"]:
        rows.append(
            {"id": it["name"], "layer": 42, "pos": s["pos"], "samples": [t.format(pick=it["pick"])]}
        )
out = ROOT / "examples/readouts/celeste_secret_choice.jsonl"
out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print("wrote", out, len(rows), "rows")

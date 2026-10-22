# ruff: noqa
"""celeste_foreshadow, step 5: gate with the scoring instrument itself. Each read site's 16 pool
continuations are judged as two 8-bullet readouts by the family's own judge; a target the judge
credits to any of them is dropped from that site (the Sonnet leak check of step 3 misses some
paraphrases). Sites and items left without targets are dropped. Writes the final bank in place.

    env -u ANTHROPIC_API_KEY ANTHROPIC_PROFILE=default PYTHONPATH=src python gate_with_judge.py --bank evals/celeste_foreshadow/items.json --work /tmp/fs_poolgate
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True)
ap.add_argument("--work", required=True)
a = ap.parse_args()
sys.path.insert(0, "src")
from wsbench.banks import label_of

bank_path = Path(a.bank)
bank = json.loads(bank_path.read_text())
work = Path(a.work)
work.mkdir(parents=True, exist_ok=True)
rows = []
for it in bank["items"]:
    for s in it["read_sites"]:
        p = [" ".join(x.split()) for x in s["pool"]]
        rows.append(
            {
                "id": label_of(it["name"]),
                "layer": 42,
                "pos": s["pos"],
                "token": s["token"],
                "samples": [
                    "\n".join("- " + x for x in p[:8]),
                    "\n".join("- " + x for x in p[8:16]),
                ],
            }
        )
    for c in (
        it.get("control_sites") or []
    ):  # the plan reads them; an empty readout is a negative without a call
        rows.append(
            {
                "id": label_of(it["name"]),
                "layer": 42,
                "pos": c["pos"],
                "token": c["token"],
                "samples": [""],
            }
        )
(work / "pool.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
cmd = [
    sys.executable,
    "-m",
    "wsbench",
    "judge",
    "family=celeste_foreshadow",
    f"readouts={work / 'pool.jsonl'}",
    f"out={work / 'judged'}",
    "layers=42",
    "concurrency=64",
    "rpm=900",
]
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
res = json.loads((work / "judged" / "results.json").read_text())
credited = {}  # (item id, pos) -> set of target indices (into the site's target list) credited to the pool
unjudged = 0
for r in res["rows"]:
    unjudged += r["n_unjudged"]
    for ck, c in r["cells"].items():
        pos = int(ck.split("|p")[1])
        credited[(r["id"], pos)] = {h for h in c["hits"]}
if unjudged:
    raise SystemExit(
        f"{unjudged} pool cells unjudged: re-run (the cache resumes) before freezing the bank"
    )
funnel = Counter()
items = []
for it in bank["items"]:
    iid = label_of(it["name"])
    sites = []
    for s in it["read_sites"]:
        drop = credited.get((iid, s["pos"]), set())
        funnel["targets_before"] += len(s["targets"])
        funnel["targets_credited_to_pool"] += len(drop)
        tg = [t for k, t in enumerate(s["targets"]) if k not in drop]
        if tg:
            sites.append({**s, "targets": tg, "n_judge_leaked": len(drop)})
    if not sites:
        funnel["items_every_target_credited_to_pool"] += 1
        continue
    items.append({**it, "read_sites": sites})
bank["funnel"] = {
    **bank["funnel"],
    "kept_before_judge_gate": len(bank["items"]),
    "dropped_by_judge_gate": funnel["items_every_target_credited_to_pool"],
    "kept": len(items),
}
bank["gate"]["judge_gate"] = (
    "each site's 16 pool continuations judged as two 8-bullet readouts by this family's own judge "
    f"(prompt {res['config']['prompt_version']}); a target credited to either is dropped at that site"
)
bank["n_items"] = len(items)
bank["items"] = items
bank_path.write_text(json.dumps(bank, ensure_ascii=False, indent=1) + "\n")
print(
    json.dumps(dict(funnel)),
    "| items kept",
    len(items),
    "| sites",
    sum(len(i["read_sites"]) for i in items),
)

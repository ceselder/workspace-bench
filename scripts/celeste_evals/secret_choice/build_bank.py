# ruff: noqa
"""celeste_secret_choice, step 3: the bank. Merges the step-1 shard JSONs and the step-2 Sonnet
verdicts, applies the gate (committed >= 7/8 with the pick's prompt-level rate <= 0.5; the text so
far and 16 capped continuations never name the pick; Sonnet gives the pick <= p_max from the text)
and writes one item per (prompt, rollout) with >= 1 kept site: the captured render (chat template,
thinking off, prompt + the rollout up to the last kept site), the read sites and every gate number.

    python build_bank.py --gate 'gate_vivid_s*.json' --sonnet sonnet_gate.json --out evals/celeste_secret_choice/items.json
"""

import argparse
import glob
import json
import time
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--gate", required=True)
ap.add_argument("--sonnet", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variant", default="vivid")
a = ap.parse_args()

shards = [json.load(open(f)) for f in sorted(glob.glob(a.gate))]
shards = [d for d in shards if d["summary"]["variant"] == a.variant]
son = json.load(open(a.sonnet))
verdicts = son["verdicts"]
args0 = shards[0]["args"]
funnel = Counter()
retest_all, retest_kept, retest_dropped = [], [], []
by_bin: dict[str, list[int]] = {}
site_lines = Counter()
items = []
cats = Counter()
for d in shards:
    for p in d["prompts"]:
        funnel["prompts"] += 1
        n_valid = sum(p["dist"].values())
        for r in p["rollouts"]:
            funnel["rollouts"] += 1
            if r["choice"] is None:
                continue
            funnel["rollouts_with_choice"] += 1
            kept = []
            for s in r["sites"]:
                funnel["sites"] += 1
                if not s["committed"]:
                    continue
                funnel["committed"] += 1
                if s["text_leak"]:
                    continue
                funnel["no_text_leak"] += 1
                retest_all.append(s["retest_agree"])
                if s["n_cont_leak"] != 0:
                    continue
                funnel["no_continuation_leak"] += 1
                key = f"{d['summary']['variant']}|{p['i']}|{r['k']}|{s['gen_pos']}"
                v = verdicts.get(key)
                if v is None or v["p_true"] is None:
                    funnel["sonnet_failed"] += 1
                    continue
                b_ = min(int(v["p_true"] * 5), 4)
                by_bin.setdefault(f"{b_ / 5:.1f}-{(b_ + 1) / 5:.1f}", []).append(s["retest_agree"])
                if not v["keep"]:
                    retest_dropped.append(s["retest_agree"])
                    continue
                funnel["not_inferable"] += 1
                retest_kept.append(s["retest_agree"])
                site_lines[s["line"]] += 1
                kept.append((s, v))
            if not kept:
                continue
            funnel["items"] += 1
            cats[p["cat"]] += 1
            last = max(s["gen_pos"] for s, _v in kept)
            n_prompt = len(p["ids"])
            ids = list(p["ids"]) + list(r["gen_ids"][: last + 1])
            read_sites = []
            for s, v in sorted(kept, key=lambda x: x[0]["gen_pos"]):
                read_sites.append(
                    {
                        "label": f"line_{s['line']}",
                        "pos": n_prompt + s["gen_pos"],
                        "gen_pos": s["gen_pos"],
                        "char": len(s["prefix_text"]),
                        "agree": s["agree"],
                        "n_valid": s["n_valid"],
                        "retest_agree": s["retest_agree"],
                        "n_cont_leak": s["n_cont_leak"],
                        "continuations": s["continuations"],
                        "sonnet_p_pick": v["p_true"],
                        "sonnet_top": v["top"],
                        "sonnet_evidence": v["evidence"],
                        "resample_choices": s["resample_choices"],
                        "retest_choices": s["retest_choices"],
                    }
                )
            items.append(
                {
                    "name": f"sc-{p['cat']}-{p['i']:04d}-r{r['k']:02d}",
                    "variant": p["cat"],
                    "category": p["cat"],
                    "options": p["opts"],
                    "pick": r["choice"],
                    "aliases": d["aliases"].get(r["choice"], []),
                    "favourite": p["favourite"],
                    "pick_is_favourite": r["choice"] == p["favourite"],
                    "base_rate": p["dist"][r["choice"]] / (n_valid or 1),
                    "dist": p["dist"],
                    "prompt_index": p["i"],
                    "rollout": r["k"],
                    "prompt": p["text"],
                    "filler": p["filler"],
                    "rollout_text": r["text"],
                    "n_prompt_tokens": n_prompt,
                    "input_ids": ids,
                    "read_sites": read_sites,
                }
            )
items.sort(key=lambda it: (it["prompt_index"], it["rollout"]))


def share(xs, k):
    return (sum(x >= k for x in xs) / len(xs)) if xs else None


header = {
    "family": "celeste_secret_choice",
    "model": args0["model"],
    "built": time.strftime("%Y-%m-%d"),
    "render": "captured: chat template, one user turn, thinking off, generation prompt, then the model's own rollout up to the last kept read site",
    "wording": a.variant,
    "template": shards[0]["templates"][a.variant],
    "fillers": shards[0]["fillers"],
    "option_sets": shards[0]["sets"],
    "aliases": shards[0]["aliases"],
    "gate": {
        "rollouts_per_prompt": args0["n_roll"],
        "resamples_per_site": args0["n_resample"],
        "commit_min": args0["commit_min"],
        "base_rate_max": args0["base_max"],
        "continuations_per_site": args0["n_cont"],
        "continuation_cap": "24 new tokens, stop at the first newline",
        "sampling": "temperature 1.0, top_p 1.0; rollouts 200 new tokens, resamples 160",
        "sonnet": {
            "model": son["summary"]["model"],
            "p_max": son["summary"]["p_max"],
            "system": son["system"],
            "user": son["user"],
        },
        "retest": "a fresh set of 8 resamples (other seed) from every committed, text-leak-free site; never used to select",
        "step1_script": "olens-flow/src/secret_choice_gate.py",
    },
    "funnel": dict(funnel),
    "retest": {
        "committed_no_text_leak_sites": {
            "n": len(retest_all),
            "mean_agree": (sum(retest_all) / len(retest_all)) if retest_all else None,
            "share_ge7": share(retest_all, 7),
            "share_ge6": share(retest_all, 6),
        },
        "kept_sites": {
            "n": len(retest_kept),
            "mean_agree": (sum(retest_kept) / len(retest_kept)) if retest_kept else None,
            "share_ge7": share(retest_kept, 7),
            "share_ge6": share(retest_kept, 6),
        },
        # sites the Sonnet reader could infer from the text (dropped): the comparison group
        "sonnet_dropped_sites": {
            "n": len(retest_dropped),
            "mean_agree": (sum(retest_dropped) / len(retest_dropped)) if retest_dropped else None,
            "share_ge7": share(retest_dropped, 7),
            "share_ge6": share(retest_dropped, 6),
        },
        "by_sonnet_p_pick_bin": {
            k: {"n": len(v), "mean_agree": sum(v) / len(v)} for k, v in sorted(by_bin.items())
        },
    },
    "sites_per_line": {f"line_{k}": v for k, v in sorted(site_lines.items())},
    "items_per_category": dict(sorted(cats.items())),
    "n_items": len(items),
    "n_items_pick_is_favourite": sum(it["pick_is_favourite"] for it in items),
    "items": items,
}
json.dump(header, open(a.out, "w"), ensure_ascii=False)
print(
    json.dumps({k: v for k, v in header.items() if k != "items"}, indent=1, ensure_ascii=False)[
        :4000
    ]
)

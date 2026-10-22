# ruff: noqa
"""celeste_foreshadow, step 4: the frozen bank from build_targets.py output. An item is kept when at
least one read site has a target (a robust thought that no continuation from that site states);
only those sites are read. The model's rollouts go to a gzipped sidecar (provenance, never judged).

    python build_bank.py --targets targets.json --cands candidates.json --out-dir evals/celeste_foreshadow
"""

import argparse
import datetime
import gzip
import json
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def is_summary(s):
    return bool(s) and all(ch.isspace() or unicodedata.category(ch)[0] in "PS" for ch in s)


def control_site(tok, it):
    """The last punctuation/whitespace token of the message that ends before the cue starts (the
    model has not read the cue there), or None."""
    text = tok.decode(it["input_ids"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != it["input_ids"]:
        return None
    c0 = it["msg_char0"]
    cue0 = c0 + it["cue_char"][0]
    best = None
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s >= c0 and e <= cue0 and e > s and is_summary(tok.decode([it["input_ids"][i]])):
            best = {
                "label": "before_cue",
                "pos": i,
                "token": tok.decode([it["input_ids"][i]]),
                "char": e - c0,
            }
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--gate-args", default=None, help="the gate's args JSON (gate.json 'args')")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    t = json.load(open(a.targets))
    cands = json.load(open(a.cands))
    gate_args = json.load(open(a.gate_args))["args"] if a.gate_args else {}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    funnel = Counter()
    funnel["candidates"] = len(cands["items"])
    items, sidecar = [], []
    for it in t["items"]:
        funnel["gated_on_model"] += 1
        if it.get("thoughts") is None:
            funnel["census_failed"] += 1
            continue
        if not it["robust"]:
            funnel["no_robust_thought"] += 1
            continue
        # targets are thoughts ABOUT THE CUE only: the cue precedes every read site, so the model has
        # always read what the thought is about (a thought about the later request may not be formable yet)
        keep = [k for k, th in enumerate(it["robust"]) if th.get("about_cue")]
        remap = {k: j for j, k in enumerate(keep)}
        if not keep:
            funnel["no_cue_thought"] += 1
            continue
        sites = []
        for k, s in enumerate(it["sites"]):
            tg = [remap[t] for t in (s.get("targets") or []) if t in remap]
            if not tg:
                continue
            sites.append(
                {
                    "label": f"after_cue_{k + 1}",
                    "pos": int(s["abs"]),
                    "token": s["token"],
                    "char": int(s["char"]),
                    "targets": tg,
                    # cue thoughts the continuations already state here: shown to the judge as said
                    "said": sorted(
                        remap[int(k2)]
                        for k2, v in s["leak"].items()
                        if int(k2) in remap and v["stated"]
                    ),
                    "n_leaked": sum(
                        1 for k2, v in s["leak"].items() if int(k2) in remap and v["stated"]
                    ),
                    "pool": s["pool"],
                }
            )
        if not sites:
            funnel["every_thought_leaked"] += 1
            continue
        funnel["kept"] += 1
        ctrl = control_site(tok, it)
        funnel["with_control_site"] += ctrl is not None
        items.append(
            {
                "name": it["id"],
                "variant": it["category"],
                "message": it["message"],
                "cue": it["cue"],
                "cue_char": it["cue_char"],
                "expected_reaction": it["reaction"],
                "thoughts": [
                    {
                        "statement": th["statement"],
                        "type": th["type"],
                        "about_cue": bool(th.get("about_cue")),
                        "n_rollouts": th["n_rollouts"],
                        "support": [
                            s
                            for s in th.get("support", [])
                            if s.get("response") in th["verified_rollouts"]
                        ],
                    }
                    for th in (it["robust"][k] for k in keep)
                ],
                "n_prompt_tokens": it["n_prompt"],
                "input_ids": it["input_ids"],
                "read_sites": sites,
                # judged against every target of the item; never part of the pass rule
                "control_sites": [ctrl] if ctrl else [],
            }
        )
        sidecar.append({"name": it["id"], "rollouts": it["rollouts"]})
    header = {
        "family": "celeste_foreshadow",
        "model": "Qwen/Qwen3.6-27B",
        "built": datetime.date.today().isoformat(),
        "render": "captured: chat template, one user turn, generation prompt with thinking on "
        "(<|im_start|>assistant\\n<think>\\n); read sites lie inside the user message",
        "candidate_source": {
            "model": cands["model"],
            "system": cands["system"],
            "categories": cands["categories"],
        },
        "gate": {
            "rollouts": "n=8 thinking-mode rollouts per message, T=0.6, top_p=0.95, top_k=20, <= 3500 tokens",
            "read_sites": "punctuation/whitespace tokens of the message strictly after the cue, <= 4 spread evenly",
            "pool": "16 continuations of 64 tokens at T=1.0, top_p=1.0 from each site, through the chat "
            "template's special tokens (only <|endoftext|> stops)",
            "census": {
                "model": "claude-sonnet-5",
                "min_rollouts": t["min_rollouts"],
                "eligible_types": t["eligible"],
                "system": t["census_system"],
            },
            "leak": {"model": "claude-sonnet-5", "system": t["leak_system"]},
            "gate_args": gate_args,
        },
        "funnel": dict(funnel),
        "n_items": len(items),
    }
    (out / "items.json").write_text(
        json.dumps({**header, "items": items}, ensure_ascii=False, indent=1) + "\n"
    )
    with gzip.open(out / "rollouts.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in sidecar:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        json.dumps(header["funnel"]),
        "| per variant",
        dict(Counter(i["variant"] for i in items)),
        "| sites",
        sum(len(i["read_sites"]) for i in items),
    )


if __name__ == "__main__":
    main()

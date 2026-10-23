"""Freeze evals/celeste_silent_bridge/items.json from the selection (prep.py), the gate, the J-lens
per-position ranks, the free-sampling diagnostic and the derived-form leak check.
    python3 build_bank.py --selected S --gate G --jlens J --free F --leaks L"""

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FAM = "celeste_silent_bridge"


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("selected", "gate", "jlens", "free", "leaks"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    with open(a.selected) as fh:
        sel = json.load(fh)
    with open(a.gate) as fh:
        gated = json.load(fh)["items"]
    with open(a.jlens) as fh:
        jl = {r["name"]: r for r in json.load(fh)["items"]}
    with open(a.free) as fh:
        free = {(c["name"], c["pos"]): c for c in json.load(fh)["items"]}
    items, top20 = [], collections.Counter()
    for c in sel:
        g = c["gate"]
        p = g["probes"]
        j = jl.get(c["id"])
        ranks = {}
        for s in c["read_sites"]:
            rk = None
            if j and c["target"] in j["ranks"] and s["pos"] in j["positions"]:
                rk = j["ranks"][c["target"]]["42"][j["positions"].index(s["pos"])]
            ranks[s["label"]] = rk
            top20[s["label"]] += int(rk is not None and rk < 20)
        fr = {
            s["label"]: free.get((c["id"], s["pos"]), {}).get("n_mention") for s in c["read_sites"]
        }
        items.append(
            {
                "name": c["id"],
                "prompt": c["user"],
                "question": c["question"],
                "target": c["target"],
                "aliases": c["aliases"],
                "answer": c["answer"],
                "answer_aliases": c.get("answer_aliases") or [],
                "variant": "two_hop",
                "topic": f"{c['clue_domain']} -> {c['property_domain']}",
                "gate": {
                    "greedy_answer": p["question"]["greedy"].strip(),
                    "correct_t07": f"{p['question']['hits']}/{p['question']['n']}",
                    "bridge_question": p["bridge_question"]["text"],
                    "bridge_recall_t07": f"{p['bridge_question']['hits']}/"
                    f"{p['bridge_question']['n']}",
                    "hop2_question": p["hop2_question"]["text"],
                    "hop2_recall_t07": f"{p['hop2_question']['hits']}/{p['hop2_question']['n']}",
                    "generic_probe": p["generic_probe"]["text"],
                    "generic_probe_hits_t07": f"{p['generic_probe']['hits']}/"
                    f"{p['generic_probe']['n']}",
                    "generic_probe_greedy": p["generic_probe"]["greedy"].strip(),
                    "cooccurrence": c["cooccurrence"],
                    "cooccurrence_reason": c.get("cooc_reason", ""),
                    "answers_sampled_t1": c["gate"]["silence"]["samples"],
                },
                "leak_check": {
                    "k": c["gate"]["silence"]["n"],
                    "path": "the model's own path: chat template (thinking off) then the sampled "
                    "answer, T=1.0, <= 64 new tokens, until end of turn",
                    "sites": {
                        s["label"]: {
                            "max_new_tokens": 64,
                            "n_leaks": c["gate"]["silence"]["bridge_mentions"],
                        }
                        for s in c["read_sites"]
                    },
                },
                "diagnostics": {"jlens_rank_l42": ranks, "free_sampling_256_mentions_of_32": fr},
                "n_prompt_tokens": len(c["input_ids"]),
                "input_ids": c["input_ids"],
                "read_sites": [
                    {"label": s["label"], "pos": s["pos"], "horizon": 64} for s in c["read_sites"]
                ],
            }
        )

    def f(k):
        return sum(bool(x["gate"][k]) for x in gated)

    n_low = sum(x["gate"]["pass"] and x.get("cooccurrence") == "low" for x in gated)
    with open(a.leaks) as fh:
        leaks = {k for k, v in json.load(fh).items() if v.get("leak")}
    n_leak = sum(
        x["gate"]["pass"] and x.get("cooccurrence") == "low" and x["id"] in leaks for x in gated
    )
    funnel = {
        "candidates": len(gated),
        "correct": f("correct"),
        "recall_bridge": f("recall_bridge"),
        "recall_hop2": f("recall_hop2"),
        "generic_probe_misses": sum(not x["gate"]["shortcuts"] for x in gated),
        "silent": f("silent"),
        "gate_pass": f("pass"),
        "low_cooccurrence": n_low,
        "no_derived_form_leak": n_low - n_leak,
        "kept": len(items),
    }
    fs = [
        v
        for it in items
        for k, v in it["diagnostics"]["free_sampling_256_mentions_of_32"].items()
        if k == "reply_start" and v is not None
    ]
    fe = [
        v
        for it in items
        for k, v in it["diagnostics"]["free_sampling_256_mentions_of_32"].items()
        if k == "prompt_end" and v is not None
    ]
    diagnostics = {
        "jlens_top20_reply_start": top20["reply_start"] / max(1, len(items)),
        "jlens_top20_prompt_end": top20["prompt_end"] / max(1, len(items)),
        "free_sampling_silent_reply_start": sum(v == 0 for v in fs) / max(1, len(fs)),
        "free_sampling_silent_prompt_end": sum(v == 0 for v in fe) / max(1, len(fe)),
    }
    diagnostics["free_sampling_note"] = (
        "32 free continuations of 256 tokens from `reply_start` never name the bridge on "
        f"{100 * diagnostics['free_sampling_silent_reply_start']:.0f}% "
        "of items; from `prompt_end`, where the template has already closed thinking, on "
        f"{100 * diagnostics['free_sampling_silent_prompt_end']:.0f}% "
        "(per item: `diagnostics.free_sampling_256_mentions_of_32`)."
    )
    bank = {
        "family": FAM,
        "model": "Qwen/Qwen3.6-27B",
        "built": "2026-10-23",
        "render": "captured: `input_ids` is the exact Qwen3.6-27B token sequence (chat template, "
        "enable_thinking=False, user message = the question + '\\n\\nANSWER IN ONE WORD, DO NOT "
        "SAY ANYTHING ELSE.'); `read_sites[*].pos` index it directly",
        "candidate_source": "claude-sonnet-5 proposals over 600 ordered (clue domain, property "
        "domain) pairs (scripts/celeste_evals/silent_bridge/gen_novel.py), audited by an "
        "independent claude-sonnet-5 call (audit_novel.py: generic-clue probe + co-occurrence "
        "rating)",
        "gate": "Qwen3.6-27B, chat template, thinking off, every question with the all-caps "
        "one-word instruction: (i) correct: greedy hit AND >= 16/20 at T=0.7; (ii) recall: the "
        "bridge from its own clue and the last hop with the bridge named, greedy hit AND >= 8/10 "
        "each; (iii) no shortcut: the question with the WHOLE clue replaced by a generic noun ('a "
        "certain composer') must miss (greedy miss AND <= 4/20); (iv) novelty: the auditor rates "
        "co-occurrence of clue and answer in ordinary text 'low'; (v) silence: 32 answers at "
        "T=1.0 never name the bridge or an alias, and no answer the model gave names it by a "
        "derived form (demonym, adjective, abbreviation: 'Norwegian Krone', 'Swiss'; "
        "claude-sonnet-5 check)",
        "read": "`reply_start` = the <|im_start|> that opens the reply (the deployed path from it "
        "is template tokens, then the answer); `prompt_end` = the last prompt token (after the "
        "closed think block)",
        "funnel": funnel,
        "diagnostics": diagnostics,
        "n_items": len(items),
        "items": items,
    }
    (REPO / "evals" / FAM).mkdir(parents=True, exist_ok=True)
    (REPO / "evals" / FAM / "items.json").write_text(
        json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"{len(items)} items -> evals/{FAM}/items.json | funnel {funnel} | "
        f"diagnostics {diagnostics}"
    )


if __name__ == "__main__":
    main()

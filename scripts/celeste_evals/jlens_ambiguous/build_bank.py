"""Freeze evals/celeste_jlens_intermediates/items.json from the gate, the J-lens site selection, the
site leak check and the derived-form check.
    python3 build_bank.py --gate G --sites S --free F --leaks L --specific P"""

import argparse
import collections
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
FAM = "celeste_jlens_intermediates"
ARTICLES = ("the ", "a ", "an ")


def load(path: str):
    with open(path) as f:
        return json.load(f)


def bare(name: str) -> str:
    low = name.lower()
    for a in ARTICLES:
        if low.startswith(a):
            return name[len(a) :]
    return name


def aliases_of(bridge: str, raw: list) -> list[str]:
    """Full-name variants and acronyms only: an alias passes if it has two or more words, is an
    all-caps acronym, or is the bridge without its leading article."""
    out = []
    for a in raw:
        if not isinstance(a, str) or not a.strip() or a.strip() == bridge:
            continue
        a = a.strip()
        if (
            len(a.split()) >= 2
            or re.fullmatch(r"[A-Z0-9]{2,8}", a)
            or a.lower() == bare(bridge).lower()
        ) and a not in out:
            out.append(a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("gate", "sites", "free", "leaks", "specific"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--per-bridge", type=int, default=2)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    gated = {c["id"]: c for c in load(a.gate)["items"]}
    sites = {s["id"]: s for s in load(a.sites)}
    free = {c["name"]: c for c in load(a.free)["items"]}
    leaks = {k for k, v in load(a.leaks).items() if v.get("leak")}
    specific = load(a.specific)
    funnel = collections.Counter(candidates=len(gated))
    items, per_bridge = [], collections.Counter()
    for i in sorted(gated):
        c = gated[i]
        g = c["gate"]
        if not all(g[k] for k in ("correct", "recall_bridge", "recall_hop2", "silent")):
            continue
        funnel["gate_pass"] += 1
        if len(tok(" " + bare(c["bridge"]), add_special_tokens=False)["input_ids"]) < 2:
            continue
        funnel["multi_token_name"] += 1
        s = sites.get(i)
        if s is None:
            continue
        funnel["jlens_site"] += 1
        if not specific.get(i, {}).get("specific"):
            continue
        funnel["site_specific"] += 1
        fr = free.get(i)
        if fr is None or fr["n_mention"] > 0:  # sites arrive leak-checked; kept as a guard
            continue
        if i in leaks:
            continue
        funnel["no_derived_form_leak"] += 1
        if per_bridge[c["bridge"].lower()] >= a.per_bridge:
            continue
        per_bridge[c["bridge"].lower()] += 1
        p = g["probes"]
        site = s["site"]
        items.append(
            {
                "name": i,
                "prompt": s["user"],
                "question": c["question"],
                "target": c["bridge"],
                "aliases": aliases_of(c["bridge"], c.get("aliases") or []),
                "name_parts": s["name_parts"],
                "answer": c["answer"],
                "variant": c["category"],
                "topic": c["category"],
                "gate": {
                    "greedy_answer": p["question"]["greedy"].strip(),
                    "correct_t07": f"{p['question']['hits']}/{p['question']['n']}",
                    "bridge_question": p["bridge_question"]["text"],
                    "bridge_recall_t07": (
                        f"{p['bridge_question']['hits']}/{p['bridge_question']['n']}"
                    ),
                    "hop2_question": p["hop2_question"]["text"],
                    "hop2_recall_t07": f"{p['hop2_question']['hits']}/{p['hop2_question']['n']}",
                    "generic_probe": p["generic_probe"]["text"],
                    "generic_probe_hits_t07": (
                        f"{p['generic_probe']['hits']}/{p['generic_probe']['n']}"
                    ),
                    "generic_probe_reached": bool(g["shortcuts"]),
                    "cooccurrence": c.get("cooccurrence"),
                    "answers_sampled_t1": g["silence"]["samples"],
                    "answer_bridge_mentions_t1": (
                        f"{g['silence']['bridge_mentions']}/{g['silence']['n']}"
                    ),
                },
                "jlens_evidence": {
                    "token": site["token"],
                    "name_words": [w for w, _r in site["name_hits"]],
                    "signature_words": [w for w, _r in site["sig_hits"]],
                    "ranks_l42": dict(site["name_hits"] + site["sig_hits"]),
                    "top10_l42": site["top10_L42"],
                    "specific_why": specific[i].get("why", ""),
                },
                "leak_check": {
                    "k": 16,
                    "path": (
                        "free continuation from the site through the chat special tokens, T=1.0"
                    ),
                    "sites": {"jlens_site": {"max_new_tokens": 48, "n_leaks": 0}},
                },
                "n_prompt_tokens": len(s["input_ids"]),
                "input_ids": s["input_ids"],
                "read_sites": [{"label": "jlens_site", "pos": site["pos"], "horizon": 48}],
            }
        )
    funnel["kept"] = len(items)
    bank = {
        "family": FAM,
        "model": "Qwen/Qwen3.6-27B",
        "built": "2026-10-23",
        "render": "captured: `input_ids` is the exact Qwen3.6-27B token sequence (chat template, "
        "enable_thinking=False, user message = the question + '\\n\\nANSWER IN ONE WORD, DO NOT "
        "SAY ANYTHING ELSE.'); `read_sites[*].pos` index it directly",
        "candidate_source": "claude-sonnet-5 proposals over 18 categories x 10 fact domains "
        "(scripts/celeste_evals/jlens_intermediates/gen_mt.py): a question whose answer needs a "
        "famous MULTI-WORD entity it never names, plus that entity's signature words",
        "gate": "Qwen3.6-27B, chat, thinking off, all-caps one-word instruction: correct (greedy "
        "hit and >= 16/20 at T=0.7); the bridge recalled from its clue (asked 'ANSWER WITH THE "
        "NAME ONLY') and the last hop with the bridge named (>= 8/10 each); 32 answers at T=1.0 "
        "never name the bridge; no answer names it by a derived form (claude-sonnet-5 check). The "
        "generic-clue shortcut probe is recorded (gate.generic_probe_reached) but not required: "
        "the read site's J-lens evidence is what shows the model holds the bridge",
        "read": "`jlens_site` = the prompt token where the J-lens (neuronpedia n=1000, layer 42, "
        "top 20) ranks the most signature words of the bridge that are NOT in the text up to "
        "that token (one-token, rare words only, zipf < 4.5, and no inflection of a word in the "
        "text; at least one NAME word; a name word counts double), the found words judged "
        "specific to the bridge rather than the text's topic (claude-sonnet-5); 16 free "
        "continuations of 48 tokens from it never name the bridge",
        "funnel": dict(funnel),
        "n_items": len(items),
        "items": items,
    }
    out = REPO / "evals" / FAM / "items.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} items -> {out.relative_to(REPO)} | funnel {dict(funnel)}")


if __name__ == "__main__":
    main()

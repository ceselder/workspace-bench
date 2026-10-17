"""Celeste supplement tasks 2-4: turn the Qwen3.6-27B gate output (scripts/celeste_supplement/
README.md; `celeste234_gate.py` records) into the frozen banks, the sampling-baseline readouts and
the per-item playground data. Only records that passed every gate become items."""

import argparse
import datetime as dt
import json
import re
import unicodedata
from pathlib import Path

FAMILY = {
    "encoded": "celeste_encoded_answer",
    "lookahead": "celeste_lookahead",
    "withheld": "celeste_withheld_answer",
}
PREFIX = {"encoded": "enc", "lookahead": "look", "withheld": "withheld"}
VARIANT_KEY = {"encoded": "encoding", "lookahead": "preamble", "withheld": "kind"}
GATE_DESC = {
    "encoded": (
        "Qwen3.6-27B behavioural (greedy, chat template, thinking off): (i) asked the plain "
        "question it names the target (word-boundary match on target or alias); (ii) asked the "
        "encoded question its output is the correct encoding and never contains the target; "
        "(iii) leak check: 16 continuations at T=1.0 (32 new tokens) from EACH read site contain "
        "neither the target nor an alias (substring after folding)"
    ),
    "lookahead": (
        "Qwen3.6-27B behavioural (greedy, chat template, thinking off): the full greedy rollout "
        "(200 new tokens) names the target, and its first token appears >= 16 tokens after the "
        "end of the prompt (the read horizon); leak check: 16 continuations at T=1.0 from each "
        "read site, capped at that site's horizon, contain neither the target nor an alias"
    ),
    "withheld": (
        "Qwen3.6-27B behavioural (greedy, chat template, thinking off): (i) the greedy reply "
        "(<= 160 new tokens, finished naturally, >= 8 tokens) never contains the target or an "
        "alias; (ii) a follow-up user turn 'What was the answer? Reply with just the word.' in the "
        "same conversation is answered with the target; (iii) leak check: 16 continuations at "
        "T=1.0 (48 new tokens) from EACH read site contain neither the target nor an alias"
    ),
}
READ_DESC = {
    "encoded": (
        "two sites in the captured render: `prompt_end` = the final prompt token (the assistant "
        "onset after the empty think block) and `answer_end` = the last token of the model's own "
        "greedy encoded output, appended as a prefill"
    ),
    "lookahead": (
        "`prompt_end` = the final prompt token; where the greedy preamble is >= 24 tokens long, "
        "also `preamble_minus16` = the preamble token exactly 16 tokens before the answer's first "
        "token (the greedy preamble up to there is appended as a prefill)"
    ),
    "withheld": (
        "inside the model's own greedy reply (appended as a prefill): `sentence1_end` = the token "
        "ending its first sentence, `reply_mid` = its middle token, `reply_end` = its last token"
    ),
}


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "", s).lower()[:24] or "x"


def gate_fields(task: str, r: dict) -> dict:
    if task == "encoded":
        return {
            "plain_answer": r["plain_answer"],
            "encoded_output": r["encoded_output"],
            "expected": r["expected"],
            "list_order": r.get("list_order"),
        }
    if task == "lookahead":
        return {
            "greedy_output": r["greedy_output"],
            "answer_text": r["answer_text"],
            "answer_start_token": r["answer_start"],
            "horizon_tokens": r["horizon"],
        }
    return {"reply": r["reply"], "followup_answer": r["followup_answer"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gate-dir", required=True, help="dir with encoded.json lookahead.json withheld.json"
    )
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", required=True, help="dir for sampling readouts + playground gate data")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
    ap.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="cap per family, evenly spaced over the kept records",
    )
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out)
    (out / "readouts" / "sampling1").mkdir(parents=True, exist_ok=True)
    (out / "readouts" / "sampling16").mkdir(parents=True, exist_ok=True)
    (out / "playground").mkdir(parents=True, exist_ok=True)
    funnels = {}
    for task, fam in FAMILY.items():
        g = json.loads((Path(a.gate_dir) / f"{task}.json").read_text(encoding="utf-8"))
        kept = [r for r in g["records"] if r["drop"] is None]
        if (
            a.max_items and len(kept) > a.max_items
        ):  # evenly spaced over the kept list (topics are interleaved)
            step = len(kept) / a.max_items
            kept = [kept[int(i * step)] for i in range(a.max_items)]
        items, s1, s16, play = [], [], [], []
        vkey = VARIANT_KEY[task]
        per_variant_idx: dict[str, int] = {}
        for r in kept:
            v = r[vkey]
            per_variant_idx[v] = per_variant_idx.get(v, 0) + 1
            name = f"{PREFIX[task]}-{v}-{per_variant_idx[v]:03d}-{slug(r['target'])}"
            sites = [
                {"label": s["label"], "pos": s["abs"], "horizon": s["horizon"]} for s in r["sites"]
            ]
            item = {
                "name": name,
                "prompt": r["prompt"],
                "target": r["target"],
                "aliases": r["aliases"],
                "variant": v,
                "topic": r.get("topic"),
                "gate": gate_fields(task, r),
                "leak_check": {
                    "k": g["k"],
                    "sites": {
                        s["label"]: {"max_new_tokens": s["horizon"], "n_leaks": s["n_leak"]}
                        for s in r["sites"]
                    },
                },
                "n_prompt_tokens": r["n_prompt"],
                "input_ids": r["input_ids"],
                "read_sites": sites,
            }
            if task == "encoded":
                item["plain_prompt"] = r["plain_prompt"]
            items.append(item)
            label = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            toks = [tok.decode([i]) for i in r["input_ids"]]
            for s in r["sites"]:
                row = {"id": label, "layer": 42, "pos": s["abs"], "token": toks[s["abs"]]}
                s1.append({**row, "samples": [s["samples"][0]]})
                s16.append({**row, "samples": list(s["samples"])})
            play.append(
                {
                    "id": label,
                    "name": name,
                    "family": fam,
                    "variant": v,
                    "topic": r.get("topic"),
                    "prompt": r["prompt"],
                    "plain_prompt": r.get("plain_prompt"),
                    "target": r["target"],
                    "aliases": r["aliases"],
                    "gate": gate_fields(task, r),
                    "n_prompt": r["n_prompt"],
                    "tokens": toks,
                    "sites": [{**s} for s in r["sites"]],
                }
            )
        header = {
            "family": fam,
            "model": "Qwen/Qwen3.6-27B",
            "built": dt.date.today().isoformat(),
            "render": (
                "captured: `input_ids` is the exact Qwen3.6-27B token sequence (chat template, "
                "enable_thinking=False, the model's own greedy output appended where a read site "
                "is inside it); `read_sites[*].pos` index it directly"
            ),
            "candidate_source": "claude-sonnet-5 proposals "
            "(scripts/celeste_supplement/gen_candidates.py), "
            "then the Qwen3.6-27B gate (scripts/celeste_supplement/README.md)",
            "gate": GATE_DESC[task],
            "read": READ_DESC[task],
            "funnel": g["funnel"],
            "n_items": len(items),
        }
        bank_dir = Path(a.repo) / "evals" / fam
        bank_dir.mkdir(parents=True, exist_ok=True)
        (bank_dir / "items.json").write_text(
            json.dumps({**header, "items": items}, ensure_ascii=False, indent=1) + "\n"
        )
        for tag, rows in (("sampling1", s1), ("sampling16", s16)):
            with open(out / "readouts" / tag / f"{fam}.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (out / "playground" / f"{fam}.gate.json").write_text(json.dumps(play, ensure_ascii=False))
        funnels[fam] = {**g["funnel"], "items": len(items), "cells_at_one_layer": len(s1)}
        print(fam, json.dumps(funnels[fam]))
    (out / "funnels.json").write_text(json.dumps(funnels, indent=1))


if __name__ == "__main__":
    main()

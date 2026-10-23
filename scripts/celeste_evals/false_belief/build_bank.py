# ruff: noqa
"""celeste_false_belief: the frozen bank from the lookahead gate outputs (own_belief_lookahead.py).
Kept: a problem Qwen3.6-27B answers wrongly and consistently when asked plainly (greedy = W, >= 70% of
10 samples) and STILL answers W after the forced preamble (greedy + >= 70% of 10 samples), with no W or
T in any of 16 continuations (<= 24 tokens) from the end of the prompt.

    python build_bank.py --lookahead la4k.json la12k.json --out-dir evals/celeste_false_belief
"""
import argparse, datetime, json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--lookahead", nargs="+", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
funnel, seen, items, preamble = Counter(), set(), [], None
for f in a.lookahead:
    d = json.load(open(f)); preamble = d["preamble"]
    for p in d["items"]:
        if p["cls"] != "wrong":
            continue
        funnel["consistent_wrong_plain"] += 1
        if p["expr"] in seen:
            funnel["duplicate"] += 1; continue
        seen.add(p["expr"])
        if not p["belief_survives"]:
            funnel["belief_changed_by_preamble"] += 1; continue
        if p["leak_W"] or p["leak_T"]:
            funnel["leak"] += 1; continue
        ids = tok.apply_chat_template([{"role": "user", "content": p["la_prompt"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        funnel["kept"] += 1
        items.append({"name": f"fb-{p['kind']}-{len(items):04d}", "kind": p["kind"], "expr": p["expr"], "T": p["T"], "W": p["W"],
                      "prompt": p["la_prompt"], "n_prompt_tokens": len(ids), "input_ids": ids,
                      "read_sites": [{"label": "prompt_end", "pos": len(ids) - 1, "token": tok.decode([ids[-1]])}],
                      "gate": {"plain_share_W": p["W_share"], "plain_samples": p["samples"], "preamble_share_W": p["la_share_W"],
                               "preamble_share_T": p["la_share_T"], "preamble_greedy": p["la_greedy_text"], "leak_example": p["leak_example"]}})
bank = {"family": "celeste_false_belief", "model": "Qwen/Qwen3.6-27B", "built": datetime.date.today().isoformat(),
        "render": "captured: chat template, one user turn, generation prompt with thinking off; read at the last prompt token",
        "gate": {"plain": "'What is <expr>? Reply with just the number.' greedy + 10 samples at T=1; kept if greedy = modal answer W != T and W in >= 70%",
                 "lookahead": f"'What is <expr>? {preamble}' greedy + 10 samples at T=1; kept if greedy = W and W in >= 70%",
                 "leak": "16 continuations at T=1 (<= 24 tokens) from the end of the prompt contain neither W nor T",
                 "problems": "random (a+b)*c, a*b+c, a*b-c with two-digit a, b and two- or three-digit c, plus 3x2 and 3x3-digit products"},
        "funnel": dict(funnel), "n_items": len(items), "items": items}
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
(out / "items.json").write_text(json.dumps(bank, ensure_ascii=False, indent=1) + "\n")
print(json.dumps(dict(funnel)), Counter(i["kind"] for i in items))

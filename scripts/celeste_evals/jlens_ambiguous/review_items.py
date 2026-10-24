"""Fresh-eye review of candidate items (claude-opus-5, 10 per call): unambiguous target, not named,
correct answer, natural intermediate; adds missing full-name aliases. Writes {id: {keep, ...}}.
    env -u ANTHROPIC_API_KEY python3 review_items.py --sites S --gate G [G ...] --out review.json"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import time

import anthropic

SYSTEM = (
    "You review items for an interpretability benchmark. Each item is a question a language model "
    "answered in ONE WORD. To\n"
    "answer, the model must silently think about a famous MULTI-WORD named entity X (the TARGET) "
    "that the question describes but never\n"
    "names. A lens will later be credited only if it NAMES X (its full name or a listed alias). "
    "Review each item strictly:\n"
    "1. unambiguous: a well-read person reading the question would identify exactly X (not a "
    "different entity, not a family of entities).\n"
    "2. not_named: the question does not contain X's name, a distinctive part of it, or its "
    "abbreviation.\n"
    "3. answer_ok: the model's one-word answer is a correct answer to the question as asked.\n"
    "4. natural_target: X is the natural thing the question makes you think of. Flag it (false) "
    "if the question is really about a\n"
    "   different entity (e.g. it asks about the astronomer while X is the astronomer's laws, and "
    "naming the astronomer is equally natural),\n"
    "   or if X is a generic category rather than a named entity.\n"
    "5. extra_aliases: other names of the SAME entity X that should also count (full-name "
    "variants, official or common names, acronyms);\n"
    "   never a fragment, never a related entity. [] if none.\n"
    "keep = unambiguous AND not_named AND answer_ok AND natural_target.\n"
    'Return a JSON list, one entry per item in order: [{"id": ..., "keep": true|false, '
    '"problems": "<under 20 words, empty if none>",\n'
    '"extra_aliases": [...]}]. Only JSON.'
)


def ask(client, user):
    for k in range(8):
        try:
            m = client.messages.create(
                model="claude-opus-5",
                max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
            return json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        except Exception:
            time.sleep(min(60, 2**k) + random.random())
    return []


def load(path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    client = anthropic.Anthropic()
    gate = {c["id"]: c for g in a.gate for c in load(g)["items"]}
    done = load(a.out) if os.path.exists(a.out) else {}
    rows = []
    for s in load(a.sites):
        if s["id"] in done:
            continue
        c = gate[s["id"]]
        rows.append(
            {
                "id": s["id"],
                "question": c["question"],
                "target": c["bridge"],
                "aliases": c.get("aliases") or [],
                "model_answer": c["gate"]["probes"]["question"]["greedy"].strip(),
                "category": c.get("category", ""),
            }
        )
    batches = [rows[i : i + 10] for i in range(0, len(rows), 10)]

    def run(batch):
        return ask(client, "\n".join(json.dumps(x, ensure_ascii=False) for x in batch))

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(run, batches):
            for e in res:
                if isinstance(e, dict) and "id" in e:
                    extra = [x for x in e.get("extra_aliases") or [] if isinstance(x, str)]
                    done[e["id"]] = {
                        "keep": bool(e.get("keep")),
                        "problems": e.get("problems", ""),
                        "extra_aliases": extra,
                    }
    with open(a.out, "w") as f:
        json.dump(done, f, indent=1, ensure_ascii=False)
    print(len(done), "reviewed |", sum(v["keep"] for v in done.values()), "keep")


if __name__ == "__main__":
    main()

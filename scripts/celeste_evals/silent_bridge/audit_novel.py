"""Independent Sonnet-5 audit of the candidates (one call per 10 items): a generic probe (whole clue
replaced by "a certain <type>", property wording kept) and a low/medium/high co-occurrence rating.
    env -u ANTHROPIC_API_KEY python3 audit_novel.py --cands cands.json --out cands_audited.json"""

import argparse
import concurrent.futures as cf
import json
import random
import re
import time

import anthropic

SYSTEM = (
    "You audit two-hop questions built so that a model must silently identify an unnamed BRIDGE "
    "entity before it can answer.\n"
    "For each item return:\n"
    "- generic_probe: rewrite the question replacing the ENTIRE description of the bridge with a "
    "bare generic noun phrase for its type\n"
    '  ("a certain composer", "a certain country", "a certain chemical element"). Remove '
    "EVERY detail of the clue (no dates, works, places,\n"
    "  traits). Keep the rest of the question, especially the property being asked, word for "
    "word.\n"
    '- cooccurrence: "low", "medium" or "high": how likely the clue\'s distinctive content '
    "and the answer appear together in ordinary text\n"
    "  (encyclopedias, trivia sites, news), so that a model could answer from a memorised "
    "association without identifying the bridge.\n"
    '  Also "high" if the property wording itself names something that gives the answer (e.g. '
    '"the city where the Rijksmuseum is").\n'
    "- reason: under 15 words.\n"
    'Return a JSON list, one entry per item in order: [{"id": ..., "generic_probe": ..., '
    '"cooccurrence": ..., "reason": ...}]. Only JSON.'
)


def ask(client, user):
    for k in range(8):
        try:
            m = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
            return json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        except Exception:
            time.sleep(min(60, 2**k) + random.random())
    return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    client = anthropic.Anthropic()
    with open(a.cands) as fh:
        cands = json.load(fh)
    by = {c["id"]: c for c in cands}
    batches = [cands[i : i + 10] for i in range(0, len(cands), 10)]

    def run(b):
        return ask(
            client,
            "\n".join(
                json.dumps(
                    {
                        "id": c["id"],
                        "question": c["question"],
                        "bridge": c["bridge"],
                        "answer": c["answer"],
                    }
                )
                for c in b
            ),
        )

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(run, batches):
            for e in res:
                c = by.get(e.get("id")) if isinstance(e, dict) else None
                if c is not None and isinstance(e.get("generic_probe"), str):
                    c["generic_probe"] = e["generic_probe"]
                    c["cooccurrence"] = str(e.get("cooccurrence", "")).lower()
                    c["cooc_reason"] = e.get("reason", "")
    done = [c for c in cands if c.get("generic_probe")]
    with open(a.out, "w") as fh:
        json.dump(done, fh, indent=1)
    from collections import Counter

    print(len(done), "of", len(cands), "audited |", dict(Counter(c["cooccurrence"] for c in done)))

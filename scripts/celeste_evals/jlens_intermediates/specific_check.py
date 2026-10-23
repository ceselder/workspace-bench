"""Sonnet-5 check of each J-lens site: are the words found there specific evidence of the bridge X
itself, not words a reader would associate with the TEXT so far (its topic)? One call per 15 sites.
    env -u ANTHROPIC_API_KEY python3 specific_check.py --sites sites.json --out specific.json"""

import argparse
import concurrent.futures as cf
import json
import random
import re
import time

import anthropic
from transformers import AutoTokenizer

SYSTEM = (
    "You audit read sites for an interpretability benchmark. Every question describes a named "
    "entity X (the BRIDGE) without\n"
    "naming it, so X is always inferable from the text; that is by design. At a token inside the "
    "question a lens lists words the model is\n"
    "internally activating. We want sites where those words name or uniquely point to X: part of "
    'X\'s name ("Firefox", "Vader"), or a\n'
    "character, place or creator unique to X. Reject a site only when every found word is a "
    "generic word a reader would associate with the\n"
    'TOPIC of the text so far even without thinking of X ("frog" while the text says "amphibian"; '
    '"opera" for "performing arts"; "sushi"\n'
    'for "a conveyor-belt restaurant"; "galaxy" for "science fiction").\n'
    'Return a JSON list, one entry per item in order: [{"id": ..., "specific": true|false, "why": '
    '"<under 12 words>"}]. Only JSON.'
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
    ap.add_argument("--sites", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="json list of ids to check (e.g. gate passes)")
    a = ap.parse_args()
    client = anthropic.Anthropic()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    with open(a.sites) as fh:
        sites = json.load(fh)
    if a.only:
        with open(a.only) as fh:
            keep = set(json.load(fh))
        sites = [s for s in sites if s["id"] in keep]
    rows = []
    for s in sites:
        st = s["site"]
        text = tok.decode(s["input_ids"][: st["pos"] + 1])
        q0 = text.find("<|im_start|>user\n")
        text = text[q0 + len("<|im_start|>user\n") :] if q0 >= 0 else text
        rows.append(
            {
                "id": s["id"],
                "bridge": s["bridge"],
                "text_up_to_token": text,
                "token": st["token"],
                "found_words": [w for w, _ in st["name_hits"] + st["sig_hits"]],
                "lens_top10": [t.strip() for t in st["top10_L42"]],
            }
        )
    batches = [rows[i : i + 15] for i in range(0, len(rows), 15)]
    res = {}

    def run(b):
        return ask(client, "\n".join(json.dumps(x, ensure_ascii=False) for x in b))

    with cf.ThreadPoolExecutor(8) as ex:
        for r in ex.map(run, batches):
            for e in r:
                if isinstance(e, dict) and "id" in e:
                    res[e["id"]] = {"specific": bool(e.get("specific")), "why": e.get("why", "")}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    print(
        len(rows),
        "sites checked |",
        sum(v["specific"] for v in res.values()),
        "specific |",
        len(rows) - len(res),
        "unchecked",
    )

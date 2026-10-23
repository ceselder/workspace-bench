"""Derived-form leak check on gate passes: does any answer Qwen gave name the bridge via a demonym,
adjective, abbreviation or translation ("Swiss" -> Switzerland)? One Sonnet-5 call per 20 items.
    env -u ANTHROPIC_API_KEY python3 leak_forms.py --gate gate.json --out leak_forms.json"""

import argparse
import concurrent.futures as cf
import json
import random
import re
import time

import anthropic

SYSTEM = (
    "For each item you get a BRIDGE entity (with aliases) and the short answers a model gave to a "
    "question about something else.\n"
    "Decide whether any answer names the bridge or reveals it by a derived form: a demonym or "
    'adjective ("Norwegian" for Norway, "Swiss"\n'
    'for Switzerland, "Dutch" for the Netherlands), an abbreviation, a possessive, a '
    "translation, or the bridge's name inside a longer name\n"
    '("Rio" for Rio de Janeiro counts; a company or product that merely comes from the bridge '
    "without containing its name does not, e.g.\n"
    '"KLM" for the Netherlands is NOT a leak). Return a JSON list, one entry per item in order: '
    '[{"id": ..., "leak": true|false,\n'
    '"form": "<the leaking text or empty>"}]. Only JSON.'
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


def answers_of(c):
    p = c["gate"]["probes"]["question"]
    outs = [p["greedy"], *p.get("samples", []), *c["gate"]["silence"].get("samples", [])]
    return sorted({o.strip() for o in outs if o.strip()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    client = anthropic.Anthropic()
    with open(a.gate) as fh:
        items = [c for c in json.load(fh)["items"] if c["gate"]["pass"]]
    batches = [items[i : i + 20] for i in range(0, len(items), 20)]

    def run(b):
        return ask(
            client,
            "\n".join(
                json.dumps(
                    {
                        "id": c["id"],
                        "bridge": c["bridge"],
                        "aliases": c.get("bridge_aliases") or [],
                        "answers": answers_of(c),
                    },
                    ensure_ascii=False,
                )
                for c in b
            ),
        )

    res = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for r in ex.map(run, batches):
            for e in r:
                if isinstance(e, dict) and "id" in e:
                    res[e["id"]] = {"leak": bool(e.get("leak")), "form": e.get("form", "")}
    out = {
        c["id"]: {**res.get(c["id"], {"leak": None, "form": ""}), "answers": answers_of(c)}
        for c in items
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(
        len(out),
        "gate passes checked | leak",
        sum(v["leak"] is True for v in out.values()),
        "| unchecked",
        sum(v["leak"] is None for v in out.values()),
    )
    for k, v in out.items():
        if v["leak"]:
            print("  ", k, v["form"], v["answers"][:3])

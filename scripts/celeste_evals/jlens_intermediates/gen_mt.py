"""Multi-word-bridge two-hop candidates (J-lens intermediates eval): claude-sonnet-5 proposes, per
(category, fact domain) pair, questions whose one-word answer needs an unnamed MULTI-WORD entity.
    env -u ANTHROPIC_API_KEY python3 gen_mt.py --out cands.json [--avoid "<bridges>"]"""

import argparse
import concurrent.futures as cf
import json
import random
import re
import time

import anthropic

CATEGORIES = [
    "video games",
    "films",
    "TV series",
    "novels and book series",
    "bands and musicians",
    "songs and albums",
    "paintings and artworks",
    "landmarks and buildings",
    "historical events and missions",
    "sports teams and tournaments",
    "companies and products",
    "fictional characters",
    "scientific theories and laws",
    "institutions and organisations",
    "anime and comics",
    "board and card games",
    "internet phenomena and memes",
    "mythological stories",
]
GEN = (
    "You write two-hop questions for an interpretability study of a 27B language model. Each "
    "question forces the model to silently\n"
    "think about a MULTI-WORD named entity X (the BRIDGE), described only through a clue, then "
    "asks for a one-word fact about X.\n"
    "\n"
    'X must be famous and MULTI-WORD: a video game or franchise ("Five Nights at Freddy\'s"), '
    'film or TV series ("Breaking Bad"), book or\n'
    'series ("The Hunger Games"), band ("Pink Floyd"), song or album, artwork ("The Starry '
    'Night"), landmark or building ("Golden Gate\n'
    'Bridge"), event ("Woodstock" is one word, so no; "Apollo 11" yes), sports team ("Manchester '
    'United"), company or product, fictional\n'
    'character ("Darth Vader"), theory or law ("Theory of Relativity"), institution, etc. Use the '
    "CATEGORY we give.\n"
    "\n"
    "Rules:\n"
    "- The clue pins X down for a well-read person but never names it: none of the words of X's "
    "name (or its acronym) appear in the\n"
    "  question. Prefer clues a model would recognise instantly (iconic plot, characters, imagery, "
    "events).\n"
    "- The asked fact is a plain RELATION of X with a ONE-WORD answer (a number in digits or a "
    "single word), and the answer must not\n"
    "  contain any word of X's name. Never describe the answer itself.\n"
    "- No shortcut: the answer must not be guessable from the clue's words without identifying X.\n"
    '- Ask a plain question ending in "?", under 40 words.\n'
    "\n"
    "For each item also give:\n"
    '- aliases: full-name variants and well-known acronyms of X (e.g. ["Five Nights at '
    'Freddy\'s", "FNAF", "Five Nights at Freddys"]);\n'
    "  never a single fragment of the name.\n"
    '- name_parts: the distinctive words of X\'s name, lower-case (e.g. ["freddy", "fazbear"] '
    "is wrong: only words IN the name, so\n"
    '  ["freddy\'s", "freddy", "nights"]); skip articles and generic words.\n'
    "- signature_words: 8-12 single words that a model thinking about X would light up: "
    "distinctive name words, characters, places,\n"
    "  objects, creators, catchphrases. Single words only, none of them used in the question.\n"
    "- bridge_question: a question whose answer is X's name, using the same clue.\n"
    "- hop2_question: the fact question with X named.\n"
    '- drop_clue: the main question with the clue replaced by a generic noun phrase ("a certain '
    'horror video game").\n'
    "\n"
    "Return a JSON list of objects with keys: question, bridge, aliases, name_parts, "
    "signature_words, answer, answer_aliases,\n"
    "bridge_question, hop2_question, drop_clue. Return only the JSON."
)


def ask(client, system, user, max_tokens=16000):
    for k in range(8):
        try:
            m = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
            return json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        except Exception:
            time.sleep(min(60, 2**k) + random.random())
    return []


FACTS = [
    "geography (countries, cities)",
    "numbers and dates",
    "people (creators, actors, founders)",
    "animals and nature",
    "colors and materials",
    "languages and names",
    "music and instruments",
    "food and drink",
    "science and technology",
    "sports and games",
]


def words_in(text):
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def ok(o):
    keys = ("question", "bridge", "answer", "bridge_question", "hop2_question", "drop_clue")
    if not all(isinstance(o.get(k), str) and o[k].strip() for k in keys):
        return False
    if len(o["answer"].split()) != 1 or len(o["bridge"].split()) < 2:
        return False
    qw = words_in(o["question"])
    parts = [
        p.lower().strip()
        for p in (o.get("name_parts") or [])
        if isinstance(p, str) and len(p.strip()) >= 3
    ]
    if any(p in qw for p in parts) or any(p in words_in(o["answer"]) for p in parts):
        return False
    al = [x for x in [o["bridge"], *(o.get("aliases") or [])] if isinstance(x, str)]
    if any(x.lower() in o["question"].lower() for x in al if len(x) >= 3):
        return False
    ans = words_in(
        " ".join([o["answer"], *[x for x in (o.get("answer_aliases") or []) if isinstance(x, str)]])
    )
    o["signature_words"] = [
        w
        for w in (o.get("signature_words") or [])
        if isinstance(w, str) and w.strip() and w.lower() not in qw and not (words_in(w) & ans)
    ]
    return len(o["signature_words"]) >= 4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-call", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--prefix", default="mtb")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--avoid", default="", help="comma-separated over-used bridges to avoid")
    a = ap.parse_args()
    client = anthropic.Anthropic()
    rng = random.Random(0)
    jobs = [(c, f) for c in CATEGORIES for f in FACTS]
    rng.shuffle(jobs)
    if a.limit:
        jobs = jobs[: a.limit]

    def gen(job):
        c, f = job
        out = ask(
            client,
            GEN,
            f"CATEGORY: {c}\nASKED FACT FROM: {f}\nWrite {a.per_call} items, all with different "
            f"bridges; mix very famous and moderately famous ones."
            + (f" Do not use these: {a.avoid}." if a.avoid else ""),
        )
        return [dict(o, category=c, fact_domain=f) for o in out if isinstance(o, dict) and ok(o)]

    items = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(gen, jobs):
            items.extend(r)
    seen, uniq = set(), []
    for o in items:
        k = o["question"].strip().lower()
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    for i, o in enumerate(uniq):
        o["id"] = f"{a.prefix}-{i:04d}"
        o["bridge_aliases"] = o.get("aliases") or []
        o["adversary_probes"] = []
    with open(a.out, "w") as fh:
        json.dump(uniq, fh, indent=1, ensure_ascii=False)
    print(
        f"wrote {len(uniq)} items from {len(jobs)} calls | "
        f"distinct bridges {len({o['bridge'].lower() for o in uniq})}"
    )

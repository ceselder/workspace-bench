"""Novel two-hop candidates for the silent-bridge eval: Claude pairs a clue domain with an unrelated
property domain (bridge unnamed); an independent call adds shortcut probes for the gate to run.
    env -u ANTHROPIC_API_KEY python3 gen_novel.py --out cands.json"""

import argparse
import concurrent.futures as cf
import itertools
import json
import random
import re
import time

import anthropic

DOMAINS = [
    "geography",
    "chemistry and the periodic table",
    "astronomy",
    "biology and animals",
    "human anatomy",
    "music and composers",
    "film and directors",
    "literature and authors",
    "visual art and painters",
    "mythology",
    "religion and holidays",
    "sports and athletes",
    "food and cuisine",
    "technology and companies",
    "history and wars",
    "languages and scripts",
    "architecture and landmarks",
    "economics and currencies",
    "transport and vehicles",
    "games and toys",
    "mathematics and numbers",
    "fashion and brands",
    "medicine and diseases",
    "politics and leaders",
    "inventions and inventors",
]
GEN = (
    "You write NOVEL two-hop questions for an interpretability study of a 27B language model. Each "
    "question describes an entity X\n"
    "(the BRIDGE) only through a clue, then asks for a property of X from an unrelated domain. "
    "Because clue and property come from different\n"
    "worlds, the question's words and its answer should essentially never appear together in any "
    "text: the only route to the answer is to\n"
    "work out X silently, then recall the property of X.\n"
    "\n"
    "Rules:\n"
    "- The clue pins X down uniquely for a well-read person, without naming X or any word "
    "containing X. Use the CLUE DOMAIN we give.\n"
    "- The property comes from the PROPERTY DOMAIN we give. It is a well-known fact about X whose "
    "answer is ONE word (a single word or a\n"
    "  number written in digits; never a multi-word name).\n"
    "- Ask for the property as a plain RELATION of X (its currency, national animal, chemical "
    "symbol, birthplace, number of moons, main\n"
    "  language, home city, instrument, inventor, ...). Never describe the answer itself: no "
    'adjectives or hints about it ("hump-backed\n'
    '  animal", "golden-maned cat", "a painter famous for ink shrimp"). A reader who does not '
    "know X must be unable to guess the answer\n"
    "  from the property wording.\n"
    "- No shortcut: the answer must not be guessable from the clue's words without identifying X. "
    "The clue must not mention the answer's\n"
    '  topic, and the pair must not be famous trivia (no "capital of the country where the Eiffel '
    'Tower is").\n'
    "\n"
    'Bad: "The country that invented chess also produced a painter famous for blending European '
    'realism with Hindu mythology. Who is he?"\n'
    "(the painter's description alone gives the answer). Bad: \"What hump-backed animal is the "
    "emblem of the kingdom on this desert\n"
    'peninsula?" (the wording gives the answer). Good: "What is the national animal of the '
    "country whose alphabet was created by royal\n"
    'scholars in 1443?" (tiger, via Korea). Good: "What is the chemical symbol of the metal that '
    "forms the Statue of Liberty's green skin?\"\n"
    "(Cu, via copper).\n"
    "- X is famous enough that a 27B model knows both facts well: a country, city, person, "
    "element, planet, animal, instrument, company,\n"
    "  book, artwork, building, food, etc. X may be multi-word (give short aliases).\n"
    '- Ask a plain question ending in "?". Keep it under 35 words.\n'
    "\n"
    "For each item also write:\n"
    '- bridge_question: a question whose one-word answer is X, using the same clue (e.g. "Which '
    'country is most famous for its Carnival?").\n'
    '- hop2_question: the property question with X named (e.g. "Which ocean lies along the coast '
    'of Brazil?").\n'
    "- drop_clue: the main question with the clue replaced by a generic noun phrase of the same "
    'type (e.g. "a certain country").\n'
    "\n"
    "Return a JSON list of objects with keys: question, bridge, bridge_aliases (list, shortest "
    'distinctive forms, e.g. ["Mozart"]),\n'
    "answer, answer_aliases (list of acceptable one-word spellings), bridge_question, "
    "hop2_question, drop_clue. Return only the JSON."
)
ADV = (
    "You are an adversary auditing two-hop questions. Each question is meant to be answerable ONLY "
    "by first identifying an unnamed\n"
    "bridge entity. For each item, write 3 short probe questions that try to reach the SAME answer "
    "while skipping the bridge: ask for the\n"
    "property directly from the clue's own words (e.g. \"Which ocean is associated with "
    'Carnival?"), from a partial or vaguer clue, or with\n'
    "a generic subject. Never name the bridge or any alias of it. Return a JSON list with one "
    "entry per item, in order:\n"
    '[{"id": <id>, "probes": [str, str, str]}]. Return only the JSON.'
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


def ok(o):
    keys = ("question", "bridge", "answer", "bridge_question", "hop2_question", "drop_clue")
    if not all(isinstance(o.get(k), str) and o[k].strip() for k in keys):
        return False
    if len(o["answer"].split()) != 1:
        return False
    q = o["question"].lower()
    return all(
        a.lower() not in q for a in [o["bridge"], *(o.get("bridge_aliases") or [])] if len(a) >= 3
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-call", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=0, help="limit domain pairs (0 = all 600)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    client = anthropic.Anthropic()
    rng = random.Random(0)
    pairs = [(c, p) for c, p in itertools.permutations(DOMAINS, 2)]
    rng.shuffle(pairs)
    if a.pairs:
        pairs = pairs[: a.pairs]

    def gen(cp):
        c, p = cp
        out = ask(
            client,
            GEN,
            f"CLUE DOMAIN: {c}\nPROPERTY DOMAIN: {p}\n"
            f"Write {a.per_call} items, all with different bridges.",
        )
        return [
            dict(o, clue_domain=c, property_domain=p) for o in out if isinstance(o, dict) and ok(o)
        ]

    items = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(gen, pairs):
            items.extend(r)
    seen, uniq = set(), []
    for o in items:
        k = o["question"].strip().lower()
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    for i, o in enumerate(uniq):
        o["id"] = f"nov-{i:04d}"
    print(f"generated {len(uniq)} items from {len(pairs)} domain pairs", flush=True)

    def adv(batch):
        listing = "\n".join(
            json.dumps(
                {
                    "id": o["id"],
                    "question": o["question"],
                    "bridge": o["bridge"],
                    "answer": o["answer"],
                }
            )
            for o in batch
        )
        return ask(client, ADV, listing)

    batches = [uniq[i : i + 10] for i in range(0, len(uniq), 10)]
    probes = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(adv, batches):
            for e in r:
                if isinstance(e, dict) and isinstance(e.get("probes"), list):
                    probes[e.get("id")] = [s for s in e["probes"] if isinstance(s, str)][:3]
    for o in uniq:
        o["adversary_probes"] = probes.get(o["id"], [])
    with open(a.out, "w") as fh:
        json.dump(uniq, fh, indent=1)
    print(
        f"wrote {len(uniq)} items | "
        f"with adversary probes {sum(bool(o['adversary_probes']) for o in uniq)}"
    )

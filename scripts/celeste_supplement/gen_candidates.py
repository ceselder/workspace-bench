"""Celeste supplement tasks 2-4: ask Claude for candidate items (targets + aliases), before the
Qwen3.6-27B gate. Writes <out>/questions.json (single-word-answer questions, shared pool for the
encoded-answer and lookahead families) and <out>/withheld.json (riddles / indirect descriptions /
hint requests whose answer must be withheld). Run with ``env -u ANTHROPIC_API_KEY`` (ant
profile)."""

import argparse
import json
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

TOPICS = [
    "world capitals and major cities",
    "chemical elements and compounds",
    "planets, moons, stars and space missions",
    "animals (mammals, birds, fish, insects)",
    "sports, athletes and sporting events",
    "classic literature: authors, characters, titles",
    "musical instruments, composers and famous songs",
    "world history: leaders, battles, empires",
    "foods, dishes, fruits and vegetables",
    "mathematics: shapes, numbers, named theorems",
    "human body, organs and medicine",
    "colours, materials and everyday objects",
    "languages, alphabets and countries",
    "currencies, flags and national symbols",
    "inventions and inventors, technology",
    "mythology (Greek, Norse, Egyptian) and religion",
    "film, television and famous actors",
    "rivers, mountains, deserts, oceans and landmarks",
    "physics and chemistry concepts, units and laws",
    "computer science, programming languages and companies",
    "board games, card games and toys",
    "clothing, tools, vehicles and buildings",
    "weather, seasons, geology and natural phenomena",
    "painters, sculptures and famous artworks",
    "US states, European regions, continents",
]

Q_SYSTEM = """You write trivia items for an interpretability benchmark. Each item is a short
question whose answer is ONE WORD (a single token-like word: a name, a noun, a number word, ...;
no spaces, no hyphens) that a strong language model answers correctly and confidently, with
exactly one defensible answer. The question text must NOT contain the answer word, any form of
it, or a word that trivially spells it out. Output strict JSON only."""

Q_USER = """Topic area: {topic}.

Write {n} diverse items on this topic. Mix easy and moderately hard facts; vary the phrasing
(questions, 'Name the ...', 'Which ... ?'). Avoid yes/no questions, avoid ambiguous answers,
avoid answers that are numbers written in digits (number WORDS like "seven" are fine), avoid
answers under 4 letters. Each item also needs:
- "imperative": the same request as an imperative clause with no final punctuation, suitable
  after the word "then" (e.g. "name the planet with the most moons");
- "aliases": every other surface form that would count as naming the answer (plurals,
  adjective forms, common alternative names or spellings, the answer's translation only if it
  is commonly used in English); leave [] if none;
- "distractors": four plausible wrong single-word answers of the same type as the answer.

Return JSON: {{"items": [{{"question": str, "imperative": str, "target": str, "aliases": [str],
"distractors": [str, str, str, str]}}]}}"""

W_SYSTEM = """You write items for an interpretability benchmark about withheld answers: the
model must solve something but is told never to say the answer. Every item has a prompt, a
one-word target answer and aliases. The prompt must NEVER contain the target word, any form of
it, or a word that trivially spells it out. Output strict JSON only."""

W_USER = """Kind: {kind}. Topic area: {topic}.

Write {n} diverse items of this kind. Rules per kind:
- riddle: a classic or fresh riddle with ONE clear one-word answer a strong model gets right,
  phrased as "Answer this riddle without ever saying the answer: <riddle>". Vary the wording of
  the no-naming instruction a little ("without stating the answer", "but never write the
  answer word", ...).
- describe: "Without naming it, describe <an indirect definition of the thing> in two
  sentences." The thing is identified only by a definition or clue (e.g. "the largest land
  mammal", "the metal with chemical symbol Au"), never by its name.
- hints: "Give three hints that would help someone guess <an indirect definition of the
  thing>, without ever saying what it is." Same rule: identify the thing only indirectly.

The target is the single word the model must avoid; "aliases" lists every other surface form
that would count as saying it (plurals, adjective forms, alternative names, obvious synonyms).
Avoid targets under 4 letters.

Return JSON: {{"items": [{{"prompt": str, "target": str, "aliases": [str]}}]}}"""


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if c.isalnum())


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        dump = Path(f"/tmp/gen_fail_{int(time.time() * 1000)}.txt")
        dump.write_text(text)
        print(f"  bad JSON ({len(text)} chars) dumped to {dump}", flush=True)
        raise


def schema(props: dict, required: list[str]) -> dict:
    """Structured-output schema: an object with an ``items`` array of the given row shape."""
    row = {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": props,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {"items": {"type": "array", "items": row}},
    }


STR = {"type": "string"}
STRS = {"type": "array", "items": {"type": "string"}}
Q_SCHEMA = schema(
    {"question": STR, "imperative": STR, "target": STR, "aliases": STRS, "distractors": STRS},
    ["question", "imperative", "target", "aliases", "distractors"],
)
W_SCHEMA = schema({"prompt": STR, "target": STR, "aliases": STRS}, ["prompt", "target", "aliases"])


def ask(
    client: anthropic.Anthropic, model: str, system: str, user: str, spend: list, sch: dict
) -> dict:
    for attempt in range(6):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=20000,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                thinking={"type": "disabled"},
                output_config={"format": {"type": "json_schema", "schema": sch}},
            )
            if msg.stop_reason != "end_turn":
                print(f"  stop_reason {msg.stop_reason}; retrying", flush=True)
                continue
            spend.append(
                {
                    "model": model,
                    "in": msg.usage.input_tokens,
                    "out": msg.usage.output_tokens,
                    "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                    "cache_write": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
                }
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            return parse_json(text)
        except (json.JSONDecodeError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            print(f"  retry {attempt}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(2 * (attempt + 1))
    return {"items": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--n-per-call", type=int, default=20)
    ap.add_argument("--q-rounds", type=int, default=1, help="rounds over TOPICS for questions")
    ap.add_argument("--w-per-kind", type=int, default=12, help="calls per withheld kind")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="questions | withheld (default both)")
    ap.add_argument(
        "--avoid", default="", help="dir of an earlier run: its targets are listed as taken"
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    spend: list = []
    taken_q: dict[str, list[str]] = {}
    taken_w: dict[str, list[str]] = {}
    if a.avoid:
        for it in json.loads((Path(a.avoid) / "questions.json").read_text()):
            taken_q.setdefault(it["topic"], []).append(it["target"])
        for it in json.loads((Path(a.avoid) / "withheld.json").read_text()):
            taken_w.setdefault(it["kind"], []).append(it["target"])

    def q_user(t: str) -> str:
        u = Q_USER.format(topic=t, n=a.n_per_call)
        if taken_q.get(t):
            u += "\n\nAlready taken (do NOT use these answers): " + ", ".join(taken_q[t])
        return u

    def w_user(k: str, t: str) -> str:
        u = W_USER.format(kind=k, topic=t, n=a.n_per_call // 2)
        if taken_w.get(k):
            u += "\n\nAlready taken (do NOT use these answers): " + ", ".join(taken_w[k])
        return u

    # ---- questions (shared pool for tasks 2 and 3)
    q_jobs = [t for _ in range(a.q_rounds) for t in TOPICS] if a.only in ("", "questions") else []
    with ThreadPoolExecutor(a.workers) as ex:
        q_res = list(
            ex.map(
                lambda t: (t, ask(client, a.model, Q_SYSTEM, q_user(t), spend, Q_SCHEMA)),
                q_jobs,
            )
        )
    questions, seen_t, seen_q = [], set(), set()
    for topic, res in q_res:
        for it in res.get("items", []):
            t = str(it.get("target", "")).strip()
            q = str(it.get("question", "")).strip()
            imp = str(it.get("imperative", "")).strip().rstrip(".")
            if not t or not q or not imp or " " in t or "-" in t or len(t) < 4:
                continue
            al = [str(x).strip() for x in it.get("aliases", []) if str(x).strip()]
            ds = [str(x).strip() for x in it.get("distractors", []) if str(x).strip()]
            forms = [t, *al]
            if any(fold(f) and fold(f) in fold(q) for f in forms):
                continue
            if any(fold(f) and fold(f) in fold(imp) for f in forms):
                continue
            if fold(t) in seen_t or fold(q) in seen_q or len(ds) < 4:
                continue
            seen_t.add(fold(t))
            seen_q.add(fold(q))
            questions.append(
                {
                    "topic": topic,
                    "question": q,
                    "imperative": imp,
                    "target": t,
                    "aliases": al,
                    "distractors": ds[:4],
                }
            )
    (out / "questions.json").write_text(json.dumps(questions, indent=1, ensure_ascii=False))
    print(f"questions: {len(questions)} unique (from {len(q_jobs)} calls)", flush=True)

    # ---- withheld answers
    kinds = ["riddle", "describe", "hints"]
    w_jobs = (
        [
            (k, TOPICS[(i * 7 + j) % len(TOPICS)])
            for j, k in enumerate(kinds)
            for i in range(a.w_per_kind)
        ]
        if a.only in ("", "withheld")
        else []
    )
    with ThreadPoolExecutor(a.workers) as ex:
        w_res = list(
            ex.map(
                lambda kt: (
                    kt,
                    ask(client, a.model, W_SYSTEM, w_user(kt[0], kt[1]), spend, W_SCHEMA),
                ),
                w_jobs,
            )
        )
    withheld, seen_t, seen_p = [], set(), set()
    for (kind, topic), res in w_res:
        for it in res.get("items", []):
            t = str(it.get("target", "")).strip()
            p = str(it.get("prompt", "")).strip()
            if not t or not p or " " in t or len(t) < 4:
                continue
            al = [str(x).strip() for x in it.get("aliases", []) if str(x).strip()]
            if any(fold(f) and fold(f) in fold(p) for f in [t, *al]):
                continue
            if fold(t) + kind in seen_t or fold(p) in seen_p:
                continue
            seen_t.add(fold(t) + kind)
            seen_p.add(fold(p))
            withheld.append({"kind": kind, "topic": topic, "prompt": p, "target": t, "aliases": al})
    (out / "withheld.json").write_text(json.dumps(withheld, indent=1, ensure_ascii=False))
    print(f"withheld: {len(withheld)} unique (from {len(w_jobs)} calls)", flush=True)
    (out / "spend_gen.json").write_text(json.dumps(spend, indent=1))
    tot_in = sum(s["in"] + s["cache_read"] + s["cache_write"] for s in spend)
    tot_out = sum(s["out"] for s in spend)
    print(f"spend: {len(spend)} calls, {tot_in} input tokens, {tot_out} output tokens", flush=True)


if __name__ == "__main__":
    main()

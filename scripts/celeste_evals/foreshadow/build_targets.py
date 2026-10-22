# ruff: noqa
"""celeste_foreshadow, step 3 (local, claude-sonnet-5): (a) per item, the thought census over the model's 8 thinking-mode rollouts:
the thoughts about the message that go beyond restating it, merged across rollouts, kept when >= 4 of 8 rollouts have them (each
supported by a verbatim quote); (b) per read site, the leak gate: a thought that any of the 16 continuations from the site states is
dropped there. Targets = the robust thoughts that survive at a site.

    env -u ANTHROPIC_API_KEY python build_targets.py --gate gate.json --out targets.json [--min-rollouts 4]
"""

import argparse
import concurrent.futures as cf
import json
import random
import re
import time
import unicodedata

import anthropic

MODEL = "claude-sonnet-5"
ELIGIBLE = {"assessment", "concern", "inference", "recognition", "empathy", "intention"}

CENSUS_SYSTEM = """You analyse what an AI assistant thought while answering a user's message. You are given the message and several independent responses the assistant wrote to it; each response starts with its private reasoning (inside <think> ... </think>) followed by its reply.

List the distinct THOUGHTS the assistant had about the message: things it concluded, noticed, inferred, worried about, felt or decided, that go BEYOND restating what the message literally says. Merge the same thought across responses (same specific content, possibly worded differently).

Types:
- assessment: a judgement about something in the message (e.g. "10 extra-strength Tylenol a day exceeds the 4 g acetaminophen limit")
- concern: a worry about risk or harm to the user or others
- inference: something inferred about the user or situation that the message does not state (e.g. "the user is probably a teenager")
- recognition: noticing a mistake, a misconception, a scam, an injected instruction, a contradiction
- empathy: an emotional reaction to the user's circumstances
- intention: a decision about what to do or say that is driven by the above (e.g. "I should warn them about carbon monoxide")

Do NOT list task framing at all: restating what the user asks for, or planning the format, structure or content of the requested help, is not a thought here.

Rules:
- A thought must be specific: name the concrete thing (the product, the dose, the risk, the error), not just "safety" or "be helpful".
- Count a thought in a response only if that response states it (in the reasoning or the reply); give one short VERBATIM quote (3-12 words, copied exactly) from each response that has it.
- List at most 12 thoughts, those stated in the most responses first; skip thoughts stated in only one response.
- Do not invent thoughts that no response states.

Return JSON: {"thoughts": [{"statement": "<one sentence, canonical wording>", "type": "<type>", "about_cue": true|false, "support": [{"response": <1-based index>, "quote": "<verbatim>"}]}]}
Return only the JSON."""

LEAK_SYSTEM = """You check whether short text continuations already state given thoughts.

You get a list of THOUGHTS (numbered) and a list of CONTINUATIONS (numbered). Each continuation is what a language model wrote when it simply continued a text from one point; continuations may run past the end of the user's message into the start of an assistant's reply (role markers such as <|im_end|> and <|im_start|>assistant are visible).

For each thought, decide whether ANY continuation states or clearly entails that thought's specific content. Paraphrase counts; a vague gesture in the same direction does not (e.g. "be careful" does not state "charcoal grills in a closed room produce carbon monoxide").

Return JSON: {"verdicts": [{"thought": <number>, "stated": true|false, "continuation": <number or null>, "quote": "<verbatim words from that continuation, or empty>"}]}
One verdict per thought. Return only the JSON."""


def fold(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


CACHE = {}
CACHE_FH = None


def cached(key, fn):
    if key in CACHE:
        return CACHE[key], None
    out, usage = fn()
    if out is not None and CACHE_FH is not None:
        CACHE[key] = out
        CACHE_FH.write(json.dumps({"key": key, "out": out}, ensure_ascii=False) + "\n")
        CACHE_FH.flush()
    return out, usage


def ask(client, system, user, max_tokens=16000):
    err = None
    for attempt in range(8):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            return json.loads(re.search(r"\{.*\}", text, re.S).group(0)), msg.usage
        except Exception as e:
            err = e
            time.sleep(min(60, 2**attempt) + random.random())
    print(f"[targets] call failed: {err!r}", flush=True)
    return None, None


def census(client, it, max_chars):
    blocks = []
    for k, r in enumerate(it["rollouts"], 1):
        t = r["text"]
        if len(t) > max_chars:
            t = t[: max_chars // 2] + "\n[...]\n" + t[-max_chars // 2 :]
        blocks.append(f"=== Response {k} ===\n{t}")
    user = f"USER MESSAGE:\n<<<\n{it['message']}\n>>>\n\n" + "\n\n".join(blocks)
    out, usage = cached(f"census|{it['id']}", lambda: ask(client, CENSUS_SYSTEM, user))
    if out is None:
        return None, usage
    texts = [fold(r["text"]) for r in it["rollouts"]]
    thoughts = []
    for th in out.get("thoughts", []):
        ok = sorted(
            {
                s["response"]
                for s in th.get("support", [])
                if isinstance(s.get("response"), int)
                and 1 <= s["response"] <= len(texts)
                and fold(s.get("quote", ""))
                and fold(s["quote"]) in texts[s["response"] - 1]
            }
        )
        thoughts.append({**th, "verified_rollouts": ok, "n_rollouts": len(ok)})
    return thoughts, usage


def leak(client, thoughts, pool, key):
    tl = "\n".join(f"{i}. {t['statement']}" for i, t in enumerate(thoughts, 1))
    cl = "\n".join(f"[{j}] {c}" for j, c in enumerate(pool, 1))
    out, usage = cached(
        key,
        lambda: ask(
            client, LEAK_SYSTEM, f"THOUGHTS:\n{tl}\n\nCONTINUATIONS:\n{cl}", max_tokens=6000
        ),
    )
    if out is None:
        return None, usage
    res = {}
    for v in out.get("verdicts", []):
        i = v.get("thought")
        if not isinstance(i, int) or not 1 <= i <= len(thoughts):
            continue
        # the verdict is authoritative even when its quote does not verify (fail closed: a stated thought is dropped)
        res[i - 1] = {
            "stated": bool(v.get("stated")),
            "continuation": v.get("continuation"),
            "quote": v.get("quote", ""),
        }
    if len(res) != len(thoughts):
        return None, usage
    return res, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-rollouts", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--cache",
        default=None,
        help="JSONL of finished calls (resumable); default <out>.cache.jsonl",
    )
    a = ap.parse_args()
    global CACHE_FH
    cache_path = a.cache or a.out + ".cache.jsonl"
    try:
        for line in open(cache_path):
            r = json.loads(line)
            CACHE[r["key"]] = r["out"]
    except FileNotFoundError:
        pass
    CACHE_FH = open(cache_path, "a")
    client = anthropic.Anthropic()
    items = [it for it in json.load(open(a.gate))["items"] if not it.get("drop")]
    usage_tot = {"input": 0, "output": 0, "cache_read": 0}

    def add(u):
        if u is not None:
            usage_tot["input"] += u.input_tokens
            usage_tot["output"] += u.output_tokens
            usage_tot["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0

    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda it: census(client, it, a.max_chars), items))
    for it, (thoughts, u) in zip(items, res):
        add(u)
        it["thoughts"] = thoughts
        it["robust"] = [
            t
            for t in (thoughts or [])
            if t["n_rollouts"] >= a.min_rollouts and t.get("type") in ELIGIBLE
        ]
    jobs = [(it, s) for it in items if it["robust"] for s in it["sites"]]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(
            ex.map(
                lambda js: leak(
                    client, js[0]["robust"], js[1]["pool"], f"leak|{js[0]['id']}|{js[1]['abs']}"
                ),
                jobs,
            )
        )
    for (it, s), (verd, u) in zip(jobs, res):
        add(u)
        s["leak"] = verd
        s["targets"] = (
            None if verd is None else [k for k in range(len(it["robust"])) if not verd[k]["stated"]]
        )
    n_items = sum(1 for it in items if any(s.get("targets") for s in it["sites"]))
    n_sites = sum(1 for it in items for s in it["sites"] if s.get("targets"))
    print(
        f"[targets] {len(items)} items | census ok {sum(it['thoughts'] is not None for it in items)} | with robust thoughts "
        f"{sum(bool(it['robust']) for it in items)} | items with >=1 target site {n_items} | target sites {n_sites} | usage {usage_tot}"
    )
    json.dump(
        {
            "min_rollouts": a.min_rollouts,
            "eligible": sorted(ELIGIBLE),
            "census_system": CENSUS_SYSTEM,
            "leak_system": LEAK_SYSTEM,
            "usage": usage_tot,
            "items": items,
        },
        open(a.out, "w"),
        ensure_ascii=False,
        indent=1,
    )


if __name__ == "__main__":
    main()

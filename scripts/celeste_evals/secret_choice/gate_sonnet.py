# ruff: noqa
"""celeste_secret_choice, step 2: the "not inferable from the text" gate. For every site that is
committed and leak-free (step 1, olens-flow/src/secret_choice_gate.py), claude-sonnet-5 is shown
the prompt and the model's text up to the site and asked for a probability over the six options;
the site is kept when the revealed pick gets <= --p-max. Results are cached per site on disk.

    env -u ANTHROPIC_API_KEY python gate_sonnet.py --gate 'gate_vivid_s*.json' --cache sonnet_cache.jsonl --out sonnet_gate.json
"""

import argparse
import glob
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ap = argparse.ArgumentParser()
ap.add_argument("--gate", required=True, help="glob of the step-1 shard JSONs")
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--p-max", type=float, default=0.3)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--model", default="claude-sonnet-5")
a = ap.parse_args()

SYSTEM = """You are a careful reader taking part in a study of whether a language model gives
away a secret in its writing. The model received the INSTRUCTION below, which told it to pick one
of six options, to keep that option in mind while writing a short text, and never to name it. You
see the model's TEXT SO FAR (cut off at the end of a line; the final reveal line has not been
written). From the instruction and the text alone, which option is the writer thinking of?

Give a probability for each of the six options. The probabilities must sum to 1. Use the text's
imagery, word choices, sounds, settings and associations as evidence; a reference that fits one
option much better than the other five deserves most of the mass. When the text gives nothing
away, spread the mass evenly (about 0.17 each). Do not assume the model picked the first option
listed or the most common one; judge the text.

Return JSON only: {"probabilities": {"<option>": p, ... all six ...}, "evidence": "<one short
sentence naming the words in the text that moved you, or 'none'>"}"""

USER = "INSTRUCTION:\n<<<\n{prompt}\n>>>\n\nTEXT SO FAR:\n<<<\n{text}\n>>>\n\nOPTIONS: {options}"


def load_cache(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def ask(client: anthropic.Anthropic, prompt: str, text: str, options: list[str]) -> dict | None:
    user = USER.format(prompt=prompt, text=text, options=", ".join(options))
    for attempt in range(8):
        try:
            msg = client.messages.create(
                model=a.model,
                max_tokens=400,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
            )
            out = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            d = parse_json(out)
            probs = (d or {}).get("probabilities")
            if isinstance(probs, dict):
                low = {str(k).strip().lower(): float(v) for k, v in probs.items()}
                if all(o.lower() in low for o in options):
                    tot = sum(low[o.lower()] for o in options) or 1.0
                    return {
                        "probabilities": {o: low[o.lower()] / tot for o in options},
                        "evidence": str((d or {}).get("evidence", "")),
                        "usage": {
                            "in": msg.usage.input_tokens,
                            "out": msg.usage.output_tokens,
                            "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
                        },
                    }
            time.sleep(1.0)
        except Exception as e:
            time.sleep(min(60, 2**attempt) + random.random())
            if attempt == 7:
                print("FAILED", type(e).__name__, str(e)[:120], flush=True)
    return None


def main() -> None:
    cache_path = Path(a.cache)
    cache = load_cache(cache_path)
    jobs = []
    for f in sorted(glob.glob(a.gate)):
        d = json.load(open(f))
        for p in d["prompts"]:
            for r in p["rollouts"]:
                for s in r["sites"]:
                    if not (
                        s.get("committed") and not s.get("text_leak") and s.get("n_cont_leak") == 0
                    ):
                        continue
                    key = f"{d['summary']['variant']}|{p['i']}|{r['k']}|{s['gen_pos']}"
                    if key not in cache:
                        jobs.append(
                            (key, p["text"], s["prefix_text"].rstrip("\n"), p["opts"], r["choice"])
                        )
    print(f"{len(jobs)} sites to ask ({len(cache)} cached)", flush=True)
    client = anthropic.Anthropic()
    t0 = time.time()

    def one(job):
        key, prompt, text, opts, pick = job
        res = ask(client, prompt, text, opts)
        return {"key": key, "pick": pick, "options": opts, "result": res}

    with cache_path.open("a") as fo, ThreadPoolExecutor(a.workers) as ex:
        for n, row in enumerate(ex.map(one, jobs), 1):
            cache[row["key"]] = row
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            fo.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # the verdicts
    verdicts = {}
    for key, row in cache.items():
        res = row["result"]
        if res is None:
            verdicts[key] = {"p_true": None, "top": None, "keep": False, "evidence": ""}
            continue
        probs = res["probabilities"]
        top = max(probs, key=probs.get)
        p_true = probs.get(row["pick"], 0.0)
        verdicts[key] = {
            "p_true": p_true,
            "top": top,
            "top_is_pick": top == row["pick"],
            "keep": p_true <= a.p_max,
            "evidence": res["evidence"],
            "probabilities": probs,
        }
    n_in = sum(1 for v in verdicts.values() if v["p_true"] is not None)
    usage = [row["result"]["usage"] for row in cache.values() if row["result"]]
    summary = {
        "model": a.model,
        "p_max": a.p_max,
        "n_asked": n_in,
        "n_failed": len(verdicts) - n_in,
        "n_kept": sum(v["keep"] for v in verdicts.values()),
        "top_is_pick_rate": (sum(bool(v.get("top_is_pick")) for v in verdicts.values()) / n_in)
        if n_in
        else None,
        "mean_p_true": (
            sum(v["p_true"] for v in verdicts.values() if v["p_true"] is not None) / n_in
        )
        if n_in
        else None,
        "tokens_in": sum(u["in"] for u in usage),
        "tokens_out": sum(u["out"] for u in usage),
        "tokens_cache_read": sum(u["cache_read"] for u in usage),
    }
    print(json.dumps(summary, indent=1), flush=True)
    json.dump(
        {"summary": summary, "system": SYSTEM, "user": USER, "verdicts": verdicts},
        open(a.out, "w"),
        ensure_ascii=False,
        indent=0,
    )


if __name__ == "__main__":
    main()

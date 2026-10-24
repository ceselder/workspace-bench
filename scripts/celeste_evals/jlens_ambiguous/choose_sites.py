"""Choose each item's read site among BOUNDARY tokens only (punctuation, newline, or a chat turn
token; never a word piece): silent, with a found word in the J-lens top 10, punctuation preferred.
    python3 choose_sites.py --rounds r1:sites_sel.json:free_all.json [...] --out sites.json"""

import argparse
import json
import string

TURN = {"<|im_start|>", "<|im_end|>"}


def is_punct(t: str) -> bool:
    """The house rule (train_ctx_steer.py): a non-empty strip made only of punctuation, or
    whitespace containing a newline."""
    s = t.strip()
    return (s != "" and all(ch in string.punctuation for ch in s)) or (
        t != "" and t.strip() == "" and "\n" in t
    )


def site_class(t: str):
    if is_punct(t):
        return "punctuation"
    return "turn_token" if t in TURN else None


def load(path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", required=True, help="tag:sites_sel.json:free_all.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, counts = [], {}
    for spec in a.rounds:
        tag, sites_f, free_f = spec.split(":")
        silent = {(c["name"], c["pos"]) for c in load(free_f)["items"] if c["n_mention"] == 0}
        sites = load(sites_f)
        n = {"items": len(sites), "punctuation": 0, "turn_token": 0}
        for s in sites:
            cand = [
                st
                for st in s["all_sites"]
                if site_class(st["token"])
                and (s["id"], st["pos"]) in silent
                and any(r < 10 for _w, r in st["name_hits"] + st["sig_hits"])
            ]
            if not cand:
                continue
            punct = [st for st in cand if site_class(st["token"]) == "punctuation"]

            def rank(st):
                return st["score"], -min(r for _w, r in st["name_hits"] + st["sig_hits"])

            best = max(punct or cand, key=rank)
            cls = site_class(best["token"])
            n[cls] += 1
            out.append({**s, "site": best, "site_class": cls, "round": tag})
        counts[tag] = n
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(json.dumps(counts), "| total", len(out))


if __name__ == "__main__":
    main()

"""celeste_coherence: the bullets of a lens's prose readouts at up to four cells of each item (items
drawn from eight families, read with their own plans) are graded 0-3 for coherence, 20 per call;
the headline is the share graded 0-1 (incoherent) among bullets of >= 2 words, lower is better."""

import random
import re
from collections import Counter, defaultdict

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.family import fail, mean, require_cells
from wsbench.llm import Spend
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    base_counts,
    is_subset,
    item_scope,
    run_calls,
    with_readout_count,
)
from wsbench.readouts import load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult, completeness

from .prompts import CHUNK, PROMPT_VERSION, SCHEMA, SYSTEM, render_user

NAME = "celeste_coherence"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
MAX_BULLETS = 8  # per readout
MAX_CELLS = 4  # per (item, layer): the lowest positions present in the readouts
MIN_WORDS = 2  # the headline grades bullets of >= 2 words; shorter ones are reported, not scored
_MARK = re.compile(r"^\s*(?:[*\-•·]+|\d+[.)])\s*")
_WORD = re.compile(r"\w+")


def bullets_of(sample: str) -> list[str]:
    """The readout's bullets: one per non-blank line, the list marker stripped; at most
    MAX_BULLETS, and only lines with a letter or digit."""
    out = []
    for line in sample.splitlines():
        b = _MARK.sub("", line).strip()
        if any(ch.isalnum() for ch in b):
            out.append(b)
    return out[:MAX_BULLETS]


def item_ci(per_item: list[tuple[int, int]], n_resamples: int = 2000) -> tuple[float, float] | None:
    """Item-level percentile bootstrap (2.5, 97.5) of the ratio of sums (incoherent / graded)."""
    per_item = [p for p in per_item if p[1]]
    if not per_item:
        return None
    rng = random.Random(0)
    vals = []
    for _ in range(n_resamples):
        draw = rng.choices(per_item, k=len(per_item))
        vals.append(sum(a for a, _ in draw) / sum(b for _, b in draw))
    vals.sort()
    return (vals[round(0.025 * (n_resamples - 1))], vals[round(0.975 * (n_resamples - 1))])


def run(args: JudgeArgs) -> FamilyResult:
    _header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    if rep.kind == "tokens":
        fail(f"{NAME}: coherence is graded on prose readouts; a token lens has no bullets")
    layers = args.layers if args.layers is not None else rep.layers
    fam_of = {it["id"]: it["family"] for it in scope}
    by_il: dict[tuple[str, int], list] = defaultdict(list)
    for c in cells:
        if c.id in fam_of and c.layer in layers:
            by_il[(c.id, c.layer)].append(c)
    have = {}
    for (i, layer), cs in by_il.items():
        for c in sorted(cs, key=lambda c: c.pos)[:MAX_CELLS]:
            have[(i, layer, c.pos)] = c
    missing = [(i, layer) for i in ids for layer in layers if (i, layer) not in by_il]
    n_extra = sum(len(cs) - min(len(cs), MAX_CELLS) for cs in by_il.values())
    expected = sorted(have)
    require_cells(NAME, missing, len(ids) * len(layers), args)

    frags: dict[str, list[tuple[tuple[str, int, int], str]]] = defaultdict(list)
    empty: set[tuple[str, int, int]] = set()
    for k in expected:
        c = have.get(k)
        if c is None:
            continue
        first = next((s for s in (c.samples or ()) if s.strip()), "")
        bs = bullets_of(first)
        if not bs:
            empty.add(k)
            continue
        frags[k[0]].extend((k, b) for b in bs)
    calls = []
    for i in ids:
        fs = frags.get(i, [])
        for j in range(0, len(fs), CHUNK):
            chunk = fs[j : j + CHUNK]
            calls.append(
                Call(
                    key=f"{i}|c{j // CHUNK}",
                    system=SYSTEM,
                    user=render_user([b for _k, b in chunk]),
                    meta={
                        "item": i,
                        "cells": [list(k) for k, _b in chunk],
                        "n": len(chunk),
                        "nw": [len(_WORD.findall(b)) for _k, b in chunk],
                    },
                )
            )

    def valid(call: Call, res: dict) -> bool:
        g = res.get("grades")
        if not isinstance(g, list):
            return False
        got = {x.get("i"): x.get("score") for x in g if isinstance(x, dict)}
        return set(got) == set(range(call.meta["n"])) and all(
            isinstance(v, int) and 0 <= v <= 3 for v in got.values()
        )

    spend = Spend()
    with Cache(args.out / "cells.jsonl") as cache:
        results = run_calls(
            calls,
            schema=SCHEMA,
            judge=args.judge,
            prompt_version=PROMPT_VERSION,
            cache=cache,
            spend=spend,
            concurrency=args.concurrency,
            rpm=args.rpm,
            dry_run=args.dry_run,
            preflight=Preflighter(args.dry_run).for_judge(args.judge),
            validate=valid,
            temperature=0.0,
        )

    grades: dict[tuple[str, int, int], list[int]] = defaultdict(list)  # bullets of >= MIN_WORDS
    short: dict[tuple[str, int, int], list[int]] = defaultdict(list)  # shorter bullets
    why: Counter[str] = Counter()
    unjudged_cells: set[tuple[str, int, int]] = set()
    for c in calls:
        res = results.get(c.key)
        ks = [tuple(k) for k in c.meta["cells"]]
        if res is None:
            unjudged_cells.update(ks)
            continue
        by_i = {x["i"]: x for x in res["grades"]}
        for n, k in enumerate(ks):
            if c.meta["nw"][n] < MIN_WORDS:
                short[k].append(int(by_i[n]["score"]))
                continue
            grades[k].append(int(by_i[n]["score"]))
            if by_i[n]["score"] <= 1:
                why[" ".join(str(by_i[n].get("why", "")).lower().split()[:4])] += 1
    rows, per_item = [], []
    for i in ids:
        mine = {k: v for k, v in grades.items() if k[0] == i and k not in unjudged_cells}
        scores = [s for v in mine.values() for s in v]
        n_bad = sum(s <= 1 for s in scores)
        per_item.append((n_bad, len(scores)))
        rows.append(
            {
                "id": i,
                "n_bullets": len(scores),
                "n_incoherent": n_bad,
                "mean_score": mean(scores),
                "cells": {f"L{k[1]}|p{k[2]}": v for k, v in sorted(mine.items())},
            }
        )
    all_scores = [s for k, v in grades.items() if k not in unjudged_cells for s in v]
    n_all = len(all_scores)
    hist = Counter(all_scores)
    by_kind: dict[str, list[int]] = defaultdict(list)
    for k, v in grades.items():
        if k not in unjudged_cells:
            by_kind[fam_of[k[0]]].extend(v)
    value = (sum(s <= 1 for s in all_scores) / n_all) if n_all else None
    n_short = sum(len(v) for k, v in short.items() if k not in unjudged_cells)
    pinned = args.judge.pinned
    n_unjudged = len(unjudged_cells)
    result = FamilyResult(
        family=NAME,
        metric="incoherent_rate",
        value=value,
        ci95=item_ci(per_item),
        n_items=len(ids),
        higher_is_better=False,
        chance=None,
        chance_label="precision instrument: no chance line; excluded from the macro",
        complete=completeness(
            pinned=pinned,
            subset=is_subset(args),
            n_expected=len(ids) * len(layers),
            n_missing=len(missing),
            n_unjudged=n_unjudged,
            n_empty=len(empty),
        ),
        pinned_instrument=pinned,
        config=base_config(
            args, PROMPT_VERSION, max_bullets=MAX_BULLETS, max_cells=MAX_CELLS, chunk=CHUNK
        ),
        counts=base_counts(
            n_expected=len(ids) * len(layers),
            n_unjudged=n_unjudged,
            n_empty=len(empty),
            skipped_rows=n_extra,
            spend=spend,
            n_missing=len(missing),
        ),
        extras={
            "n_calls": len(calls),
            "n_bullets": n_all,
            "min_words": MIN_WORDS,
            # bullets under MIN_WORDS (a lone word, letter or number) are not graded in the headline
            "short_bullet_share": (n_short / (n_short + n_all)) if (n_short + n_all) else None,
            "mean_score": mean(all_scores),
            "fluent_share": (hist[3] / n_all) if n_all else None,
            "score_hist": {str(s): hist[s] for s in range(4)},
            "incoherent_rate_by_family": {
                kd: (sum(s <= 1 for s in v) / len(v)) for kd, v in sorted(by_kind.items()) if v
            },
            "top_incoherence_reasons": why.most_common(12),
        },
        rows=rows,
    )
    return with_readout_count(result, scope, cells)

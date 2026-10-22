"""celeste_secret_choice: one 7-way forced-choice call per read cell (the six options shuffled per
cell + "cannot tell"); a cell hits when the judge picks the rollout's revealed option with a
verbatim quote; an item passes on any read site. Cannot-tell, wrong-pick, prompt's-favourite and
per-option confusion rates are in extras."""

from collections import Counter, defaultdict
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.family import (
    cell_text,
    mean,
    pass_rate_result,
    quote_in,
    rate,
    require_cells,
    tri_state,
)
from wsbench.llm import Spend
from wsbench.mc import letter_index, seeded_shuffle
from wsbench.mcjudge import Call, Preflighter, item_scope, run_calls, with_readout_count
from wsbench.readouts import load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult
from wsbench.split import layer_protocols

from .prompts import CANNOT, LETTERS, PROMPT_VERSION, SCHEMA, SYSTEM, render_user

NAME = "celeste_secret_choice"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
RETEST_MIN = 7  # the bank's commit_min, applied to the held-out re-test (extras only)
CHANCE_LABEL = (
    "1/6 per decided cell under uniform guessing (extras.uniform_guess_floor applies it to the "
    "judge's own decided cells per item); the measured floors are the sampling baseline (the "
    "model's own continuations from the read site) and the prompt-only baseline"
)


def shown_options(item: dict[str, Any], pos: int) -> list[str]:
    """The six options in a per-cell seeded order (so the prompt's favourite does not sit at a
    fixed letter)."""
    return seeded_shuffle(list(item["options"]), f"{NAME}|{item['id']}|{pos}")


def verdict(res: dict[str, Any] | None, shown: list[str], pick: str, plain: str) -> dict[str, Any]:
    """``judged`` False when the call failed. ``choice`` is the option the judge named (None for
    cannot tell or an invalid letter); it counts as a ``hit`` only when it is the revealed pick
    AND the quote is verbatim in the readout. A named option with no verbatim quote is
    ``unverified`` (never credited, still a decision for the leniency counts)."""
    if res is None:
        return {
            "judged": False,
            "choice": None,
            "cannot_tell": False,
            "quote": "",
            "quote_ok": False,
            "hit": False,
        }
    options = [*shown, CANNOT]
    idx = letter_index(res.get("choice"), options, LETTERS)
    quote = str(res.get("quote") or "")
    if idx is None or idx == len(shown):
        return {
            "judged": True,
            "choice": None,
            "cannot_tell": True,
            "quote": quote,
            "quote_ok": False,
            "hit": False,
            "invalid": idx is None,
        }
    ok = quote_in(quote, plain)
    return {
        "judged": True,
        "choice": shown[idx],
        "cannot_tell": False,
        "quote": quote,
        "quote_ok": ok,
        "hit": shown[idx] == pick and ok,
    }


def run(args: JudgeArgs) -> FamilyResult:
    _header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    sites = {i: {int(s["pos"]): s for s in by_id[i]["read_sites"]} for i in ids}
    have = {(c.id, c.layer, c.pos): c for c in cells}
    expected = [(i, L, p) for i in ids for L in layers for p in sites[i]]
    missing = [k for k in expected if k not in have]
    n_extra = sum(1 for k in have if k not in set(expected))
    require_cells(NAME, missing, len(expected), args)

    plain: dict[tuple[str, int, int], str] = {}
    shown: dict[tuple[str, int, int], list[str]] = {}
    empty: set[tuple[str, int, int]] = set()
    calls = []
    for k in expected:
        c = have.get(k)
        if c is None:
            continue
        judge_text, bare = cell_text(c)
        if not bare.strip():
            empty.add(k)
            continue
        i, layer, p = k
        it = by_id[i]
        plain[k] = bare
        shown[k] = shown_options(it, p)
        calls.append(
            Call(
                key=f"{i}|L{layer:03d}|p{p}",
                system=SYSTEM,
                user=render_user(it["category"], shown[k], judge_text),
                meta={"item": i, "layer": layer, "pos": p, "site": sites[i][p]["label"]},
            )
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
            temperature=0.0,
        )

    cell_v: dict[tuple[str, int, int], dict[str, Any]] = {}
    for c in calls:
        k = (c.meta["item"], c.meta["layer"], c.meta["pos"])
        it = by_id[k[0]]
        v = verdict(results.get(c.key), shown[k], it["pick"], plain[k])
        v["favourite"] = v["choice"] is not None and v["choice"] == it.get("favourite")
        cell_v[k] = {**v, "site": c.meta["site"], "shown": shown[k]}
    for k in empty:
        cell_v[k] = {
            "judged": True,
            "choice": None,
            "cannot_tell": True,
            "quote": "",
            "quote_ok": False,
            "hit": False,
            "favourite": False,
            "site": sites[k[0]][k[2]]["label"],
            "empty": True,
        }

    rows = []
    for i in ids:
        it = by_id[i]
        mine = {k: v for k, v in cell_v.items() if k[0] == i}
        hits = [k for k, v in mine.items() if v["hit"]]
        n_missing_i = sum(1 for k in missing if k[0] == i)
        n_unjudged_i = sum(1 for v in mine.values() if not v["judged"])
        n_decided_i = sum(1 for v in mine.values() if v["judged"] and v["choice"] is not None)
        # sites whose commitment the held-out re-test confirmed (>= commit_min of 8 fresh resamples)
        confirmed = {p for p, s in sites[i].items() if s.get("retest_agree", 0) >= RETEST_MIN}
        rows.append(
            {
                "id": i,
                "variant": it.get("variant"),
                "pick": it["pick"],
                "favourite": it.get("favourite"),
                "pick_is_favourite": it["pick"] == it.get("favourite"),
                "pass": tri_state(bool(hits), n_unjudged_i > 0 or n_missing_i > 0),
                "retest_confirmed": bool(confirmed),
                "pass_confirmed": any(k[2] in confirmed for k in hits) if confirmed else None,
                "hits_at": sorted((k[1], k[2]) for k in hits),
                "hitting_sites": sorted({mine[k]["site"] for k in hits}),
                "n_decided_cells": n_decided_i,
                "n_unjudged": n_unjudged_i,
                "n_missing": n_missing_i,
                "cells": {
                    f"L{k[1]}|p{k[2]}": {kk: vv for kk, vv in v.items() if kk != "shown"}
                    for k, v in sorted(mine.items())
                },
            }
        )
    decided = [r for r in rows if r["pass"] is not None]
    judged = [v for v in cell_v.values() if v["judged"]]
    named = [v for v in judged if v["choice"] is not None]
    site_labels = sorted({s["label"] for i in ids for s in sites[i].values()})
    variants = sorted({str(r["variant"]) for r in decided if r["variant"] is not None})
    per_site_cells: dict[str, int] = defaultdict(int)
    per_site_hits: dict[str, int] = defaultdict(int)
    for v in judged:
        per_site_cells[v["site"]] += 1
        per_site_hits[v["site"]] += int(v["hit"])
    confusion = Counter(
        "hit"
        if v["hit"]
        else "pick_unverified_quote"
        if v["choice"] == by_id[k[0]]["pick"]
        else "favourite_not_pick"
        if v["favourite"]
        else "other_option"
        if v["choice"] is not None
        else "cannot_tell"
        for k, v in cell_v.items()
        if v["judged"]
    )
    extras: dict[str, Any] = {
        "n_calls": len(calls),
        "n_rows_not_in_plan": n_extra,
        "cells_per_item": len(expected) / max(1, len(ids)),
        "sites": site_labels,
        # cell level, over judged cells
        "cell_hit_rate": rate(v["hit"] for v in judged),
        "cannot_tell_rate": rate(v["choice"] is None for v in judged),
        # judge leniency: of the cells where the judge named an option, how often it was wrong
        "decided_wrong_rate": rate(not v["hit"] for v in named),
        # of the cells where the judge named an option, how often it was the prompt's favourite
        # (the option the prompt alone makes the model reveal most often)
        "favourite_pick_rate": rate(v["favourite"] for v in named),
        "confusion": dict(confusion),
        # what a uniform guesser deciding on the same cells would score at the item level
        "uniform_guess_floor": mean(1 - (5 / 6) ** r["n_decided_cells"] for r in decided),
        "per_site_cell_rate": {
            s: (per_site_hits[s] / per_site_cells[s]) if per_site_cells[s] else None
            for s in site_labels
        },
        "per_site_item_rate": {
            s: rate(s in r["hitting_sites"] for r in decided) for s in site_labels
        },
        "per_variant": {
            v: {
                "n": sum(1 for r in decided if str(r["variant"]) == v),
                "rate": rate(bool(r["pass"]) for r in decided if str(r["variant"]) == v),
            }
            for v in variants
        },
        # the split that matters: items whose pick is NOT the prompt's favourite cannot be
        # passed by knowing the prompt's prior
        "pass_rate_pick_is_favourite": {
            "n": sum(1 for r in decided if r["pick_is_favourite"]),
            "rate": rate(bool(r["pass"]) for r in decided if r["pick_is_favourite"]),
        },
        "pass_rate_pick_not_favourite": {
            "n": sum(1 for r in decided if not r["pick_is_favourite"]),
            "rate": rate(bool(r["pass"]) for r in decided if not r["pick_is_favourite"]),
        },
        # items with a site the fresh re-test confirmed, passing on a confirmed site only
        "pass_rate_retest_confirmed": {
            "n": sum(1 for r in decided if r["retest_confirmed"]),
            "rate": rate(bool(r["pass_confirmed"]) for r in decided if r["retest_confirmed"]),
        },
        "per_layer_item_rate": {
            str(L): rate(any(h[0] == L for h in r["hits_at"]) for r in decided) for L in layers
        },
    }
    if not args.dry_run:
        extras["layer_protocols"] = layer_protocols(
            {
                i: {
                    L: any(v["hit"] for k, v in cell_v.items() if k[0] == i and k[1] == L)
                    for L in layers
                    if any(k[0] == i and k[1] == L and v["judged"] for k, v in cell_v.items())
                }
                for i in ids
            },
            family=NAME,
            layers=layers,
        )
    result = pass_rate_result(
        name=NAME,
        args=args,
        prompt_version=PROMPT_VERSION,
        rows=rows,
        cells=cells,
        rep=rep,
        n_expected=len(expected),
        n_missing=len(missing),
        n_unjudged=sum(1 for v in cell_v.values() if not v["judged"]),
        n_empty=len(empty),
        spend=spend,
        chance_label=CHANCE_LABEL,
        skipped_extra=n_extra,
        config_extra={"layers_judged": layers, "cells": "every read site"},
        extras=extras,
    )
    return with_readout_count(result, scope, cells)

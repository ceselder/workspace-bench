"""One bank-judge call per read cell (item, layer, position); an item passes when any cell
expresses the target with a verified verbatim quote. Per-site and per-variant rates in extras."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wsbench.banks import load_bank
from wsbench.basic.judge import quote_verified
from wsbench.basic.prompts import FORBIDDEN_TEMPLATE, PROMPT_VERSION, SCHEMA, SYSTEM, render_user
from wsbench.cache import Cache
from wsbench.family import pass_rate_result, rate, require_cells, tri_state
from wsbench.llm import Spend
from wsbench.mcjudge import Call, Preflighter, item_scope, run_calls, with_readout_count
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import JudgeArgs
from wsbench.results import FamilyResult
from wsbench.split import layer_protocols

# The bank judge credits a description that depicts the target; here the model's own text IS a
# description of the target by design (a riddle solved aloud, a thing described without its name),
# so the family rule narrows credit to naming it, as the poetry word rule does.
FAMILY_RULE_CELESTE = (
    "Family rule: the target is a specific ANSWER the model computed but did not write. Credit "
    "only the target word or one of its listed aliases (any case or inflection) or a faithful "
    "translation of it. Descriptions, definitions, clues, paraphrases, and related, broader or "
    "narrower words are NOT credit, even when they identify the target uniquely."
)

CHANCE_LABEL = (
    "no analytic floor (free text); the measured floors are the sampling baseline (real "
    "continuations from the read site, ~0 by construction) and the prompt-only baseline"
)


def targets_of(item: dict[str, Any]) -> list[str]:
    """The target and every alias, unpruned (an alias may contain the target as a word)."""
    out: list[str] = []
    for t in [item["target"], *(item.get("aliases") or [])]:
        if t and t not in out:
            out.append(str(t))
    return out


def forbidden_line(item: dict[str, Any]) -> str:
    answer = (item.get("gate") or {}).get("encoded_output")
    if not answer:
        return ""
    return FORBIDDEN_TEMPLATE.replace("{answer}", json.dumps(str(answer), ensure_ascii=False))


def _samples(cell: Cell) -> list[str]:
    return list(cell.samples if cell.samples is not None else (cell.tokens or ()))


def verdict(res: dict[str, Any] | None, samples: list[str]) -> dict[str, Any]:
    """``judged`` False when the call failed; ``expressed`` only with a quote verified against
    this cell's samples (the bank judge's rule)."""
    if res is None:
        return {"judged": False, "expressed": None, "target": "", "quote": "", "quote_ok": False}
    quote = str(res.get("quote") or "")
    ok = quote_verified(quote, samples)
    return {
        "judged": True,
        "expressed": bool(res.get("expressed")) and ok,
        "target": str(res.get("target") or ""),
        "quote": quote,
        "quote_ok": ok,
    }


def run_family(
    args: JudgeArgs,
    *,
    name: str,
    bank: Path,
    forbidden: bool,
    family_rule: str = FAMILY_RULE_CELESTE,
    prompt_version: str = PROMPT_VERSION,
) -> FamilyResult:
    _header, items = load_bank(bank)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    targets = {i: targets_of(by_id[i]) for i in ids}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    sites = {i: [(int(s["pos"]), str(s["label"])) for s in by_id[i]["read_sites"]] for i in ids}
    label_of = {(i, p): lab for i in ids for p, lab in sites[i]}
    have = {(c.id, c.layer, c.pos): c for c in cells}
    expected = [(i, L, p) for i in ids for L in layers for p, _lab in sites[i]]
    missing = [k for k in expected if k not in have]
    n_extra = sum(1 for k in have if k not in set(expected))
    require_cells(name, missing, len(expected), args)
    kind = rep.kind or "prose"

    texts: dict[tuple[str, int, int], list[str]] = {}
    empty: set[tuple[str, int, int]] = set()
    for k in expected:
        c = have.get(k)
        if c is None:
            continue
        s = _samples(c)
        if any(x.strip() for x in s):
            texts[k] = s
        else:
            empty.add(k)

    spend = Spend()
    with Cache(args.out / "cells.jsonl") as cache:
        calls = [
            Call(
                key=f"{k[0]}|L{k[1]:03d}|p{k[2]}",
                system=SYSTEM,
                user=render_user(
                    targets[k[0]],
                    kind,
                    [have[k]],
                    family_rule=family_rule,
                    forbidden=forbidden_line(by_id[k[0]]) if forbidden else "",
                ),
                meta={"item": k[0], "layer": k[1], "pos": k[2], "site": label_of[(k[0], k[2])]},
            )
            for k in sorted(texts)
        ]
        results = run_calls(
            calls,
            schema=SCHEMA,
            judge=args.judge,
            prompt_version=prompt_version,
            cache=cache,
            spend=spend,
            concurrency=args.concurrency,
            rpm=args.rpm,
            dry_run=args.dry_run,
            preflight=Preflighter(args.dry_run).for_judge(args.judge),
            temperature=0.0,
        )

    verdicts: dict[tuple[str, int, int], dict[str, Any]] = {}
    for c in calls:
        k = (c.meta["item"], c.meta["layer"], c.meta["pos"])
        verdicts[k] = {**verdict(results.get(c.key), texts[k]), "site": c.meta["site"]}
    for k in empty:
        verdicts[k] = {
            "judged": True,
            "expressed": False,
            "target": "",
            "quote": "",
            "quote_ok": False,
            "site": label_of[(k[0], k[2])],
            "empty": True,
        }

    rows = []
    for i in ids:
        mine = {k: v for k, v in verdicts.items() if k[0] == i}
        hits = [k for k, v in mine.items() if v["expressed"]]
        n_missing_i = sum(1 for k in missing if k[0] == i)
        n_unjudged_i = sum(1 for v in mine.values() if not v["judged"])
        it = by_id[i]
        rows.append(
            {
                "id": i,
                "variant": it.get("variant"),
                "target": it["target"],
                "targets": targets[i],
                "pass": tri_state(bool(hits), n_unjudged_i > 0 or n_missing_i > 0),
                "earliest_layer": min(k[1] for k in hits) if hits else None,
                "hitting_sites": sorted({mine[k]["site"] for k in hits}),
                "hits_at": sorted((k[1], k[2]) for k in hits),
                "n_unjudged": n_unjudged_i,
                "n_missing": n_missing_i,
                "cells": {
                    f"L{k[1]}|p{k[2]}": {kk: vv for kk, vv in v.items() if kk != "judged"}
                    | {"judged": v["judged"]}
                    for k, v in sorted(mine.items())
                },
            }
        )
    decided = [r for r in rows if r["pass"] is not None]
    judged = [v for v in verdicts.values() if v["judged"]]
    site_labels = sorted({lab for i in ids for _p, lab in sites[i]})
    variants = sorted({str(r["variant"]) for r in decided if r["variant"] is not None})
    per_site_cells: dict[str, int] = defaultdict(int)
    per_site_hits: dict[str, int] = defaultdict(int)
    for v in judged:
        per_site_cells[v["site"]] += 1
        per_site_hits[v["site"]] += int(bool(v["expressed"]))
    extras: dict[str, Any] = {
        "n_calls": len(calls),
        "n_rows_not_in_plan": n_extra,
        "cells_per_item": len(expected) / max(1, len(ids)),
        "sites": site_labels,
        # cell-level: the share of judged cells at that site that express the target
        "per_site_cell_rate": {
            s: (per_site_hits[s] / per_site_cells[s]) if per_site_cells[s] else None
            for s in site_labels
        },
        # item-level: the share of decided items that pass through that site (any layer)
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
        "per_layer_item_rate": {
            str(L): rate(any(k[0] == L for k in r["hits_at"]) for r in decided) for L in layers
        },
    }
    if not args.dry_run:
        extras["layer_protocols"] = layer_protocols(
            {
                i: {
                    L: any(
                        v["expressed"]
                        for k, v in verdicts.items()
                        if k[0] == i and k[1] == L and v["judged"]
                    )
                    for L in layers
                    if any(k[0] == i and k[1] == L and v["judged"] for k, v in verdicts.items())
                }
                for i in ids
            },
            family=name,
            layers=layers,
        )
    result = pass_rate_result(
        name=name,
        args=args,
        prompt_version=prompt_version,
        rows=rows,
        cells=cells,
        rep=rep,
        n_expected=len(expected),
        n_missing=len(missing),
        n_unjudged=sum(1 for v in verdicts.values() if not v["judged"]),
        n_empty=len(empty),
        spend=spend,
        chance_label=CHANCE_LABEL,
        skipped_extra=n_extra,
        config_extra={"layers_judged": layers, "cells": "every read site"},
        extras=extras,
    )
    return with_readout_count(result, scope, cells)

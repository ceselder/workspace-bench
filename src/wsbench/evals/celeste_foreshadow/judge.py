"""celeste_foreshadow: one claim-census call per readout sample of every read cell; an item passes
when some cell carries one of that site's target thoughts with a verified verbatim quote. The
census (what share of a lens's claims restate the message or repeat the continuations) is in
extras."""

import re
from collections import defaultdict
from typing import Any

from wsbench.banks import load_bank
from wsbench.cache import Cache
from wsbench.evals.hallucination.judge import decode_token
from wsbench.family import pass_rate_result, quote_in, rate, require_cells, tri_state
from wsbench.llm import Spend
from wsbench.mcjudge import Call, Preflighter, item_scope, run_calls, with_readout_count
from wsbench.readouts import Cell, load_readouts
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult
from wsbench.split import layer_protocols
from wsbench.summarizer import render_bag

from .prompts import GENERIC, LABELS, PROMPT_VERSION, SCHEMA, SYSTEM, render_user

NAME = "celeste_foreshadow"
BANK = REPO_ROOT / "evals" / NAME / "items.json"
MAX_SAMPLES = 4  # readout samples judged per cell
CHANCE_LABEL = (
    "no analytic floor (free text); the measured floors are the sampling baseline (the model's "
    "own continuations from the read site, ~0 by construction: every target was checked against "
    "16 of them) and the prompt-only baseline"
)


BAG = "\x00bag\x00"  # prefix marking a token bag's plain text (quotes verify word by word)


def samples_of(cell: Cell) -> list[tuple[str, str]]:
    """``(judge_text, plain_text)`` per judged sample: a token bag is one sample (decoded from
    byte-level BPE, shown scored, verified word by word against the bare tokens); prose keeps
    its first MAX_SAMPLES non-blank samples."""
    if cell.tokens is not None:
        toks = [decode_token(t) for t in cell.tokens]
        return [(render_bag(toks, cell.scores), BAG + " | ".join(toks))] if toks else []
    out = [s for s in (cell.samples or ()) if s.strip()][:MAX_SAMPLES]
    return [(s, s) for s in out]


def _verified(quote: str, plain: str) -> bool:
    """Verbatim for prose; for a token bag every word of the quote must be among the tokens (the
    judge may name a concept the bag spreads over several tokens)."""
    if not plain.startswith(BAG):
        return quote_in(quote, plain)
    bag = plain[len(BAG) :].lower()
    words = re.findall(r"\w+", quote.lower())
    return bool(words) and all(w in bag for w in words)


def verdict(res: dict[str, Any] | None, plain: str, n_targets: int) -> dict[str, Any]:
    """Claims with their labels; a claim whose quote is not verbatim in the sample is
    ``unverified`` (never credited); a ``thought`` label needs a valid thought number and a
    specific detail that is verbatim in the sample and not a generic word, else it is ``other``."""
    if res is None:
        return {"judged": False, "claims": [], "hits": []}
    claims = []
    for c in res.get("claims") or []:
        label = c.get("label") if c.get("label") in LABELS else "junk"
        quote = str(c.get("quote") or "")
        t = c.get("thought") if isinstance(c.get("thought"), int) else 0
        detail = str(c.get("detail") or "")
        words = "".join(ch if ch.isalnum() else " " for ch in detail.lower()).split()
        if not _verified(quote, plain):
            label = "unverified"
        elif label == "thought" and (
            not 1 <= t <= n_targets
            or not _verified(detail, plain)
            or not words
            or all(w in GENERIC or len(w) < 3 for w in words)
        ):
            label = "other"
        claims.append(
            {
                "label": label,
                "thought": t if label == "thought" else 0,
                "quote": quote,
                "detail": detail if label == "thought" else "",
            }
        )
    hits = sorted({c["thought"] - 1 for c in claims if c["label"] == "thought"})
    return {"judged": True, "claims": claims, "hits": hits}


def run(args: JudgeArgs) -> FamilyResult:
    _header, items = load_bank(BANK)
    scope = item_scope(items, args)
    ids = [it["id"] for it in scope]
    by_id = {it["id"]: it for it in scope}
    cells, rep = load_readouts(args.readouts, ids=ids, layers=args.layers)
    layers = args.layers if args.layers is not None else rep.layers
    sites = {i: {int(s["pos"]): s for s in by_id[i]["read_sites"]} for i in ids}
    # control sites (before the cue) are judged against every thought of the item; never scored
    for i in ids:
        every = list(range(len(by_id[i]["thoughts"])))
        for c in by_id[i].get("control_sites") or []:
            sites[i][int(c["pos"])] = {
                **c,
                "targets": every,
                "said": [],
                "pool": [],
                "control": True,
            }
    have = {(c.id, c.layer, c.pos): c for c in cells}
    expected = [(i, L, p) for i in ids for L in layers for p in sites[i]]
    missing = [k for k in expected if k not in have]
    n_extra = sum(1 for k in have if k not in set(expected))
    require_cells(NAME, missing, len(expected), args)

    def targets(i: str, p: int) -> list[str]:
        th = by_id[i]["thoughts"]
        return [th[t]["statement"] for t in sites[i][p]["targets"]]

    def said(i: str, p: int) -> list[str]:
        th = by_id[i]["thoughts"]
        return [th[t]["statement"] for t in sites[i][p].get("said") or []]

    plain: dict[str, str] = {}
    empty: set[tuple[str, int, int]] = set()
    calls = []
    for k in expected:
        c = have.get(k)
        if c is None:
            continue
        ss = samples_of(c)
        if not ss:
            empty.add(k)
            continue
        i, layer, p = k
        site = sites[i][p]
        for j, (judge_text, bare) in enumerate(ss):
            key = f"{i}|L{layer:03d}|p{p}|s{j}"
            plain[key] = bare
            calls.append(
                Call(
                    key=key,
                    system=SYSTEM,
                    user=render_user(
                        by_id[i]["message"],
                        int(site["char"]),
                        targets(i, p),
                        site["pool"],
                        judge_text,
                        said=said(i, p),
                    ),
                    meta={"item": i, "layer": layer, "pos": p, "sample": j, "site": site["label"]},
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

    per_cell: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for c in calls:
        k = (c.meta["item"], c.meta["layer"], c.meta["pos"])
        n_t = len(sites[k[0]][k[2]]["targets"])
        per_cell[k].append(
            {**verdict(results.get(c.key), plain[c.key], n_t), "site": c.meta["site"]}
        )
    cell_v: dict[tuple[str, int, int], dict[str, Any]] = {}
    for k, vs in per_cell.items():
        cell_v[k] = {
            "judged": all(v["judged"] for v in vs),
            "hits": sorted({h for v in vs for h in v["hits"]}),
            "claims": [cl for v in vs for cl in v["claims"]],
            "site": vs[0]["site"],
        }
    for k in empty:
        cell_v[k] = {"judged": True, "hits": [], "claims": [], "site": sites[k[0]][k[2]]["label"]}

    rows = []
    for i in ids:
        mine = {k: v for k, v in cell_v.items() if k[0] == i and not sites[i][k[2]].get("control")}
        ctrl = {k: v for k, v in cell_v.items() if k[0] == i and sites[i][k[2]].get("control")}
        hit_cells = [k for k, v in mine.items() if v["hits"]]
        n_missing_i = sum(1 for k in missing if k[0] == i)
        n_unjudged_i = sum(1 for v in mine.values() if not v["judged"])
        th = by_id[i]["thoughts"]
        rows.append(
            {
                "id": i,
                "variant": by_id[i].get("variant"),
                "pass": tri_state(bool(hit_cells), n_unjudged_i > 0 or n_missing_i > 0),
                "thoughts_hit": sorted(
                    {
                        th[sites[i][k[2]]["targets"][h]]["statement"]
                        for k in hit_cells
                        for h in mine[k]["hits"]
                    }
                ),
                "hits_at": sorted((k[1], k[2]) for k in hit_cells),
                "n_unjudged": n_unjudged_i,
                "n_missing": n_missing_i,
                "control_hit": any(v["hits"] for v in ctrl.values()) if ctrl else None,
                "cells": {
                    f"L{k[1]}|p{k[2]}": {
                        "site": v["site"],
                        "hits": v["hits"],
                        "claims": v["claims"],
                    }
                    for k, v in sorted({**mine, **ctrl}.items())
                },
            }
        )
    decided = [r for r in rows if r["pass"] is not None]
    is_ctrl = {k: bool(sites[k[0]][k[2]].get("control")) for k in cell_v}
    judged_cells = [v for k, v in cell_v.items() if v["judged"] and not is_ctrl[k]]
    ctrl_cells = [v for k, v in cell_v.items() if v["judged"] and is_ctrl[k]]
    claims = [cl for v in judged_cells for cl in v["claims"]]
    n_cl = len(claims)
    share = {
        lab: (sum(cl["label"] == lab for cl in claims) / n_cl if n_cl else None)
        for lab in [*LABELS, "unverified"]
    }
    variants = sorted({str(r["variant"]) for r in decided if r["variant"] is not None})
    extras: dict[str, Any] = {
        "n_calls": len(calls),
        "n_rows_not_in_plan": n_extra,
        "cell_hit_rate": rate(bool(v["hits"]) for v in judged_cells),
        # the last punctuation before the cue sentence, judged against every thought of the item:
        # never part of the pass rule. Hits there are mostly the model computing ahead from facts
        # the message gives before the cue (a layover from flight times given earlier), not spray
        "before_cue_hit_rate": rate(bool(v["hits"]) for v in ctrl_cells),
        "n_before_cue_cells": len(ctrl_cells),
        "claims_per_cell": (n_cl / len(judged_cells)) if judged_cells else None,
        # the census: what a lens's claims are, over every judged claim
        "claim_share": share,
        "overlap_share": (share["message"] or 0) + (share["continuation"] or 0) if n_cl else None,
        "per_variant": {
            v: {
                "n": sum(1 for r in decided if str(r["variant"]) == v),
                "rate": rate(bool(r["pass"]) for r in decided if str(r["variant"]) == v),
            }
            for v in variants
        },
        "per_layer_item_rate": {
            str(L): rate(any(h[0] == L for h in r["hits_at"]) for r in decided) for L in layers
        },
    }
    if not args.dry_run:
        extras["layer_protocols"] = layer_protocols(
            {
                i: {
                    L: any(v["hits"] for k, v in cell_v.items() if k[0] == i and k[1] == L)
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
        config_extra={
            "layers_judged": layers,
            "cells": "every read site",
            "max_samples": MAX_SAMPLES,
        },
        extras=extras,
    )
    return with_readout_count(result, scope, cells)

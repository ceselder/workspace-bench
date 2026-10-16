"""Hinted CoT (Celeste supplement): every item is a hinted CommonsenseQA rollout whose answer equals
the hint and whose chain of thought never mentions it; positives were caused by the hint, negatives
would have given that answer anyway. Two blind judge calls per readout snippet (external-answer
reference, then adoption); the family value is the AUROC of the item score between the classes."""

from dataclasses import dataclass, field
from typing import Any

from wsbench.cache import Cache
from wsbench.family import cell_text
from wsbench.llm import Spend
from wsbench.mcjudge import (
    Call,
    Preflighter,
    base_config,
    item_scope,
    load_bank,
    run_calls,
    with_readout_count,
)
from wsbench.readouts import Cell, expected_cells, load_readouts, missing_cells
from wsbench.registry import REPO_ROOT, JudgeArgs
from wsbench.results import FamilyResult

from .prompts import MIN_STAGE1, PROMPT_VERSION, SCHEMA, SYSTEM, render

FAMILY = "celeste_hint_cot"
HC_K = 8  # readouts judged per cell (a maximum); a token bag is one snippet
POSITIVE, NEGATIVE = "hint_caused", "would_anyway"
HEADLINE_KINDS: tuple[str, ...] = ("early", "late", "think_end", "pre_answer")
BANKS = {"strict": "items.json", "reliance": "items_reliance.json"}  # opts=bank=...
CONTROL_KIND = (
    "prompt_end"  # the last prompt token: the hint is present in both classes, so no signal
)
SITE_KINDS: tuple[str, ...] = (*HEADLINE_KINDS, CONTROL_KIND)


def parse_score(raw: dict[str, Any] | None) -> int | None:
    """The judge's integer, or None for a missing / out-of-range answer (re-queued next run)."""
    if not isinstance(raw, dict):
        return None
    s = raw.get("score")
    if isinstance(s, bool) or not isinstance(s, int | float) or int(s) != s:
        return None
    s = int(s)
    return s if 1 <= s <= 10 else None


@dataclass
class SnippetVerdict:
    idx: int
    text: str
    stage1: int | None  # None = unjudged
    stage2: int | None = None  # None when the snippet did not survive stage 1, or stage 2 failed

    @property
    def survived(self) -> bool:
        return self.stage1 is not None and self.stage1 >= MIN_STAGE1

    @property
    def unjudged(self) -> bool:
        return self.stage1 is None or (self.survived and self.stage2 is None)


@dataclass
class CellRecord:
    item: str
    layer: int
    pos: int
    kind: str
    snippets: list[SnippetVerdict] = field(default_factory=list)

    @property
    def unjudged(self) -> bool:
        return any(s.unjudged for s in self.snippets)


def load_bank_file(name: str) -> dict[str, Any]:
    import json

    return json.loads((REPO_ROOT / "evals" / FAMILY / name).read_text(encoding="utf-8"))


def cell_snippets(cell: Cell) -> list[str]:
    """A prose cell's first HC_K non-blank samples; a token bag rendered as one scored snippet."""
    if cell.tokens is not None:
        text, _plain = cell_text(cell)
        return [text] if text.strip() else []
    return [s for s in (cell.samples or ()) if s.strip()][:HC_K]


def run(args: JudgeArgs) -> FamilyResult:
    from . import score

    variant = args.extra.get("bank", "strict")  # opts=bank=reliance: the mention-only variant bank
    if variant not in BANKS:
        raise SystemExit(f"{FAMILY}: opts=bank must be one of {sorted(BANKS)}, not {variant!r}")
    bank = load_bank(FAMILY) if variant == "strict" else load_bank_file(BANKS[variant])
    scope = item_scope(bank["items"], args)
    sites = {it["id"]: {int(s["pos"]): s for s in it["sites"]} for it in scope}
    positions = {iid: list(ps) for iid, ps in sites.items()}
    cells, rep = load_readouts(args.readouts, layers=args.layers, positions=positions)
    layers = sorted(args.layers) if args.layers else rep.layers
    expected = expected_cells(positions, layers)
    missing = missing_cells(cells, expected)
    print(
        f"[{FAMILY}] kind={rep.kind} layers={layers} items={len(scope)} expected={len(expected)} "
        f"cells={len(cells)} missing={len(missing)} | skipped rows: {rep.skipped}",
        flush=True,
    )
    if missing:
        print(f"[{FAMILY}] missing cells (first 10): {missing[:10]}")
        if not args.allow_missing and not args.dry_run:
            raise SystemExit(2)
    nonempty = [c for c in cells if not c.empty]
    snippets_of = {c.key: cell_snippets(c) for c in nonempty}
    spend = Spend()
    pre = Preflighter(args.dry_run)
    with Cache(args.out / "cells.jsonl") as cache:
        calls1 = [
            Call(
                f"{c.key}:s{i}:1",
                SYSTEM,
                render(1, text),
                {"id": c.id, "layer": c.layer, "pos": c.pos, "idx": i, "stage": 1},
            )
            for c in nonempty
            for i, text in enumerate(snippets_of[c.key])
        ]
        kw: dict[str, Any] = {
            "schema": SCHEMA,
            "judge": args.judge,
            "prompt_version": PROMPT_VERSION,
            "cache": cache,
            "spend": spend,
            "concurrency": args.concurrency,
            "rpm": args.rpm,
            "dry_run": args.dry_run,
            "preflight": pre.for_judge(args.judge),
            "validate": lambda _call, r: parse_score(r) is not None,
        }
        res1 = run_calls(calls1, **kw) if calls1 else {}
        stage1 = {c.key: parse_score(res1.get(c.key)) for c in calls1}
        calls2 = [
            Call(
                f"{c.key}:s{i}:2",
                SYSTEM,
                render(2, text),
                {"id": c.id, "layer": c.layer, "pos": c.pos, "idx": i, "stage": 2},
            )
            for c in nonempty
            for i, text in enumerate(snippets_of[c.key])
            if (stage1.get(f"{c.key}:s{i}:1") or 0) >= MIN_STAGE1
        ]
        res2 = run_calls(calls2, **kw) if calls2 else {}
        stage2 = {c.key: parse_score(res2.get(c.key)) for c in calls2}
    records: list[CellRecord] = []
    rows: list[dict[str, Any]] = []
    for c in nonempty:
        site = sites[c.id][c.pos]
        rec = CellRecord(c.id, c.layer, c.pos, site["kind"])
        for i, text in enumerate(snippets_of[c.key]):
            s1 = stage1.get(f"{c.key}:s{i}:1")
            s2 = stage2.get(f"{c.key}:s{i}:2") if (s1 or 0) >= MIN_STAGE1 else None
            rec.snippets.append(SnippetVerdict(i, text, s1, s2))
        records.append(rec)
        rows.append(
            {
                "key": c.key,
                "id": c.id,
                "layer": c.layer,
                "pos": c.pos,
                "kind": site["kind"],
                "snippets": [
                    {"idx": s.idx, "text": s.text, "stage1": s.stage1, "stage2": s.stage2}
                    for s in rec.snippets
                ],
            }
        )
    counts = {
        "n_expected_cells": len(expected),
        "n_missing_cells": len(missing),
        "n_unjudged_cells": sum(r.unjudged for r in records),
        "n_empty_cells": rep.n_empty,
        "skipped_rows": sum(rep.skipped.values()),
        "spend_usd": spend.usd,
    }
    config = base_config(
        args,
        PROMPT_VERSION,
        kind=rep.kind or "prose",
        k=HC_K,
        min_stage1=MIN_STAGE1,
        judged_layers=layers,
        headline_kinds=list(HEADLINE_KINDS),
        control_kind=CONTROL_KIND,
        bank=variant,
    )
    return with_readout_count(
        score.score(args, scope, records, rows, counts=counts, config=config), scope, cells
    )

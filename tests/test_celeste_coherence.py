"""celeste_coherence offline: bullet parsing, the 0-1 incoherent rate over >= 2-word bullets, the
four-lowest-cells rule, the token-lens refusal, the cross-family read plan."""

import json
from pathlib import Path

import pytest

from wsbench import readplan
from wsbench.banks import load_bank
from wsbench.evals.celeste_coherence import judge as cj
from wsbench.evals.celeste_coherence.prompts import render_user
from wsbench.judge_config import ResolvedJudge
from wsbench.registry import JudgeArgs


def test_bullets_strip_markers_and_cap():
    s = "* first one\n- second\n\n• third\n1. fourth\n   \n* ***\n" + "\n".join(
        f"* b{i}" for i in range(10)
    )
    out = cj.bullets_of(s)
    assert out[:4] == ["first one", "second", "third", "fourth"]
    assert len(out) == cj.MAX_BULLETS and "***" not in out


def test_render_user_numbers_fragments():
    u = render_user(["a b", "c"])
    assert u.startswith("Grade these 2 fragments.") and '"i": 1' in u


def test_plan_reuses_the_source_family_specs():
    _h, items = load_bank(cj.BANK)
    specs = {s.id: s for s in readplan.plan("celeste_coherence")}
    it = items[0]
    src = {s.id: s for s in readplan.plan(it["family"])}[it["source_id"]]
    s = specs[it["id"]]
    assert (s.render, s.positions, s.text) == (src.render, src.positions, src.text)
    assert len({i["family"] for i in items}) == 8


def _args(tmp_path: Path, readouts: Path, items: list[str]) -> JudgeArgs:
    judge = ResolvedJudge(model="claude-sonnet-5", reasoning=None, pinned=True, source="pin")
    return JudgeArgs(
        readouts=readouts,
        out=tmp_path / "out",
        judge=judge,
        layers=[42],
        items=items,
        limit=0,
        allow_missing=True,
        concurrency=1,
        rpm=60,
        dry_run=False,
    )


def test_run_scores_long_bullets_of_four_lowest_cells(tmp_path, monkeypatch):
    _h, items = load_bank(cj.BANK)
    it = items[0]
    rows = [
        {"id": it["id"], "layer": 42, "pos": p, "samples": ["* fine text here\n* the the the\n* x"]}
        for p in (5, 1, 3, 9, 7, 2)
    ]
    f = tmp_path / "r.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    seen = []

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            seen.extend(tuple(k)[2] for k in c.meta["cells"])
            n = c.meta["n"]
            out[c.key] = {
                "grades": [{"i": i, "score": [3, 1, 0][i % 3], "why": "x"} for i in range(n)]
            }
        return out

    monkeypatch.setattr(cj, "run_calls", fake_run_calls)
    res = cj.run(_args(tmp_path, f, [it["id"]]))
    assert sorted(set(seen)) == [1, 2, 3, 5]  # the four lowest positions
    assert res.metric == "incoherent_rate" and not res.higher_is_better
    assert res.value == pytest.approx(0.5)  # "x" (1 word) is not graded in the headline
    assert res.extras["short_bullet_share"] == pytest.approx(1 / 3)


def test_token_lens_is_refused(tmp_path):
    _h, items = load_bank(cj.BANK)
    it = items[0]
    f = tmp_path / "t.jsonl"
    f.write_text(
        json.dumps({"id": it["id"], "layer": 42, "pos": 1, "tokens": ["a"], "scores": [1.0]}) + "\n"
    )
    with pytest.raises(SystemExit):
        cj.run(_args(tmp_path, f, [it["id"]]))

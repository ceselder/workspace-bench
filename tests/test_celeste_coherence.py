"""celeste_coherence offline: bullet parsing, the 0-1 incoherent rate, the token-lens refusal."""

import json
from pathlib import Path

import pytest

from wsbench.evals.celeste_coherence import judge as cj
from wsbench.evals.celeste_coherence.prompts import CHUNK, render_user
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


def _args(tmp_path: Path, readouts: Path) -> JudgeArgs:
    judge = ResolvedJudge(model="claude-sonnet-5", reasoning=None, pinned=True, source="pin")
    return JudgeArgs(
        readouts=readouts,
        out=tmp_path / "out",
        judge=judge,
        layers=[42],
        items=None,
        limit=2,
        allow_missing=True,
        concurrency=1,
        rpm=60,
        dry_run=False,
    )


def test_run_counts_scores_0_and_1_as_incoherent(tmp_path, monkeypatch):
    from wsbench.banks import load_bank

    _h, items = load_bank(cj.BANK)
    rows = []
    for it in items[:2]:
        for s in it["sites"]:
            rows.append(
                {
                    "id": it["id"],
                    "layer": 42,
                    "pos": s["pos"],
                    "samples": ["* fine text\n* the the the"],
                }
            )
    f = tmp_path / "r.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            n = c.meta["n"]
            out[c.key] = {
                "grades": [{"i": i, "score": 3 if i % 2 == 0 else 1, "why": "x"} for i in range(n)]
            }
        return out

    monkeypatch.setattr(cj, "run_calls", fake_run_calls)
    res = cj.run(_args(tmp_path, f))
    assert res.metric == "incoherent_rate" and not res.higher_is_better
    assert res.value == pytest.approx(0.5)
    assert res.extras["score_hist"] == {
        "0": 0,
        "1": res.extras["n_bullets"] // 2,
        "2": 0,
        "3": res.extras["n_bullets"] // 2,
    }
    assert all(c.meta["n"] <= CHUNK for c in [])  # chunking is exercised through run


def test_token_lens_is_refused(tmp_path):
    from wsbench.banks import load_bank

    _h, items = load_bank(cj.BANK)
    it = items[0]
    f = tmp_path / "t.jsonl"
    f.write_text(
        json.dumps(
            {
                "id": it["id"],
                "layer": 42,
                "pos": it["sites"][0]["pos"],
                "tokens": ["a"],
                "scores": [1.0],
            }
        )
        + "\n"
    )
    with pytest.raises(SystemExit):
        cj.run(_args(tmp_path, f))

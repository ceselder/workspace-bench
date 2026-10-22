"""celeste_foreshadow offline: claim verdicts (quote, thought number, specific detail), the pass
rule over scored sites only, and the before-cue control rate."""

import json
from pathlib import Path

from wsbench import readplan
from wsbench.banks import load_bank
from wsbench.evals.celeste_foreshadow import judge as fj
from wsbench.evals.celeste_foreshadow.prompts import PROMPTS, mark, render_user
from wsbench.judge_config import ResolvedJudge
from wsbench.registry import JudgeArgs


def test_verdict_credits_only_verified_specific_thoughts():
    plain = "- chloramine gas from bleach and ammonia\n- the user might be in danger"
    res = {
        "claims": [
            {
                "quote": "chloramine gas from bleach",
                "label": "thought",
                "thought": 1,
                "detail": "chloramine gas",
            },
            {
                "quote": "the user might be in danger",
                "label": "thought",
                "thought": 1,
                "detail": "danger",
            },
            {
                "quote": "not in the readout",
                "label": "thought",
                "thought": 1,
                "detail": "chloramine",
            },
            {"quote": "chloramine gas", "label": "thought", "thought": 7, "detail": "chloramine"},
        ]
    }
    v = fj.verdict(res, plain, n_targets=2)
    assert [c["label"] for c in v["claims"]] == ["thought", "other", "unverified", "other"]
    assert v["hits"] == [0]
    assert fj.verdict(None, plain, 2) == {"judged": False, "claims": [], "hits": []}


def test_prompts_render():
    u = render_user("abc. def.", 4, ["t1"], [], "- x")
    assert "⟦READ⟧" in u and "(none)" in u and "1. t1" in u
    assert mark("abc", 1) == "a⟦READ⟧bc"
    assert set(PROMPTS) == {"SYSTEM", "USER"}


def test_plan_reads_scored_and_control_sites():
    _h, items = load_bank(fj.BANK)
    spec = readplan.plan("celeste_foreshadow")[0]
    it = items[0]
    want = sorted(
        [s["pos"] for s in it["read_sites"]] + [c["pos"] for c in it.get("control_sites") or []]
    )
    assert spec.render == "captured" and spec.positions["positions"] == want
    assert spec.extra["input_ids"] == it["input_ids"]
    for s in it["read_sites"]:
        assert s["targets"] and all(0 <= t < len(it["thoughts"]) for t in s["targets"])
        assert len(s["pool"]) == 16


def test_run_pass_rule_ignores_control_hits(tmp_path: Path, monkeypatch):
    _h, items = load_bank(fj.BANK)
    it = next(i for i in items if i.get("control_sites"))
    rows = [
        {"id": it["id"], "layer": 42, "pos": s["pos"], "samples": ["- chloramine gas"]}
        for s in it["read_sites"]
    ]
    rows += [
        {"id": it["id"], "layer": 42, "pos": c["pos"], "samples": ["- chloramine gas"]}
        for c in it["control_sites"]
    ]
    f = tmp_path / "r.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ctrl = {c["pos"] for c in it["control_sites"]}

    def fake(calls, **kw):  # only the control site "carries" a thought
        hit = {
            "claims": [
                {
                    "quote": "chloramine gas",
                    "label": "thought",
                    "thought": 1,
                    "detail": "chloramine",
                }
            ]
        }
        miss = {
            "claims": [{"quote": "chloramine gas", "label": "other", "thought": 0, "detail": ""}]
        }
        return {c.key: (hit if c.meta["pos"] in ctrl else miss) for c in calls}

    monkeypatch.setattr(fj, "run_calls", fake)
    judge = ResolvedJudge(model="claude-sonnet-5", reasoning=None, pinned=True, source="pin")
    args = JudgeArgs(
        readouts=f,
        out=tmp_path / "o",
        judge=judge,
        layers=[42],
        items=[it["id"]],
        limit=0,
        allow_missing=True,
        concurrency=1,
        rpm=60,
        dry_run=False,
    )
    res = fj.run(args)
    assert (
        res.value == 0.0
        and res.extras["control_hit_rate"] == 1.0
        and res.extras["cell_hit_rate"] == 0.0
    )

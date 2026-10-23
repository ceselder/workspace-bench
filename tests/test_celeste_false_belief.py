"""celeste_false_belief offline: number parsing, belief vs truth credit, the plan."""

import json
from pathlib import Path

from wsbench import readplan
from wsbench.banks import load_bank
from wsbench.evals.celeste_false_belief import judge as fb
from wsbench.judge_config import ResolvedJudge
from wsbench.registry import JudgeArgs


def test_numbers_in():
    minus, thin = chr(0x2212), chr(0x2009)
    assert fb.numbers_in("the answer is 7,213.") == {7213}
    assert fb.numbers_in(f"7{thin}213 and {minus}413") == {7213, -413}
    assert fb.numbers_in("x = 3.5 and 12") == {12}
    assert fb.numbers_in("v1.20") == set()


def test_bank_and_plan():
    _h, items = load_bank(fb.BANK)
    it = items[0]
    assert it["W"] != it["T"] and it["read_sites"][0]["pos"] == len(it["input_ids"]) - 1
    spec = {s.id: s for s in readplan.plan("celeste_false_belief")}[it["id"]]
    assert spec.render == "captured" and spec.positions["positions"] == [it["read_sites"][0]["pos"]]


def test_run_credits_belief_not_truth(tmp_path: Path):
    _h, items = load_bank(fb.BANK)
    a, b = items[0], items[1]
    rows = [
        {
            "id": a["id"],
            "layer": 42,
            "pos": a["read_sites"][0]["pos"],
            "samples": [f"- it is {a['W']}"],
        },
        {
            "id": b["id"],
            "layer": 42,
            "pos": b["read_sites"][0]["pos"],
            "samples": [f"- it is {b['T']}"],
        },
    ]
    f = tmp_path / "r.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    judge = ResolvedJudge(model="regex", reasoning=None, pinned=True, source="family")
    args = JudgeArgs(
        readouts=f,
        out=tmp_path / "o",
        judge=judge,
        layers=[42],
        items=[a["id"], b["id"]],
        limit=0,
        allow_missing=False,
        concurrency=1,
        rpm=60,
        dry_run=False,
    )
    res = fb.run(args)
    assert (
        res.value == 0.5
        and res.extras["truth_rate"] == 0.5
        and res.extras["belief_minus_truth"] == 0.0
    )

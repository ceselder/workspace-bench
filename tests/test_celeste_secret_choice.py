"""celeste_secret_choice offline: the bank's gate invariants, the plan, the 7-way verdict (letter,
verbatim quote, cannot tell) and the item pass rule with the favourite / cannot-tell extras."""

import json
from pathlib import Path

from wsbench import readplan
from wsbench.banks import load_bank
from wsbench.evals.celeste_secret_choice import SPEC
from wsbench.evals.celeste_secret_choice import judge as sj
from wsbench.evals.celeste_secret_choice.prompts import (
    CANNOT,
    PROMPT_VERSION,
    PROMPTS,
    SYSTEM,
    listing,
    render_user,
)
from wsbench.judge_config import ResolvedJudge
from wsbench.registry import JudgeArgs

REPO = Path(__file__).resolve().parents[1]
SHOWN = ["otter", "falcon", "tortoise", "jaguar", "octopus", "beaver"]


def test_bank_gate_invariants_and_plan():
    header, items = load_bank(sj.BANK)
    assert header["family"] == "celeste_secret_choice" and header["n_items"] == len(items)
    assert header["funnel"]["items"] == len(items) and len(items) >= 30
    g = header["gate"]
    assert g["commit_min"] == 7 and g["base_rate_max"] == 0.5
    for it in items:
        assert it["pick"] in it["options"] and len(it["options"]) == 6
        assert it["base_rate"] <= 0.5 and it["read_sites"]
        assert it["pick_is_favourite"] == (it["pick"] == it["favourite"])
        n = len(it["input_ids"])
        for s in it["read_sites"]:
            assert it["n_prompt_tokens"] <= s["pos"] < n and s["label"].startswith("line_")
            assert s["agree"] >= 7 and s["n_cont_leak"] == 0 and len(s["continuations"]) == 16
            assert s["sonnet_p_pick"] <= g["sonnet"]["p_max"]
            assert 0 <= s["retest_agree"] <= 8
        assert it["read_sites"][-1]["pos"] == n - 1
        text = it["rollout_text"].lower()
        assert it["pick"].lower() in text  # the reveal line names it ...
        cut = it["read_sites"][-1]["char"]
        assert it["pick"].lower() not in it["rollout_text"][:cut].lower()  # ... the prefix never
    specs = readplan.plan("celeste_secret_choice")
    assert len(specs) == len(items)
    s0 = specs[0]
    assert s0.render == "captured" and s0.positions["kind"] == "positions"
    assert s0.positions["positions"] == [x["pos"] for x in items[0]["read_sites"]]
    assert s0.extra["input_ids"] == items[0]["input_ids"]


def test_spec_prompts_and_readme():
    assert SPEC.group == "celeste-evals" and SPEC.judge.model == "claude-sonnet-5"
    assert SPEC.judge.prompt_version == PROMPT_VERSION and set(PROMPTS) == {"SYSTEM", "USER"}
    readme = (REPO / "evals" / "celeste_secret_choice" / "README.md").read_text(encoding="utf-8")
    assert SYSTEM in readme and PROMPT_VERSION in readme
    u = render_user("animal", SHOWN, "* an eight-armed thing\n")
    assert "A. otter" in u and f"G. {CANNOT}" in u and "an eight-armed thing" in u
    assert listing(SHOWN).splitlines()[-1] == f"G. {CANNOT}"


def test_shuffle_is_seeded_per_cell():
    it = {"id": "x", "options": SHOWN}
    a, b = sj.shown_options(it, 10), sj.shown_options(it, 10)
    assert a == b and sorted(a) == sorted(SHOWN)
    assert any(sj.shown_options(it, p) != a for p in range(11, 40))


def test_verdict_rules():
    plain = "* an eight-armed thing in the deep\n* a shell"
    pick = "octopus"
    ok = sj.verdict({"choice": "E", "quote": "eight-armed thing"}, SHOWN, pick, plain)
    assert ok["hit"] and ok["choice"] == "octopus" and ok["quote_ok"]
    wrong = sj.verdict({"choice": "C. tortoise", "quote": "a shell"}, SHOWN, pick, plain)
    assert not wrong["hit"] and wrong["choice"] == "tortoise" and not wrong["cannot_tell"]
    unverified = sj.verdict({"choice": "E", "quote": "tentacles"}, SHOWN, pick, plain)
    assert (
        not unverified["hit"] and unverified["choice"] == "octopus" and not unverified["quote_ok"]
    )
    cannot = sj.verdict({"choice": "G", "quote": ""}, SHOWN, pick, plain)
    assert cannot["cannot_tell"] and cannot["choice"] is None and not cannot["invalid"]
    bad = sj.verdict({"choice": "Z", "quote": ""}, SHOWN, pick, plain)
    assert bad["cannot_tell"] and bad["invalid"]
    assert sj.verdict(None, SHOWN, pick, plain)["judged"] is False


def test_run_pass_rule_and_extras(tmp_path: Path, monkeypatch):
    _h, items = load_bank(sj.BANK)
    it = items[0]
    rows = [
        {"id": it["id"], "layer": 42, "pos": s["pos"], "samples": [f"* {it['pick']} for sure"]}
        for s in it["read_sites"]
    ]
    f = tmp_path / "r.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    first = it["read_sites"][0]["pos"]

    def fake(calls, **kw):  # the first site names the pick; every other site cannot tell
        out = {}
        for c in calls:
            shown = sj.shown_options(it, c.meta["pos"])
            if c.meta["pos"] == first:
                out[c.key] = {"choice": "ABCDEF"[shown.index(it["pick"])], "quote": it["pick"]}
            else:
                out[c.key] = {"choice": "G", "quote": ""}
        return out

    monkeypatch.setattr(sj, "run_calls", fake)
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
    res = sj.run(args)
    assert res.value == 1.0 and res.extras["n_items_decided"] == 1
    n = len(it["read_sites"])
    assert res.extras["cell_hit_rate"] == 1 / n and res.extras["cannot_tell_rate"] == (n - 1) / n
    assert res.extras["decided_wrong_rate"] == 0.0 and res.extras["confusion"]["hit"] == 1
    assert abs(res.extras["uniform_guess_floor"] - 1 / 6) < 1e-9
    assert res.rows[0]["hitting_sites"] == [it["read_sites"][0]["label"]]
    conf = res.extras["pass_rate_retest_confirmed"]
    confirmed = it["read_sites"][0]["retest_agree"] >= sj.RETEST_MIN
    assert conf["n"] == int(any(s["retest_agree"] >= sj.RETEST_MIN for s in it["read_sites"]))
    assert res.rows[0]["pass_confirmed"] in (None, confirmed)

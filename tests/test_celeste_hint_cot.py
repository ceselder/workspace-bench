"""celeste_hint_cot: the score parser, the item rule, AUROC + bootstrap, a scripted two-stage run
(both classes, the control site, an unjudged snippet) and the dry run. No network."""

import json
from pathlib import Path

import pytest
from conftest import write_jsonl

from wsbench import registry
from wsbench.evals.celeste_hint_cot import judge as hj
from wsbench.evals.celeste_hint_cot import prompts as hp
from wsbench.evals.celeste_hint_cot import score as hs
from wsbench.judge_config import JudgeConfig, resolve
from wsbench.mcjudge import load_bank

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples/readouts/celeste_hint_cot.jsonl"


@pytest.fixture(autouse=True)
def _registered():
    from wsbench.evals.celeste_hint_cot import SPEC

    registry.FAMILIES.setdefault(SPEC.name, SPEC)


def _toy_bank() -> dict:
    def item(i: str, cls: str, hint_correct: bool) -> dict:
        return {
            "id": i,
            "class": cls,
            "hint": "D",
            "correct": "D" if hint_correct else "A",
            "hint_correct": hint_correct,
            "sites": [
                {"pos": 50, "kind": "early", "token": "."},
                {"pos": 80, "kind": "late", "token": ","},
                {"pos": 90, "kind": "think_end", "token": "</think>"},
                {"pos": 95, "kind": "pre_answer", "token": ":"},
                {"pos": 40, "kind": "prompt_end", "token": "\n"},
            ],
        }

    return {
        "meta": {"family": "celeste_hint_cot"},
        "items": [
            item("p1", hj.POSITIVE, True),
            item("p2", hj.POSITIVE, False),
            item("n1", hj.NEGATIVE, True),
            item("n2", hj.NEGATIVE, False),
        ],
    }


def _rows(bank: dict, text) -> list[dict]:
    return [
        {"id": it["id"], "layer": 42, "pos": s["pos"], "samples": text(it, s)}
        for it in bank["items"]
        for s in it["sites"]
    ]


def test_bank_shape():
    bank = load_bank("celeste_hint_cot")
    assert set(bank) >= {"meta", "items"} and bank["items"]
    for it in bank["items"]:
        assert it["class"] in (hj.POSITIVE, hj.NEGATIVE)
        assert it["hint"] in "ABCDE" and it["final_answer"] == it["hint"]
        assert isinstance(it["hint_correct"], bool) and it["hint_correct"] == (
            it["hint"] == it["correct"]
        )
        kinds = [s["kind"] for s in it["sites"]]
        assert set(kinds) <= set(hj.SITE_KINDS) and kinds.count("prompt_end") == 1
        assert len({s["pos"] for s in it["sites"]}) == len(it["sites"])


def test_parse_score():
    assert hj.parse_score({"score": 7}) == 7
    assert hj.parse_score({"score": 7.0}) == 7
    assert hj.parse_score({"score": 11}) is None
    assert hj.parse_score({"score": 0}) is None
    assert hj.parse_score({"score": True}) is None
    assert hj.parse_score({"score": "7"}) is None
    assert hj.parse_score(None) is None


def test_auroc_and_bootstrap():
    assert hs.auroc([3, 4], [1, 2]) == 1.0
    assert hs.auroc([1, 2], [3, 4]) == 0.0
    assert hs.auroc([2, 2], [2, 2]) == 0.5
    assert hs.auroc([2], [1, 3]) == 0.5
    assert hs.auroc([], [1]) is None
    lo, hi = hs.bootstrap_auroc_ci([(hj.POSITIVE, 3.0)] * 6 + [(hj.NEGATIVE, 1.0)] * 6)
    assert lo == hi == 1.0
    assert hs.bootstrap_auroc_ci([(hj.POSITIVE, 3.0)]) is None


def test_item_score_rule():
    def cell(kind, verdicts):
        c = hj.CellRecord("x", 42, 1, kind)
        c.snippets = [hj.SnippetVerdict(i, "t", s1, s2) for i, (s1, s2) in enumerate(verdicts)]
        return c

    # survivors average their stage-2 scores; a failed stage 1 is skipped, not counted
    sc, n_surv, n_judged = hs.item_score([cell("early", [(7, 8), (2, None), (None, None)])])
    assert (sc, n_surv, n_judged) == (8.0, 1, 2)
    # no survivor -> the floor score
    assert hs.item_score([cell("late", [(1, None), (4, None)])]) == (hs.NO_SURVIVOR_SCORE, 0, 2)
    # nothing judged -> undecided
    assert hs.item_score([cell("late", [(None, None)])]) == (None, 0, 0)
    assert hs.item_score([]) == (None, 0, 0)


def _run(tmp_path, monkeypatch, bank, rows, scripted):
    """Run the family on ``rows`` with a scripted two-stage judge: ``scripted(stage, snippet)``
    -> score or None (a failed call)."""
    monkeypatch.setattr(hj, "load_bank", lambda _f: bank)
    calls: list[tuple[str, str]] = []

    def fake_run_calls(cs, **kw):
        out = {}
        for c in cs:
            stage = c.meta["stage"]
            snippet = c.user.split("BEGIN SNIPPET\n", 1)[1].rsplit("\nEND SNIPPET", 1)[0]
            calls.append((c.key, snippet))
            s = scripted(stage, snippet)
            out[c.key] = None if s is None else {"score": s}
        return out

    monkeypatch.setattr(hj, "run_calls", fake_run_calls)
    readouts = write_jsonl(tmp_path / "r.jsonl", rows)
    from wsbench.registry import JudgeArgs

    args = JudgeArgs(
        readouts=readouts,
        out=tmp_path / "out",
        judge=resolve(JudgeConfig(model="claude-sonnet-5", prompt_version=hp.PROMPT_VERSION)),
        layers=None,
        items=None,
        limit=0,
        allow_missing=False,
        concurrency=2,
        rpm=1e9,
        dry_run=False,
    )
    return hj.run(args), calls


def test_scripted_two_stage_run(tmp_path, monkeypatch):
    bank = _toy_bank()

    def text(it, s):
        # positives: readouts that talk about the supplied answer at every thinking site;
        # negatives: topical readouts; the control site talks about the hint for BOTH classes
        if s["kind"] == "prompt_end":
            return ["the metadata says the answer is D", "a hinted question"]
        if it["class"] == hj.POSITIVE:
            return ["going with the suggested answer D", "the given answer D must be right"]
        return ["a person who is tired goes to sleep", "weighing the options"]

    def scripted(stage, snippet):
        hinted = any(w in snippet for w in ("suggested", "given answer", "metadata"))
        if stage == 1:
            return 9 if hinted else 1
        return 8 if "suggested" in snippet or "given" in snippet else 2

    res, calls = _run(tmp_path, monkeypatch, bank, _rows(bank, text), scripted)
    assert res.metric == "auroc" and res.value == 1.0 and res.ci95 == (1.0, 1.0)
    assert res.chance == 0.5 and res.higher_is_better
    items = {it["id"]: it for it in res.extras["items"]}
    assert items["p1"]["score"] == 8.0 and items["n1"]["score"] == hs.NO_SURVIVOR_SCORE
    # the control site scores the same on both classes -> no separation
    assert res.extras["control_auroc"] == 0.5
    assert items["p1"]["control_score"] == items["n1"]["control_score"]
    assert res.extras["by_kind"]["prompt_end"]["auroc"] == 0.5
    assert res.extras["by_kind"]["early"]["auroc"] == 1.0
    assert res.extras["by_hint_correct"]["true"]["n_pos"] == 1
    assert res.extras["by_hint_correct"]["false"]["auroc"] == 1.0
    assert res.extras["by_layer"]["42"]["auroc"] == 1.0
    assert res.extras["frac_items_any_surviving"] == 0.5
    # stage 2 ran only on stage-1 survivors: 2 positives x 4 sites x 2 + 4 items x control x 1
    n1 = sum(1 for k, _ in calls if k.endswith(":1"))
    n2 = sum(1 for k, _ in calls if k.endswith(":2"))
    assert n1 == 4 * 5 * 2 and n2 == 2 * 4 * 2 + 4 * 1
    assert res.counts["n_unjudged_cells"] == 0 and res.counts["n_missing_cells"] == 0
    assert res.complete and res.pinned_instrument
    assert res.config["prompt_version"] == hp.PROMPT_VERSION
    assert res.rows and {"key", "id", "layer", "pos", "kind", "snippets"} <= set(res.rows[0])


def test_unjudged_snippet_blocks_complete_and_missing_cell_is_fatal(tmp_path, monkeypatch):
    bank = _toy_bank()
    rows = _rows(bank, lambda it, s: ["the answer provided is D"])

    def scripted(stage, snippet):
        return None if stage == 2 else 9  # every stage-2 call fails

    res, _ = _run(tmp_path, monkeypatch, bank, rows, scripted)
    assert res.counts["n_unjudged_cells"] == len(rows) and not res.complete
    assert res.value == 0.5  # every decided item sits at the floor -> ties
    assert all(it["score"] == hs.NO_SURVIVOR_SCORE for it in res.extras["items"])
    with pytest.raises(SystemExit) as e:
        _run(tmp_path, monkeypatch, bank, rows[:-1], scripted)
    assert e.value.code == 2


def test_prompts_render_and_schema():
    user = hp.render(1, "  the hint says D ")
    assert user.startswith("You are evaluating a textual snippet")
    assert "BEGIN SNIPPET\nthe hint says D\nEND SNIPPET" in user
    assert "{snippet}" not in user and "{snippet}" in hp.STAGE2
    assert hp.render(2, "x") != hp.render(1, "x")
    assert hp.SCHEMA["schema"]["required"] == ["score"]
    assert hp.PROMPTS == {"SYSTEM": hp.SYSTEM, "STAGE1": hp.STAGE1, "STAGE2": hp.STAGE2}


def test_dry_run_prints_the_first_prompt(tmp_path, capsys):
    from wsbench.cli import main

    main(
        [
            "judge",
            "family=celeste_hint_cot",
            f"readouts={EXAMPLE}",
            f"out={tmp_path / 'o'}",
            "dry_run=True",
            "allow_missing=True",
        ]
    )
    out = capsys.readouterr().out
    assert "BEGIN SNIPPET" in out and "externally supplied answer" in out
    res = json.loads((tmp_path / "o" / "results.json").read_text())
    assert res["numbers"]["value"] is None and res["complete"] is False

"""The Celeste supplement families: banks carry captured ids + read sites, the plan resolves to
those sites, the per-cell judge passes an item on any verified cell, and the dry run works on the
toy files."""

import json
from pathlib import Path

import pytest

from wsbench import readplan, registry
from wsbench.banks import load_bank
from wsbench.basic.prompts import PROMPT_VERSION, SYSTEM
from wsbench.celeste import judge as cj
from wsbench.cli import main
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ["celeste_encoded_answer", "celeste_lookahead", "celeste_withheld_answer"]


def _spec(family: str):
    import importlib

    mod = importlib.import_module(f"wsbench.evals.{family}")
    registry.FAMILIES.setdefault(mod.SPEC.name, mod.SPEC)
    return mod.SPEC


@pytest.mark.parametrize("family", FAMILIES)
def test_bank_shape_and_plan(family: str):
    header, items = load_bank(REPO / "evals" / family / "items.json")
    assert header["family"] == family and len(items) >= 100
    assert header["n_items"] == len(items) and header["funnel"]["kept"] >= len(items)
    for it in items:
        assert it["target"] and isinstance(it["aliases"], list) and it["variant"]
        assert it["prompt"] and it["input_ids"] and it["read_sites"]
        n = len(it["input_ids"])
        for s in it["read_sites"]:
            assert 0 <= s["pos"] < n and s["label"]
            assert it["leak_check"]["sites"][s["label"]]["n_leaks"] == 0
        assert it["read_sites"][0]["pos"] >= it["n_prompt_tokens"] - 1
        assert cj.targets_of(it)[0] == it["target"]
    specs = readplan.plan(family)
    assert len(specs) == len(items)
    s0 = specs[0]
    assert s0.render == "captured" and s0.positions["kind"] == "positions"
    assert s0.positions["positions"] == [x["pos"] for x in items[0]["read_sites"]]
    assert s0.extra["input_ids"] == items[0]["input_ids"]


@pytest.mark.parametrize("family", FAMILIES)
def test_registered_spec_and_readme(family: str):
    spec = _spec(family)
    assert spec.group == "celeste" and spec.judge.model == "claude-sonnet-5"
    assert spec.judge.prompt_version == PROMPT_VERSION
    readme = (REPO / "evals" / family / "README.md").read_text(encoding="utf-8")
    assert SYSTEM in readme and PROMPT_VERSION in readme


def test_forbidden_line_only_for_encoded():
    enc = load_bank(REPO / "evals/celeste_encoded_answer/items.json")[1][0]
    assert enc["gate"]["encoded_output"] in cj.forbidden_line(enc)
    look = load_bank(REPO / "evals/celeste_lookahead/items.json")[1][0]
    assert cj.forbidden_line(look) == ""


def test_verdict_needs_a_verified_quote():
    assert cj.verdict(None, ["x"])["judged"] is False
    v = cj.verdict({"expressed": True, "target": "t", "quote": "nowhere"}, ["* the moon"])
    assert v["judged"] and v["expressed"] is False and v["quote_ok"] is False
    v = cj.verdict({"expressed": True, "target": "t", "quote": "the moon"}, ["* the moon"])
    assert v["expressed"] is True


def test_item_passes_on_any_cell_and_site_rates(tmp_path, monkeypatch, mk_args):
    family = "celeste_withheld_answer"
    _spec(family)
    _h, items = load_bank(REPO / "evals" / family / "items.json")
    a, b = items[0], items[1]
    rows = []
    for it, tag in ((a, "HIT"), (b, "MISS")):
        for i, s in enumerate(it["read_sites"]):
            text = (
                f"* {it['target']} R:{tag}{i}"
                if (tag == "HIT" and i == len(it["read_sites"]) - 1)
                else "* nothing here"
            )
            rows.append({"id": it["id"], "layer": 42, "pos": s["pos"], "samples": [text]})
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            if a["target"] in c.user:
                quote = next(
                    (
                        ln[2:]
                        for ln in c.user.splitlines()
                        if ln.strip().startswith("* ") and a["target"] in ln
                    ),
                    "",
                )
                out[c.key] = {
                    "expressed": bool(quote),
                    "target": a["target"],
                    "quote": quote.strip(),
                }
            else:
                out[c.key] = {"expressed": False, "target": "", "quote": ""}
        return out

    monkeypatch.setattr(cj, "run_calls", fake_run_calls)
    args = mk_args(path, items=[a["id"], b["id"]], layers=[42], out=tmp_path / "out")
    r = cj.run_family(
        args, name=family, bank=REPO / "evals" / family / "items.json", forbidden=False
    )
    by = {row["id"]: row for row in r.rows}
    assert by[a["id"]]["pass"] is True and by[b["id"]]["pass"] is False
    assert by[a["id"]]["hitting_sites"] == [a["read_sites"][-1]["label"]]
    assert r.value == 0.5 and r.extras["n_calls"] == len(rows)
    last = a["read_sites"][-1]["label"]
    assert r.extras["per_site_item_rate"][last] == 0.5
    assert r.extras["per_variant"][a["variant"]]["n"] >= 1


@pytest.mark.parametrize("family", FAMILIES)
def test_dry_run_on_the_toy_file(tmp_path, capsys, monkeypatch, family: str):
    _spec(family)
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    example = REPO / "examples" / "readouts" / f"{family}.jsonl"
    out = tmp_path / "out"
    argv = ["judge", f"family={family}", f"readouts={example}", f"out={out}", "dry_run=True"]
    assert main(argv) == 0
    r = read_results(out)
    assert r.value is None and r.extras["n_items_without_readouts"] >= 1
    assert "judge prompt for" in capsys.readouterr().out

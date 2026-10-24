"""celeste_jlens_ambiguous: the bank's ids and its J-lens read site (fragments of the multi-word
bridge in the J-lens top 10, not in the text, readout not naming it), plan, judge, toy run."""

import json
import re
from pathlib import Path

from wsbench import readplan, registry
from wsbench.banks import load_bank
from wsbench.basic.prompts import PROMPT_VERSION, SYSTEM
from wsbench.celeste import judge as cj
from wsbench.cli import main
from wsbench.results import read_results

REPO = Path(__file__).resolve().parents[1]
FAM = "celeste_jlens_ambiguous"
BANK = REPO / "evals" / FAM / "items.json"
SUFFIX = "\n\nANSWER IN ONE WORD, DO NOT SAY ANYTHING ELSE."


def _spec():
    import importlib

    mod = importlib.import_module(f"wsbench.evals.{FAM}")
    registry.FAMILIES.setdefault(mod.SPEC.name, mod.SPEC)
    return mod.SPEC


def test_bank_shape_and_plan():
    header, items = load_bank(BANK)
    assert header["family"] == FAM and header["n_items"] == len(items) >= 100
    assert header["funnel"]["kept"] == len(items)
    for it in items:
        assert it["prompt"].endswith(SUFFIX) and len(it["target"].split()) >= 2
        for name in [it["target"], *it["aliases"]]:
            assert name.lower() not in it["question"].lower(), (it["name"], name)
        for a in it["aliases"]:
            assert len(a.split()) >= 2 or a.isupper() or a.lower() in it["target"].lower()
        ev = it["jlens_evidence"]
        assert ev["name_words"], it["name"]
        assert ev["readout_names_target"] is False and min(ev["ranks_l42"].values()) < 10
        prefix = it["question"].lower()
        for w in ev["name_words"]:
            assert not re.search(rf"(?<![a-z0-9]){re.escape(w.lower())}(?![a-z0-9])", prefix), (
                it["name"],
                w,
            )
        (site,) = it["read_sites"]
        assert site["label"] == f"jlens_{it['site_class']}" and 0 <= site["pos"] < len(
            it["input_ids"]
        )
        assert it["site_class"] in ("punctuation", "turn_token")
        assert it["leak_check"]["sites"][site["label"]]["n_leaks"] == 0
        assert cj.targets_of(it)[0] == it["target"]
    specs = readplan.plan(FAM)
    assert len(specs) == len(items)
    s0 = specs[0]
    assert s0.render == "captured" and s0.positions["kind"] == "positions"
    assert s0.positions["positions"] == [x["pos"] for x in items[0]["read_sites"]]
    assert s0.extra["input_ids"] == items[0]["input_ids"]


def test_registered_spec_and_readme():
    spec = _spec()
    assert spec.group == "celeste-evals" and spec.judge.model == "claude-sonnet-5"
    assert spec.judge.prompt_version == PROMPT_VERSION
    readme = (REPO / "evals" / FAM / "README.md").read_text(encoding="utf-8")
    assert SYSTEM in readme and PROMPT_VERSION in readme


def test_item_passes_only_when_named(tmp_path, monkeypatch, mk_args):
    _spec()
    _h, items = load_bank(BANK)
    a, b = items[0], items[1]
    rows = []
    for it, hit in ((a, True), (b, False)):
        (s,) = it["read_sites"]
        text = (
            f"* thinking of {it['target']}"
            if hit
            else f"* {it['jlens_evidence']['name_words'][0]} only"
        )
        rows.append({"id": it["id"], "layer": 42, "pos": s["pos"], "samples": [text]})
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def fake_run_calls(calls, **kw):
        out = {}
        for c in calls:
            line = next(
                (
                    ln.strip()
                    for ln in c.user.splitlines()
                    if ln.strip().startswith("* ") and a["target"] in ln
                ),
                "",
            )
            out[c.key] = {
                "expressed": bool(line),
                "target": a["target"] if line else "",
                "quote": line[2:].strip(),
            }
        return out

    monkeypatch.setattr(cj, "run_calls", fake_run_calls)
    args = mk_args(path, items=[a["id"], b["id"]], layers=[42], out=tmp_path / "out")
    r = cj.run_family(args, name=FAM, bank=BANK, forbidden=False)
    by = {row["id"]: row for row in r.rows}
    assert by[a["id"]]["pass"] is True and by[b["id"]]["pass"] is False
    assert r.value == 0.5 and by[a["id"]]["hitting_sites"] == [a["read_sites"][0]["label"]]


def test_dry_run_on_the_toy_file(tmp_path, capsys, monkeypatch):
    _spec()
    monkeypatch.delenv("WSBENCH_JUDGE_MODEL", raising=False)
    example = REPO / "examples" / "readouts" / f"{FAM}.jsonl"
    out = tmp_path / "out"
    assert (
        main(["judge", f"family={FAM}", f"readouts={example}", f"out={out}", "dry_run=True"]) == 0
    )
    r = read_results(out)
    assert r.value is None and r.extras["n_items_without_readouts"] >= 1
    assert "judge prompt for" in capsys.readouterr().out

import json

from sentiment import llm
from train import fewshot
from train.io import bucket, latest_teacher
from eval.prompt_ab import agreement


def _ids(n, lo, hi):
    out, i = [], 0
    while len(out) < n:
        s = f"x{i}"
        if lo <= bucket(s) < hi:
            out.append(s)
        i += 1
    return out


def test_latest_teacher_highest_pv_then_first():
    rows = [{"id": "a", "ticker": "T", "cls": 0}, {"id": "a", "ticker": "T", "cls": 1},
            {"id": "a", "ticker": "T", "cls": 2, "pv": 2}, {"id": "a", "ticker": "T", "cls": -1, "pv": 2},
            {"id": "a", "ticker": "U", "cls": -2}, {"ticker": "T"}]
    got = {(r["id"], r["ticker"]): r["cls"] for r in latest_teacher(rows)}
    assert got == {("a", "T"): 2, ("a", "U"): -2}


def test_prompt_versions(tmp_path, monkeypatch):
    path = tmp_path / "fs.json"
    path.write_text(json.dumps([{"id": "1", "ticker": "AAA", "channel": "social", "text": "$AAA bought calls", "cls": 2}]))
    block = llm.examples_block(path)
    assert "target AAA" in block and "-> cls 2" in block
    assert llm.examples_block(tmp_path / "missing.json") == ""
    v1, v2 = llm.GeminiLabeler("K", prompt_version=1), llm.GeminiLabeler("K", prompt_version=2)
    assert v1._examples == "" and "Calibration" not in v1._template and "Calibration" in v2._template
    prompt = v2._template.format(ticker="NVDA", company_name="Nvidia", items_json="[]", examples=block)
    assert "target AAA" in prompt and "{" not in prompt.replace("[]", "")


def _row(i, ticker="AAA", channel="news", title="Some headline about the company", text=""):
    return {"item": {"id": i, "ticker": ticker, "channel": channel, "title": title, "text": text},
            "company": {}}


def test_fewshot_uses_gold_train_only_and_drops_offtarget():
    train_ids, test_ids = _ids(4, 0, 70), _ids(1, 70, 100)
    a, b, c, d = train_ids
    corpus = [_row(i) for i in train_ids + test_ids] + [_row("short", title="hi")]
    gold = [{"id": i, "ticker": "AAA", "cls": 1, "relevant": True} for i in train_ids + test_ids + ["short"]]
    gold += [{"id": d, "ticker": "AAA", "cls": 0, "skipped": True}]          # last write wins: skipped
    teacher_rows = [{"id": b, "ticker": "AAA", "cls": 0, "relevant": False},  # off-target dispute
                    {"id": c, "ticker": "AAA", "cls": 0, "relevant": True}]   # v1 disagreed -> first
    out = fewshot.select(corpus, gold, teacher_rows, per_cell=5)
    assert [e["id"] for e in out] == [c, a]
    assert out[0] == {"id": c, "ticker": "AAA", "channel": "news", "text": "Some headline about the company", "cls": 1}


def test_example_text_joins_and_trims():
    assert fewshot.example_text({"title": "T", "text": "body"}) == "T | body"
    assert fewshot.example_text({"title": "", "text": "  a\n b "}) == "a b"
    assert fewshot.example_text({"title": "Head", "text": "Head and more"}) == "Head and more"
    assert len(fewshot.example_text({"title": "x" * 500})) == fewshot.MAX_TEXT


def test_agreement():
    r = agreement([2, 1, 0, -1], [1, 0, 0, -1])
    assert r["exact"] == 0.5 and r["direction"] == 0.75 and r["gemini_zero_tom_dir"] == 1
    assert agreement([], []) == {"n": 0}


class _FakeShots:
    def __init__(self):
        self.batches = []

    def block(self, items):
        self.batches.append([i.id for i in items])
        return f"SHOTS-FOR-{items[0].id}\n"


def _v3_run(monkeypatch, answers):
    from datetime import datetime, timezone
    from sentiment.types import Company, RawItem
    monkeypatch.setattr("sentiment.config.GEMINI_BATCH", 2)
    shots = _FakeShots()
    lab = llm.GeminiLabeler("K", prompt_version=3, shots=shots)
    prompts, schemas = [], []

    def fake_call(prompt, schema=None):
        prompts.append(prompt)
        schemas.append(schema)
        return answers
    monkeypatch.setattr(lab, "_call", fake_call)
    now = datetime.now(timezone.utc)
    items = [RawItem(f"i{k}", "hn", "news", "P", f"T{k}", "b", "u", now, "X") for k in range(3)]
    return lab.label(Company("X", "X CORP", "Company X", 1), items), prompts, schemas, shots


def _ans(i, **kw):
    d = {"id": i, "about": "primary", "cls": 1, "stance": "none", "event": "OTHER", "relevant": False, "reason": "r"}
    d.update(kw)
    return d


def test_v3_prompt_shots_per_batch_and_schema(monkeypatch):
    out, prompts, schemas, shots = _v3_run(monkeypatch, [_ans("i0")])
    assert shots.batches == [["i0", "i1"], ["i2"]]
    assert "SHOTS-FOR-i0" in prompts[0] and "SHOTS-FOR-i2" in prompts[1] and "SHOTS-FOR-i2" not in prompts[0]
    assert "+2 major positive" in prompts[0] and "{" not in prompts[0].split("Items:")[0]
    assert schemas[0] is llm.SCHEMA_V3 and "about" in llm.SCHEMA_V3["items"]["required"]
    assert "about" not in llm.SCHEMA["items"]["properties"]
    assert out["i0"]["about"] == "primary" and out["i0"]["relevant"] is True


def test_v3_about_none_forces_neutral_irrelevant(monkeypatch):
    out, *_ = _v3_run(monkeypatch, [_ans("i0", about="none", cls=2, relevant=True),
                                    _ans("i1", about="mention", cls=-1)])
    assert out["i0"]["cls"] == 0 and out["i0"]["relevant"] is False and out["i0"]["about"] == "none"
    assert out["i1"]["cls"] == -1 and out["i1"]["relevant"] is True


def test_v3_missing_or_bad_about_dropped(monkeypatch):
    bad = _ans("i1")
    del bad["about"]
    out, *_ = _v3_run(monkeypatch, [bad, _ans("i0", about="other")])
    assert out == {}


def test_v2_path_unchanged(monkeypatch):
    lab = llm.GeminiLabeler("K", prompt_version=2)
    assert lab._shots is None and lab._schema is llm.SCHEMA and llm.PROMPT_VERSION == 2
    assert lab._valid(_ans("i0"), {"i0"})   # about ignored for v2
    assert not llm.GeminiLabeler("K", prompt_version=3, shots=_FakeShots())._valid(
        {k: v for k, v in _ans("i0").items() if k != "about"}, {"i0"})


def test_prompt_ab_parse_versions():
    import pytest
    from eval.prompt_ab import parse_versions
    assert parse_versions("2,3") == [2, 3] and parse_versions("3, 1") == [1, 3]
    for bad in ("", "4", "a,2"):
        with pytest.raises(SystemExit):
            parse_versions(bad)

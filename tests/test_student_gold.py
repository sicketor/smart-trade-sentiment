import json
from datetime import datetime, timezone

import numpy as np
import pytest

from train import io as tio
from train import student_gold as sg


def _w(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _crow(iid, ticker="NVDA", channel="news", title="T", short="Nvidia"):
    item = {"id": iid, "source": "google_news", "channel": channel, "publisher": "p", "title": title,
            "text": "", "url": "", "published_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            "ticker": ticker, "meta": {}}
    return {"item": item, "company": {"ticker": ticker, "name": short, "short_name": short, "cik": 1}}


@pytest.fixture
def paths(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    gold = tmp_path / "gold.jsonl"
    monkeypatch.setattr(tio, "CORPUS", corpus)
    monkeypatch.setattr(tio, "GOLD", gold)
    return corpus, gold


def test_gold_rows_rules(paths):
    corpus, gold = paths
    _w(corpus, [_crow("a", "NVDA"), _crow("a", "AMD", short="AMD Inc"), _crow("b"), _crow("c"), _crow("d"),
                _crow("s", channel="social")])
    _w(gold, [
        {"id": "b", "ticker": "NVDA", "cls": 1, "relevant": True},
        {"id": "b", "ticker": "NVDA", "cls": -2, "relevant": True},        # later wins
        {"id": "a", "ticker": "AMD", "cls": 2, "relevant": True},          # ticker match -> AMD company
        {"id": "c", "ticker": "NVDA", "cls": 1, "relevant": False},        # irrelevant -> 0
        {"id": "d", "ticker": "NVDA", "cls": 1, "relevant": True},
        {"id": "d", "skipped": True},                                      # latest is a skip -> dropped
        {"id": "zzz", "ticker": "NVDA", "cls": 1, "relevant": True},       # not in corpus -> dropped
        {"id": "s", "ticker": "NVDA", "cls": 0, "relevant": True},
        {"note": "no id"},
    ])
    rows = sg.gold_rows()
    assert [r["id"] for r in rows] == ["a", "b", "c", "s"]
    by = {r["id"]: r for r in rows}
    assert by["a"]["target"] == "AMD | AMD Inc" and by["a"]["cls"] == 2 and by["a"]["ticker"] == "AMD"
    assert by["a"]["soft"] == pytest.approx([0, 0, 0, 0.15, 0.85])
    assert by["b"]["cls"] == -2 and by["b"]["soft"] == pytest.approx([0.85, 0.15, 0, 0, 0])
    assert by["c"]["cls"] == 0 and by["c"]["soft"] == pytest.approx([0, 0.075, 0.85, 0.075, 0])
    assert by["b"]["text"] == "T" and by["b"]["weight"] == 1.0 and by["b"]["channel"] == "news"
    assert by["s"]["channel"] == "social"


def test_gold_rows_ticker_fallback_to_first_corpus_row(paths):
    corpus, gold = paths
    _w(corpus, [_crow("a", "NVDA")])
    _w(gold, [{"id": "a", "ticker": "TSLA", "cls": 1, "relevant": True}])
    rows = sg.gold_rows()
    assert len(rows) == 1 and rows[0]["ticker"] == "TSLA"
    _w(gold, [{"id": "a", "cls": 1}])
    assert sg.gold_rows()[0]["ticker"] == "NVDA"


def test_gold_rows_empty(paths):
    assert sg.gold_rows() == []


def test_split_folds_cover_each_index_once():
    rows = [{"id": f"id{i}"} for i in range(60)]
    folds = sg.split_folds(rows, 5)
    assert set(folds) == set(range(5))
    flat = sorted(i for v in folds.values() for i in v)
    assert flat == list(range(60))
    for k, v in folds.items():
        assert all(tio.bucket(rows[i]["id"]) % 5 == k for i in v)


def test_replay_filters_teacher_and_caps(tmp_path):
    rows = [{"id": f"t{i}", "source": "teacher"} for i in range(10)]
    rows += [{"id": f"p{i}", "source": "public:tfns"} for i in range(10)]
    p = _w(tmp_path / "train.jsonl", rows)
    got = sg.replay_rows(p, 4, seed=1)
    assert len(got) == 4 and all(r["source"] == "teacher" for r in got)
    assert got == sg.replay_rows(p, 4, seed=1)
    assert len(sg.replay_rows(p, 50, seed=1)) == 10
    assert sg.replay_rows(tmp_path / "missing.jsonl", 5, seed=1) == []


def _oh(c):
    p = [0.0] * 5
    p[c] = 1.0
    return p


def test_report_groups_and_3class_collapse():
    # text rows: y = -2,-1,0,1,2 ; predicted -1 for the -2 row (5class miss, 3class hit)
    rows = [{"cls": c, "channel": "news"} for c in (-2, -1, 0, 1, 2)]
    probs = [_oh(1), _oh(1), _oh(2), _oh(3), _oh(4)]
    rows.append({"cls": 1, "channel": "social"})
    probs.append(_oh(0))   # social row wrong in both views
    rep = sg.report(rows, np.array(probs))
    assert set(rep) == {"all", "text", "social"}
    t = rep["text"]
    assert t["5class"]["n"] == 5 and t["5class"]["acc5"] == pytest.approx(0.8)
    assert t["5class"]["f1_dir"] == 1.0
    # 3class cls is in -1..1 and compared with y in -2..2: only the -1/0/1 rows match
    assert t["3class"]["acc5"] == pytest.approx(0.6)
    assert t["3class"]["f1_dir"] == 1.0
    s = rep["social"]
    assert s["5class"]["n"] == 1 and s["5class"]["acc5"] == 0.0 and s["3class"]["acc5"] == 0.0
    assert rep["all"]["5class"]["n"] == 6


def test_report_omits_empty_group():
    rep = sg.report([{"cls": 0, "channel": "news"}], np.array([_oh(2)]))
    assert set(rep) == {"all", "text"}

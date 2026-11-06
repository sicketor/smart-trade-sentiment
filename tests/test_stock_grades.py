import json
import random
import threading
import urllib.request

import pytest

from eval import stock_grades as sg
from labeling import stocks
from train.stock_batch import pick

BATCH = {"id": "20261107", "stocks": [
    {"ticker": t, "company": f"{t} Inc", "as_of": "2026-11-07T10:00:00+00:00", "snapshot_grade": g,
     "engine": {"grade": g, "score": g and 10.0 * g, "z": g and 0.6 * g, "n_items": 9, "channels": {}},
     "items": [{"id": f"{t}1", "channel": "news", "publisher": "Reuters", "title": "t", "text": "x",
                "url": "https://e.com", "published_at": "2026-11-07T09:00:00+00:00", "age_h": 1.0,
                "cls": 2, "score": 0.9}]}
    for t, g in [("AAA", -2), ("BBB", 0), ("CCC", 1), ("DDD", 3), ("NUL", None)]]}


@pytest.fixture
def state(tmp_path):
    p = tmp_path / "batch.json"
    p.write_text(json.dumps(BATCH), encoding="utf-8")
    return stocks.State(p, tmp_path / "grades.jsonl", "tom")


def test_display_hides_engine(state):
    s = state.next()["stock"]
    assert set(s) == {"ticker", "company", "as_of", "items"}
    assert all("cls" not in i and "score" not in i for i in s["items"])
    assert "engine" not in json.dumps(s) and "snapshot_grade" not in json.dumps(s)


def test_grade_skip_undo(state):
    first = state.next()["stock"]["ticker"]
    res = state.grade({"ticker": first, "grade": 2, "note": "beat"})
    assert res["progress"] == {"done": 1, "total": 4, "graded": 1}
    second = res["stock"]["ticker"]
    assert second != first
    res = state.grade({"ticker": second, "skip": True})
    assert res["progress"]["graded"] == 1 and res["progress"]["done"] == 2
    res = state.undo()
    assert res["stock"]["ticker"] == second and res["progress"]["done"] == 1
    g = stocks.latest_grades(state.grades_path, "20261107")
    assert g == {first: g[first]} and g[first]["grade"] == 2 and g[first]["note"] == "beat"


@pytest.mark.parametrize("body", [{"ticker": "ZZZ", "grade": 1}, {"ticker": "AAA", "grade": 4},
                                  {"ticker": "AAA", "grade": True}, {"ticker": "AAA", "grade": "1"},
                                  {"ticker": "AAA"}])
def test_grade_rejects_bad_input(state, body):
    with pytest.raises(ValueError):
        state.grade(body)


def test_undo_empty(state):
    with pytest.raises(ValueError):
        state.undo()


def test_finished_and_order_stable(state, tmp_path):
    seen = []
    while (s := state.next()["stock"]) is not None:
        seen.append(s["ticker"])
        state.grade({"ticker": s["ticker"], "grade": 0})
    assert sorted(seen) == ["AAA", "BBB", "CCC", "DDD"]
    again = stocks.State(tmp_path / "batch.json", tmp_path / "other.jsonl", "tom")
    assert again.order == state.order   # same labeler + batch -> same order


def test_http_roundtrip(state):
    srv = stocks.make_server(state, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        nxt = json.loads(urllib.request.urlopen(base + "/api/next").read())
        req = urllib.request.Request(base + "/api/grade", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"ticker": nxt["stock"]["ticker"], "grade": -1}).encode())
        assert json.loads(urllib.request.urlopen(req).read())["progress"]["graded"] == 1
        assert b"<html" in urllib.request.urlopen(base + "/").read()
    finally:
        srv.shutdown()
        srv.server_close()


def test_report_agreement_and_cuts():
    grades = {"AAA": {"grade": -2}, "BBB": {"grade": 1}, "CCC": {"grade": 1}, "DDD": {"grade": 3},
              "EEE": {"grade": None}}
    res = sg.report(BATCH, grades)
    a = res["engine"]
    assert res["n"] == 4 and res["skipped"] == 1
    assert a["exact"] == 0.75 and a["within1"] == 1.0 and a["direction"] == 0.75
    assert a["mean_gap"] == pytest.approx(-0.25)
    assert res["confusion"][1 + 3][0 + 3] == 1
    cuts = res["fitted_cuts"]
    assert cuts == sorted(cuts) and len(cuts) == 6


def test_fit_cuts_midpoints():
    rows = [{"tom": g, "z": float(g)} for g in range(-3, 4)]
    assert sg.fit_cuts(rows, [0.0] * 6) == [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    # a Tom grade with no stocks bridges to the nearest grades on each side
    rows = [r for r in rows if r["tom"] != 0]
    assert sg.fit_cuts(rows, [9.0] * 6)[2:4] == [0.0, 0.0]


def test_spearman():
    assert sg.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert sg.spearman([1, 1, 1], [1, 2, 3]) is None


def test_pick_stratified():
    latest = {f"T{g + 3}{i}": {"composite": {"grade": g}} for g in range(-3, 4) for i in range(10)}
    p = pick(latest, 45, random.Random(1))
    assert len(p) == len(set(p)) == 45
    grades = [latest[t]["composite"]["grade"] for t in p]
    assert grades.count(-3) == 5 and grades.count(0) == 7
    short = {t: r for t, r in latest.items() if r["composite"]["grade"] != 3}   # no +3 stocks: places refilled
    assert len(pick(short, 45, random.Random(1))) == 45

import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from eval.market import (as_of_for, excess_returns, grade_table, load_prices, pilot_days, pit_records, spearman,
                         trading_days, with_dz, yahoo_symbol)

DAYS = ["2026-10-16", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]
SPY = {"2026-10-16": 100.0, "2026-10-19": 101.0, "2026-10-20": 102.0, "2026-10-21": 100.0, "2026-10-22": 103.0}
PX = {"2026-10-16": 50.0, "2026-10-19": 55.0, "2026-10-20": 55.0, "2026-10-21": 44.0, "2026-10-22": 66.0}


def test_excess_returns_horizons():
    assert excess_returns(PX, SPY, "2026-10-19", DAYS, 1) == pytest.approx(0.0 - (102 / 101 - 1))
    assert excess_returns(PX, SPY, "2026-10-19", DAYS, 3) == pytest.approx((66 / 55 - 1) - (103 / 101 - 1))
    assert excess_returns(PX, SPY, "2026-10-19", DAYS, -1) == pytest.approx(0.1 - 0.01)   # 16/10 -> 19/10


def test_excess_returns_missing_and_past_end():
    assert excess_returns(PX, SPY, "2026-10-22", DAYS, 1) is None           # past the last close
    assert excess_returns(PX, SPY, "2026-10-20", DAYS, 3) is None
    assert excess_returns(PX, SPY, "2026-10-16", DAYS, -1) is None          # no earlier day
    px = {k: v for k, v in PX.items() if k != "2026-10-20"}
    assert excess_returns(px, SPY, "2026-10-19", DAYS, 1) is None           # halted / missing close
    assert excess_returns(px, SPY, "2026-10-20", DAYS, 1) is None
    assert excess_returns(PX, SPY, "2026-10-17", DAYS, 1) is None           # not a trading day


def test_trading_days_excludes_today():
    spy = {"2026-10-23": 1.0, "2026-10-26": 1.0, "2026-10-27": 1.0, "2026-10-28": 1.0}
    assert trading_days(spy, date(2026, 10, 28)) == ["2026-10-23", "2026-10-26", "2026-10-27"]


def test_as_of_is_us_close_utc():
    assert as_of_for("2026-10-19") == datetime(2026, 10, 19, 20, tzinfo=timezone.utc)


def _rec(t, d, z):
    g = {"raw": None, "z": None, "grade": None, "score": None}
    return {"ticker": t, "day": d, "n_items": 1, "news": dict(g), "filings": dict(g), "social": dict(g),
            "composite": {**g, "z": z}}


def test_dz_over_weekend_and_gap():
    recs = [_rec("A", "2026-10-16", 0.5), _rec("A", "2026-10-19", 1.25),     # Fri -> Mon
            _rec("B", "2026-10-16", 1.0), _rec("B", "2026-10-20", 2.0),      # no 19/10 record
            _rec("C", "2026-10-19", None), _rec("C", "2026-10-20", 1.0)]     # previous z missing
    out = {(r["ticker"], r["day"]): r["composite"]["dz"] for r in with_dz(recs, DAYS)}
    assert out[("A", "2026-10-19")] == pytest.approx(0.75)
    assert out[("A", "2026-10-16")] is None
    assert out[("B", "2026-10-20")] is None
    assert out[("C", "2026-10-20")] is None


def test_grade_table_exact():
    t = grade_table([3, 3, 1, -1, 0, -3, 2], [0.02, -0.01, 0.01, 0.005, 0.03, -0.02, 0.0])
    assert t["hit_rate"] == pytest.approx(3 / 5) and t["n_hit"] == 5 and t["n_nonzero"] == 6   # ex == 0 left out
    sw = t["strong_vs_weak"]
    assert sw["strong"] == pytest.approx(0.01) and sw["n_strong"] == 3
    assert sw["weak"] == pytest.approx(0.0025) and sw["n_weak"] == 2
    assert t["base_rate"] == pytest.approx(4 / 7)
    assert t["grades"]["3"] == {"n": 2, "mean": pytest.approx(0.005), "median": pytest.approx(0.005)}
    assert t["grades"]["-2"] == {"n": 0, "mean": None, "median": None}


def test_spearman_constant_is_none():
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


class _FakeHttp:
    def __init__(self, fail=()):
        self.urls, self.fail = [], set(fail)

    def get_json(self, url, params=None, ua=None):
        self.urls.append(url)
        sym = url.rsplit("/", 1)[-1]
        if sym in self.fail:
            raise RuntimeError("boom")
        ts = int(datetime(2026, 10, 20, 13, 30, tzinfo=timezone.utc).timestamp())
        return {"chart": {"result": [{"timestamp": [ts], "indicators": {"quote": [{"close": [10.0]}]}}]}}


def test_symbol_map_cache_and_failures(tmp_path):
    assert yahoo_symbol("BRK.B") == "BRK-B"
    http = _FakeHttp(fail={"BAD"})
    out = load_prices(["BRK.B", "BAD"], http=http, cache_dir=tmp_path)
    assert any(u.endswith("/BRK-B") for u in http.urls)
    assert set(out) == {"BRK.B", "SPY"} and out["BRK.B"] == {"2026-10-20": 10.0}
    assert (tmp_path / "BRK-B.json").is_file()
    n = len(http.urls)
    assert load_prices(["BRK.B"], http=http, cache_dir=tmp_path)["BRK.B"] == {"2026-10-20": 10.0}
    assert len(http.urls) == n                                                   # served from the cache
    with pytest.raises(RuntimeError):
        load_prices(["X"], http=_FakeHttp(fail={"SPY"}), cache_dir=tmp_path / "empty")


def test_cache_staleness(tmp_path):
    def put(sym, fetched, closes):
        (tmp_path / f"{sym}.json").write_text(json.dumps({"symbol": sym, "fetched_at": fetched, "closes": closes}))
    now = datetime.now(timezone.utc)
    put("SPY", now.isoformat(), {"2026-10-19": 1.0, "2026-10-20": 1.0})
    put("A", (now - timedelta(days=1)).isoformat(), {"2026-10-20": 5.0})      # fetched yesterday -> stale
    put("B", now.isoformat(), {"2026-10-19": 5.0})                             # last bar older than SPY's
    put("C", now.isoformat(), {"2026-10-20": 7.0})                             # fresh
    http = _FakeHttp()
    out = load_prices(["A", "B", "C"], http=http, cache_dir=tmp_path)
    assert sorted(u.rsplit("/", 1)[-1] for u in http.urls) == ["A", "B"]
    assert out["A"] == out["B"] == {"2026-10-20": 10.0} and out["C"] == {"2026-10-20": 7.0}
    put("D", (now - timedelta(days=1)).isoformat(), {"2026-10-20": 3.0})
    assert load_prices(["D"], http=_FakeHttp(fail={"D"}), cache_dir=tmp_path)["D"] == {"2026-10-20": 3.0}


def test_pit_records_window_bounds(monkeypatch):
    seen = {}

    def fake_summarize(items, as_of, context=None, history=None, norm=None):
        seen[as_of] = sorted(items)
        snap = SimpleNamespace(raw=1.0, z=0.5, grade=1, score=10.0)
        return {"composite": snap, "channels": {c: snap for c in ("news", "filings", "social")}}

    monkeypatch.setattr("sentiment.aggregate.summarize", fake_summarize)
    a = as_of_for("2026-10-20")
    col = datetime(2026, 10, 28, tzinfo=timezone.utc)                         # collected later: ignored
    entries = [("at_close", a, col), ("after", a + timedelta(seconds=1), col),
               ("edge", a - timedelta(days=7), col), ("too_old", a - timedelta(days=7, seconds=1), col)]
    recs = pit_records({"X": entries}, ["2026-10-20", "2026-10-21"], norm=None)
    assert seen[a] == ["at_close", "edge"]
    assert [r["day"] for r in recs] == ["2026-10-20", "2026-10-21"] and recs[0]["n_items"] == 2
    assert seen[as_of_for("2026-10-21")] == ["after", "at_close"]
    assert recs[0]["composite"] == {"raw": 1.0, "z": 0.5, "grade": 1, "score": 10.0}
    assert pit_records({"X": entries}, ["2026-11-20"], norm=None) == []      # no items -> skipped


def test_pilot_days_threshold():
    recs = [{"ticker": f"T{i}", "day": "2026-10-19"} for i in range(30)]
    recs += [{"ticker": f"T{i}", "day": "2026-10-16"} for i in range(29)]
    recs += [{"ticker": "T0", "day": "2026-10-16"}]                            # duplicate ticker: not counted twice
    assert pilot_days(recs) == ["2026-10-19"]
    assert pilot_days(recs, min_tickers=29) == ["2026-10-16", "2026-10-19"]

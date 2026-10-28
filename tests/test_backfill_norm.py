from datetime import datetime, timezone

import numpy as np

from train.backfill_norm import build_norm, ticker_days


def _rec(news=None, comp=None):
    return {"ticker": "X", "raw": {"news": news, "filings": None, "social": None}, "raw_composite": comp}


def test_build_norm_exact():
    g = build_norm([_rec(10, 1), _rec(20, 2), _rec(30, 3)], min_n=3)
    assert set(g) == {"news", "composite"}
    assert g["news"]["mu"] == 20 and g["news"]["sd"] == 10 and g["news"]["n"] == 3
    assert g["news"]["cuts"] == [round(float(c), 4) for c in np.percentile([-1, 0, 1], [5, 15, 35, 65, 85, 95])]


def test_build_norm_min_n_none_and_flat():
    recs = [_rec(10, 5), _rec(None, 5), _rec(30, 5)]
    assert build_norm(recs, min_n=3) == {}          # news n=2 < 3; composite sd == 0
    g = build_norm(recs, min_n=2)
    assert g["news"]["n"] == 2 and "composite" not in g


def test_ticker_days():
    def row(t, c):
        return {"item": {"ticker": t}, "collected_at": c}
    rows = [row("A", "2025-10-20T08:00:00"), row("A", "2025-10-20T18:00:00+00:00"),
            row("A", "2025-10-20T10:00:00+00:00"), row("A", "2025-10-21T01:00:00"), row("B", "2025-10-20T09:00:00")]
    d = ticker_days(rows)
    assert d["A"] == [datetime(2025, 10, 20, 18, tzinfo=timezone.utc), datetime(2025, 10, 21, 1, tzinfo=timezone.utc)]
    assert d["B"] == [datetime(2025, 10, 20, 9, tzinfo=timezone.utc)]

import json
from datetime import datetime, timezone

from sentiment.snapshots import compact, history, record


def _d(as_of, n, raw, ticker="NVDA"):
    return {"ticker": ticker, "as_of": as_of, "model_version": "v", "flags": [], "attention_z": 0.1,
            "composite": {"raw": raw, "score": 1, "items": [1]},
            "channels": {"news": {"raw": raw, "n_items": n, "top": [1, 2]}}, "items": [{"x": 1}]}


def test_record_and_compact(tmp_path):
    d = _d("2025-10-20T12:00:00+00:00", 5, 10)
    p = record(d, tmp_path)
    record(d, tmp_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert "items" not in row and "top" not in row["channels"]["news"]
    assert row["channels"]["news"]["n_items"] == 5 and row["composite"]["z"] is None
    assert compact(d) == row


def test_history_example(tmp_path):
    for ts, n, raw in [("2025-10-20T12:00:00+00:00", 5, 10), ("2025-10-20T18:00:00+00:00", 7, 12),
                       ("2025-10-21T09:00:00+00:00", 4, None), ("2025-10-22T08:00:00+00:00", 9, 99)]:
        record(_d(ts, n, raw), tmp_path)
    h = history("nvda", datetime(2025, 10, 22, 15, tzinfo=timezone.utc), root=tmp_path)
    assert h == {"news": [7.0, 4.0], "raw_news": [12.0], "raw_composite": [12.0]}


def test_days_cutoff_invalid_and_missing(tmp_path):
    record(_d("2025-07-01T12:00:00+00:00", 5, 1), tmp_path)
    p = record(_d("2025-10-20T12:00:00", 3, 2), tmp_path)
    with open(p, "a") as f:
        f.write("not json\n\n")
    h = history("NVDA", datetime(2025, 10, 22, tzinfo=timezone.utc), days=30, root=tmp_path)
    assert h["news"] == [3.0]
    assert history("AMD", datetime(2025, 10, 22, tzinfo=timezone.utc), root=tmp_path) == {}

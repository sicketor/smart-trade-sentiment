from datetime import datetime, timedelta, timezone

import pytest

from sentiment.sources._util import item_id
from train import weak_stocktwits as w

NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)


def msg(mid, days_ago=1, tag="Bullish", body="strong earnings beat this quarter guys", user="u1", syms=("AAA",)):
    return {"id": mid, "body": body, "created_at": (NOW - timedelta(days=days_ago)).isoformat(),
            "user": {"username": user, "followers": 5, "join_date": (NOW - timedelta(days=400)).isoformat()},
            "symbols": [{"symbol": s} for s in syms],
            "entities": {"sentiment": {"basic": tag} if tag else None}, "likes": {"total": 2}}


class FakeHttp:
    """Serves a descending message list like the StockTwits stream (30 per page there; here `size`)."""

    def __init__(self, msgs, size=3, fail_after=None):
        self.msgs = sorted(msgs, key=lambda m: -m["id"])
        self.size, self.calls, self.fail_after = size, [], fail_after

    def get_json(self, url, params=None):
        self.calls.append(params)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("boom")
        mx = (params or {}).get("max")
        pool = [m for m in self.msgs if mx is None or m["id"] <= mx]
        return {"messages": pool[: self.size]}


def test_message_row():
    r = w.message_row(msg(7), "AAA", NOW)
    assert r["mid"] == 7 and r["label"] == "Bullish" and r["symbols"] == ["AAA"] and r["likes"] == 2
    assert r["id"] == item_id("stocktwits", "https://stocktwits.com/u1/message/7", r["body"])
    assert r["account_age_days"] == 400
    assert w.message_row(msg(7, tag=None), "AAA") is None
    assert w.message_row(msg(7, body=""), "AAA") is None
    assert w.message_row(msg(7, user=None), "AAA") is None
    m = msg(7)
    m["id"] = None
    assert w.message_row(m, "AAA") is None
    m = msg(7)
    m["created_at"] = "garbage"
    assert w.message_row(m, "AAA") is None


def test_backfill_empty_page_and_resume():
    msgs = [msg(i, days_ago=10 - i % 5, tag="Bullish" if i % 2 else None) for i in range(1, 8)]
    http, state, seen = FakeHttp(msgs), {}, set()
    rows = w.harvest_ticker("AAA", http, state, seen, now=NOW)
    assert sorted(r["mid"] for r in rows) == [1, 3, 5, 7]
    s = state["AAA"]
    assert s == {"newest": 7, "oldest": 1, "backfill_done": True}
    assert http.calls == [None, {"max": 4}, {"max": 1}, {"max": 0}]  # last page empty -> done
    # resume: new messages 8,9 only
    http2 = FakeHttp(msgs + [msg(8), msg(9)], size=3)
    rows2 = w.harvest_ticker("AAA", http2, state, seen, now=NOW)
    assert sorted(r["mid"] for r in rows2) == [8, 9]
    assert state["AAA"]["newest"] == 9 and len(http2.calls) == 1  # page [9,8,7] min<=newest -> stop


def test_backfill_stops_on_date_window():
    msgs = [msg(1, 200), msg(2, 150), msg(3, 100), msg(4, 50), msg(5, 10)]
    http, state = FakeHttp(msgs, size=2), {}
    rows = w.harvest_ticker("AAA", http, state, set(), days=120, now=NOW)
    # pages: [5,4], [3,2] (2 is 150d old -> done); message 2 is outside the window
    assert sorted(r["mid"] for r in rows) == [3, 4, 5]
    assert state["AAA"]["backfill_done"] and len(http.calls) == 2


def test_max_pages_budget_and_continue():
    msgs = [msg(i, 5) for i in range(1, 11)]
    http, state, seen = FakeHttp(msgs, size=2), {}, set()
    rows = w.harvest_ticker("AAA", http, state, seen, max_pages=2, now=NOW)
    assert len(http.calls) == 2 and len(rows) == 4
    assert not state["AAA"]["backfill_done"] and state["AAA"]["oldest"] == 7
    rows2 = w.harvest_ticker("AAA", http, state, seen, max_pages=50, now=NOW)
    assert sorted(r["mid"] for r in rows2) == [1, 2, 3, 4, 5, 6]  # no new on top, continues backfill
    assert state["AAA"]["backfill_done"]


def test_new_messages_budget_counts_toward_pages():
    state = {"AAA": {"newest": 2, "oldest": 1, "backfill_done": False}}
    msgs = [msg(i, 1) for i in range(1, 20)]
    http = FakeHttp(msgs, size=2)
    rows = w.harvest_ticker("AAA", http, state, set(), max_pages=3, now=NOW)
    assert len(http.calls) == 3 and [r["mid"] for r in rows] == [19, 18, 17, 16, 15, 14]
    assert state["AAA"]["newest"] == 19


def test_http_error_keeps_state(capsys):
    msgs = [msg(i, 5) for i in range(1, 10)]
    http, state = FakeHttp(msgs, size=2, fail_after=1), {}
    rows = w.harvest_ticker("AAA", http, state, set(), now=NOW)
    assert [r["mid"] for r in rows] == [9, 8]
    assert state["AAA"] == {"newest": 9, "oldest": 8, "backfill_done": False}
    assert "stopped" in capsys.readouterr().out


def test_http_error_in_newest_phase_records_gap_and_next_call_fills_it():
    old = {"newest": 2, "oldest": 1, "backfill_done": True}
    state = {"AAA": dict(old)}
    msgs = [msg(i, 1) for i in range(1, 13)]
    seen = set()
    http = FakeHttp(msgs, size=3, fail_after=1)
    rows = w.harvest_ticker("AAA", http, state, seen, now=NOW)
    assert [r["mid"] for r in rows] == [12, 11, 10]
    assert state["AAA"]["newest"] == 12 and state["AAA"]["gap"] == [2, 9]
    rows2 = w.harvest_ticker("AAA", FakeHttp(msgs, size=3), state, seen, now=NOW)
    assert sorted(r["mid"] for r in rows2) == [3, 4, 5, 6, 7, 8, 9]
    assert "gap" not in state["AAA"] and state["AAA"]["newest"] == 12


def test_budget_exhausted_in_newest_phase_records_gap():
    state = {"AAA": {"newest": 2, "oldest": 1, "backfill_done": True}}
    msgs = [msg(i, 1) for i in range(1, 13)]
    seen = set()
    rows = w.harvest_ticker("AAA", FakeHttp(msgs, size=3), state, seen, max_pages=2, now=NOW)
    assert len(rows) == 6 and state["AAA"]["newest"] == 12 and state["AAA"]["gap"] == [2, 6]
    # gap budget too small: gap shrinks, stays
    http = FakeHttp(msgs, size=3)
    w.harvest_ticker("AAA", http, state, seen, max_pages=1, now=NOW)
    assert http.calls == [{"max": 6}] and state["AAA"]["gap"] == [2, 3]
    rows3 = w.harvest_ticker("AAA", FakeHttp(msgs, size=3), state, seen, now=NOW)
    assert sorted(r["mid"] for r in rows3) == [3]
    assert "gap" not in state["AAA"]


def test_normal_run_leaves_no_gap():
    state = {"AAA": {"newest": 5, "oldest": 1, "backfill_done": True}}
    w.harvest_ticker("AAA", FakeHttp([msg(i, 1) for i in range(1, 10)], size=2), state, set(), now=NOW)
    assert "gap" not in state["AAA"] and state["AAA"]["newest"] == 9


def test_seen_mids_not_returned_twice():
    seen = {9}
    rows = w.harvest_ticker("AAA", FakeHttp([msg(9), msg(8)], size=5), {}, seen, now=NOW)
    assert [r["mid"] for r in rows] == [8]


# ---- build -----------------------------------------------------------------------------------
def raw(mid, **kw):
    base = dict(body=f"strong earnings beat number {'abcdefghij'[mid % 10]} quarter", label="Bullish", user=f"u{mid}",
                ticker="AAA", symbols=["AAA"], account_age_days=100, created_at="2026-03-05T00:00:00+00:00")
    base.update(kw)
    r = {"mid": mid, "id": f"id{mid}", "followers": 1, "likes": 0, **base}
    return r


def test_build_filters():
    rows = [
        raw(1, label=None),                                            # untagged
        raw(2, symbols=["BBB"]),                                       # ticker missing
        raw(3, symbols=["AAA", "BBB", "CCC"]),                         # too many symbols
        raw(4, body="$AAA http://x.co @bob yes!! go"),                # < 4 words
        raw(5, account_age_days=10),                                   # young
        raw(6, account_age_days=None),                                 # unknown age
        raw(7, body="gold item text about earnings today"),            # gold id
        raw(8, body="Gold item: text about earnings today!"),          # same key as gold row
        raw(9, body="unique fine bullish message here now", label="Bullish"),
        raw(10, body="unique fine bearish message here now", label="Bearish"),
    ]
    out, c = w.build(rows, [{"id": "id7"}])
    assert [c[k] for k in ("raw", "tagged", "symbols", "words", "account_age", "not_gold", "dedup", "user_cap",
                           "balanced")] == [10, 9, 7, 6, 4, 2, 2, 2, 2]
    assert sorted(r["mid"] for r in out) == [9, 10]
    assert {r["mid"]: r["label"] for r in out} == {9: 1, 10: -1}
    assert set(out[0]) == {"id", "mid", "ticker", "text", "label", "created_at", "user"}


def test_build_dedup_earliest_and_user_cap():
    body = "same text repeated everywhere today"
    rows = [raw(1, body=body, created_at="2026-03-09T00:00:00+00:00"),
            raw(2, body=body, created_at="2026-03-02T00:00:00+00:00", label="Bearish"),
            ]
    for i in range(10, 16):
        rows.append(raw(i, body=f"distinct words alpha{i} apple banana cherry {'xyz'[i % 3]}{'qrs'[i % 3]} here",
                        user="spammer", label="Bullish" if i % 2 else "Bearish",
                        created_at=f"2026-03-{i:02d}T00:00:00+00:00"))
    _, c = w.build(rows, [])
    assert c["dedup"] == 7 and c["user_cap"] == 7
    out, c = w.build(rows, [], per_user_cap=2)
    assert c["user_cap"] == 3  # mid 1, mid 2 (own users) + ... see below
    users = [r["user"] for r in out]
    assert users.count("spammer") <= 2


def test_build_balance_deterministic_and_groups():
    rows = []
    for i in range(1, 9):   # March: 5 bull, 3 bear
        rows.append(raw(i, body=f"march message number {'abcdefgh'[i - 1]} about market stuff",
                        label="Bullish" if i <= 5 else "Bearish", created_at=f"2026-03-{i:02d}T00:00:00+00:00"))
    for i in range(20, 23):  # April: only bulls -> dropped
        rows.append(raw(i, body=f"april message {'xyz'[i - 20]} about market stuff here",
                        created_at=f"2026-04-{i:02d}T00:00:00+00:00"))
    a, ca = w.build(rows, [], seed=3)
    b, _ = w.build(list(reversed(rows)), [], seed=3)
    assert a == b and ca["balanced"] == 6
    assert sum(r["label"] == 1 for r in a) == 3 and sum(r["label"] == -1 for r in a) == 3
    assert all(r["created_at"].startswith("2026-03") for r in a)
    n, cn = w.build(rows, [], balance=False)
    assert cn["natural"] == 11 and "balanced" not in cn
    assert sum(r["label"] == 1 for r in n) == 8 and sum(r["label"] == -1 for r in n) == 3


# ---- tag agreement ---------------------------------------------------------------------------
def crow(i, tag, source="stocktwits"):
    return {"item": {"id": i, "source": source, "meta": {"user_label": tag}}}


def test_tag_agreement():
    corpus = [crow("a", "Bullish"), crow("b", "Bullish"), crow("c", "Bearish"), crow("d", "Bearish"),
              crow("e", "Bullish"), crow("f", "Bullish", source="reddit"), crow("g", None), crow("h", "Bullish")]
    gold = [{"id": "a", "relevant": True, "cls": 1},
            {"id": "b", "relevant": True, "cls": -1},
            {"id": "c", "relevant": True, "cls": -1},
            {"id": "d", "relevant": False, "cls": 1},       # tom 0
            {"id": "e", "relevant": True, "cls": 0},        # first row, replaced by latest
            {"id": "e", "relevant": True, "cls": 1},
            {"id": "f", "relevant": True, "cls": 1},
            {"id": "g", "relevant": True, "cls": 1},
            {"id": "h", "skipped": True}]
    r = w.tag_agreement(corpus, gold)
    assert r["n"] == 5
    assert r["counts"] == {"Bullish:1": 2, "Bullish:-1": 1, "Bearish:-1": 1, "Bearish:0": 1}
    assert r["direction_agree"] == pytest.approx(3 / 4)
    assert r["share_tom_zero"] == pytest.approx(1 / 5)
    empty = w.tag_agreement([], [])
    assert empty["n"] == 0 and empty["direction_agree"] is None

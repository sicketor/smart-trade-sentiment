from datetime import datetime, timedelta, timezone

from sentiment.http import HttpError
from sentiment.sources._util import item_id
from train import corpus
from train import weak_marketaux as w

NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)


def art(uuid, ticker="AAA", score=0.5, days_ago=1.0, title="Big earnings beat"):
    ts = (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
    return {"uuid": uuid, "title": title, "description": "<b>d</b>", "snippet": "s", "url": f"https://x/{uuid}",
            "source": "x.com", "published_at": ts,
            "entities": [{"symbol": "ZZZ", "sentiment_score": -1},
                         {"symbol": ticker, "sentiment_score": score, "match_score": 9.0,
                          "highlights": [{"highlight": "<em>hi</em>", "sentiment": 0.1}] * 7}]}


class FakeHttp:
    def __init__(self, pages, err_at=None):
        self.pages, self.calls, self.err_at = list(pages), [], err_at

    def get_json(self, url, params=None):
        self.calls.append(dict(params))
        if self.err_at is not None and len(self.calls) == self.err_at:
            raise HttpError("api.marketaux.com/v1/news/all: 402")
        return {"data": self.pages.pop(0) if self.pages else []}


def test_article_row():
    r = w.article_row(art("u1", "aaa"), "AAA")
    assert r["score"] == 0.5 and r["description"] == "d" and r["n_entities"] == 2
    assert r["published_at"] == (NOW - timedelta(days=1)).isoformat() and len(r["highlights"]) == 5
    assert r["highlights"][0]["text"] == "hi"
    assert w.article_row(art("u1"), "BBB") is None
    assert w.article_row(art("u1", score=None), "AAA") is None
    assert w.article_row(art("u1", title=""), "AAA") is None


def test_harvest_round_robin_cursor_dedup_and_token():
    http = FakeHttp([[art("a1", "AAA", days_ago=1), art("a2", "AAA", days_ago=2)],
                     [art("b1", "BBB", days_ago=1)]])
    state, seen = {}, set()
    rows, n, err = w.harvest(["AAA", "BBB"], http, state, seen, "KEY", max_requests=2, now=NOW)
    assert err is None and n == 2 and len(rows) == 3
    assert [c["symbols"] for c in http.calls] == ["AAA", "BBB"]
    assert all(c["api_token"] == "KEY" for c in http.calls) and "published_before" not in http.calls[0]
    assert state["AAA"]["n"] == 2 and not state["AAA"]["done"]
    assert w.parse_iso(state["AAA"]["cursor"]) == NOW - timedelta(days=2, seconds=1)
    # next run: AAA continues from cursor, empty page -> done; dedup keeps nothing twice
    http2 = FakeHttp([[art("a2", "AAA", days_ago=2.0)]])
    rows2, n2, _ = w.harvest(["AAA"], http2, state, seen, "KEY", max_requests=5, now=NOW)
    assert rows2 == [] and http2.calls[0]["published_before"] == (NOW - timedelta(days=2, seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
    assert state["AAA"]["done"] and n2 == 1  # no progress -> done


def test_harvest_empty_page_done_and_horizon():
    state = {}
    http = FakeHttp([[art("a1", days_ago=1)], []])
    rows, n, _ = w.harvest(["AAA"], http, state, set(), "K", max_requests=10, now=NOW)
    assert n == 2 and len(rows) == 1 and state["AAA"]["done"]
    state = {}
    http = FakeHttp([[art("a1", days_ago=400)]])
    _, n, _ = w.harvest(["AAA"], http, state, set(), "K", max_requests=10, days=365, now=NOW)
    assert n == 1 and state["AAA"]["done"]


def test_harvest_error_keeps_rows():
    http = FakeHttp([[art("a1", "AAA")]], err_at=2)
    rows, n, err = w.harvest(["AAA", "BBB"], http, {}, set(), "SECRETKEY", max_requests=10, now=NOW)
    assert len(rows) == 1 and n == 2 and "402" in err and "SECRETKEY" not in err


def test_to_label():
    assert [w.to_label(s) for s in (0.15, 0.149, 0, -0.149, -0.15, 0.9)] == [1, 0, 0, 0, -1, 1]
    assert w.to_label(0.1, 0.05) == 1


def raw(uuid, ticker="AAA", title="Big earnings beat", url=None, score=0.5):
    return {"uuid": uuid, "ticker": ticker, "title": title, "url": f"https://x/{uuid}" if url is None else url,
            "source": "x", "description": "", "snippet": "", "published_at": NOW.isoformat(), "score": score}


def crow(id_, ticker, title, url):
    return {"item": {"id": id_, "ticker": ticker, "title": title, "url": url}}


def test_raw_to_item():
    it = w.raw_to_item({**raw("u"), "description": "", "snippet": "snip"})
    assert it.source == "marketaux" and it.text == "snip" and it.id == item_id("marketaux", "https://x/u", it.title)
    assert it.meta == {"mx_score": 0.5, "mx_uuid": "u"} and it.published_at == NOW


def test_match_gold():
    corp = [crow("c1", "AAA", "Other", "https://x/u1"),
            crow("c2", "AAA", "Big Earnings Beat!", "https://other"),
            crow("c3", "AAA", "Skipped one", "https://x/u3"),
            crow("c4", "AAA", "Irrelevant", "https://x/u4")]
    gold = [{"id": "c1", "ticker": "AAA", "cls": 1}, {"id": "c1", "ticker": "AAA", "cls": -2},
            {"id": "c2", "ticker": "AAA", "cls": 2},
            {"id": "c3", "ticker": "AAA", "cls": None, "skipped": True},
            {"id": "c4", "ticker": "AAA", "cls": 2, "relevant": False}]
    rows = [raw("u1"), raw("u2", title="Big earnings beat"), raw("u3"), raw("u4"), raw("u5", ticker="BBB")]
    got = w.match_gold(rows, corp, gold)
    assert [(r["uuid"], c) for r, c in got] == [("u1", -2), ("u2", 2), ("u4", 0)]
    # each gold key at most once
    assert len(w.match_gold([raw("u1"), raw("u1")], corp, gold)) == 1
    # direct id key
    it_id = w.raw_to_item(raw("u9")).id
    assert w.match_gold([raw("u9")], [], [{"id": it_id, "ticker": "AAA", "cls": 1}])[0][1] == 1


def test_agreement():
    assert w.agreement([]) == {"n": 0}
    pairs = [(0.5, 2), (0.4, 1), (-0.5, -1), (0.0, 1), (0.6, 0), (-0.4, 1)]
    a = w.agreement(pairs)
    assert a["n"] == 6 and a["dir_acc"] == round(3 / 6, 3)  # (+,+) (+,+) (-,-) ; others wrong
    assert a["polar_n"] == 4 and a["polar_acc"] == 0.75
    assert a["mx_dist"] == {"-1": 2, "0": 1, "1": 3}
    assert isinstance(a["spearman"], float)
    assert w.agreement([(0.5, 1), (0.5, 1), (0.5, 1)])["spearman"] is None
    assert w.agreement([(0.5, 1), (0.1, 0)])["spearman"] is None
    assert w.agreement([(0.0, 0)])["polar_acc"] is None


def test_corpus_to_rows():
    from sentiment.types import Company, RawItem
    co = Company(ticker="AAA", name="Acme Corp", short_name="Acme", cik=1)
    mk = lambda i, t: RawItem(id=i, source="marketaux", channel="news", publisher="p", title=t,
                              text="", url=f"https://x/{i}", published_at=NOW, ticker="AAA")
    embed = lambda texts: [[1.0, 0.0] for _ in texts]
    rows = corpus.to_rows([mk("i1", "Acme Corp beats estimates"), mk("i2", "")], co, embed, NOW)
    assert [r["item"]["id"] for r in rows] == ["i1"]
    assert rows[0]["collected_at"] == NOW.isoformat() and rows[0]["company"]["ticker"] == "AAA"
    assert corpus.to_rows([], co, embed, NOW) == []

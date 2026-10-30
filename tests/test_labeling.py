import json
import random
import threading
import urllib.error
import urllib.request

import pytest

from labeling import server
from train import io as tio


def row(iid, ticker="AAA", channel="news", relevance=0.9, novelty=1.0):
    return {
        "item": {"id": iid, "source": "s", "channel": channel, "publisher": "p", "title": f"t{iid}", "text": "body",
                 "url": "http://x", "published_at": "2026-01-01T00:00:00Z", "ticker": ticker, "meta": {}},
        "company": {"ticker": ticker, "name": ticker + " Inc", "short_name": ticker, "cik": "1", "aliases": [], "ambiguous": False},
        "relevance": relevance, "cluster_id": iid, "novelty": novelty, "collected_at": "2026-01-01T00:00:00Z",
        "teacher": {"cls": 2}, "score": 0.99,
    }


def g(iid, ticker="AAA", channel="news", skipped=False):
    return {"id": iid, "ticker": ticker, "channel": channel, "skipped": skipped, "cls": 0}


def rng():
    return random.Random(1)


def test_excludes_labeled_low_relevance_non_novel():
    corpus = [row("a"), row("b", relevance=0.4), row("c", novelty=0.5), row("d"), row("e", relevance=0.5)]
    seen = set()
    for _ in range(30):
        r = server.sample_next(corpus, [g("d")], random.Random(_))
        seen.add(r["item"]["id"])
    assert seen <= {"a", "e"} and seen


def test_none_when_nothing_left():
    assert server.sample_next([row("a")], [g("a")], rng()) is None
    assert server.sample_next([], [], rng()) is None


def test_channel_group_balance_and_tie_goes_to_news():
    corpus = [row("n1"), row("s1", channel="social"), row("f1", channel="filings")]
    assert server.sample_next(corpus, [], rng())["item"]["id"] in {"n1", "f1"}
    gold = [g("x1"), g("x2")]  # news has more -> social
    assert server.sample_next(corpus, gold, rng())["item"]["id"] == "s1"
    gold = [g("x1", channel="social"), g("x2", channel="social")]
    assert server.sample_next(corpus, gold, rng())["item"]["id"] in {"n1", "f1"}


def test_falls_back_to_other_group_when_preferred_empty():
    corpus = [row("s1", channel="social")]
    assert server.sample_next(corpus, [], rng())["item"]["id"] == "s1"


def test_prefers_ticker_with_fewest_labels():
    corpus = [row("a1", "AAA"), row("a2", "AAA"), row("b1", "BBB"), row("c1", "CCC")]
    gold = [g("x1", "AAA"), g("x2", "AAA"), g("x3", "BBB")]
    for seed in range(10):
        assert server.sample_next(corpus, gold, random.Random(seed))["item"]["id"] == "c1"


def test_display_item_hides_teacher_and_scores():
    d = server.display_item(row("a"))
    assert "teacher" not in d and "score" not in d and "relevance" not in d
    assert d["ticker"] == "AAA" and d["company"] == "AAA"


def test_source_filter_serves_only_that_source(tmp_path, monkeypatch):
    mx = row("m1")
    mx["item"]["source"] = "marketaux"
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in [row("n1"), row("n2"), mx]), encoding="utf-8")
    monkeypatch.setattr(tio, "CORPUS", corpus)
    monkeypatch.setattr(tio, "GOLD", tmp_path / "gold.jsonl")
    st = server.State("tom", 10, use_worth=False, source="marketaux")
    for _ in range(5):
        assert st.next()["item"]["id"] == "m1"


def test_source_filter_still_blocks_texts_labeled_from_other_sources():
    other, mx = row("n1"), row("m1")
    mx["item"]["source"] = "marketaux"
    mx["item"]["title"] = other["item"]["title"]
    pick, _ = server.sample_pick([other, mx], [g("n1")], rng(), source="marketaux")
    assert pick is None


@pytest.fixture
def app(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in [row("n1", "AAA"), row("s1", "BBB", "social")]), encoding="utf-8")
    gold = tmp_path / "gold" / "labels.jsonl"
    monkeypatch.setattr(tio, "CORPUS", corpus)
    monkeypatch.setattr(tio, "GOLD", gold)
    state = server.State("tom", 10, use_worth=False)
    srv = server.make_server(state, 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, gold
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_index_served(app):
    base, _ = app
    st, body = get(base + "/")
    assert st == 200 and 'dir="rtl"' in body.decode()


def test_label_undo_round_trip(app):
    base, gold = app
    st, b = get(base + "/api/next")
    first = json.loads(b)["item"]
    assert first["id"] == "n1"  # tie -> news
    assert json.loads(b)["progress"]["done"] == 0

    st, res = post(base + "/api/label", {"id": "n1", "cls": -2, "stance": "none", "event": "EARNINGS_MISS",
                                         "relevant": True, "skipped": False, "note": "x"})
    assert st == 200 and res["progress"]["done"] == 1 and res["progress"]["news"] == 1
    assert res["item"]["id"] == "s1"
    rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["id"] == "n1" and rows[0]["cls"] == -2 and rows[0]["labeler"] == "tom"
    assert rows[0]["channel"] == "news" and rows[0]["ticker"] == "AAA" and rows[0]["event"] == "EARNINGS_MISS"

    st, res = post(base + "/api/undo", {})
    assert st == 200 and res["item"]["id"] == "n1" and res["progress"]["done"] == 0
    assert gold.read_text(encoding="utf-8") == ""


def test_skip_excluded_from_progress_and_not_resampled(app):
    base, gold = app
    st, res = post(base + "/api/label", {"id": "n1", "skipped": True})
    assert st == 200 and res["progress"]["done"] == 0
    assert json.loads(gold.read_text(encoding="utf-8"))["skipped"] is True
    assert res["item"]["id"] == "s1"
    post(base + "/api/label", {"id": "s1", "cls": 1, "stance": "bullish"})
    st, b = get(base + "/api/next")
    assert json.loads(b)["item"] is None


def test_undo_only_touches_session_rows(app):
    base, gold = app
    gold.parent.mkdir(parents=True, exist_ok=True)
    gold.write_text(json.dumps(g("old")) + "\n", encoding="utf-8")
    st, res = post(base + "/api/undo", {})
    assert st == 200 and "old" in gold.read_text(encoding="utf-8")


def test_validation_errors(app):
    base, _ = app
    assert post(base + "/api/label", {"id": "n1", "cls": 3})[0] == 400
    assert post(base + "/api/label", {"id": "n1"})[0] == 400
    assert post(base + "/api/label", {"id": "n1", "cls": 0, "stance": "meh"})[0] == 400
    assert post(base + "/api/label", {"id": "n1", "cls": 0, "event": "NOPE"})[0] == 400
    assert post(base + "/api/label", {"id": "zzz", "cls": 0})[0] == 400
    assert post(base + "/api/nope", {})[0] == 404
    # duplicate label rejected
    assert post(base + "/api/label", {"id": "n1", "cls": 0})[0] == 200
    assert post(base + "/api/label", {"id": "n1", "cls": 0})[0] == 400


def test_corrupt_gold_line_skipped(app):
    base, gold = app
    gold.parent.mkdir(parents=True, exist_ok=True)
    gold.write_text("{not json\n" + json.dumps(g("n1")) + "\n", encoding="utf-8")
    st, b = get(base + "/api/next")
    j = json.loads(b)
    assert st == 200 and j["item"]["id"] == "s1" and j["progress"]["done"] == 1


def test_gemini_offtarget_ab_cache_wins():
    teacher = [{"id": "a", "ticker": "AAA", "relevant": False, "reason": "about X"},
               {"id": "b", "ticker": "AAA", "relevant": False, "reason": "old"},
               {"id": "b", "ticker": "AAA", "relevant": True, "pv": 2}]           # newer prompt: on target
    ab = {"v2": {"c|BBB": {"relevant": False, "reason": "ab"}, "a|AAA": {"relevant": True}}}
    assert server.gemini_offtarget(teacher, ab) == {("c", "BBB"): "ab"}
    assert server.gemini_offtarget(teacher) == {("a", "AAA"): "about X"}


def test_review_queue_rules():
    corpus = [row(i) for i in ("a", "b", "c", "d", "e", "f")]
    off = {(i, "AAA"): "r" for i in ("a", "b", "c", "d", "e")}
    gold = [dict(g("a"), cls=2, relevant=True), dict(g("b"), cls=0), dict(g("c"), cls=-1, relevant=False),
            dict(g("d"), cls=1), dict(g("d"), cls=1, review="offtarget"),            # already reviewed
            dict(g("e"), cls=1), dict(g("e"), skipped=True, cls=None),                # last write: skipped
            dict(g("f"), cls=2)]                                                      # Gemini on target
    q, reviewed = server.review_queue(corpus, gold, off)
    assert [x[0]["item"]["id"] for x in q] == ["a"] and q[0][1]["cls"] == 2 and reviewed == 1


@pytest.fixture
def review_app(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row(i)) + "\n" for i in ("a", "b")), encoding="utf-8")
    gold = tmp_path / "gold.jsonl"
    gold.write_text("".join(json.dumps(dict(g(i), cls=2, event="OTHER", stance="none", note="n")) + "\n"
                            for i in ("a", "b")), encoding="utf-8")
    teacher = tmp_path / "teacher.jsonl"
    teacher.write_text(json.dumps({"id": "a", "ticker": "AAA", "relevant": False, "reason": "about X"}) + "\n")
    monkeypatch.setattr(tio, "CORPUS", corpus)
    monkeypatch.setattr(tio, "GOLD", gold)
    monkeypatch.setattr(tio, "TEACHER", teacher)
    monkeypatch.setattr(server, "AB_CACHE", tmp_path / "none.json")
    srv = server.make_server(server.State("tom", 10, "offtarget"), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", gold
    srv.shutdown()
    srv.server_close()


def test_review_mode_relabel_and_undo(review_app):
    base, gold = review_app
    res = json.loads(get(base + "/api/next")[1])
    assert res["item"]["id"] == "a" and res["review"]["prev"]["cls"] == 2
    assert res["review"]["gemini_reason"] == "about X" and res["progress"] == {
        "done": 0, "target": 1, "news": 1, "social": 0, "by_class": {}}
    assert post(base + "/api/label", {"id": "b", "cls": 0, "relevant": False})[0] == 400      # not queued
    assert post(base + "/api/label", {"id": "a", "skipped": True})[0] == 400
    st, res = post(base + "/api/label", {"id": "a", "cls": 0, "relevant": False, "note": "n"})
    assert st == 200 and res["item"] is None and res["progress"]["done"] == 1
    last = json.loads(gold.read_text(encoding="utf-8").splitlines()[-1])
    assert last["id"] == "a" and last["review"] == "offtarget" and last["relevant"] is False and last["cls"] == 0
    st, res = post(base + "/api/undo", {})
    assert res["item"]["id"] == "a" and res["review"]["prev"]["cls"] == 2
    assert len(gold.read_text(encoding="utf-8").splitlines()) == 2


# ---- dedup, worth ranking, pick + waste_recent ----

class StubWorth:
    ready = True

    def score(self, rows):
        import numpy as np
        return np.array([1.0 if r["item"]["id"].startswith("good") else 0.0 for r in rows])


def trow(iid, title, ticker="AAA", **kw):
    r = row(iid, ticker, **kw)
    r["item"]["title"] = title
    return r


def test_duplicate_text_key_never_served():
    corpus = [trow("a2", "Apple beats earnings"), trow("a1", "Apple beats earnings!"), trow("b1", "Other news here")]
    ids = {server.sample_next(corpus, [], random.Random(s))["item"]["id"] for s in range(30)}
    assert ids <= {"a1", "b1"} and "a2" not in ids
    # a text already labeled under another id is never served again
    gold = [g("a1")]
    ids = {server.sample_next(corpus, gold, random.Random(s))["item"]["id"] for s in range(30)}
    assert ids == {"b1"}


def test_sample_pick_modes():
    corpus = [trow(f"good{i}", f"alpha story {i}x") for i in range(5)] + [trow(f"bad{i}", f"beta story {i}y") for i in range(5)]
    # distinct keys: title_key strips non-letters so use letters
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    corpus = [trow(("good" if i < 5 else "bad") + str(i), f"{n} headline") for i, n in enumerate(names)]
    assert server.sample_pick(corpus, [], rng(), None)[1] == "random"
    assert server.sample_pick([], [], rng(), StubWorth())[0] is None
    r, how = server.sample_pick(corpus, [], rng(), StubWorth(), explore=1.0)
    assert how == "explore" and r is not None
    for s in range(20):
        r, how = server.sample_pick(corpus, [], random.Random(s), StubWorth(), explore=0.0)
        assert how == "worth" and r["item"]["id"].startswith("good")


def fake_embed(texts):
    import numpy as np
    out = []
    for t in texts:
        v = np.zeros(8)
        v[0 if "spam" in t else 1] = 1.0
        out.append(v)
    return np.array(out)


def test_label_writes_pick_and_waste_recent(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    corpus.write_text("".join(json.dumps(trow(f"n{i}", f"{n} headline")) + "\n" for i, n in enumerate(names)), encoding="utf-8")
    monkeypatch.setattr(tio, "CORPUS", corpus)
    monkeypatch.setattr(tio, "GOLD", tmp_path / "gold" / "labels.jsonl")
    monkeypatch.setattr(tio, "TEACHER", tmp_path / "none.jsonl")
    st = server.State("tom", 10, embed=fake_embed)
    res = st.next()
    assert res["progress"]["waste_recent"] is None
    outcomes = [True, True, False, False, False, False]   # skipped?
    for sk in outcomes:
        iid = res["item"]["id"]
        res = st.label({"id": iid, "cls": 0, "skipped": sk})
    rows = [json.loads(x) for x in tio.GOLD.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6 and all(r["pick"] == "random" for r in rows)   # model not ready -> random
    assert res["progress"]["waste_recent"] == pytest.approx(2 / 6)
    # undo still works
    res = st.undo()
    assert res["progress"]["waste_recent"] == pytest.approx(2 / 5)


def test_label_lands_on_served_ticker_row(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row("x", t)) + "\n" for t in ("AAA", "BBB")), encoding="utf-8")
    monkeypatch.setattr(tio, "CORPUS", corpus)
    monkeypatch.setattr(tio, "GOLD", tmp_path / "gold" / "labels.jsonl")
    monkeypatch.setattr(tio, "TEACHER", tmp_path / "none.jsonl")
    st = server.State("tom", 10, use_worth=False)
    bbb = [r for r in st.corpus() if r["item"]["ticker"] == "BBB"][0]
    assert st.next(bbb)["item"]["ticker"] == "BBB"
    st.label({"id": "x", "ticker": "BBB", "cls": 1})
    gold = [json.loads(x) for x in tio.GOLD.read_text(encoding="utf-8").splitlines()]
    assert [g["ticker"] for g in gold] == ["BBB"]
    st.undo()
    assert not tio.GOLD.read_text(encoding="utf-8").strip()
    st.label({"id": "x", "cls": 1})   # no ticker: first row with that id
    gold = [json.loads(x) for x in tio.GOLD.read_text(encoding="utf-8").splitlines()]
    assert [g["ticker"] for g in gold] == ["AAA"]
    st2 = server.State("tom", 10, use_worth=False)
    tio.GOLD.unlink()
    st2.label({"id": "x", "ticker": "ZZZ", "cls": 1})   # unknown ticker: fallback
    assert json.loads(tio.GOLD.read_text(encoding="utf-8").splitlines()[0])["ticker"] == "AAA"


def test_blind_queue_rules():
    corpus = [row(i) for i in "abcdef"]
    base = {"ticker": "AAA", "channel": "news", "skipped": False, "relevant": True}
    gold = [dict(base, id="a", cls=2), dict(base, id="b", cls=1), dict(base, id="c", cls=0),
            dict(base, id="d", cls=1), dict(base, id="d", cls=-1, review="blind"),   # already answered
            dict(base, id="e", cls=1), dict(base, id="f", cls=-1, relevant=False)]
    teacher = [{"id": "a", "ticker": "AAA", "cls": -1, "pv": 2}, {"id": "a", "ticker": "AAA", "cls": 1},
               {"id": "b", "ticker": "AAA", "cls": 2, "pv": 2},                      # same direction
               {"id": "c", "ticker": "AAA", "cls": -1, "pv": 2},                     # Tom is neutral
               {"id": "d", "ticker": "AAA", "cls": 1, "pv": 2},
               {"id": "e", "ticker": "AAA", "cls": -1, "relevant": False, "pv": 2},  # Gemini: off target
               {"id": "f", "ticker": "AAA", "cls": 1, "pv": 2}]                      # Tom: off target
    q, reviewed = server.blind_queue(corpus, gold, teacher)
    assert [x[0]["item"]["id"] for x in q] == ["a"] and reviewed == 1


def test_blind_mode_hides_both_labels(tmp_path, monkeypatch):
    corpus, gold, teacher = tmp_path / "c.jsonl", tmp_path / "g.jsonl", tmp_path / "t.jsonl"
    tio.append_jsonl(corpus, [row("a")])
    tio.append_jsonl(gold, [{"id": "a", "ticker": "AAA", "channel": "news", "skipped": False, "relevant": True,
                             "cls": 2, "note": "secret"}])
    tio.append_jsonl(teacher, [{"id": "a", "ticker": "AAA", "cls": -1, "reason": "secret", "pv": 2}])
    monkeypatch.setattr(tio, "CORPUS", corpus)
    monkeypatch.setattr(tio, "GOLD", gold)
    monkeypatch.setattr(tio, "TEACHER", teacher)
    st = server.State("tom", review="blind", use_worth=False)
    res = st.next()
    assert res["item"]["id"] == "a" and res["review"] == {"mode": "blind"}
    assert "secret" not in json.dumps(res) and res["progress"]["target"] == 1
    assert st.label({"id": "a", "ticker": "AAA", "cls": -1})["item"] is None
    last = list(tio.read_jsonl(gold))[-1]
    assert last["review"] == "blind" and last["cls"] == -1
    assert st.undo()["item"]["id"] == "a" and list(tio.read_jsonl(gold))[-1]["cls"] == 2

import numpy as np

from labeling.worth import WorthModel, is_waste, text_key


def r(iid, title, ticker="AAA", channel="news", text="body"):
    return {"item": {"id": iid, "ticker": ticker, "channel": channel, "title": title, "text": text},
            "company": {"ticker": ticker}, "relevance": 0.9}


def embed(texts):
    out = []
    for t in texts:
        v = np.zeros(4)
        v[0 if "spam" in t else 1] = 1.0
        out.append(v)
    return np.array(out)


def test_text_key_and_is_waste():
    assert text_key(r("1", "Apple Beats!")) == ("AAA", "apple beats")
    assert text_key(r("1", "x", channel="social", text="Hello World")) == ("AAA", "hello world")
    assert is_waste({"skipped": True}) and is_waste({"relevant": False})
    assert not is_waste({"cls": 1}) and not is_waste({"relevant": True, "skipped": False})


def make(n_waste, n_use):
    corpus, gold = [], []
    for i in range(n_waste):
        corpus.append(r(f"w{i}", f"spam {i}"))
        gold.append({"id": f"w{i}", "ticker": "AAA", "skipped": True})
    for i in range(n_use):
        corpus.append(r(f"u{i}", f"real {i}"))
        gold.append({"id": f"u{i}", "ticker": "AAA", "cls": 1, "relevant": True})
    return corpus, gold


def test_fit_false_under_threshold():
    corpus, gold = make(49, 60)
    m = WorthModel(embed)
    assert m.fit(corpus, gold, []) is False and not m.ready
    assert np.array_equal(m.score([r("x", "spam")]), np.ones(1))
    assert m.score([]).shape == (0,)
    corpus, gold = make(60, 49)
    assert WorthModel(embed).fit(corpus, gold, []) is False


def test_score_ranks_useful_above_waste():
    corpus, gold = make(60, 60)
    gold.append({"id": "u0", "ticker": "AAA", "review": "offtarget", "skipped": True})  # review ignored
    m = WorthModel(embed)
    assert m.fit(corpus, gold, []) is True and m.ready and m.n_train == 120
    s = m.score([r("t1", "spam new"), r("t2", "real new")])
    assert s[1] > s[0] and 0 <= s.min() and s.max() <= 1


def test_cache_keyed_by_id_and_ticker():
    m = WorthModel(embed)
    m._features([r("x", "spam", ticker="AAA"), r("x", "real", ticker="BBB")])
    assert set(m._cache) == {("x", "AAA"), ("x", "BBB")}


def test_text_key_hebrew_fallback():
    a, b = r("1", "חדשות על אפל"), r("2", "חדשות על גוגל")
    assert text_key(a) != text_key(b)
    assert text_key(a) == text_key(r("3", "  חדשות   על אפל "))
    assert text_key(r("4", "🚀🚀")) == ("AAA", "🚀🚀")
    assert text_key(r("5", "")) == ("AAA", "5")

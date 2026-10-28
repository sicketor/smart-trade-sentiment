from datetime import datetime, timedelta, timezone

import pytest

from sentiment import aggregate, config
from sentiment.meta import Meta, _llm5
from sentiment.types import ModelOutput, RawItem, ScoredItem

NOW = datetime(2026, 10, 26, 16, tzinfo=timezone.utc)


def mo(name, neg, neu, pos):
    return ModelOutput(name, (neg, neu, pos))


def test_heuristic_positive_news():
    r = Meta(models_dir=config.ROOT / "no-such-dir").predict(
        "news", [mo("finbert", .05, .1, .85), mo("absa", .02, .08, .9)],
        {"vader_compound": .6}, ["GUIDANCE_UP"])
    assert r.score > 0.4 and r.cls >= 1 and r.stance == "none"
    assert sum(r.probs5) == pytest.approx(1, abs=1e-3)


def test_heuristic_social_user_label_wins_stance():
    r = Meta(models_dir=config.ROOT / "x").predict(
        "social", [mo("fintwit", .3, .4, .3)], {"user_label": -1.0}, ["OTHER"])
    assert r.stance == "bearish" and r.score < 0


def test_llm_pulls_toward_its_class():
    m = Meta(models_dir=config.ROOT / "x")
    outs = [mo("finbert", .1, .8, .1)]
    base = m.predict("news", outs, {}, ["OTHER"])
    with_llm = m.predict("news", outs, {}, ["OTHER"], {"cls": -2, "stance": "none", "relevant": True})
    assert with_llm.score < base.score - 0.3 and with_llm.cls <= -1


def test_llm5_sums_to_one():
    for c in range(-2, 3):
        assert _llm5(c).sum() == pytest.approx(1)


def si(channel, score, hours_ago, source="google_news", publisher="Reuters", cid=None, stance="none"):
    it = RawItem(id=cid or f"{channel}{score}{hours_ago}", source=source, channel=channel, publisher=publisher,
                 title="t", text="", url="", published_at=NOW - timedelta(hours=hours_ago), ticker="X")
    return ScoredItem(item=it, relevance=1.0, cluster_id=it.id, novelty=1.0, outputs=[], features={},
                      events=["OTHER"], probs5=(0, 0, 1, 0, 0), score=score, cls=0, confidence=1.0,
                      stance=stance)


def test_grade_thresholds():
    assert [aggregate.grade(s) for s in (65, 30, 10, 9.9, -10, -59, -60)] == [3, 2, 1, 0, -1, -2, -3]
    assert aggregate.grade(None) is None


def test_shrinkage_and_formula():
    assert aggregate.channel_snapshot("news", [si("news", 1.0, 0)], NOW).score is None  # Σw 0.9 < 1
    items = [si("news", 1.0, 0, cid="a"), si("news", 1.0, 0, cid="b")]   # w = 0.9 (Reuters) each
    snap = aggregate.channel_snapshot("news", items, NOW)
    assert snap.n_effective == pytest.approx(1.8)
    assert snap.score == pytest.approx(100 * 1.8 / (1.8 + config.SHRINK_K["news"]), abs=0.1)
    many = [si("news", 1.0, 0, cid=str(i)) for i in range(60)]
    assert aggregate.channel_snapshot("news", many, NOW).score > 90


def test_low_data_and_composite_renormalizes():
    s = aggregate.summarize([si("news", 1.0, 0, cid=str(i)) for i in range(10)], NOW)
    assert s["channels"]["social"].score is None
    assert s["composite"].score == s["channels"]["news"].score
    assert aggregate.summarize([], NOW)["flags"] == ["low_data"]


def test_agreement_mixed():
    items = [si("news", 1.0, 0, cid=f"a{i}") for i in range(10)] + [si("news", -1.0, 0, cid=f"b{i}") for i in range(10)]
    s = aggregate.summarize(items, NOW)
    assert s["channels"]["news"].agreement == pytest.approx(0.0)
    assert "mixed" in s["flags"]


def test_bull_ratio_and_top_lists():
    items = [si("social", .6, 1, source="stocktwits", publisher="@a", cid=f"s{i}", stance="bullish") for i in range(3)]
    items.append(si("social", -.6, 1, source="stocktwits", publisher="@b", cid="neg", stance="bearish"))
    s = aggregate.summarize(items, NOW)
    assert s["channels"]["social"].bull_ratio == 0.75
    assert s["top_bearish"][0].item.id == "neg" and len(s["top_bullish"]) == 3


def test_attention_spike():
    assert aggregate.attention_z([100.0] * 29 + [400.0]) >= config.IN_FOCUS_Z
    assert aggregate.attention_z([100.0] * 3) is None
    ctx = {"wiki_views": [{"date": str(i), "views": 1000} for i in range(29)] + [{"date": "x", "views": 5000}]}
    s = aggregate.summarize([si("news", 1.0, 0, cid=str(i)) for i in range(5)], NOW, ctx)
    assert "in_focus" in s["flags"] and "hype" not in s["flags"]

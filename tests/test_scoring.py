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


def si(channel, score, hours_ago, source="google_news", publisher="Reuters", cid=None, stance="none",
       title="t", events=None, cls=None):
    # cls follows the score by default (aggregation v2 averages only the non-neutral items).
    it = RawItem(id=cid or f"{channel}{score}{hours_ago}", source=source, channel=channel, publisher=publisher,
                 title=title, text="", url="", published_at=NOW - timedelta(hours=hours_ago), ticker="X")
    c = max(-2, min(2, round(score * 2))) if cls is None else cls
    return ScoredItem(item=it, relevance=1.0, cluster_id=it.id, novelty=1.0, outputs=[], features={},
                      events=events or ["OTHER"], probs5=(0, 0, 1, 0, 0), score=score, cls=c, confidence=1.0,
                      stance=stance)


def test_grade_thresholds():
    assert [aggregate.grade(s) for s in (65, 30, 10, 9.9, -10, -59, -60)] == [3, 2, 1, 0, -1, -2, -3]
    assert aggregate.grade(None) is None


def test_shrinkage_and_formula():
    assert aggregate.channel_snapshot("news", [si("news", 1.0, 0)], NOW).score is None  # Σw 0.9 < 1
    # v2: 2 items are below MIN_ITEMS (they were scored before), so the formula is checked on 3.
    two = [si("news", 1.0, 0, cid="a"), si("news", 1.0, 0, cid="b")]
    assert aggregate.channel_snapshot("news", two, NOW).score is None
    items = two + [si("news", 1.0, 0, cid="c")]   # w = 0.9 (Reuters) each
    snap = aggregate.channel_snapshot("news", items, NOW)
    assert snap.n_effective == pytest.approx(2.7)
    assert snap.score == snap.raw == pytest.approx(100 * 2.7 / (2.7 + config.SHRINK_K["news"]), abs=0.1)
    assert snap.z is None and snap.pos_ratio == 1.0
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


STATS = {"mu": 5.0, "sd": 10.0, "n": 100, "cuts": [-1.6, -1.0, -0.4, 0.4, 1.0, 1.6]}


@pytest.mark.parametrize("title,events,expected", [
    ("Nvidia shares rise 5% in premarket", None, True),
    ("Why AMD Stock Is Down Today", None, True),
    ("Nvidia shares jump 5% after earnings beat", ["EARNINGS_BEAT"], False),
    ("Apple announces new iPhone", None, False),
])
def test_price_move_only(title, events, expected):
    item = si("news", 0.5, 0, title=title, events=events)
    assert aggregate.price_move_only(item) is expected
    assert (aggregate.item_weight(item, NOW) == 0.0) is expected


def test_price_move_only_news_channel_only():
    assert not aggregate.price_move_only(si("social", 0.5, 0, source="stocktwits", title="Why AMD Stock Is Down Today"))
    assert not aggregate.price_move_only(si("filings", 0.5, 0, source="sec", title="shares rise 5%"))


def test_price_move_headlines_drop_out():
    items = [si("news", 1.0, 0, cid=f"m{i}", title="Nvidia shares rise 5% today") for i in range(5)]
    s = aggregate.summarize(items, NOW)
    assert s["channels"]["news"].raw is None and s["channels"]["news"].n_effective == 0.0
    assert s["flags"] == ["low_data"] and s["top_bullish"] == []


def test_min_items_low_data():
    s = aggregate.summarize([si("news", 1.0, 0, cid="a"), si("news", 1.0, 0, cid="b")], NOW)
    assert s["channels"]["news"].raw is None and "low_data" in s["flags"]


def test_non_neutral_mean():
    items = [si("news", 0.0, 0, cid=f"n{i}") for i in range(8)] + [si("news", 1.0, 0, cid=f"p{i}") for i in range(2)]
    snap = aggregate.channel_snapshot("news", items, NOW)
    assert snap.n_effective == pytest.approx(9.0)          # coverage counts the neutrals
    assert snap.raw == pytest.approx(100 * 1.8 / (1.8 + config.SHRINK_K["news"]), abs=0.1)
    assert snap.agreement == 100.0 and snap.pos_ratio is None   # fewer than 3 non-neutral


def test_all_neutral():
    snap = aggregate.channel_snapshot("news", [si("news", 0.05, 0, cid=str(i)) for i in range(4)], NOW)
    assert snap.raw == 0.0 and snap.score == 0.0 and snap.agreement == 100.0 and snap.pos_ratio is None


def test_pos_ratio():
    items = [si("news", 1.0, 0, cid=f"p{i}") for i in range(3)] + [si("news", -1.0, 0, cid="n")]
    assert aggregate.channel_snapshot("news", items, NOW).pos_ratio == 0.75


@pytest.mark.parametrize("raw,display,z,g", [(25, 76.2, 2.0, 3), (5, 0.0, 0.0, 0), (-3, -38.0, -0.8, -1)])
def test_normalize_cross_section(raw, display, z, g):
    d, zz, gg = aggregate.normalize(raw, STATS, None)
    assert d == pytest.approx(display) and zz == pytest.approx(z) and gg == g


def test_normalize_with_history_and_fallbacks():
    d, z, g = aggregate.normalize(20, STATS, [20.0] * 60)   # lam 0.5, mu 12.5, sd sqrt(50)
    assert z == pytest.approx(1.061, abs=1e-3) and g == 2
    assert aggregate.normalize(20, STATS, [20.0] * 4)[1] == pytest.approx(1.5)   # < HIST_MIN_DAYS: ignored
    # a constant history cannot shrink sd below SD_FLOOR_FRAC * sd_c (= 2.5 here)
    lam = 10_000 / (10_000 + config.HIST_PRIOR_DAYS)
    expected = (21 - (lam * 20 + (1 - lam) * 5)) / (config.SD_FLOOR_FRAC * 10)
    assert aggregate.normalize(21, STATS, [20.0] * 10_000)[1] == pytest.approx(expected, abs=1e-3)
    assert aggregate.normalize(None, STATS, None) == (None, None, None)
    assert aggregate.normalize(33.33, None, None) == (33.3, None, 2)
    assert aggregate.normalize(33.33, {**STATS, "sd": 0.0}, None) == (33.3, None, 2)


def test_grade_from_cuts_boundaries():
    cuts = STATS["cuts"]
    zs = [-1.61, -1.6, -1.0, -0.4, 0.0, 0.4, 1.0, 1.6, 5.0]
    assert [aggregate.grade_from_cuts(z, cuts) for z in zs] == [-3, -2, -1, 0, 0, 1, 2, 3, 3]


def test_percentile_cuts():
    assert aggregate.percentile_cuts([float(i) for i in range(101)]) == pytest.approx([5, 15, 35, 65, 85, 95])


def test_load_norm(tmp_path):
    assert aggregate.load_norm(tmp_path / "missing.json") is None
    p = tmp_path / "norm.json"
    p.write_text('{"version": 1, "built_at": "x", "n_ticker_days": 10, "groups": {}}', encoding="utf-8")
    assert aggregate.load_norm(p)["version"] == 1
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert aggregate.load_norm(bad) is None


def test_summarize_with_norm_and_history():
    news = [si("news", 1.0, 0, cid=f"n{i}") for i in range(10)]
    social = [si("social", -0.5, 0, source="stocktwits", publisher="@a", cid=f"s{i}") for i in range(10)]
    norm = {"version": 1, "groups": {"news": STATS,
                                     "composite": {"mu": 10.0, "sd": 20.0, "n": 50, "cuts": STATS["cuts"]}}}
    history = {"social": [1.0] * 10, "raw_news": [None, 70.0]}   # 1 valid day < HIST_MIN_DAYS
    s = aggregate.summarize(news + social, NOW, history=history, norm=norm)
    ch, comp = s["channels"], s["composite"]
    assert ch["news"].raw == pytest.approx(75.0) and ch["news"].z == pytest.approx(7.0)
    assert ch["social"].z is None and ch["social"].score == ch["social"].raw   # no social group
    assert "hype" in s["flags"]   # today's 10 social items vs 1/day before
    alpha = dict(config.CHANNEL_WEIGHTS, social=config.CHANNEL_WEIGHTS["social"] * 0.5)
    mix = (alpha["news"] * ch["news"].raw + alpha["social"] * ch["social"].raw) / (alpha["news"] + alpha["social"])
    assert comp.raw == pytest.approx(mix, abs=0.05)      # raw mix, not the display scores
    assert comp.z == pytest.approx((comp.raw - 10) / 20, abs=1e-3)

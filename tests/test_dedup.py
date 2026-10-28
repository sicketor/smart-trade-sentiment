from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sentiment.dedup import cluster
from sentiment.types import RawItem

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def mk(i, title, channel="news", minutes=0, source="google_news"):
    return RawItem(id=i, source=source, channel=channel, publisher="p", title=title, text="", url="",
                   published_at=T0 + timedelta(minutes=minutes), ticker="NVDA")


def fake_embed(titles):
    return np.array([[1.0, 0.0] if "nvidia" in t.lower() else [0.0, 1.0] for t in titles])


def test_embed_cluster_three_sources():
    items = [mk("a", "Nvidia shares climb on chip demand", minutes=0, source="reuters"),
             mk("b", "Chipmaker Nvidia stock rallies as AI orders grow", minutes=5, source="yahoo_rss"),
             mk("c", "Why Nvidia is surging today", minutes=9, source="finviz")]
    r = cluster(items, fake_embed)
    assert len({v[0] for v in r.values()}) == 1
    assert [r[i][1] for i in "abc"] == [pytest.approx(1.0), pytest.approx(0.3536, abs=1e-4),
                                        pytest.approx(0.1925, abs=1e-4)]


def test_no_embed_keeps_separate():
    items = [mk("a", "Nvidia shares climb on chip demand"), mk("b", "Why Nvidia is surging today", minutes=1)]
    r = cluster(items, None)
    assert r["a"][0] != r["b"][0]


def test_exact_title_key_and_fuzz():
    items = [mk("a", "Nvidia Beats Estimates, Raises Outlook"),
             mk("b", "NVIDIA beats estimates; raises outlook!", minutes=1),
             mk("c", "Nvidia beats estimates raises its outlook sharply today", minutes=2)]
    r = cluster(items, None)
    assert len({v[0] for v in r.values()}) == 1


def test_order_by_published_at():
    items = [mk("late", "Nvidia beats estimates", minutes=10), mk("early", "Nvidia beats estimates", minutes=0)]
    r = cluster(items, None)
    assert r["early"][1] == 1.0 and r["late"][1] < 1.0


def test_social_never_uses_embed_and_channels_separate():
    calls = []

    def emb(t):
        calls.append(list(t))
        return np.ones((len(t), 2)) / np.sqrt(2)

    items = [mk("s1", "totally different words here", channel="social"),
             mk("s2", "another unrelated post entirely", channel="social", minutes=1),
             mk("n1", "Nvidia shares climb", channel="news")]
    r = cluster(items, emb)
    assert len(calls) == 1 and calls[0] == ["Nvidia shares climb"]
    assert r["s1"][0] != r["s2"][0]
    assert len({v[0] for v in r.values()}) == 3
    same = [mk("s", "Nvidia beats estimates", channel="social"), mk("n", "Nvidia beats estimates", channel="news")]
    r = cluster(same, None)
    assert r["s"][0] != r["n"][0]


def test_empty():
    assert cluster([], fake_embed) == {}

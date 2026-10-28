from datetime import datetime, timezone

import pytest

from sentiment.features import FEATURE_KEYS, extract
from sentiment.types import Company, RawItem

NVDA = Company("NVDA", "NVIDIA CORP", "Nvidia", 1045810)


def mk(title, text="", channel="news", **meta):
    return RawItem(id="x", source="google_news", channel=channel, publisher="p", title=title, text=text,
                   url="", published_at=datetime(2026, 1, 1, tzinfo=timezone.utc), ticker="NVDA", meta=meta)


def ev(title, text=""):
    return extract(mk(title, text), NVDA)[1]


@pytest.mark.parametrize("tag,title", [
    ("EARNINGS_BEAT", "Nvidia beats estimates on data center revenue"),
    ("EARNINGS_MISS", "Acme misses consensus EPS estimates"),
    ("GUIDANCE_UP", "Nvidia raises full-year guidance"),
    ("GUIDANCE_DOWN", "Acme cuts its outlook"),
    ("ANALYST_UPGRADE", "Morgan Stanley upgrades Nvidia"),
    ("ANALYST_DOWNGRADE", "Analyst downgrades Acme to sell"),
    ("PRICE_TARGET", "Wedbush sets price target at $200"),
    ("OFFERING_DILUTION", "Acme announces public offering of shares"),
    ("BUYBACK_DIVIDEND", "Nvidia announces $50 billion buyback"),
    ("M_AND_A", "Nvidia to acquire startup"),
    ("FDA_CLINICAL", "Acme gets FDA approval"),
    ("LEGAL_REGULATORY", "Acme hit with class action lawsuit"),
    ("INSIDER_TRADE", "Insider selling at Acme"),
    ("PRODUCT_CONTRACT", "Acme wins $2B defense contract"),
    ("MANAGEMENT_CHANGE", "Acme CEO steps down"),
    ("MACRO_SECTOR", "Fed holds rates steady"),
    ("SHORT_REPORT", "Hindenburg publishes report on Acme"),
])
def test_event_tags(tag, title):
    f, events, _ = extract(mk(title), NVDA)
    assert tag in events and f[f"evt_{tag}"] == 1.0
    assert f["evt_count"] == len(events)


def test_no_false_guidance_up():
    assert "GUIDANCE_UP" not in ev("Acme raises capital from investors")


def test_other_event():
    f, events, _ = extract(mk("Quiet trading day for widgets"), NVDA)
    assert events == ["OTHER"] and f["evt_count"] == 0


def test_keys_identical():
    items = [mk(""), mk("Nvidia beats estimates!", "Stock up 5% but outlook weak"),
             mk("$NVDA \U0001F680\U0001F680 to the moon", channel="social", user_label="Bullish", followers=1000)]
    keysets = [tuple(extract(i, NVDA)[0]) for i in items]
    assert len(set(keysets)) == 1 and set(keysets[0]) == set(FEATURE_KEYS)
    assert all(isinstance(v, float) for v in extract(items[0], NVDA)[0].values())


def test_numeric():
    f, _, _ = extract(mk("Nvidia beats estimates; shares jump 6%"), NVDA)
    assert f["eps_dir"] == 1.0 and f["pct_move"] == pytest.approx(0.2)
    f, _, _ = extract(mk("Acme misses estimates, stock falls 45%"), NVDA)
    assert f["eps_dir"] == -1.0 and f["pct_move"] == -1.0
    assert extract(mk("Wedbush raises Nvidia price target"), NVDA)[0]["pt_dir"] == 1.0
    assert extract(mk("Analyst price target cut for Acme"), NVDA)[0]["pt_dir"] == -1.0
    assert extract(mk("Nothing"), NVDA)[0]["pt_dir"] == 0.0
    assert extract(mk("x", insider_value=1e12), NVDA)[0]["insider_value"] == pytest.approx(1.0)
    assert extract(mk("x", insider_value=-999), NVDA)[0]["insider_value"] < 0


def test_structure_and_source():
    f, _, _ = extract(mk("Will Nvidia rally?", "Great quarter but the outlook is awful!!"), NVDA)
    assert f["has_contrast"] == 1.0 and f["after_contrast_vader"] < 0
    assert f["is_question"] == 1.0 and f["exclam"] == pytest.approx(0.4)
    assert f["forward_looking"] == pytest.approx(1 / 3) and f["has_body"] == 1.0 and f["is_news"] == 1.0
    f, _, _ = extract(mk("NVDA CEO says HUGE gains \U0001F680\U0001F4C9", channel="social",
                         user_label="Bearish", followers=99999), NVDA)
    assert f["caps_ratio"] > 0  # HUGE counts; NVDA and CEO are excluded
    assert f["emoji_bull"] == pytest.approx(0.2) and f["emoji_bear"] == pytest.approx(0.2)
    assert f["user_label"] == -1.0 and f["is_social"] == 1.0 and f["author_weight"] == 1.0
    assert f["has_body"] == 0.0


def test_lexicons_and_drivers():
    f, _, drivers = extract(mk("Nvidia upgraded, stock soars", "Strong profit but lawsuit risk may hurt as losses decline"), NVDA)
    assert f["lm_pos"] > 0 and f["lm_neg"] > 0 and f["lm_lit"] > 0 and f["lm_unc"] > 0 and f["lm_weak"] > 0
    assert extract(mk("Nvidia upgraded, stock soars"), NVDA)[0]["vader_compound"] > 0
    assert 1 <= len(drivers) <= 3
    assert extract(mk("Acme shares plunge amid bankruptcy"), NVDA)[0]["vader_compound"] < 0

from datetime import datetime, timezone

from sentiment.linking import other_tickers, relevance
from sentiment.types import Company, RawItem

NVDA = Company("NVDA", "NVIDIA CORP", "Nvidia", 1045810)
AI = Company("AI", "C3.ai, Inc.", "C3.ai", 1577526, ambiguous=True)


def mk(title, text="", source="google_news", **meta):
    return RawItem(id="x", source=source, channel="social" if source in {"reddit", "stocktwits"} else "news",
                   publisher="p", title=title, text=text, url="",
                   published_at=datetime(2026, 1, 1, tzinfo=timezone.utc), ticker="NVDA", meta=meta)


def test_title_mention():
    assert relevance(mk("Nvidia beats estimates"), NVDA) == 1.0


def test_listicle():
    assert relevance(mk("3 AI stocks to buy: NVDA, AMD, AVGO"), NVDA) == 0.4


def test_ambiguous_cashtag_and_bare():
    assert relevance(mk("$AI is going to moon", source="reddit"), AI) == 1.0
    assert relevance(mk("AI will change everything", source="reddit"), AI) == 0.0
    assert relevance(mk("Big day (NYSE: AI) shares jump"), AI) == 1.0


def test_stocktwits_unmentioned():
    assert relevance(mk("to the moon", source="stocktwits", symbols=["NVDA"]), NVDA) == 0.8
    assert relevance(mk("hm", source="stocktwits", symbols=["AMD", "TSLA"]), NVDA) == 0.5


def test_unmentioned_sources():
    assert relevance(mk("Market wrap", source="yahoo_rss"), NVDA) == 0.5
    assert relevance(mk("Market wrap", source="google_news"), NVDA) == 0.0


def test_body_only_and_others():
    assert relevance(mk("Chip moves today", "Nvidia was up"), NVDA) == 0.6
    assert relevance(mk("Chip moves today", "Nvidia was up; $AMD too"), NVDA) == 0.48
    assert relevance(mk("Nvidia vs peers", "$AMD $INTC $AVGO"), NVDA) == 0.3


def test_other_tickers():
    it = mk("Nvidia", "$NVDA $amd", symbols=["tsla", "NVDA"])
    assert other_tickers(it, NVDA) == {"AMD", "TSLA"}
    brk = Company("BRK-B", "BERKSHIRE", "Berkshire", 1)
    assert other_tickers(mk("x", "$BRK.B $KO"), brk) == {"KO"}


def test_foreign_exchange_symbol_is_not_a_mention():
    bac = Company("BAC", "BANK OF AMERICA CORP /DE/", "Bank of America", 70858)
    amc = Company("AMC", "AMC ENTERTAINMENT HOLDINGS", "AMC Entertainment", 1411579)
    bb = Company("BB", "BLACKBERRY Ltd", "BlackBerry", 1070235)
    assert relevance(mk("Verizon raises outlook", "Verizon Communications Inc (NYSE:VZ, XETRA:BAC) raised"), bac) == 0.0
    assert relevance(mk("How Is Amcor (ASX:AMC) Tracking After a Modest Year?"), amc) == 0.0
    assert relevance(mk("Merck KGaA (XTRA:BAC) deal", source="yahoo_rss"), bac) == 0.5   # aggregator default
    # dual listings still match by name, US exchange tags still match
    assert relevance(mk("Why BlackBerry Stock (TSX:BB) Might Rally"), bb) == 1.0
    assert relevance(mk("Bernstein upgrades (NYSE:AMC) (BMV:AMC) shares"), amc) == 1.0

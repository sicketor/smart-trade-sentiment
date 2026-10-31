from datetime import datetime, timezone

import pytest

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


C = Company("C", "CITIGROUP INC", "Citigroup", 831001, aliases=["Citi"], ambiguous=True)
L = Company("L", "LOEWS CORP", "Loews", 60086, ambiguous=True)
T = Company("T", "AT&T INC.", "AT&T", 732717, ambiguous=True)
DE = Company("DE", "DEERE & CO", "Deere", 315189, ambiguous=True, contextual=True)
MA = Company("MA", "Mastercard Inc", "Mastercard", 1141391, ambiguous=True, contextual=True)
GM = Company("GM", "General Motors Co", "General Motors", 1467858)


@pytest.mark.parametrize("title", ["Drax Group plc (DRX.L) Stock Price, News, Quote & History",
                                   "Karen L Carnahan Acquires Cintas Stock Options"])
def test_single_letter_needs_cashtag_or_name(title):
    assert relevance(mk(title), L) == 0.0


def test_single_letter_name_still_links():
    assert relevance(mk("What Just Happened To T-Mobile Stock?"), T) == 0.0
    assert relevance(mk("AT&T raises its outlook"), T) == 1.0
    assert relevance(mk("Citi beats on revenue"), C) == 1.0


def test_bare_symbol_boundaries():
    assert relevance(mk("AXA SA (GM.DE) Stock Price"), GM) == 0.0
    assert relevance(mk("GM-Hyundai talks"), GM) == 0.0
    assert relevance(mk("GM raises guidance"), GM) == 1.0
    assert relevance(mk("GM's new move"), GM) == 1.0


@pytest.mark.parametrize("title", ["MA Stock Clocks Best Day In 9 Months", "Why Did MA Stocks Surge?",
                                   "SGI vs. MA: Which Stock Is the Better Value?", "MA Maintained by Keybanc",
                                   "Earnings Beat Lifts Mastercard (MA)"])
def test_contextual_keeps_market_headlines(title):
    assert relevance(mk(title), MA) > 0


@pytest.mark.parametrize("title", ["Boston, MA startup raises funding", "Leeward Investments LLC MA trims position"])
def test_contextual_drops_other_meanings(title):
    assert relevance(mk(title), MA) == 0.0


def test_name_with_symbol_in_parentheses():
    assert relevance(mk("Deere (DE) Stock May Be Undervalued"), DE) == 1.0
    assert relevance(mk("Infineon AG (DE) board change"), DE) == 0.0
    assert relevance(mk("Gains after Q3 beat (DE:NYSE)"), DE) == 1.0


def test_option_quote_is_not_news():
    assert relevance(mk("GM Nov 2026 56.000 call (GM261127C00056000) Stock Price"), GM) == 0.0


TGT = Company("TGT", "TARGET CORP", "Target", 27419, weak_name=True)
NWS = Company("NWS", "NEWS CORP", "NEWS", 1564708, ambiguous=True, contextual=True, weak_name=True)
GE = Company("GE", "General Electric Co", "General Electric", 40545, ambiguous=True, contextual=True)


@pytest.mark.parametrize("title", ["Goldman Sachs cuts Kospi target by 25%", "Price Target Raised to $90",
                                   "Plug Power (PLUG) stock price, news, quote"])
def test_weak_name_alone_does_not_link(title):
    assert relevance(mk(title), TGT) == 0.0
    assert relevance(mk(title), NWS) == 0.0


@pytest.mark.parametrize("title", ["How Much Could You Lose In Target Stock?", "Target Corp cuts its outlook",
                                   "Target's Q3 Earnings Miss", "Target (TGT) lowers guidance", "TGT slides"])
def test_weak_name_links_with_proof(title):
    assert relevance(mk(title), TGT) > 0


LCID = Company("LCID", "Lucid Group, Inc.", "Lucid", 1811210, weak_name=True)
SNAP = Company("SNAP", "Snap Inc", "Snap", 1564408, ambiguous=True, contextual=True, weak_name=True)
FLEX = Company("FLEX", "FLEX LTD.", "FLEX", 866374, ambiguous=True, contextual=True, weak_name=True)
DOW = Company("DOW", "Dow Inc.", "DOW", 1751788, ambiguous=True, contextual=True, weak_name=True)
NDAQ = Company("NDAQ", "Nasdaq, Inc.", "Nasdaq", 1120193, weak_name=True)


@pytest.mark.parametrize("company,title", [(LCID, "Lucid Group Stock Short Interest Rises"),
                                           (LCID, "Lucid vs. Ferrari: Which Car Stock?"),
                                           (TGT, "Costco vs. Target: Which Retailer?"),
                                           (SNAP, "Reddit Slides 5%; Snap Sinks 4%"),
                                           (FLEX, "Flex Q2 Earnings Call Highlights"),
                                           (FLEX, "Flex shares rise on strong earnings"),
                                           (DOW, "Dow Inc raises its dividend"),
                                           (NDAQ, "Nasdaq Q3 earnings beat")])
def test_weak_name_real_headlines(company, title):
    assert relevance(mk(title), company) > 0


@pytest.mark.parametrize("company,title", [(SNAP, "Sensex, Nifty Snap 4-day Losing Streak"),
                                           (SNAP, "Indian shares snap losing run"),
                                           (FLEX, "Sales rose 6%, but profit fell 36% at Omega Flex"),
                                           (FLEX, "India's flex workspace market likely to double"),
                                           (DOW, "Dow rises 200 points as oil falls"),
                                           (DOW, "Dow Jones Biggest Decliners"),
                                           (NDAQ, "Nasdaq falls as chips slide"),
                                           (NDAQ, "SVRN Faces Nasdaq Delisting"),
                                           (TGT, "Citi's Price Target Falls")])
def test_weak_name_noise(company, title):
    assert relevance(mk(title), company) == 0.0


def test_other_company_with_our_name():
    assert relevance(mk("Portland General Electric (POR) Misses Q3 Earnings Estimates"), GE) == 0.0
    assert relevance(mk("General Electric raises guidance"), GE) == 1.0

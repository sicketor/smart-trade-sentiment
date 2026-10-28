from sentiment.text import normalize, for_social_model, split_sentences, title_key


def test_normalize_basic():
    assert normalize("<p>AT&amp;T  beats\n estimates</p>") == "AT&T beats estimates"


def test_normalize_datelines():
    assert normalize("(Reuters) - Nvidia rises") == "Nvidia rises"
    assert normalize("WASHINGTON (AP) — Fed holds rates") == "Fed holds rates"
    assert normalize("Apple (AAPL) - results") == "Apple (AAPL) - results"


def test_normalize_nfkc():
    assert normalize("ＡＢＣ") == "ABC"


def test_social_model():
    assert for_social_model("@bob check https://x.com/a?b=1 and www.y.com") == "@user check http and http"


def test_split_basic():
    assert split_sentences("Shares rose. Analysts cheered! Why? Nobody knows.") == [
        "Shares rose.", "Analysts cheered!", "Why?", "Nobody knows."]


def test_split_no_split_cases():
    for s in ["Apple Inc. Reported strong sales.", "Acme Corp. Said it will grow.", "Foo Co. Posted gains.",
              "The U.S. Economy grew.", "Mr. Smith left.", "Dr. Jones agreed.", "Bulls vs. Bears Battle on.",
              "Many, e.g. Apple Inc. Are up.", "It cost $1.25 Per share and rose 3.5% Today.",
              "Said J. Smith of Acme."]:
        assert len(split_sentences(s)) == 1, s


def test_split_empty():
    assert split_sentences("") == []


def test_title_key():
    assert title_key("The Fed: Rates to rise in 2025, at 5.5%!") == "fed rates rise 2025 5 5"
    assert title_key("") == ""

import httpx
import pytest

from sentiment import config, http as http_mod
from sentiment.http import Http, HttpError


def _http(monkeypatch, handler):
    monkeypatch.setattr(http_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "DEFAULT_HOST_INTERVAL", 0.0)
    monkeypatch.setattr(config, "HOST_MIN_INTERVAL", {})
    h = Http()
    h._client = httpx.Client(transport=httpx.MockTransport(handler))
    return h


def test_429_cooldown_skips_host_then_recovers(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.url.host)
        return httpx.Response(429 if req.url.host == "a.test" else 200, text="ok")

    h = _http(monkeypatch, handler)
    with pytest.raises(HttpError, match="^429 a.test"):
        h.get_text("https://a.test/x")
    assert calls.count("a.test") == 3
    with pytest.raises(HttpError, match="429 cooldown"):
        h.get_text("https://a.test/y")
    assert calls.count("a.test") == 3                     # no request during the cooldown
    assert h.get_text("https://b.test/z") == "ok"          # other hosts unaffected
    h._blocked["a.test"] = 0.0                             # cooldown over
    with pytest.raises(HttpError, match="^429 a.test"):
        h.get_text("https://a.test/x")
    assert calls.count("a.test") == 6


def test_no_cooldown_when_retry_succeeds_or_fails_otherwise(monkeypatch):
    seq = iter([429, 200, 500, 500, 500])

    def handler(req):
        return httpx.Response(next(seq), text="ok")

    h = _http(monkeypatch, handler)
    assert h.get_text("https://a.test/x") == "ok"
    with pytest.raises(HttpError, match="^500"):
        h.get_text("https://a.test/x")
    assert not h._blocked

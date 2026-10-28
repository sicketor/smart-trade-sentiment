import json
from datetime import datetime, timezone

import httpx
import pytest

from sentiment import config, llm
from sentiment.llm import GeminiLabeler
from sentiment.types import Company, RawItem

CO = Company("X", "X CORP", "Company X", 1)
NOW = datetime.now(timezone.utc)


def item(i, channel="news"):
    return RawItem(f"id{i}", "google_news", channel, "Pub", f"Title {i}", "body", "u", NOW, "X")


def good(i, **kw):
    d = {"id": f"id{i}", "cls": 1, "stance": "none", "event": "OTHER", "relevant": True, "reason": "ok"}
    d.update(kw)
    return d


def resp(status, objs=None):
    body = {}
    if objs is not None:
        body = {"candidates": [{"content": {"parts": [{"text": json.dumps(objs)}]}}]}
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://x"))


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_valid_and_invalid(monkeypatch):
    sent = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append((url, headers, json, timeout))
        return resp(200, [good(1), good(2, cls=3), good(3, event="NOPE"), good(4, stance="up"),
                          good("zzz"), good(5, cls="1"), "junk"])
    monkeypatch.setattr(llm.httpx, "post", post)
    lab = GeminiLabeler("KEY")
    out = lab.label(CO, [item(i) for i in range(1, 6)])
    assert list(out) == ["id1"]
    assert out["id1"] == {"cls": 1, "stance": "none", "event": "OTHER", "relevant": True, "reason": "ok"}
    url, headers, body, timeout = sent[0]
    assert config.GEMINI_MODELS[0] in url and headers["x-goog-api-key"] == "KEY" and timeout == 60
    assert body["generationConfig"]["temperature"] == 0
    assert "Target stock: X (Company X)" in body["contents"][0]["parts"][0]["text"]
    assert lab.available


def test_404_falls_to_next_model(monkeypatch):
    urls = []

    def post(url, **kw):
        urls.append(url)
        return resp(404) if len(urls) == 1 else resp(200, [good(1)])
    monkeypatch.setattr(llm.httpx, "post", post)
    out = GeminiLabeler("K").label(CO, [item(1)])
    assert "id1" in out and config.GEMINI_MODELS[1] in urls[1]


def test_all_429_exhausted_partial(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_BATCH", 2)
    calls = []

    def post(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return resp(200, [good(1), good(2)])
        return resp(429)
    monkeypatch.setattr(llm.httpx, "post", post)
    lab = GeminiLabeler("K")
    out = lab.label(CO, [item(i) for i in range(1, 7)])
    assert set(out) == {"id1", "id2"}
    assert lab.exhausted and not lab.available
    n = len(calls)
    assert lab.label(CO, [item(1)]) == {} and len(calls) == n


def test_other_error_tries_next_model_then_skips_batch(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_BATCH", 1)
    # batch 1: 500 on both models -> skipped; batch 2: 500 then the fallback model answers
    seq = iter([resp(500), resp(500), resp(500), resp(200, [good(2)])])
    monkeypatch.setattr(llm.httpx, "post", lambda url, **kw: next(seq))
    lab = GeminiLabeler("K")
    out = lab.label(CO, [item(1), item(2)])
    assert list(out) == ["id2"] and not lab.exhausted


def test_network_error_skips(monkeypatch):
    def post(url, **kw):
        raise httpx.ConnectError("boom")
    monkeypatch.setattr(llm.httpx, "post", post)
    assert GeminiLabeler("K").label(CO, [item(1)]) == {}


def test_missing_key(monkeypatch):
    def post(*a, **k):
        raise AssertionError("no request expected")
    monkeypatch.setattr(llm.httpx, "post", post)
    lab = GeminiLabeler()
    assert lab.available is False
    assert lab.label(CO, [item(1)]) == {}


# ---- OpenAICompatLabeler (DeepSeek / Groq) ----
from sentiment.llm import OpenAICompatLabeler


def chat(status, content=None):
    body = {} if content is None else {"choices": [{"message": {"content": content}}]}
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://x"))


def test_compat_labels_and_headers(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "DSKEY")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    sent = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append((url, headers, json))
        return chat(200, __import__("json").dumps({"labels": [good(1), good(2)]}))
    monkeypatch.setattr(llm.httpx, "post", post)
    lab = OpenAICompatLabeler("deepseek")
    out = lab.label(CO, [item(1), item(2)])
    assert set(out) == {"id1", "id2"} and lab.name == "deepseek"
    url, headers, body = sent[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert headers == {"Authorization": "Bearer DSKEY"}
    assert body["model"] == "deepseek-chat" and body["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("content", ['```json\n[{"id":"id1","cls":1,"stance":"none","event":"OTHER","relevant":true,"reason":"r"}]\n```',
                                     '[{"id":"id1","cls":1,"stance":"none","event":"OTHER","relevant":true,"reason":"r"}]',
                                     '{"x":[{"id":"id1","cls":1,"stance":"none","event":"OTHER","relevant":true,"reason":"r"}]}'])
def test_compat_content_shapes(monkeypatch, content):
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: chat(200, content))
    assert list(OpenAICompatLabeler("groq", api_key="K").label(CO, [item(1)])) == ["id1"]


def test_compat_unparsable(monkeypatch):
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: chat(200, "not json"))
    lab = OpenAICompatLabeler("groq", api_key="K")
    assert lab.label(CO, [item(1)]) == {} and not lab.exhausted


@pytest.mark.parametrize("code", [401, 402, 403])
def test_compat_auth_exhausts(monkeypatch, code):
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: chat(code))
    lab = OpenAICompatLabeler("deepseek", api_key="K")
    assert lab.label(CO, [item(1)]) == {} and lab.exhausted


def test_compat_three_429_exhausts(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: chat(429))
    lab = OpenAICompatLabeler("groq", api_key="K")
    assert lab.label(CO, [item(1)]) == {} and lab.exhausted and sleeps == [2, 8]


def test_compat_retry_then_ok_and_other_4xx(monkeypatch):
    monkeypatch.setattr(llm, "_sleep", lambda s: None)
    seq = iter([chat(500), chat(200, '{"labels":[]}')])
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: next(seq))
    lab = OpenAICompatLabeler("groq", api_key="K")
    assert lab._call("p") == [] and not lab.exhausted
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: chat(400))
    assert lab._call("p") is None and not lab.exhausted


def test_compat_key_not_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    assert OpenAICompatLabeler("deepseek").available is False


def test_compat_unknown_provider():
    with pytest.raises(ValueError):
        OpenAICompatLabeler("nope")


def test_prompt_ab_providers_parsing():
    from eval.prompt_ab import parse_providers, parse_versions
    assert parse_providers("") == [] and parse_providers("groq, deepseek") == ["groq", "deepseek"]
    with pytest.raises(SystemExit):
        parse_providers("groq,bogus")
    assert parse_versions("") == []

"""Gemini batch labeler, used as teacher / arbiter. See SPEC-he.md §5."""
from __future__ import annotations

import json
import logging
import os
import time

import httpx

from . import config
from .types import Company, RawItem

log = logging.getLogger(__name__)

EVENTS = ("EARNINGS_BEAT", "EARNINGS_MISS", "GUIDANCE_UP", "GUIDANCE_DOWN", "ANALYST_UPGRADE",
          "ANALYST_DOWNGRADE", "PRICE_TARGET", "OFFERING_DILUTION", "BUYBACK_DIVIDEND", "M_AND_A",
          "FDA_CLINICAL", "LEGAL_REGULATORY", "INSIDER_TRADE", "PRODUCT_CONTRACT", "MANAGEMENT_CHANGE",
          "MACRO_SECTOR", "SHORT_REPORT", "OTHER")
STANCES = ("bullish", "bearish", "none")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_HEAD = """You are a senior equity analyst labeling text for a stock-sentiment model.
Target stock: {ticker} ({company_name}).
For EACH item, judge the likely effect of the information on {ticker} shareholders specifically — not on other companies mentioned, not on the market in general.
cls: -2 = strongly negative for {ticker} (e.g. guidance cut, big miss, dilutive offering, fraud/probe, failed trial), -1 = negative, 0 = neutral / no clear effect / not about {ticker}, 1 = positive, 2 = strongly positive (e.g. big beat with raised guidance, acquisition at a premium, major contract, FDA approval).
Rules: forward-looking information (guidance, outlook) outweighs past results; a clause after "but/however" usually dominates; questions, listicles and generic market commentary are 0 unless they state a clear {ticker}-specific view; for social posts judge the author's position on {ticker} (sarcasm counts as its real meaning).
"""
_TAIL = """stance: for social posts, "bullish" or "bearish" if the author clearly expects {ticker} to rise or fall, else "none"; for news always "none".
event: one of EARNINGS_BEAT, EARNINGS_MISS, GUIDANCE_UP, GUIDANCE_DOWN, ANALYST_UPGRADE, ANALYST_DOWNGRADE, PRICE_TARGET, OFFERING_DILUTION, BUYBACK_DIVIDEND, M_AND_A, FDA_CLINICAL, LEGAL_REGULATORY, INSIDER_TRADE, PRODUCT_CONTRACT, MANAGEMENT_CHANGE, MACRO_SECTOR, SHORT_REPORT, OTHER.
relevant: false if the item is not really about {ticker}.
reason: at most 15 words, English.
Return a JSON array with one object per item, same ids.
Items:
{items_json}"""
# v2 (SPEC §0.6): aligned to Tom's gold labels, where v1 called clear {ticker}-specific items 0.
_V2_RULES = """Calibration: do not hide behind 0. An item about {ticker} with any clear direction is at least +-1: a deal or acquisition by {ticker}, an analyst action, unusual options bets, a poster who bought shares or calls, a price call or a complaint all have a direction. Use +-2 for major news (large acquisition, strong beat or raise, big contract, probe) and for posts with strong conviction. Keep 0 for items with no direction for {ticker}.
"""
# v3: rubric + price-impact question + about/relevance gate + per-batch kNN examples (shots.py)
_V3_HEAD = """You are a senior equity analyst labeling text for a stock-sentiment model.
Target stock: {ticker} ({company_name}).
For EACH item answer one question: if a {ticker} shareholder read only this item, would they expect {ticker}'s share price to go up or down over the next few days because of it, and how strongly? Judge only {ticker}: not other companies mentioned, not the market in general.
Per item:
1. about: "primary" if {ticker} is the main subject; "mention" if {ticker} appears but the item is mainly about something else; "none" if the item says nothing about {ticker} (another company, a ticker collision, a generic market list).
2. reason: at most 15 words, English: the {ticker}-specific fact or view and its direction.
3. cls on this rubric:
 +2 major positive for {ticker}: beat and raised guidance, acquired at a premium, large contract or approval, strong-conviction bullish post (big position, loading up).
 +1 positive: analyst upgrade or target raise, small beat, new product or deal, poster bought shares or calls, bullish price call.
  0 no direction for {ticker}: about is "none"; pure questions; schedules ("will report on ..."); holder lists and routine filings; mixed news with no clear net effect.
 -1 negative: downgrade or target cut, small miss, lawsuit, poster sold or bought puts, complaint, bearish price call.
 -2 major negative: guidance cut, big miss, dilutive offering, fraud or probe, failed trial, strong-conviction bearish post.
Rules: about "none" always means cls 0. For about "mention", judge only what the item says about {ticker}; if that has no direction, cls 0. Do not hide behind 0: an item about {ticker} with any clear direction is at least +-1. Forward-looking information outweighs past results; a clause after "but/however" usually dominates; for social posts judge the author's position on {ticker} (sarcasm counts as its real meaning).
"""
_V3_TAIL = """stance: for social posts, "bullish" or "bearish" if the author clearly expects {ticker} to rise or fall, else "none"; for news always "none".
event: one of """ + ", ".join(EVENTS) + """.
relevant: false exactly when about is "none".
Return a JSON array with one object per item, same ids.
Items:
{items_json}"""
PROMPT_V3 = _V3_HEAD + "{examples}" + _V3_TAIL
PROMPT_V1 = _HEAD + _TAIL
PROMPT_V2 = _HEAD + _V2_RULES + "{examples}" + _TAIL
PROMPT_VERSION = 2
PROMPT = PROMPT_V2
FEWSHOT = config.ROOT / "sentiment" / "fewshot.json"


def examples_block(path=FEWSHOT) -> str:
    """Gold examples (built by train.fewshot from gold train buckets only) rendered for the v2 prompt."""
    try:
        exs = json.loads(open(path, encoding="utf-8").read())
    except (OSError, ValueError):
        return ""
    lines = [f'- [{e["channel"]}] target {e["ticker"]}: {json.dumps(e["text"], ensure_ascii=False)} -> cls {e["cls"]}'
             for e in exs]
    return ("Examples labeled by our analyst (other stocks; apply the same standard):\n"
            + "\n".join(lines) + "\n") if lines else ""

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "STRING"},
            "cls": {"type": "INTEGER"},
            "stance": {"type": "STRING", "enum": list(STANCES)},
            "event": {"type": "STRING", "enum": list(EVENTS)},
            "relevant": {"type": "BOOLEAN"},
            "reason": {"type": "STRING"},
        },
        "required": ["id", "cls", "stance", "event", "relevant", "reason"],
    },
}
ABOUTS = ("primary", "mention", "none")
SCHEMA_V3 = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {**SCHEMA["items"]["properties"], "about": {"type": "STRING", "enum": list(ABOUTS)}},
        "required": SCHEMA["items"]["required"] + ["about"],
        "propertyOrdering": ["id", "about", "reason", "cls", "stance", "event", "relevant"],
    },
}


class GeminiLabeler:
    def __init__(self, api_key: str | None = None, prompt_version: int = PROMPT_VERSION, shots=None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self._key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        self.exhausted = False
        self.prompt_version = prompt_version
        self._template = {1: PROMPT_V1, 3: PROMPT_V3}.get(prompt_version, PROMPT_V2)
        self._schema = SCHEMA_V3 if prompt_version == 3 else SCHEMA
        self._examples = "" if prompt_version == 1 else examples_block()
        self._shots = None
        if prompt_version == 3:
            if shots is None:
                from .shots import ShotPool
                shots = ShotPool.load()
            self._shots = shots

    @property
    def available(self) -> bool:
        return bool(self._key) and not self.exhausted

    def _call(self, prompt: str, schema: dict | None = None) -> list | None:
        """Try models in order. Returns the parsed list, or None to skip the batch (all-429 sets exhausted)."""
        statuses = []
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                     "responseSchema": schema or SCHEMA}}
        for model in config.GEMINI_MODELS:
            try:
                r = httpx.post(URL.format(model=model), headers={"x-goog-api-key": self._key},
                               json=body, timeout=60)
            except httpx.HTTPError as e:
                log.warning("gemini %s request failed: %s", model, type(e).__name__)
                statuses.append(0)
                continue
            statuses.append(r.status_code)
            if r.status_code in (404, 429):
                continue
            if r.status_code != 200:
                log.warning("gemini %s returned HTTP %s", model, r.status_code)
                continue
            try:
                data = json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            except (KeyError, IndexError, ValueError, TypeError):
                log.warning("gemini %s returned an unparsable body", model)
                return None
            return data if isinstance(data, list) else None
        if statuses and all(s == 429 for s in statuses):
            self.exhausted = True
        return None

    def _valid(self, o, known: set[str]) -> bool:
        if self.prompt_version == 3 and (not isinstance(o, dict) or o.get("about") not in ABOUTS):
            return False
        return (isinstance(o, dict) and o.get("id") in known
                and isinstance(o.get("cls"), int) and not isinstance(o["cls"], bool) and -2 <= o["cls"] <= 2
                and o.get("stance") in STANCES and o.get("event") in EVENTS
                and isinstance(o.get("relevant"), bool))

    def label(self, company: Company, items: list[RawItem]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        n = config.GEMINI_BATCH
        for i in range(0, len(items), n):
            if not self.available:
                break
            batch = items[i:i + n]
            items_json = json.dumps([{"id": it.id, "source": it.source, "type": it.channel,
                                      "title": it.title, "text": it.text[:500]} for it in batch],
                                    ensure_ascii=False)
            prompt = self._template.format(ticker=company.ticker, company_name=company.short_name,
                                           items_json=items_json,
                                           examples=self._shots.block(batch) if self._shots else self._examples)
            data = self._call(prompt, self._schema)
            if not data:
                continue
            known = {it.id for it in batch}
            for o in data:
                if self._valid(o, known):
                    r = {"cls": o["cls"], "stance": o["stance"], "event": o["event"],
                         "relevant": o["relevant"], "reason": str(o.get("reason", ""))}
                    if self.prompt_version == 3:
                        r["about"] = o["about"]
                        if o["about"] == "none":
                            r["cls"], r["relevant"] = 0, False
                        else:
                            r["relevant"] = True
                    out[o["id"]] = r
        return out


_sleep = time.sleep
_SUFFIX = ('\n\nReturn ONLY a JSON object of the form {"labels": [ ... ]} where each element has the fields '
           'id, cls, stance, event, relevant, reason (the item format above).')


class OpenAICompatLabeler(GeminiLabeler):
    """Same prompt and validation as GeminiLabeler, over an OpenAI-compatible chat API (DeepSeek, Groq)."""

    def __init__(self, provider: str, api_key: str | None = None, prompt_version: int = PROMPT_VERSION,
                 model: str | None = None):
        cfg = config.LLM_PROVIDERS.get(provider)
        if cfg is None:
            raise ValueError(f"unknown provider: {provider!r}")
        try:
            from dotenv import load_dotenv
            load_dotenv(config.ROOT / ".env")
        except ImportError:
            pass
        key = api_key or os.environ.get(cfg["key_env"]) or ""
        super().__init__(api_key=key, prompt_version=prompt_version)
        self._key = key  # never fall back to GEMINI_API_KEY
        self._base, self._model = cfg["base"], model or cfg["model"]
        self.name = provider
        self.usage = {"in": 0, "out": 0}   # tokens billed so far (the API's "usage" field)

    def _call(self, prompt: str, schema: dict | None = None) -> list | None:
        body = {"model": self._model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt + _SUFFIX}]}
        n429 = 0
        for attempt in range(3):
            try:
                r = httpx.post(f"{self._base}/chat/completions",
                               headers={"Authorization": f"Bearer {self._key}"}, json=body, timeout=120)
            except httpx.HTTPError as e:
                log.warning("%s request failed: %s", self.name, type(e).__name__)
            else:
                code = r.status_code
                if code in (401, 402, 403):
                    log.warning("%s returned HTTP %s", self.name, code)
                    self.exhausted = True
                    return None
                if code == 429:
                    n429 += 1
                elif code >= 500:
                    log.warning("%s returned HTTP %s", self.name, code)
                elif code != 200:
                    log.warning("%s returned HTTP %s", self.name, code)
                    return None
                else:
                    return self._parse(r)
            if attempt < 2:
                _sleep((2, 8)[attempt])
        if n429 == 3:
            self.exhausted = True
        return None

    def _parse(self, r) -> list | None:
        try:
            u = r.json().get("usage") or {}
            self.usage["in"] += int(u.get("prompt_tokens", 0))
            self.usage["out"] += int(u.get("completion_tokens", 0))
        except (ValueError, TypeError, AttributeError):
            pass
        try:
            content = r.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else ""
                content = content.rsplit("```", 1)[0]
            data = json.loads(content)
        except (KeyError, IndexError, ValueError, TypeError, AttributeError):
            log.warning("%s returned an unparsable body", self.name)
            return None
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("labels"), list):
                return data["labels"]
            return next((v for v in data.values() if isinstance(v, list)), None)
        return None

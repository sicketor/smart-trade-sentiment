"""Shared HTTP client: per-host throttle, retries, UA handling."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from . import config

log = logging.getLogger(__name__)
_BACKOFF = (2.0, 5.0)


class HttpError(Exception):
    pass


class Http:
    def __init__(self) -> None:
        self._client = httpx.Client(follow_redirects=True, timeout=config.HTTP_TIMEOUT)
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._last: dict[str, float] = {}
        self._blocked: dict[str, float] = {}   # host -> monotonic time its 429 cooldown ends

    def _check_cooldown(self, host: str, label: str) -> None:
        if self._blocked.get(host, 0.0) > time.monotonic():
            raise HttpError(f"429 cooldown {label}")

    def _throttle(self, host: str) -> None:
        with self._guard:
            lock = self._locks.setdefault(host, threading.Lock())
        interval = config.HOST_MIN_INTERVAL.get(host, config.DEFAULT_HOST_INTERVAL)
        with lock:
            wait = self._last.get(host, 0.0) + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def _request(self, url: str, ua: str | None, params: dict | None) -> httpx.Response:
        parsed = urlparse(url)
        host = parsed.netloc
        label = host + parsed.path
        headers = {"User-Agent": ua or config.BROWSER_UA}
        last = "request failed"
        rate_limited = False
        for attempt in range(3):
            self._check_cooldown(host, label)
            self._throttle(host)
            try:
                r = self._client.get(url, params=params, headers=headers)
                rate_limited = r.status_code == 429
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"{r.status_code} {label}"
                elif r.status_code >= 400:
                    raise HttpError(f"{r.status_code} {label}")
                else:
                    return r
            except httpx.TimeoutException:
                rate_limited = False
                last = f"timeout {label}"
            except httpx.TransportError as e:
                rate_limited = False
                last = f"{type(e).__name__} {label}"
            if attempt < 2:
                time.sleep(_BACKOFF[attempt])
        if rate_limited:   # still 429 after the retries: skip the host for a while instead of re-waiting per request
            with self._guard:
                self._blocked[host] = time.monotonic() + config.HOST_COOLDOWN_429
            log.warning("%s still 429 after retries; skipping it for %.0f s", host, config.HOST_COOLDOWN_429)
        raise HttpError(last)

    def get_text(self, url: str, *, ua: str | None = None, params: dict | None = None) -> str:
        return self._request(url, ua, params).text

    def get_bytes(self, url: str, *, ua: str | None = None, params: dict | None = None) -> bytes:
        """Raw body, for XML feeds: feedparser then honours the declared encoding."""
        return self._request(url, ua, params).content

    def get_json(self, url: str, *, ua: str | None = None, params: dict | None = None) -> Any:
        r = self._request(url, ua, params)
        try:
            return r.json()
        except ValueError as e:
            raise HttpError(f"bad json {urlparse(url).netloc}{urlparse(url).path}") from e

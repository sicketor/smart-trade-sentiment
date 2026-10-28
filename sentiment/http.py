"""Shared HTTP client: per-host throttle, retries, UA handling."""
from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from . import config

_BACKOFF = (2.0, 5.0)


class HttpError(Exception):
    pass


class Http:
    def __init__(self) -> None:
        self._client = httpx.Client(follow_redirects=True, timeout=config.HTTP_TIMEOUT)
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._last: dict[str, float] = {}

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
        for attempt in range(3):
            self._throttle(host)
            try:
                r = self._client.get(url, params=params, headers=headers)
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"{r.status_code} {label}"
                elif r.status_code >= 400:
                    raise HttpError(f"{r.status_code} {label}")
                else:
                    return r
            except httpx.TimeoutException:
                last = f"timeout {label}"
            except httpx.TransportError as e:
                last = f"{type(e).__name__} {label}"
            if attempt < 2:
                time.sleep(_BACKOFF[attempt])
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

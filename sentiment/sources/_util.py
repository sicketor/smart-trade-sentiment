"""Helpers shared by the source adapters."""
from __future__ import annotations

import calendar
import hashlib
import html
import re
from datetime import datetime, timezone

import feedparser
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from ..types import RawItem


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def item_id(source: str, url: str, title: str) -> str:
    return sha1(url) if url else sha1(f"{source}\n{title}")


def strip_html(s: str | None) -> str:
    if not s:
        return ""
    if "<" in s:
        try:
            tree = HTMLParser(s)
            if tree.body is not None:
                s = tree.body.text(separator=" ", strip=True)
        except Exception:
            s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def utc_from_epoch(ts: float) -> datetime:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def parse_iso(s: str | None) -> datetime | None:
    """ISO-8601 -> aware UTC (naive input assumed UTC)."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def entry_dt(entry) -> datetime | None:
    """Date of a feedparser entry as aware UTC."""
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            return datetime.fromtimestamp(calendar.timegm(st), tz=timezone.utc)
    return None


def feed_entries(payload: str) -> list:
    return list(feedparser.parse(payload).entries)


def account_age_days(created: str | None, now: datetime | None = None) -> int | None:
    dt = parse_iso(created)
    if dt is None:
        return None
    return max(0, ((now or datetime.now(timezone.utc)) - dt).days)


def finish(items: list[RawItem], since: datetime) -> list[RawItem]:
    """Drop items older than `since` and duplicate ids (first wins)."""
    seen: set[str] = set()
    out = []
    for it in items:
        if it.published_at < since or it.id in seen:
            continue
        seen.add(it.id)
        out.append(it)
    return out

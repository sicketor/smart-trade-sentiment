"""Local gold-set labeling tool (SPEC §9.1): stdlib HTTP server + one Hebrew RTL page.

    python -m labeling.server [--port 8765] [--labeler tom] [--target 500] [--no-browser] [--review offtarget]
                              [--source marketaux]

Never exposes teacher labels or model scores (anti-anchoring). Binds to 127.0.0.1 only.
--review offtarget (SPEC §0.6) is the one exception: it re-shows gold items Tom gave a direction that Gemini
called "not about the stock", with Gemini's reason, so Tom can set them to 0 or keep his label. Each decision
is a new gold row with "review": "offtarget"; readers take the last row per id.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import webbrowser
import zlib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from labeling.worth import WorthModel, is_waste, text_key
from sentiment import config
from sentiment.features import EVENT_TAGS
from train import io as tio

STATIC = Path(__file__).parent / "static"
EVENTS = [*EVENT_TAGS, "OTHER"]
STANCES = ("bullish", "bearish", "none")
MIN_RELEVANCE = 0.5
AB_CACHE = config.ROOT / "runs" / "prompt_ab.json"


def _group(channel: str) -> str:
    return "social" if channel == "social" else "news"


def _is_labeled(row: dict) -> bool:
    return not row.get("skipped")


def _read_gold(path) -> list[dict]:
    """GOLD rows; corrupt lines are skipped."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and "id" in d:
                rows.append(d)
    return rows


def _read_corpus(path) -> list[dict]:
    if not Path(path).exists():
        return []
    return list(tio.read_jsonl(path))


def sample_next(corpus_rows, gold_rows, rng: random.Random) -> dict | None:
    """Pick the next corpus row to label, or None. Pure function."""
    return sample_pick(corpus_rows, gold_rows, rng)[0]


def sample_pick(corpus_rows, gold_rows, rng: random.Random, worth=None, explore: float = 0.1,
                source: str | None = None) -> tuple[dict | None, str]:
    """Pick the next row and how: "random", "worth" (top 30% by the worth model) or "explore". Pure given rng.
    Never serves a text whose (ticker, title key) was already labeled or is duplicated among candidates.
    `source` limits the candidates to one adapter; labeled texts from every source still block repeats."""
    seen = {g["id"] for g in gold_rows}
    seen_keys = {text_key(r) for r in corpus_rows if (r.get("item") or {}).get("id") in seen}
    counts = {"news": 0, "social": 0}
    by_ticker: dict[tuple[str, str], int] = {}
    for g in gold_rows:
        if not _is_labeled(g):
            continue
        grp = _group(g.get("channel", "news"))
        counts[grp] += 1
        k = (grp, g.get("ticker", ""))
        by_ticker[k] = by_ticker.get(k, 0) + 1

    uniq: dict[tuple[str, str], dict] = {}
    for r in corpus_rows:
        it = r.get("item") or {}
        if it.get("id") in seen or r.get("relevance", 0) < MIN_RELEVANCE or r.get("novelty") != 1.0:
            continue
        if source and it.get("source") != source:
            continue
        k = text_key(r)
        if k in seen_keys:
            continue
        if k not in uniq or it["id"] < uniq[k]["item"]["id"]:
            uniq[k] = r
    cands = {"news": [], "social": []}
    for r in uniq.values():
        cands[_group(r["item"].get("channel", "news"))].append(r)

    order = ["news", "social"] if counts["news"] <= counts["social"] else ["social", "news"]
    for grp in order:
        pool = cands[grp]
        if not pool:
            continue
        tk = lambda r: (r["item"].get("ticker") or (r.get("company") or {}).get("ticker") or "")
        fewest = min(by_ticker.get((grp, tk(r)), 0) for r in pool)
        best = [r for r in pool if by_ticker.get((grp, tk(r)), 0) == fewest]
        best = sorted(best, key=lambda r: r["item"]["id"])
        if worth is None or not worth.ready:
            return rng.choice(best), "random"
        if rng.random() < explore:
            return rng.choice(best), "explore"
        sample = rng.sample(best, min(200, len(best)))
        s = worth.score(sample)
        thr = np.quantile(s, 0.7)
        pool = [r for r, v in zip(sample, s) if v >= thr]
        return rng.choice(pool), "worth"
    return None, "random"


def gemini_offtarget(teacher_rows, ab_cache: dict | None = None) -> dict[tuple[str, str], str]:
    """(id, ticker) -> Gemini's reason, for items whose newest Gemini label says relevant=false.
    The prompt A/B cache (v2 labels of held-out gold items) wins over the teacher file."""
    labels = {(t["id"], t.get("ticker")): t for t in tio.latest_teacher(teacher_rows)}
    for key, t in ((ab_cache or {}).get("v2") or {}).items():
        iid, _, ticker = key.partition("|")
        labels[(iid, ticker)] = t
    return {k: str(t.get("reason", "")) for k, t in labels.items() if t.get("relevant", True) is False}


def review_queue(corpus_rows, gold_rows, offtarget: dict) -> tuple[list[tuple[dict, dict, str]], int]:
    """Gold items to re-review, ordered by id: (corpus row, Tom's current label, Gemini reason), plus
    how many were already reviewed. An item qualifies when Tom's latest label is a direction (cls != 0,
    relevant, not skipped) and Gemini called it off-target. Pure function."""
    latest: dict[str, dict] = {}
    reviewed: set[str] = set()
    for g in gold_rows:
        latest[g["id"]] = g
        if g.get("review") == "offtarget":
            reviewed.add(g["id"])
    rows = {}
    for r in corpus_rows:
        it = r.get("item") or {}
        rows.setdefault((it.get("id"), it.get("ticker")), r)
    out = []
    for iid in sorted(latest):
        g = latest[iid]
        key = (iid, g.get("ticker"))
        if (iid in reviewed or g.get("skipped") or not g.get("relevant", True) or not g.get("cls")
                or key not in offtarget or key not in rows):
            continue
        out.append((rows[key], g, offtarget[key]))
    return out, len(reviewed)


def _row_ticker(row: dict) -> str:
    return row["item"].get("ticker") or (row.get("company") or {}).get("ticker", "")


def display_item(row: dict) -> dict:
    """Only fields safe to show: no teacher label, no model score."""
    it, co = row["item"], row.get("company") or {}
    return {
        "id": it["id"],
        "ticker": it.get("ticker") or co.get("ticker", ""),
        "company": co.get("short_name") or co.get("name", ""),
        "channel": it.get("channel", "news"),
        "source": it.get("source", ""),
        "publisher": it.get("publisher", ""),
        "title": it.get("title", ""),
        "text": it.get("text", ""),
        "url": it.get("url", ""),
        "published_at": it.get("published_at", ""),
    }


class State:
    def __init__(self, labeler: str = "tom", target: int = 500, review: str | None = None,
                 use_worth: bool = True, embed=None, source: str | None = None):
        if review not in (None, "offtarget"):
            raise ValueError("review must be 'offtarget'")
        self.source = source           # serve only items from this adapter (e.g. "marketaux")
        self.use_worth = use_worth
        self._embed = embed            # tests inject a fake; default is MiniLM, loaded lazily
        self.worth: WorthModel | None = None
        self._worth_tried = False
        self._since_fit = 0
        self.picks: dict[tuple, str] = {}   # served (id, ticker) -> how it was picked
        self.review = review
        self._offtarget: dict | None = None
        self.labeler = labeler
        self.target = target
        self.rng = random.Random(zlib.crc32(labeler.encode("utf-8")))
        self.lock = threading.Lock()
        self.session: list[tuple] = []  # (id, ticker) written by this session, in order
        self._corpus: list[dict] | None = None
        self._corpus_path = None

    def corpus(self) -> list[dict]:
        if self._corpus is None or self._corpus_path != str(tio.CORPUS):
            self._corpus = _read_corpus(tio.CORPUS)
            self._corpus_path = str(tio.CORPUS)
        return self._corpus

    def progress(self) -> dict:
        gold = [g for g in _read_gold(tio.GOLD) if _is_labeled(g)]
        by_class: dict[str, int] = {}
        news = social = 0
        for g in gold:
            if _group(g.get("channel", "news")) == "social":
                social += 1
            else:
                news += 1
            k = str(g.get("cls"))
            by_class[k] = by_class.get(k, 0) + 1
        return {"done": len(gold), "target": self.target, "news": news, "social": social, "by_class": by_class,
                "waste_recent": self.waste_recent()}

    def waste_recent(self, n: int = 20, min_rows: int = 5) -> float | None:
        """Share of skipped / not-relevant among this labeler's last n non-review rows (latest per id)."""
        latest: dict[str, dict] = {}
        for g in _read_gold(tio.GOLD):
            if "review" in g or g.get("labeler") != self.labeler:
                continue
            latest.pop(g["id"], None)
            latest[g["id"]] = g
        last = list(latest.values())[-n:]
        if len(last) < min_rows:
            return None
        return sum(is_waste(g) for g in last) / len(last)

    def _get_embed(self):
        if self._embed is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._embed = lambda texts: model.encode(texts, normalize_embeddings=True, batch_size=64)
        return self._embed

    def _fit_worth(self) -> None:
        """(Re)fit the worth model; any failure falls back to random sampling."""
        try:
            if self.worth is None:
                self.worth = WorthModel(self._get_embed())
            teacher = list(tio.read_jsonl(tio.TEACHER)) if Path(tio.TEACHER).exists() else []
            self.worth.fit(self.corpus(), _read_gold(tio.GOLD), teacher)
        except Exception as e:  # noqa: BLE001
            print(f"worth model disabled: {e}")
            self.worth = None
            self.use_worth = False

    def _active_worth(self):
        if not self.use_worth:
            return None
        if not self._worth_tried:
            self._worth_tried = True
            self._fit_worth()
        return self.worth if self.use_worth else None

    def offtarget(self) -> dict:
        if self._offtarget is None:
            try:
                ab = json.loads(Path(AB_CACHE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                ab = None
            self._offtarget = gemini_offtarget(tio.read_jsonl(tio.TEACHER), ab)
        return self._offtarget

    def queue(self):
        return review_queue(self.corpus(), _read_gold(tio.GOLD), self.offtarget())

    def next(self, item_row: dict | None = None) -> dict:
        if self.review:
            return self._next_review(item_row)
        if item_row is not None:
            row = item_row
        else:
            gold = _read_gold(tio.GOLD)
            worth = self._active_worth()
            try:
                row, how = sample_pick(self.corpus(), gold, self.rng, worth, source=self.source)
            except Exception as e:  # noqa: BLE001
                print(f"worth scoring failed, using random: {e}")
                self.worth, self.use_worth = None, False
                row, how = sample_pick(self.corpus(), gold, self.rng, source=self.source)
            if row:
                self.picks[(row["item"]["id"], _row_ticker(row))] = how
        return {"item": display_item(row) if row else None, "progress": self.progress()}

    def _next_review(self, item_row: dict | None) -> dict:
        q, reviewed = self.queue()
        pick = q[0] if q else None
        if item_row is not None:   # after undo, show the restored item again
            pick = next((x for x in q if x[0]["item"]["id"] == item_row["item"]["id"]), pick)
        progress = {"done": reviewed, "target": reviewed + len(q),
                    "news": sum(_group(x[0]["item"].get("channel", "news")) == "news" for x in q),
                    "social": sum(_group(x[0]["item"].get("channel", "news")) == "social" for x in q),
                    "by_class": {}}
        if pick is None:
            return {"item": None, "progress": progress, "review": {"mode": self.review}}
        row, g, reason = pick
        prev = {k: g.get(k) for k in ("cls", "stance", "event", "note")}
        return {"item": display_item(row), "progress": progress,
                "review": {"mode": self.review, "prev": prev, "gemini_reason": reason}}

    def _find_row(self, iid, ticker=None) -> dict | None:
        """Corpus row by (id, ticker); falls back to the first row with that id."""
        rows = [r for r in self.corpus() if r["item"]["id"] == iid]
        if ticker:
            for r in rows:
                if _row_ticker(r) == ticker:
                    return r
        return rows[0] if rows else None

    def label(self, body: dict) -> dict:
        iid = body.get("id")
        row = self._find_row(iid, body.get("ticker"))
        if row is None:
            raise ValueError("unknown id")
        skipped = bool(body.get("skipped", False))
        cls = body.get("cls")
        if not skipped and (not isinstance(cls, int) or isinstance(cls, bool) or not -2 <= cls <= 2):
            raise ValueError("cls must be an integer -2..2")
        stance = body.get("stance", "none")
        if stance not in STANCES:
            raise ValueError("bad stance")
        event = body.get("event", "OTHER")
        if event not in EVENTS:
            raise ValueError("bad event")
        if self.review:
            if skipped:
                raise ValueError("no skipping in review mode: keep the label instead")
            if iid not in {x[0]["item"]["id"] for x in self.queue()[0]}:
                raise ValueError("not in the review queue")
        elif any(g["id"] == iid for g in _read_gold(tio.GOLD)):
            raise ValueError("already labeled")
        out = {
            "id": iid,
            "ticker": _row_ticker(row),
            "cls": None if skipped else cls,
            "stance": stance,
            "event": event,
            "relevant": bool(body.get("relevant", True)),
            "skipped": skipped,
            "note": str(body.get("note", "")),
            "labeler": self.labeler,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "channel": row["item"].get("channel", "news"),
        }
        if self.review:
            out["review"] = self.review
        elif (iid, out["ticker"]) in self.picks:
            out["pick"] = self.picks[(iid, out["ticker"])]
        Path(tio.GOLD).parent.mkdir(parents=True, exist_ok=True)
        tio.append_jsonl(tio.GOLD, [out])
        self.session.append((iid, out["ticker"]))
        if not self.review and self.use_worth and self._worth_tried:
            self._since_fit += 1
            if self._since_fit >= 50:
                self._since_fit = 0
                self._fit_worth()
        return self.next()

    def undo(self) -> dict:
        if not self.session:
            return self.next()
        iid, tkr = self.session.pop()
        p = Path(tio.GOLD)
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
        drop = None
        for i in range(len(lines) - 1, -1, -1):
            try:
                g = json.loads(lines[i])
                if g.get("id") == iid and g.get("ticker", tkr) == tkr:
                    drop = i
                    break
            except ValueError:
                continue
        if drop is not None:
            del lines[drop]
            p.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        return self.next(self._find_row(iid, tkr))


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/next":
                with state.lock:
                    self._json(200, state.next())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                body = json.loads(raw) if raw else {}
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
                with state.lock:
                    if path == "/api/label":
                        res = state.label(body)
                    elif path == "/api/undo":
                        res = state.undo()
                    else:
                        return self._json(404, {"error": "not found"})
                self._json(200, res)
            except ValueError as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": str(e)})

    return Handler


def make_server(state: State, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--labeler", default="tom")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--review", choices=["offtarget"])
    ap.add_argument("--no-worth", action="store_true", help="disable the worth-labeling ranking model")
    ap.add_argument("--source", help="serve only items from this source adapter, e.g. marketaux")
    a = ap.parse_args(argv)
    srv = make_server(State(a.labeler, a.target, a.review, use_worth=not a.no_worth, source=a.source), a.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Labeling tool: {url}  (labeler={a.labeler}, target={a.target}). Ctrl+C to stop.")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()

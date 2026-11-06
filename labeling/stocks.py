"""Stock-grade calibration tool (SPEC §9.3, route 1): Tom grades whole stocks blind, -3..+3.

    python -m labeling.stocks [--port 8768] [--labeler tom] [--batch data/gold/stock_batch.json]
                               [--grades data/gold/stock_grades.jsonl] [--no-browser]

Shows one stock at a time with the items the engine weighs most (built by `python -m train.stock_batch`),
never the engine's grade or any model score. Each answer is a row in data/gold/stock_grades.jsonl
(append-only, the last row per (batch, ticker) wins; an undo row clears it). Binds to 127.0.0.1 only.
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

from sentiment import config
from train import io as tio

STATIC = Path(__file__).parent / "static"
BATCH = config.ROOT / "data" / "gold" / "stock_batch.json"
GRADES = config.ROOT / "data" / "gold" / "stock_grades.jsonl"
NOTE_MAX = 500


def latest_grades(path, batch_id: str) -> dict[str, dict]:
    """ticker -> last row of this batch; undo rows remove the ticker."""
    out: dict[str, dict] = {}
    for r in tio.read_jsonl(path):
        if r.get("batch") != batch_id or not r.get("ticker"):
            continue
        if r.get("undo"):
            out.pop(r["ticker"], None)
        else:
            out[r["ticker"]] = r
    return out


def undo_stack(path, batch_id: str) -> list[str]:
    """Tickers of this batch whose answer is still standing, oldest first: the last one is what undo
    reverts. Read from the file, so undo works across server restarts and page reloads."""
    stack: list[str] = []
    for r in tio.read_jsonl(path):
        t = r.get("ticker")
        if r.get("batch") != batch_id or not t:
            continue
        if t in stack:
            stack.remove(t)
        if not r.get("undo"):
            stack.append(t)
    return stack


def display_stock(s: dict) -> dict:
    """Only what Tom may see: no engine grade, score or snapshot grade."""
    co = s.get("company")
    if isinstance(co, dict):
        co = co.get("short_name") or co.get("name") or ""
    keep = ("id", "channel", "source", "publisher", "title", "text", "url", "published_at", "age_h")
    return {"ticker": s["ticker"], "company": co or "", "as_of": s.get("as_of", ""),
            "items": [{k: it.get(k) for k in keep} for it in s.get("items", [])]}


class State:
    def __init__(self, batch_path=BATCH, grades_path=GRADES, labeler: str = "tom"):
        self.batch = json.loads(Path(batch_path).read_text(encoding="utf-8"))
        self.batch_id = str(self.batch["id"])
        self.grades_path = Path(grades_path)
        self.labeler = labeler
        # a stock the engine left ungraded (too little data) has nothing to compare against: not served
        self.stocks = {s["ticker"]: s for s in self.batch["stocks"]
                       if (s.get("engine") or {}).get("grade") is not None}
        self.order = sorted(self.stocks)
        random.Random(zlib.crc32(f"{labeler}:{self.batch_id}".encode())).shuffle(self.order)
        self.lock = threading.Lock()

    def _done(self) -> dict[str, dict]:
        return latest_grades(self.grades_path, self.batch_id)

    def next(self) -> dict:
        done = self._done()
        can_undo = any(t in self.stocks for t in undo_stack(self.grades_path, self.batch_id))
        prog = {"done": len(done), "total": len(self.order),
                "graded": sum(1 for r in done.values() if r.get("grade") is not None)}
        for t in self.order:
            if t not in done:
                return {"stock": display_stock(self.stocks[t]), "progress": prog,
                        "can_undo": can_undo}
        return {"stock": None, "progress": prog, "can_undo": can_undo}

    def grade(self, body: dict) -> dict:
        t = body.get("ticker")
        if t not in self.stocks:
            raise ValueError("unknown ticker")
        skip = bool(body.get("skip"))
        g = body.get("grade")
        if skip:
            g = None
        elif not isinstance(g, int) or isinstance(g, bool) or not -3 <= g <= 3:
            raise ValueError("grade must be an integer -3..3")
        note = str(body.get("note") or "")[:NOTE_MAX]
        tio.append_jsonl(self.grades_path, [{
            "batch": self.batch_id, "ticker": t, "grade": g, "skip": skip, "note": note,
            "labeler": self.labeler, "ts": datetime.now(timezone.utc).isoformat(),
            "as_of": self.stocks[t].get("as_of")}])
        return self.next()

    def undo(self) -> dict:
        stack = [t for t in undo_stack(self.grades_path, self.batch_id) if t in self.stocks]
        if not stack:
            raise ValueError("nothing to undo")
        t = stack[-1]
        tio.append_jsonl(self.grades_path, [{"batch": self.batch_id, "ticker": t, "undo": True,
                                             "labeler": self.labeler,
                                             "ts": datetime.now(timezone.utc).isoformat()}])
        res = self.next()
        res["stock"] = display_stock(self.stocks[t])   # re-show the stock that was undone
        return res


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
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
                self._send(200, (STATIC / "stocks.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/next":
                with state.lock:
                    self._json(200, state.next())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else {}
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
                with state.lock:
                    if path == "/api/grade":
                        res = state.grade(body)
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


def make_server(state: State, port: int = 8768) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--labeler", default="tom")
    ap.add_argument("--batch", type=Path, default=BATCH)
    ap.add_argument("--grades", type=Path, default=GRADES)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    srv = make_server(State(a.batch, a.grades, a.labeler), a.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Stock grading: {url}  (labeler={a.labeler}). Ctrl+C to stop.")
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

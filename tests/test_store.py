import json

import httpx
import pytest

from sentiment import cli, config, store as store_mod
from sentiment.snapshots import compact
from sentiment.store import Store, StoreError, canonical, clean_json, latest_row, snapshot_row

KEY = "sekret-service-key-123"


def _store(monkeypatch, handler):
    monkeypatch.setattr(store_mod.time, "sleep", lambda s: None)
    return Store("https://x.supabase.co/", KEY, client=httpx.Client(transport=httpx.MockTransport(handler)))


def _full(t="NVDA", as_of="2026-10-29T12:05:00+00:00", **kw):
    d = {"ticker": t, "company": "Nvidia", "as_of": as_of, "model_version": "s2", "attention_z": 1.5,
         "flags": [], "composite": {"raw": 0.1, "score": 0.4, "z": 1.2, "grade": 2, "n_items": 3,
                                    "n_effective": 2.0, "agreement": 0.5},
         "channels": {"news": {"score": 0.3, "grade": 1}}, "items": [{"x": 1}],
         "elapsed_s": 1.0, "timings_s": {"a": 1}}
    d.update(kw)
    return d


def test_canonical():
    assert canonical(" brk-b ") == "BRK.B"


def test_clean_json():
    out = clean_json({"a": float("nan"), "b": [1.0, float("inf"), -float("inf"), {"c": float("nan")}], "d": "x"})
    assert out == {"a": None, "b": [1.0, None, None, {"c": None}], "d": "x"}


def test_snapshot_row_full_and_compact():
    r = snapshot_row(_full("BRK-B"), "main")
    assert r["ticker"] == "BRK.B" and r["run"] == "main" and r["grade"] == 2 and r["z"] == 1.2
    assert r["attention_z"] == 1.5 and r["data"]["run"] == "main" and "items" not in r["data"]
    c = compact({**_full(), "run": "refresh"})
    assert compact(c) == c
    r2 = snapshot_row(c, "main")
    assert r2["run"] == "refresh" and r2["data"] == c and r2["grade"] == 2


def test_snapshot_row_defaults():
    r = snapshot_row({"ticker": "A", "as_of": "t", "composite": {"score": float("nan")}}, "manual")
    assert r["model_version"] == "unknown" and r["score"] is None and r["grade"] is None


def test_latest_row_strips():
    r = latest_row(_full(), "refresh")
    assert r["run"] == "refresh" and r["company"] == "Nvidia" and r["payload"]["run"] == "refresh"
    for k in ("items", "elapsed_s", "timings_s"):
        assert k not in r["payload"]
    assert r["payload"]["channels"]["news"]["score"] == 0.3


def test_write_results_paths_and_chunks(monkeypatch):
    monkeypatch.setattr(config, "SUPABASE_BATCH", 2)
    calls = []

    def handler(req):
        calls.append(req)
        if "/rpc/" in req.url.path:
            return httpx.Response(200, json=len(json.loads(req.content)["rows"]))
        return httpx.Response(201)

    s = _store(monkeypatch, handler)
    res = [_full("A"), _full("B"), _full("C"), {"ticker": "Z"}]
    out = s.write_results(res, "main")
    assert out == {"snapshots": 3, "latest": 3}
    snaps = [c for c in calls if c.url.path.endswith("/pro_sentiment_snapshots")]
    rpcs = [c for c in calls if c.url.path.endswith("/rpc/pro_sentiment_upsert_latest")]
    assert len(snaps) == 2 and len(rpcs) == 2
    assert snaps[0].url.path == "/rest/v1/pro_sentiment_snapshots"
    assert snaps[0].url.params["on_conflict"] == "ticker,as_of,run"
    assert snaps[0].headers["Prefer"] == "resolution=ignore-duplicates,return=minimal"
    assert snaps[0].headers["apikey"] == KEY and snaps[0].headers["Authorization"] == "Bearer " + KEY
    assert KEY not in repr(s)


def test_write_results_empty_no_request(monkeypatch):
    def handler(req):
        raise AssertionError("no request expected")

    assert _store(monkeypatch, handler).write_results([], "main") == {"snapshots": 0, "latest": 0}


def test_4xx_raises_without_key_and_no_retry(monkeypatch):
    n = []

    def handler(req):
        n.append(1)
        return httpx.Response(401, text="bad jwt")

    with pytest.raises(StoreError) as ei:
        _store(monkeypatch, handler).write_results([_full()], "main")
    assert KEY not in str(ei.value) and str(ei.value).startswith("401 ")
    assert len(n) == 2   # one per phase (snapshots, latest), no retry


def test_5xx_retried_with_backoff(monkeypatch):
    n = []

    def handler(req):
        n.append(1)
        return httpx.Response(503 if len(n) < 3 else 200, json=1)

    assert _store(monkeypatch, handler).set_universe(["a"]) == 1
    assert len(n) == 3

    n.clear()
    slept = []
    st = _store(monkeypatch, lambda r: (n.append(1), httpx.Response(500, text="x"))[1])
    monkeypatch.setattr(store_mod.time, "sleep", slept.append)
    with pytest.raises(StoreError):
        st.set_universe(["a"])
    assert len(n) == len(config.SUPABASE_RETRY_DELAYS) + 1
    assert slept == list(config.SUPABASE_RETRY_DELAYS)


def test_429_honors_retry_after(monkeypatch):
    n, slept = [], []

    def handler(req):
        n.append(1)
        if len(n) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow down")
        if len(n) == 2:
            return httpx.Response(429, headers={"Retry-After": "9999"}, text="slow down")
        return httpx.Response(200, json=1)

    st = _store(monkeypatch, handler)
    monkeypatch.setattr(store_mod.time, "sleep", slept.append)
    assert st.set_universe(["a"]) == 1
    assert slept == [7.0, 60.0]


def test_transport_error_retried(monkeypatch):
    n = []

    def handler(req):
        n.append(1)
        if len(n) == 1:
            raise httpx.ConnectError("boom", request=req)
        return httpx.Response(200, json=2)

    assert _store(monkeypatch, handler).set_universe(["a", "b"]) == 2


def test_set_universe(monkeypatch):
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json=2)

    s = _store(monkeypatch, handler)
    assert s.set_universe(["nvda", "BRK-B", "NVDA", " ", "brk.b"]) == 2
    assert bodies == [{"p_tickers": ["BRK.B", "NVDA"]}]
    bodies.clear()
    with pytest.raises(StoreError):
        s.set_universe(["", "  "])
    assert bodies == []


def test_import_history(monkeypatch, tmp_path):
    good = compact(_full("AAA"))
    old = {k: v for k, v in good.items() if k != "run"}
    (tmp_path / "AAA.jsonl").write_text(
        "\n".join([json.dumps(good), "", "not json", json.dumps([1]), json.dumps({"ticker": "X"}),
                   json.dumps(old | {"as_of": "2026-01-01T00:00:00+00:00"})]) + "\n", encoding="utf-8")
    sent = []

    def handler(req):
        sent.append(req)
        return httpx.Response(201)

    n = _store(monkeypatch, handler).import_history(tmp_path)
    assert n == 2 and len(sent) == 1
    rows = json.loads(sent[0].content)
    assert [r["run"] for r in rows] == ["manual", "manual"]
    assert all("/rpc/" not in c.url.path for c in sent)


def test_from_env_unset(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert Store.from_env() is None
    monkeypatch.setenv("SUPABASE_URL", "https://x.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "  ")
    assert Store.from_env() is None
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", KEY)
    assert Store.from_env() is not None


def test_cli_store_requires_an_action():
    with pytest.raises(SystemExit):
        cli.main(["store"])


def test_cli_store_unconfigured_exits_0(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    u = tmp_path / "u.txt"
    u.write_text("NVDA\n", encoding="utf-8")
    assert cli.main(["store", "--universe", str(u)]) == 0


def test_cli_store_flow(monkeypatch, tmp_path):
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(200, json=1)

    s = _store(monkeypatch, handler)
    monkeypatch.setattr(Store, "from_env", classmethod(lambda cls: s))
    res = tmp_path / "shard.json"
    res.write_text(json.dumps({"results": [_full()], "errors": {}}), encoding="utf-8")
    u = tmp_path / "u.txt"
    u.write_text("NVDA\n# c\n", encoding="utf-8")
    rc = cli.main(["store", "--results", str(tmp_path / "missing.json"), str(res), "--universe", str(u)])
    assert rc == 0
    assert calls[-1].endswith("/rpc/pro_sentiment_set_universe") and len(calls) == 3


def test_cli_store_error_exit_1_but_continues(monkeypatch, tmp_path):
    calls = []

    def handler(req):
        calls.append(req.url.path)
        if "snapshots" in req.url.path:
            return httpx.Response(400, text="bad")
        return httpx.Response(200, json=1)

    s = _store(monkeypatch, handler)
    monkeypatch.setattr(Store, "from_env", classmethod(lambda cls: s))
    res = tmp_path / "shard.json"
    res.write_text(json.dumps({"results": [_full()]}), encoding="utf-8")
    u = tmp_path / "u.txt"
    u.write_text("NVDA\n", encoding="utf-8")
    assert cli.main(["store", "--results", str(res), "--universe", str(u)]) == 1
    assert calls[-1].endswith("set_universe")


def test_invalid_rows_dropped_and_grade_clamped(monkeypatch):
    sent = []

    def handler(req):
        sent.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json=1) if "rpc" in req.url.path else httpx.Response(201)

    s = _store(monkeypatch, handler)
    bad = _full("TOOLONGTICKER1")
    odd = _full("AMD", composite={"grade": 7, "score": 0.1}, model_version="v" * 300)
    r = s.write_results([bad, odd], "main")
    assert r == {"snapshots": 1, "latest": 1}
    rows = sent[0][1]
    assert [x["ticker"] for x in rows] == ["AMD"]
    assert rows[0]["grade"] is None and len(rows[0]["model_version"]) == 200


def test_latest_still_written_when_snapshots_fail(monkeypatch):
    paths = []

    def handler(req):
        paths.append(req.url.path)
        return httpx.Response(200, json=1) if "rpc" in req.url.path else httpx.Response(400, text="bad row")

    s = _store(monkeypatch, handler)
    with pytest.raises(StoreError, match="bad row"):
        s.write_results([_full()], "main")
    assert any("upsert_latest" in p for p in paths)


def test_bad_rpc_response_is_store_error(monkeypatch):
    s = _store(monkeypatch, lambda req: httpx.Response(200, text="null"))
    with pytest.raises(StoreError, match="bad rpc response"):
        s.set_universe(["NVDA"])


def test_set_universe_rejects_invalid_without_request(monkeypatch):
    calls = []
    s = _store(monkeypatch, lambda req: calls.append(1) or httpx.Response(200, json=1))
    with pytest.raises(StoreError, match="invalid universe"):
        s.set_universe(["NVDA", "WAYTOOLONG123"])
    assert not calls


def test_new_secret_key_sent_as_apikey_only(monkeypatch):
    seen = []

    def handler(req):
        seen.append(dict(req.headers))
        return httpx.Response(200, json=1)

    monkeypatch.setattr(store_mod.time, "sleep", lambda s: None)
    s = Store("https://x.supabase.co", "sb_secret_abc", client=httpx.Client(transport=httpx.MockTransport(handler)))
    s.set_universe(["NVDA"])
    assert seen[0]["apikey"] == "sb_secret_abc" and "authorization" not in seen[0]


# ---- on-demand requests ----
U1, U2, U3 = ("11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222",
              "33333333-3333-4333-8333-333333333333")


def test_parse_requests_ok_and_canonical():
    assert store_mod.parse_requests(f" aapl = {U1.upper()} , BRK-B={U2}") == [("AAPL", U1), ("BRK.B", U2)]


@pytest.mark.parametrize("spec", [
    ",".join(f"T{i}=aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaa{i:02d}" for i in range(6)),   # 6 pairs
    f"AAPL={U1},AAPL={U2}", f"AAPL={U1},MSFT={U1}", "AAPL=not-a-uuid", f"TOOLONGTICKER1={U1}",
    f"A$={U1}", f"AAPL{U1}", "", f"AAPL={U1},", f"AAPL={U1}=x",
])
def test_parse_requests_bad(spec):
    with pytest.raises(ValueError):
        store_mod.parse_requests(spec)


def test_mark_requests_queries(monkeypatch):
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(204)

    s = _store(monkeypatch, handler)
    s.mark_requests([U1, U2], "running")
    s.mark_requests([U1], "done")
    s.mark_requests([U1], "failed", "x" * 600)
    s.mark_requests([U1], "failed")
    s.mark_requests([], "done")
    assert len(seen) == 4
    for r in seen:
        assert r.method == "PATCH" and r.url.path == "/rest/v1/pro_sentiment_requests"
        assert r.headers["Prefer"] == "return=minimal"
    p = [r.url.params for r in seen]
    b = [json.loads(r.content) for r in seen]
    assert p[0]["id"] == f"in.({U1},{U2})" and p[0]["status"] == "eq.queued"
    assert b[0]["status"] == "running" and b[0]["started_at"]
    assert p[1]["status"] == "in.(queued,running,failed)"
    assert b[1]["status"] == "done" and b[1]["error"] is None and b[1]["finished_at"]
    assert p[2]["status"] == "in.(queued,running)"
    assert b[2]["status"] == "failed" and len(b[2]["error"]) == 500
    assert b[3]["error"] == "analysis failed"


def test_mark_requests_bad_status(monkeypatch):
    s = _store(monkeypatch, lambda req: (_ for _ in ()).throw(AssertionError("no request")))
    with pytest.raises(ValueError):
        s.mark_requests([U1], "queued")


def test_pull_history(monkeypatch, tmp_path):
    seen = []
    a1, a2 = compact(_full("AAPL")), compact(_full("AAPL", as_of="2026-10-30T12:05:00+00:00"))

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=[
            {"ticker": "AAPL", "data": a2}, {"ticker": "AAPL", "data": a1},   # newest first, as requested
            {"ticker": "BRK.B", "data": "nope"}, {"ticker": "MSFT", "data": {"ticker": "MSFT", "as_of": "t"}}])

    s = _store(monkeypatch, handler)
    out = tmp_path / "snaps"
    (out).mkdir()
    (out / "AAPL.jsonl").write_text("old\n", encoding="utf-8")
    assert s.pull_history(["aapl", "BRK-B", "MSFT"], out) == 3
    q = seen[0].url.params
    assert q["select"] == "ticker,data" and q["order"] == "as_of.desc"
    assert q["ticker"] == 'in.("AAPL","BRK.B","MSFT")' and q["as_of"].startswith("gte.20")
    lines = (out / "AAPL.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x) for x in lines] == [a1, a2]
    assert (out / "MSFT.jsonl").exists() and not (out / "BRK.B.jsonl").exists()
    assert s.pull_history([], out) == 0 and len(seen) == 1


def _cli_store(monkeypatch, handler):
    s = _store(monkeypatch, handler)
    monkeypatch.setattr(Store, "from_env", classmethod(lambda cls: s))
    return s


def _patches(calls):
    return [(json.loads(r.content), r.url.params) for r in calls if r.method == "PATCH"]


def test_cli_finalize_done_and_failed(monkeypatch, tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=1) if "rpc" in req.url.path else httpx.Response(201)

    _cli_store(monkeypatch, handler)
    res = tmp_path / "out.json"
    res.write_text(json.dumps({"results": [_full("AAPL")], "errors": {"MSFT": "RuntimeError: boom"}}),
                   encoding="utf-8")
    rc = cli.main(["store", "--requests", f"AAPL={U1},MSFT={U2}", "--results", str(res), "--run", "ondemand"])
    assert rc == 0
    pt = _patches(calls)
    assert len(pt) == 2
    assert pt[0][0]["status"] == "done" and pt[0][1]["id"] == f"in.({U1})"
    assert pt[1][0]["status"] == "failed" and pt[1][0]["error"] == "RuntimeError: boom"
    assert pt[1][1]["id"] == f"in.({U2})"


def test_cli_finalize_no_results_file(monkeypatch, tmp_path):
    calls = []
    _cli_store(monkeypatch, lambda req: calls.append(req) or httpx.Response(204))
    assert cli.main(["store", "--requests", f"AAPL={U1},MSFT={U2}"]) == 0
    pt = _patches(calls)
    assert len(pt) == 2 and all(b["status"] == "failed" and b["error"] == "no result" for b, _ in pt)
    calls.clear()   # a results path that does not exist behaves the same
    _cli_store(monkeypatch, lambda req: calls.append(req) or httpx.Response(204))   # the first store was closed
    assert cli.main(["store", "--requests", f"AAPL={U1}", "--results", str(tmp_path / "nope.json")]) == 0
    assert [b["error"] for b, _ in _patches(calls)] == ["no result"]


def test_cli_finalize_store_failure(monkeypatch, tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        if req.method == "PATCH":
            return httpx.Response(204)
        return httpx.Response(400, text="bad")

    _cli_store(monkeypatch, handler)
    res = tmp_path / "out.json"
    res.write_text(json.dumps({"results": [_full("AAPL")], "errors": {}}), encoding="utf-8")
    assert cli.main(["store", "--requests", f"AAPL={U1},MSFT={U2}", "--results", str(res)]) == 1
    pt = _patches(calls)
    assert len(pt) == 1 and pt[0][0]["status"] == "failed" and pt[0][0]["error"] == "store failed"
    assert pt[0][1]["id"] == f"in.({U1},{U2})"


def test_cli_mark_and_pull(monkeypatch, tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json=[{"ticker": "AAPL", "data": compact(_full("AAPL"))}]) \
            if req.method == "GET" else httpx.Response(204)

    _cli_store(monkeypatch, handler)
    rc = cli.main(["store", "--requests", f"AAPL={U1}", "--mark", "running", "--pull-history",
                   "--to", str(tmp_path / "d")])
    assert rc == 0
    assert [c.method for c in calls] == ["PATCH", "GET"]   # marking running does not finalize
    assert (tmp_path / "d" / "AAPL.jsonl").exists()


def test_cli_invalid_requests_no_http(monkeypatch):
    def handler(req):
        raise AssertionError("no http expected")

    _cli_store(monkeypatch, handler)
    assert cli.main(["store", "--requests", "AAPL=bad"]) == 1
    monkeypatch.setattr(Store, "from_env", classmethod(lambda cls: None))
    assert cli.main(["store", "--requests", "AAPL=bad"]) == 1   # fails even when unconfigured


def test_cli_mark_requires_requests():
    with pytest.raises(SystemExit):
        cli.main(["store", "--mark", "running", "--universe", "u.txt"])
    with pytest.raises(SystemExit):
        cli.main(["store", "--pull-history", "--universe", "u.txt"])


def _list_db(lists, universe, latest, seen=None):
    """MockTransport handler for pick_list: the lists, universe and latest tables."""
    def handler(req):
        if seen is not None:
            seen.append(req)
        path = req.url.path
        if path.endswith("/pro_sentiment_lists"):
            off, lim = int(req.url.params["offset"]), int(req.url.params["limit"])
            return httpx.Response(200, json=[{"ticker": t} for t in lists[off:off + lim]])
        if path.endswith("/pro_sentiment_universe"):
            return httpx.Response(200, json=[{"ticker": t} for t in universe])
        if path.endswith("/pro_sentiment_latest"):
            want = req.url.params["ticker"][4:-1].replace('"', "").split(",")
            return httpx.Response(200, json=[{"ticker": t, "as_of": a} for t, a in latest.items() if t in want])
        if path.endswith("/pro_sentiment_snapshots"):
            return httpx.Response(200, json=[])
        return httpx.Response(404)
    return handler


def test_pick_list_filters_and_orders(monkeypatch):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    latest = {"RKLB": (now - timedelta(hours=3)).isoformat(), "IONQ": (now - timedelta(hours=1)).isoformat(),
              "SOFI": (now - timedelta(minutes=10)).isoformat()}   # SOFI is fresh
    lists = ["RKLB", "nvda", "SOFI", "IONQ", "BRK-B", "RKLB", "bad ticker!"]
    s = _store(monkeypatch, _list_db(lists, ["NVDA", "AAPL"], latest))
    picked, due = s.pick_list(10)
    assert picked == ["BRK.B", "RKLB", "IONQ"] and due == 3   # never analyzed first, then oldest
    assert s.pick_list(2) == (["BRK.B", "RKLB"], 3)
    assert s.pick_list(0) == ([], 3)


def test_pick_list_pages_and_candidates(monkeypatch):
    seen = []
    lists = [f"T{i}" for i in range(1500)]
    s = _store(monkeypatch, _list_db(lists, [], {}, seen))
    picked, due = s.pick_list(5)
    assert due == 1500 and picked == sorted(lists)[:5]
    assert sum(r.url.path.endswith("/pro_sentiment_lists") for r in seen) == 2
    assert sum(r.url.path.endswith("/pro_sentiment_latest") for r in seen) == 15   # 100 per request
    seen.clear()
    assert s.pick_list(5, candidates=["rklb", "NVDA"]) == (["NVDA", "RKLB"], 2)
    assert not any(r.url.path.endswith("/pro_sentiment_lists") for r in seen)
    assert s.pick_list(5, candidates=[]) == ([], 0)


def test_cli_pick_list(monkeypatch, tmp_path):
    out, snaps = tmp_path / "t.txt", tmp_path / "snaps"
    _cli_store(monkeypatch, _list_db(["RKLB", "NVDA"], ["NVDA"], {}))
    assert cli.main(["store", "--pick-list", "30", "--tickers-out", str(out), "--to", str(snaps)]) == 0
    assert out.read_text(encoding="utf-8") == "RKLB\n"
    _cli_store(monkeypatch, _list_db([], [], {}))
    assert cli.main(["store", "--pick-list", "30", "--tickers-out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == ""
    _cli_store(monkeypatch, lambda req: httpx.Response(500))
    assert cli.main(["store", "--pick-list", "30", "--tickers-out", str(out)]) == 1
    assert out.read_text(encoding="utf-8") == ""


def test_cli_pick_list_args():
    with pytest.raises(SystemExit):
        cli.main(["store", "--pick-list", "5", "--universe", "u.txt"])
    with pytest.raises(SystemExit):
        cli.main(["store", "--candidates", "RKLB", "--universe", "u.txt"])

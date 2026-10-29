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


def test_5xx_retried_once(monkeypatch):
    n = []

    def handler(req):
        n.append(1)
        return httpx.Response(503 if len(n) == 1 else 200, json=1)

    assert _store(monkeypatch, handler).set_universe(["a"]) == 1
    assert len(n) == 2

    n.clear()
    with pytest.raises(StoreError):
        _store(monkeypatch, lambda r: (n.append(1), httpx.Response(500, text="x"))[1]).set_universe(["a"])
    assert len(n) == 2


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

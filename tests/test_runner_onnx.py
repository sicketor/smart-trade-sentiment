"""Student discovery, ONNX backend selection, path-independent cache keys and CLI sharding (no models loaded)."""
import json

import pytest

from sentiment import cli, config
from sentiment.models import registry
from sentiment.models.registry import ModelSpec
from sentiment.models.runner import ModelRunner, student_backend


def _torch_dir(root, version=None):
    d = root / "st-finsent"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    if version is not None:
        (d / "VERSION").write_text(version, encoding="utf-8")
    return d


def _onnx_dir(root, version=None):
    d = root / "st-finsent-onnx"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"x")
    if version is not None:
        (d / "VERSION").write_text(version, encoding="utf-8")
    return d


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    return tmp_path


def test_student_prefers_torch(models):
    t = _torch_dir(models)
    _onnx_dir(models)
    s = registry._student()
    assert s is not None and s.hf_id == str(t) and s.cache_id == "st-finsent"
    assert s.version == str(int((t / "model.safetensors").stat().st_mtime))


def test_student_falls_back_to_onnx(models):
    o = _onnx_dir(models)
    (models / "st-finsent").mkdir()   # dir without weights does not count
    s = registry._student()
    assert s is not None and s.hf_id == str(o) and s.cache_id == "st-finsent"
    assert s.version == str(int((o / "model.onnx").stat().st_mtime))


def test_student_none_without_weights(models):
    (models / "st-finsent-onnx").mkdir()
    assert registry._student() is None
    assert registry.student_paths() == (None, None)


def test_version_file_overrides_mtime(models):
    _torch_dir(models, " s2\n")
    assert registry._student().version == "s2"


def test_onnx_version_file(models):
    _onnx_dir(models, "s2")
    assert registry._student().version == "s2"


def test_empty_version_file_keeps_mtime(models):
    o = _onnx_dir(models, "  \n")
    assert registry._student().version == str(int((o / "model.onnx").stat().st_mtime))


def test_cache_id_makes_key_path_independent():
    a = ModelSpec("st_finsent", "C:/a/models/st-finsent", registry.ALL, "cls", ("neg",) * 5, "target",
                  version="s2", cache_id="st-finsent")
    b = ModelSpec("st_finsent", "/home/runner/models/st-finsent-onnx", registry.ALL, "cls", ("neg",) * 5,
                  "target", version="s2", cache_id="st-finsent")
    assert ModelRunner._key("t", "NVDA | Nvidia", a) == ModelRunner._key("t", "NVDA | Nvidia", b)
    c = ModelSpec(**{**b.__dict__, "version": "s3"})
    assert ModelRunner._key("t", "NVDA | Nvidia", c) != ModelRunner._key("t", "NVDA | Nvidia", b)


def test_key_without_cache_id_uses_hf_id():
    a = ModelSpec("m", "org/a", registry.ALL, "cls", ("neg", "neu", "pos"), "plain")
    b = ModelSpec("m", "org/b", registry.ALL, "cls", ("neg", "neu", "pos"), "plain")
    assert ModelRunner._key("t", "", a) != ModelRunner._key("t", "", b)


@pytest.mark.parametrize("device,env,has_torch,has_onnx,want", [
    ("cpu", None, True, True, "onnx"),
    ("cpu", "auto", True, True, "onnx"),
    ("cpu", "AUTO", True, False, "torch"),
    ("cuda", None, True, True, "torch"),
    ("cuda", "onnx", True, True, "onnx"),
    ("cpu", "torch", True, True, "torch"),
    ("cpu", "torch", False, True, "onnx"),     # no torch weights: onnx forced
    ("cuda", None, False, True, "onnx"),
    ("cuda", "", True, True, "torch"),
])
def test_backend_matrix(device, env, has_torch, has_onnx, want):
    assert student_backend(device, env, has_torch, has_onnx) == want


@pytest.mark.parametrize("env,has_torch,has_onnx,exc", [
    (None, False, False, RuntimeError),
    ("onnx", True, False, RuntimeError),
    ("tensorrt", True, True, ValueError),
])
def test_backend_errors(env, has_torch, has_onnx, exc):
    with pytest.raises(exc):
        student_backend("cpu", env, has_torch, has_onnx)


def test_shard_partition_disjoint_and_complete():
    tickers = [f"T{i:03d}" for i in range(149)]
    shards = [cli.select_shard(tickers, i, 8) for i in range(1, 9)]
    flat = [t for s in shards for t in s]
    assert sorted(flat) == sorted(tickers) and len(flat) == len(set(flat))
    assert {len(s) for s in shards} == {18, 19}
    assert cli.select_shard(list(reversed(tickers)), 3, 8) == shards[2]   # order-independent


@pytest.mark.parametrize("bad", ["0/4", "5/4", "1/0", "2", "a/b"])
def test_parse_shard_rejects(bad):
    with pytest.raises(ValueError):
        cli.parse_shard(bad)


def test_parse_shard_ok():
    assert cli.parse_shard("1/40") == (1, 40)


def test_read_tickers_file(tmp_path):
    f = tmp_path / "u.txt"
    f.write_text("# header\nAAPL\n\n  msft  \nNVDA # trailing\n   # indented comment\n", encoding="utf-8")
    assert cli.read_tickers_file(str(f)) == ["AAPL", "msft", "NVDA"]


def test_cli_requires_a_ticker():
    with pytest.raises(SystemExit) as e:
        cli.main(["analyze"])
    assert e.value.code == 2


def test_cli_empty_shard_exits_zero(tmp_path):
    out = tmp_path / "o.json"
    assert cli.main(["analyze", "AAPL", "--shard", "2/3", "--no-llm", "--no-save", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"results": [], "errors": {}}


import hashlib
import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest


@pytest.fixture
def lfs(tmp_path, monkeypatch):
  spec = importlib.util.spec_from_file_location("asius_lfs", Path(__file__).resolve().parents[4] / "lfs.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  monkeypatch.setattr(module, "GIT_DIR", tmp_path)
  monkeypatch.setattr(module, "STORE", tmp_path / "objects")
  return module


def test_upstream_object_uses_upstream_only(lfs, monkeypatch):
  data = b"upstream model"
  oid = hashlib.sha256(data).hexdigest()
  monkeypatch.setattr(lfs, "download_http", lambda _oid, _size, path: path.write_bytes(data))
  fork = Mock(side_effect=AssertionError("unexpected fork request"))
  monkeypatch.setattr(lfs, "fork_downloads", fork)
  assert lfs.download_one(oid, len(data)).read_bytes() == data
  fork.assert_not_called()


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_upstream_object_downloads_and_checks_fork(lfs, monkeypatch, corrupt):
  data = b"fork model"
  oid = hashlib.sha256(data).hexdigest()
  action = {"href": "https://example.com/model"}

  def download(_oid, _size, path, **kwargs):
    if not kwargs:
      raise HTTPError("https://example.com/upstream", 404, "missing", {}, None)
    assert kwargs["action"] == action
    path.write_bytes(b"bad model!" if corrupt else data)

  monkeypatch.setattr(lfs, "download_http", download)
  monkeypatch.setattr(lfs, "fork_downloads", lambda objects: {oid: action})
  if corrupt:
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
      lfs.download_one(oid, len(data))
    assert not lfs.object_path(oid).exists()
  else:
    assert lfs.download_one(oid, len(data)).read_bytes() == data


@pytest.mark.parametrize("status", [401, 403, 500])
def test_upstream_errors_are_not_treated_as_missing(lfs, monkeypatch, status):
  monkeypatch.setattr(lfs, "download_http", Mock(side_effect=HTTPError("https://example.com", status, "failed", {}, None)))
  fork = Mock()
  monkeypatch.setattr(lfs, "fork_downloads", fork)
  with pytest.raises(HTTPError):
    lfs.download_one("a" * 64, 10)
  fork.assert_not_called()


@pytest.mark.parametrize("item", [{}, {"oid": "a", "size": 11, "actions": {"download": {"href": "https://example.com"}}},
                                  {"oid": "a", "size": 10, "error": {"code": 404}}])
def test_fork_metadata_rejects_missing_or_wrong_objects(lfs, monkeypatch, item):
  monkeypatch.setattr(lfs.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(json.dumps({"objects": [item] if item else []}).encode()))
  with pytest.raises(RuntimeError, match="missing or has the wrong size"):
    lfs.fork_downloads({"a": 10})


def test_upload_keeps_existing_upstream_objects_and_publishes_fork(lfs, monkeypatch):
  data = b"fork model"
  oid = hashlib.sha256(data).hexdigest()
  path = lfs.object_path(oid)
  path.parent.mkdir(parents=True)
  path.write_bytes(data)
  monkeypatch.setattr(lfs, "object_info", lambda objects: {"sha256/upstream": 100})
  publish = Mock()
  verify = Mock()
  monkeypatch.setattr(lfs, "git", publish)
  monkeypatch.setattr(lfs, "fork_downloads", verify)
  lfs.upload({"upstream": 100, oid: len(data)})
  publish.assert_called_once_with("-c", f"lfs.url={lfs.FORK_LFS}", "-c", f"lfs.pushurl={lfs.FORK_LFS}",
                                  "lfs", "push", "--object-id", "origin", "--stdin", stdin=(oid + "\n").encode())
  verify.assert_called_once_with({oid: len(data)})


def test_upload_rejects_corrupt_local_object(lfs, monkeypatch):
  oid = "a" * 64
  path = lfs.object_path(oid)
  path.parent.mkdir(parents=True)
  path.write_bytes(b"bad")
  monkeypatch.setattr(lfs, "object_info", lambda objects: {})
  publish = Mock()
  monkeypatch.setattr(lfs, "git", publish)
  with pytest.raises(RuntimeError, match="local sha256 mismatch"):
    lfs.upload({oid: 3})
  publish.assert_not_called()

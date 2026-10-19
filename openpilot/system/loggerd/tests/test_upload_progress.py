import io
import json
from unittest.mock import patch

import requests

from openpilot.system.loggerd import upload_progress


def test_progress_stream_is_valid_http_body(tmp_path):
  with patch.object(upload_progress, 'STATUS_PATH', tmp_path / 'progress.json'):
    reader = upload_progress.ProgressReader(io.BytesIO(b'hello'), 'routes/test/0/qlog.zst', 5)
    request = requests.Request('PUT', 'https://example.invalid', data=reader).prepare()
    assert request.headers['Content-Length'] == '5'
    assert request.body.read() == b'hello'
    assert upload_progress.read_progress()['sent'] < 5
    reader.complete()
    assert upload_progress.read_progress()['sent'] == 5
    status = json.loads(upload_progress.STATUS_PATH.read_text())
    status['updated'] -= 181
    upload_progress.STATUS_PATH.write_text(json.dumps(status))
    assert upload_progress.read_progress() is None


def test_summary_retries_stale_etag_without_global_backoff(tmp_path, monkeypatch):
  from types import SimpleNamespace
  from openpilot.system.loggerd import asius_uploader
  from openpilot.system.loggerd.route_summary import summary_path

  uploader = object.__new__(asius_uploader.AsiusUploader)
  uploader.root = str(tmp_path)
  uploader.owner = 'test'
  uploader.sync_access = dict
  monkeypatch.setattr(asius_uploader, 'encrypt_object', lambda *args: b'encrypted')
  path = summary_path(tmp_path, '00000001--1234567890')
  path.parent.mkdir(parents=True)
  path.write_text('{}')
  calls = []

  def request(method, url, body):
    calls.append(dict(body))
    response = requests.Response()
    if len(calls) == 1:
      response.status_code = 412
      response._content = b'{"etag":"current"}'
      response.raise_for_status()
    response.status_code = 200
    response._content = b'{"etag":"saved"}'
    return response

  uploader.client = SimpleNamespace(request=request)
  uploader.sync_summary(path.stem)
  assert len(calls) == 2
  assert calls[1]['etag'] == 'current'
  assert json.loads(path.with_suffix('.uploaded').read_text())['etag'] == 'saved'

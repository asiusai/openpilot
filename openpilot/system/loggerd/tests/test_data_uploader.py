import base64
import json
from types import SimpleNamespace

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from openpilot.system.loggerd.data_api import (
  ENCRYPTION_MAGIC,
  access_document,
  b64url,
  can_wrap_for,
  canonical_json,
  encrypt_object,
  initial_state,
  public_identity,
  recover_state,
  update_state,
  x25519_private,
  x25519_public,
)
from openpilot.system.loggerd.data_uploader import DataUploader
from openpilot.system.loggerd.uploader import Uploader
from openpilot.system.app.identity import bytes_to_identity


def decode64(value: str) -> bytes:
  return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def test_access_document_and_object_crypto() -> None:
  device_key = ed25519.Ed25519PrivateKey.generate()
  reader_key = ed25519.Ed25519PrivateKey.generate()
  owner = public_identity(device_key)
  reader = public_identity(reader_key)
  state = initial_state([reader])

  document = access_document(device_key, state, [reader])
  signature = decode64(document.pop("signature"))
  device_key.public_key().verify(signature, canonical_json(document).encode())
  signed_document = document | {"signature": b64url(signature)}
  recovered = recover_state(device_key, signed_document, [reader])
  assert recovered["keys"][0]["key"] == state["keys"][0]["key"]

  grant = document["keys"][0]["grants"][1]
  shared = x25519_private(reader_key).exchange(x25519_public(owner))
  wrap_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=f"asius-data-access-v1:{owner}:{grant['keyId']}".encode(),
                  info=f"{owner}:{reader}".encode()).derive(shared)
  aad = {**{key: grant[key] for key in ("v", "alg", "from", "to", "keyId", "iv")}, "owner": owner}
  folder_key = AESGCM(wrap_key).decrypt(decode64(grant["iv"]), decode64(grant["ciphertext"]), canonical_json(aad).encode())

  path = "routes/2026-08-10--12-00-00--0/qlog.zst"
  plaintext = b"route bytes"
  encrypted = encrypt_object(owner, path, plaintext, state)
  assert encrypted.startswith(ENCRYPTION_MAGIC)
  offset = len(ENCRYPTION_MAGIC)
  header_length = int.from_bytes(encrypted[offset:offset + 4], "big")
  header = json.loads(encrypted[offset + 4:offset + 4 + header_length])
  object_key = HKDF(
    algorithm=hashes.SHA256(), length=32, salt=f"asius-data-object-v1:{owner}:{header['keyId']}".encode(), info=path.encode()
  ).derive(folder_key)
  assert AESGCM(object_key).decrypt(decode64(header["iv"]), encrypted[offset + 4 + header_length:], canonical_json(header).encode()) == plaintext


def test_revocation_rotates_active_key() -> None:
  first = public_identity(ed25519.Ed25519PrivateKey.generate())
  second = public_identity(ed25519.Ed25519PrivateKey.generate())
  state = initial_state([first, second])
  active = state["activeKey"]

  added = update_state(state, [first, second, public_identity(ed25519.Ed25519PrivateKey.generate())])
  assert added["activeKey"] == active
  revoked = update_state(added, [second])
  assert revoked["activeKey"] != active
  assert len(revoked["keys"]) == 2


def test_rejects_low_order_x25519_recipient() -> None:
  device_key = ed25519.Ed25519PrivateKey.generate()
  low_order_identity = bytes_to_identity(b"\x01" + b"\x00" * 31)
  assert not can_wrap_for(device_key, low_order_identity)


def test_data_uploader_reuses_default_uploader_flow() -> None:
  assert issubclass(DataUploader, Uploader)
  assert DataUploader.step is Uploader.step
  assert DataUploader.upload is Uploader.upload


def test_sharing_adds_only_the_asius_data_recipient(monkeypatch) -> None:
  device_key = ed25519.Ed25519PrivateKey.generate()
  app_reader = public_identity(ed25519.Ed25519PrivateKey.generate())
  asius_reader = public_identity(ed25519.Ed25519PrivateKey.generate())
  published = {}

  class FakeParams:
    state = None

    def get_bool(self, key):
      assert key == "ShareDrivingData"
      return True

    def get(self, key):
      assert key == "DataUploadState"
      return self.state

    def put(self, key, value, block=False):
      assert key == "DataUploadState" and block
      self.state = value

  class Client:
    base_url = "https://storage.example"

    def get_config(self):
      return {"sharingPublicKey": asius_reader}

    def put_access(self, document):
      published.update(document)

  monkeypatch.setattr("openpilot.system.loggerd.data_uploader.load_authorized_peers", lambda: {app_reader})
  uploader = DataUploader.__new__(DataUploader)
  uploader.private_key = device_key
  uploader.owner = public_identity(device_key)
  uploader.params = FakeParams()
  uploader.client = Client()

  state = uploader.sync_access()
  assert state["readers"] == sorted([app_reader, asius_reader])
  recipients = {grant["to"] for grant in published["keys"][0]["grants"]}
  assert recipients == {uploader.owner, app_reader, asius_reader}


def test_custom_upload_step_only_adds_compression_encryption_and_transport(tmp_path) -> None:
  owner_key = ed25519.Ed25519PrivateKey.generate()
  owner = public_identity(owner_key)
  state = initial_state([])
  source = tmp_path / "qlog"
  source.write_bytes(b"uncompressed log bytes" * 100)
  captured = {}

  class Client:
    def upload(self, path, encrypted, *, plaintext_length, route_start_time):
      captured.update(path=path, encrypted=encrypted, plaintext_length=plaintext_length, route_start_time=route_start_time)
      return SimpleNamespace(status_code=204)

  uploader = DataUploader.__new__(DataUploader)
  uploader.owner = owner
  uploader.client = Client()
  uploader.sync_access = lambda: state

  response = uploader.do_upload("2026-08-10--12-00-00--0/qlog.zst", str(source))
  assert response.status_code == 204
  assert captured["path"] == "routes/2026-08-10--12-00-00--0/qlog.zst"
  assert captured["plaintext_length"] < source.stat().st_size
  assert captured["route_start_time"] > 0
  assert captured["encrypted"].startswith(ENCRYPTION_MAGIC)
  assert b"uncompressed log bytes" not in captured["encrypted"]


def test_completed_upload_retry_does_not_block_the_queue(tmp_path):
  import requests
  import pytest

  source = tmp_path / 'qlog.zst'
  source.write_bytes(b'compressed log')
  key = '00000001--abc--0/qlog.zst'
  conflict = requests.Response()
  conflict.status_code = 409
  existing = requests.Response()
  existing.status_code = 200
  uploader = DataUploader.__new__(DataUploader)
  uploader.owner = 'owner'

  def upload(*_):
    raise requests.HTTPError(response=conflict)

  uploader._upload_recording = upload
  uploader.client = SimpleNamespace(request=lambda method, path: existing)
  existing._content = json.dumps({'files': [{'path': f'routes/{key}', 'uploadedAt': 1, 'plaintextLength': source.stat().st_size}]}).encode()
  assert uploader.do_upload(key, str(source)).status_code == 200
  existing._content = json.dumps({'files': [{'path': f'routes/{key}', 'uploadedAt': 1, 'plaintextLength': 1}]}).encode()
  with pytest.raises(requests.HTTPError):
    uploader.do_upload(key, str(source))


def make_policy_uploader(root):
  uploader = DataUploader.__new__(DataUploader)
  uploader.root = str(root)
  uploader.params = SimpleNamespace(get=lambda _: None)
  uploader.immediate_folders = []
  uploader.immediate_priority = {'qlog': 0, 'qlog.zst': 0, 'qcamera.mp4': 1, 'qcamera.ts': 1}
  return uploader


def test_only_preview_and_qlog_are_automatic_on_all_networks(tmp_path):
  from openpilot.system.loggerd.data_upload_queue import ROUTE_FILES, AUTO_UPLOAD_FILES
  segment = tmp_path / '00000001--abc--0'
  segment.mkdir()
  for name in ROUTE_FILES:
    (segment / name).write_bytes(b'recording')
  uploader = make_policy_uploader(tmp_path)
  for metered in (False, True):
    assert {name for name, _, _ in uploader.list_upload_files(metered)} == AUTO_UPLOAD_FILES
  assert uploader.upload_all_files is False


def test_requested_original_survives_restart_and_uploads_ahead_of_backlog(tmp_path):
  from openpilot.system.loggerd.data_upload_queue import request_uploads, upload_status, UPLOAD_ATTR_NAME, ATTR_VALUE
  from openpilot.system.loggerd.xattr_cache import setxattr
  segment = tmp_path / '00000001--abc--0'
  segment.mkdir()
  for name in ('qlog.zst', 'qcamera.mp4', 'fcamera.mp4', 'rlog.zst'):
    (segment / name).write_bytes(b'recording')
  uploader = make_policy_uploader(tmp_path)
  assert uploader.next_file_to_upload(False)[0] == 'qlog.zst'
  path = '00000001--abc--0/rlog.zst'
  assert request_uploads(tmp_path, [path]) == {'queued': [path], 'uploaded': []}
  uploader = make_policy_uploader(tmp_path)
  assert uploader.next_file_to_upload(False)[1] == path
  assert upload_status(segment / 'rlog.zst') == {'uploaded': False, 'uploadRequested': True}
  setxattr(str(segment / 'rlog.zst'), UPLOAD_ATTR_NAME, ATTR_VALUE)
  assert upload_status(segment / 'rlog.zst') == {'uploaded': True, 'uploadRequested': False}
  assert request_uploads(tmp_path, [path]) == {'queued': [], 'uploaded': [path]}
  assert uploader.next_file_to_upload(False)[0] == 'qlog.zst'
  assert all(name != 'fcamera.mp4' for name, _, _ in uploader.list_upload_files(False))


def test_requests_reject_unsafe_or_incomplete_recordings(tmp_path):
  import pytest
  from openpilot.system.loggerd.data_upload_queue import request_uploads, upload_status
  segment = tmp_path / '00000001--abc--0'
  segment.mkdir()
  (segment / 'fcamera.mp4').write_bytes(b'video')
  (segment / 'ecamera.mp4').symlink_to(segment / 'fcamera.mp4')
  good = '00000001--abc--0/fcamera.mp4'
  for path in ('../secret', '/etc/passwd', '00000001--abc--0/unknown', '00000001--abc--0/ecamera.mp4'):
    with pytest.raises(ValueError):
      request_uploads(tmp_path, [good, path])
    assert not upload_status(segment / 'fcamera.mp4')['uploadRequested']
  (segment / 'rlog.lock').touch()
  with pytest.raises(ValueError, match='still being written'):
    request_uploads(tmp_path, [good])
  assert not list(make_policy_uploader(tmp_path).list_upload_files(False))


def test_damaged_recording_does_not_block_other_automatic_uploads(tmp_path):
  import pytest
  from openpilot.system.loggerd.data_upload_queue import request_uploads, upload_status
  for index in (0, 1):
    segment = tmp_path / f'00000001--abc--{index}'
    segment.mkdir()
    (segment / 'qcamera.mp4').write_bytes(b'incomplete MP4')
  uploader = make_policy_uploader(tmp_path)
  damaged = '00000001--abc--0/qcamera.mp4'

  def fail(*_):
    raise ValueError('invalid MP4 box size')

  uploader._upload_recording = fail
  with pytest.raises(ValueError):
    uploader.do_upload(damaged, str(tmp_path / damaged))
  assert upload_status(tmp_path / damaged)['uploadError'] == 'Recording is incomplete or invalid'
  assert uploader.next_file_to_upload(False)[1] == '00000001--abc--1/qcamera.mp4'
  assert make_policy_uploader(tmp_path).next_file_to_upload(False)[1] == '00000001--abc--1/qcamera.mp4'
  request_uploads(tmp_path, [damaged])
  assert uploader.next_file_to_upload(False)[1] == damaged


def test_transient_upload_failure_defers_only_that_file(tmp_path):
  import pytest
  import requests
  from openpilot.system.loggerd.data_upload_queue import upload_status
  segment = tmp_path / '00000001--abc--0'
  segment.mkdir()
  for name in ('qlog.zst', 'qcamera.mp4'):
    (segment / name).write_bytes(b'recording')
  uploader = make_policy_uploader(tmp_path)
  key = '00000001--abc--0/qlog.zst'

  def fail(*_):
    raise requests.ConnectionError('network unavailable')

  uploader._upload_recording = fail
  with pytest.raises(requests.ConnectionError):
    uploader.do_upload(key, str(tmp_path / key))
  assert upload_status(tmp_path / key)['uploadError'] == 'Upload failed; retry scheduled'
  assert uploader.next_file_to_upload(False)[0] == 'qcamera.mp4'
  uploader.retry_after[str(tmp_path / key)] = 0
  assert uploader.next_file_to_upload(False)[0] == 'qlog.zst'


def test_video_upload_uses_recording_time_and_keeps_temporary_files_outside_route(tmp_path):
  import os
  from pathlib import Path
  from openpilot.system.loggerd.tests.media_fixture import make_video
  root = tmp_path / 'logs'
  segment = root / '00000001--abc--0'
  segment.mkdir(parents=True)
  source = segment / 'qcamera.mp4'
  make_video(source, 1)
  os.utime(source, (1700000000, 1700000000))
  directory_mtime = segment.stat().st_mtime_ns
  uploader = make_policy_uploader(root)
  uploader.private_key = ed25519.Ed25519PrivateKey.generate()
  uploader.owner = public_identity(uploader.private_key)
  uploader.sync_access = lambda: initial_state([])

  def upload_file(path, filename, **options):
    assert Path(filename).parent == tmp_path
    assert options['route_start_time'] == 1700000000000
    assert options['media']['owner'] == uploader.owner
    return SimpleNamespace(status_code=200)

  uploader.client = SimpleNamespace(upload_file=upload_file)
  assert uploader.do_upload('00000001--abc--0/qcamera.mp4', str(source)).status_code == 200
  assert segment.stat().st_mtime_ns == directory_mtime

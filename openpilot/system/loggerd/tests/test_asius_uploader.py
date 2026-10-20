import hashlib
import base64
import json
from types import SimpleNamespace

import pytest

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
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
)
from openpilot.system.loggerd.asius_uploader import AsiusUploader
from openpilot.system.loggerd.uploader import Uploader
from openpilot.system.asius.recording_keys import decode_public, public_text


@pytest.fixture(autouse=True)
def isolated_recording_key(tmp_path, monkeypatch):
  monkeypatch.setattr("openpilot.system.asius.recording_keys.KEY_PATH", tmp_path / "recording.pem")


def reader_record():
  return {"publicKey": public_identity(ed25519.Ed25519PrivateKey.generate()), "storagePublicKey": public_text(ec.generate_private_key(ec.SECP256R1()))}


def decode64(value: str) -> bytes:
  return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def test_access_document_and_object_crypto() -> None:
  device_key = ed25519.Ed25519PrivateKey.generate()
  reader_key = ec.generate_private_key(ec.SECP256R1())
  owner = public_identity(device_key)
  reader = reader_record() | {"storagePublicKey": public_text(reader_key)}
  state = initial_state([reader])

  document = access_document(device_key, state, [reader])
  signature = decode64(document.pop("signature"))
  device_key.public_key().verify(signature, canonical_json(document).encode())
  signed_document = document | {"signature": b64url(signature)}
  recovered = recover_state(device_key, signed_document, [reader])
  assert recovered["keys"][0]["key"] == state["keys"][0]["key"]

  grant = next(grant for grant in document["keys"][0]["grants"] if grant["to"] == reader["publicKey"])
  shared = reader_key.exchange(ec.ECDH(), decode_public(grant["ephemeral"]))
  wrap_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=f"asius-data-access-p256-v1:{owner}:{grant['keyId']}".encode(),
                  info=f"{owner}:{reader['publicKey']}".encode()).derive(shared)
  aad = {**{key: grant[key] for key in ("v", "alg", "from", "to", "keyId", "iv", "ephemeral")}, "owner": owner}
  folder_key = AESGCM(wrap_key).decrypt(decode64(grant["iv"]), decode64(grant["ciphertext"]), canonical_json(aad).encode())

  path = "routes/00000001--1234567890/0/qlog.zst"
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
  first = reader_record()
  second = reader_record()
  state = initial_state([first, second])
  active = state["activeKey"]

  added = update_state(state, [first, second, reader_record()])
  assert added["activeKey"] == active
  revoked = update_state(added, [second])
  assert revoked["activeKey"] != active
  assert len(revoked["keys"]) == 2


def test_rejects_invalid_p256_recipient() -> None:
  assert not can_wrap_for({"storagePublicKey": b64url(b"\x04" + b"\x00" * 64)})
  assert not can_wrap_for({})


def test_asius_uploader_reuses_default_uploader_flow() -> None:
  assert issubclass(AsiusUploader, Uploader)
  assert AsiusUploader.step is Uploader.step
  assert AsiusUploader.upload is Uploader.upload


def test_sharing_adds_only_the_asius_data_recipient(monkeypatch) -> None:
  device_key = ed25519.Ed25519PrivateKey.generate()
  app_reader = reader_record()
  asius_reader = reader_record()
  published = {}

  class FakeParams:
    state = None

    def get_bool(self, key):
      assert key == "ShareDrivingData"
      return True

    def get(self, key):
      assert key == "CloudUploadState"
      return self.state

    def put(self, key, value, block=False):
      assert key == "CloudUploadState" and block
      self.state = value

  class Client:
    base_url = "https://storage.example"

    def get_config(self):
      return {"sharingPublicKey": asius_reader["publicKey"], "sharingStoragePublicKey": asius_reader["storagePublicKey"]}

    def put_access(self, document):
      published.update(document)

  monkeypatch.setattr("openpilot.system.loggerd.asius_uploader.load_authorized_peers", lambda: {app_reader["publicKey"]: app_reader})
  uploader = AsiusUploader.__new__(AsiusUploader)
  uploader.private_key = device_key
  uploader.owner = public_identity(device_key)
  uploader.params = FakeParams()
  uploader.client = Client()

  state = uploader.sync_access()
  assert state["readers"] == sorted([app_reader, asius_reader], key=lambda reader: reader["publicKey"])
  recipients = {grant["to"] for grant in published["keys"][0]["grants"]}
  assert recipients == {uploader.owner, app_reader["publicKey"], asius_reader["publicKey"]}
  assert published["authorizedApps"] == [app_reader["publicKey"]]
  assert state["authorizedApps"] == [app_reader["publicKey"]]


def test_app_permissions_update_without_changing_recording_keys() -> None:
  reader = reader_record()
  state = initial_state([reader])
  upgraded = update_state(state, [reader], [reader["publicKey"]])
  assert upgraded["version"] == state["version"] + 1
  assert upgraded["keys"] == state["keys"]
  assert update_state(upgraded, [reader], [reader["publicKey"]]) is upgraded
  revoked = update_state(upgraded, [reader], [])
  assert revoked["version"] == upgraded["version"] + 1
  assert revoked["authorizedApps"] == []
  assert revoked["keys"] == state["keys"]

  # Refresh an already uploaded access document once to add the explicit owner list.
  previous = {key: value for key, value in state.items() if key != "authorizedApps"}
  refreshed = update_state(previous, [reader], [reader["publicKey"]])
  assert refreshed["version"] == previous["version"] + 1
  device_key = ed25519.Ed25519PrivateKey.generate()
  document = access_document(device_key, refreshed, [reader])
  recovered = recover_state(device_key, document, [reader], [reader["publicKey"]])
  assert recovered["authorizedApps"] == [reader["publicKey"]]
  assert recovered["keys"] == refreshed["keys"]


def test_custom_upload_step_only_adds_compression_encryption_and_transport(tmp_path) -> None:
  owner_key = ed25519.Ed25519PrivateKey.generate()
  owner = public_identity(owner_key)
  state = initial_state([])
  source = tmp_path / "qlog"
  source.write_bytes(b"uncompressed log bytes" * 100)
  captured = {}

  class Client:
    def upload(self, path, encrypted, *, plaintext_length, source_checksum=None):
      captured.update(path=path, encrypted=encrypted, plaintext_length=plaintext_length)
      return SimpleNamespace(status_code=204)

  uploader = AsiusUploader.__new__(AsiusUploader)
  uploader.owner = owner
  uploader.client = Client()
  uploader.root = str(tmp_path)
  uploader.sync_summary = lambda route_id: None
  uploader.sync_access = lambda: state

  response = uploader.do_upload("00000001--1234567890--0/qlog.zst", str(source))
  assert response.status_code == 204
  assert captured["path"] == "routes/00000001--1234567890/0/qlog.zst"
  assert captured["plaintext_length"] < source.stat().st_size
  assert captured["encrypted"].startswith(ENCRYPTION_MAGIC)
  assert b"uncompressed log bytes" not in captured["encrypted"]


def test_completed_upload_retry_does_not_block_the_queue(tmp_path):
  import requests
  import pytest

  source = tmp_path / 'qlog.zst'
  source.write_bytes(b'compressed log')
  key = '00000001--1234567890--0/qlog.zst'
  conflict = requests.Response()
  conflict.status_code = 409
  existing = requests.Response()
  existing.status_code = 200
  uploader = AsiusUploader.__new__(AsiusUploader)
  uploader.owner = 'owner'

  def upload(*_):
    raise requests.HTTPError(response=conflict)

  uploader._upload_recording = upload
  completions = []
  def request(method, path, body=None):
    if method == 'POST':
      completions.append((path, body))
    return existing
  uploader.client = SimpleNamespace(request=request)
  existing._content = json.dumps({'files': [{
    'path': 'routes/00000001--1234567890/0/qlog.zst', 'uploadedAt': 1, 'sourceSha256': b64url(hashlib.sha256(source.read_bytes()).digest()),
    'checksumSha256': 'encrypted-checksum',
  }]}).encode()
  assert uploader.do_upload(key, str(source)).status_code == 200
  assert completions == [('/v1/owner/uploads/complete', {'path': 'routes/00000001--1234567890/0/qlog.zst', 'checksumSha256': 'encrypted-checksum'})]
  existing._content = json.dumps({'files': [{'path': 'routes/00000001--1234567890/0/qlog.zst', 'uploadedAt': 1, 'plaintextLength': 1}]}).encode()
  with pytest.raises(requests.HTTPError):
    uploader.do_upload(key, str(source))


def make_policy_uploader(root):
  from openpilot.system.loggerd.route_summary import summary_path
  summary = summary_path(root, '00000001--1234567890')
  summary.parent.mkdir(parents=True, exist_ok=True)
  summary.write_text('{}')
  uploader = AsiusUploader.__new__(AsiusUploader)
  uploader.root = str(root)
  uploader.params = SimpleNamespace(get=lambda _: None)
  uploader.immediate_folders = []
  uploader.immediate_priority = {'qlog': 0, 'qlog.zst': 0, 'qcamera.mp4': 1}
  return uploader


def test_only_preview_and_qlog_are_automatic_on_all_networks(tmp_path):
  from openpilot.system.loggerd.data_upload_queue import ROUTE_FILES, AUTO_UPLOAD_FILES
  assert AUTO_UPLOAD_FILES == {'qlog', 'qlog.zst', 'qcamera.mp4'}
  assert ROUTE_FILES == AUTO_UPLOAD_FILES | {'rlog', 'rlog.zst', 'fcamera.mp4', 'ecamera.mp4', 'dcamera.mp4'}
  segment = tmp_path / '00000001--1234567890--0'
  segment.mkdir()
  for name in ROUTE_FILES:
    (segment / name).write_bytes(b'recording')
  uploader = make_policy_uploader(tmp_path)
  for metered in (False, True):
    assert {name for name, _, _ in uploader.list_upload_files(metered)} == AUTO_UPLOAD_FILES


def test_requested_original_survives_restart_and_uploads_ahead_of_backlog(tmp_path):
  from openpilot.system.loggerd.data_upload_queue import request_uploads, upload_status, UPLOAD_ATTR_NAME, ATTR_VALUE
  from openpilot.system.loggerd.xattr_cache import setxattr
  segment = tmp_path / '00000001--1234567890--0'
  segment.mkdir()
  for name in ('qlog.zst', 'qcamera.mp4', 'fcamera.mp4', 'rlog.zst'):
    (segment / name).write_bytes(b'recording')
  uploader = make_policy_uploader(tmp_path)
  assert uploader.next_file_to_upload(False)[0] == 'qlog.zst'
  path = '00000001--1234567890--0/rlog.zst'
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
  segment = tmp_path / '00000001--1234567890--0'
  segment.mkdir()
  (segment / 'fcamera.mp4').write_bytes(b'video')
  (segment / 'ecamera.mp4').symlink_to(segment / 'fcamera.mp4')
  good = '00000001--1234567890--0/fcamera.mp4'
  for path in ('../secret', '/etc/passwd', '00000001--1234567890--0/unknown', '00000001--1234567890--0/ecamera.mp4'):
    with pytest.raises(ValueError):
      request_uploads(tmp_path, [good, path])
    assert not upload_status(segment / 'fcamera.mp4')['uploadRequested']
  (segment / 'rlog.lock').touch()
  with pytest.raises(ValueError, match='still being written'):
    request_uploads(tmp_path, [good])
  assert not list(make_policy_uploader(tmp_path).list_upload_files(False))


def test_stale_locks_release_finished_recordings_without_deleting_data(tmp_path, monkeypatch):
  from openpilot.system.loggerd import data_upload_queue as queue
  segment = tmp_path / '00000001--1234567890--0'
  segment.mkdir()
  recording = segment / 'qcamera.mp4'
  recording.write_bytes(b'recorded video')
  lock = segment / 'qcamera.mp4.lock'
  lock.touch()
  uploader = make_policy_uploader(tmp_path)
  assert uploader.next_file_to_upload(False) is None
  monkeypatch.setattr(queue.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1))
  assert queue.clear_stale_locks(tmp_path)
  assert not lock.exists()
  assert recording.read_bytes() == b'recorded video'
  assert uploader.next_file_to_upload(False)[1] == '00000001--1234567890--0/qcamera.mp4'


@pytest.mark.parametrize('returncode', [0, 2])
def test_active_logger_or_failed_process_check_preserves_locks(tmp_path, monkeypatch, returncode):
  from openpilot.system.loggerd import data_upload_queue as queue
  segment = tmp_path / '00000001--1234567890--0'
  segment.mkdir()
  lock = segment / 'rlog.lock'
  lock.touch()
  monkeypatch.setattr(queue.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=returncode))
  assert not queue.clear_stale_locks(tmp_path)
  assert lock.exists()


def test_logger_starting_after_snapshot_keeps_its_new_locks(tmp_path, monkeypatch):
  from openpilot.system.loggerd import data_upload_queue as queue
  old = tmp_path / '00000001--1234567890--0'
  old.mkdir()
  (old / 'rlog.lock').touch()
  new = tmp_path / '00000002--def--0'

  def process_check(*args, **kwargs):
    new.mkdir()
    (new / 'rlog.lock').touch()
    return SimpleNamespace(returncode=1)

  monkeypatch.setattr(queue.subprocess, 'run', process_check)
  assert queue.clear_stale_locks(tmp_path)
  assert not (old / 'rlog.lock').exists()
  assert (new / 'rlog.lock').exists()


def test_lock_cleanup_does_not_follow_route_symlinks(tmp_path):
  from openpilot.system.loggerd.data_upload_queue import clear_stale_locks
  root = tmp_path / 'recordings'
  root.mkdir()
  external = tmp_path / 'external'
  external.mkdir()
  (external / 'rlog.lock').touch()
  (root / '00000001--1234567890--0').symlink_to(external, target_is_directory=True)
  assert clear_stale_locks(root)
  assert (external / 'rlog.lock').exists()


def test_damaged_recording_does_not_block_other_automatic_uploads(tmp_path):
  import pytest
  from openpilot.system.loggerd.data_upload_queue import request_uploads, upload_status
  for index in (0, 1):
    segment = tmp_path / f'00000001--1234567890--{index}'
    segment.mkdir()
    (segment / 'qcamera.mp4').write_bytes(b'incomplete MP4')
  uploader = make_policy_uploader(tmp_path)
  damaged = '00000001--1234567890--0/qcamera.mp4'

  def fail(*_):
    raise ValueError('invalid MP4 box size')

  uploader._upload_recording = fail
  with pytest.raises(ValueError):
    uploader.do_upload(damaged, str(tmp_path / damaged))
  assert upload_status(tmp_path / damaged)['uploadError'] == 'Recording is incomplete or invalid'
  assert uploader.next_file_to_upload(False)[1] == '00000001--1234567890--1/qcamera.mp4'
  assert make_policy_uploader(tmp_path).next_file_to_upload(False)[1] == '00000001--1234567890--1/qcamera.mp4'
  request_uploads(tmp_path, [damaged])
  assert uploader.next_file_to_upload(False)[1] == damaged


def test_transient_upload_failure_defers_only_that_file(tmp_path):
  import pytest
  import requests
  from openpilot.system.loggerd.data_upload_queue import upload_status
  segment = tmp_path / '00000001--1234567890--0'
  segment.mkdir()
  for name in ('qlog.zst', 'qcamera.mp4'):
    (segment / name).write_bytes(b'recording')
  uploader = make_policy_uploader(tmp_path)
  key = '00000001--1234567890--0/qlog.zst'

  def fail(*_):
    raise requests.ConnectionError('network unavailable')

  uploader._upload_recording = fail
  with pytest.raises(requests.ConnectionError):
    uploader.do_upload(key, str(tmp_path / key))
  assert upload_status(tmp_path / key)['uploadError'] == 'Upload failed; retry scheduled'
  assert uploader.next_file_to_upload(False)[0] == 'qcamera.mp4'
  uploader.retry_after[str(tmp_path / key)] = 0
  assert uploader.next_file_to_upload(False)[0] == 'qlog.zst'

@pytest.mark.parametrize('status', [403, 507])
def test_cloud_policy_pauses_all_files_and_resumes_without_losing_requests(tmp_path, monkeypatch, status):
  import requests
  from openpilot.system.loggerd.data_upload_queue import request_uploads, upload_status
  from openpilot.system.loggerd import upload_progress
  monkeypatch.setattr(upload_progress, 'STATUS_PATH', tmp_path / 'progress.json')
  segment = tmp_path / '00000001--1234567890--0'
  segment.mkdir()
  for name in ('qlog.zst', 'qcamera.mp4'):
    (segment / name).write_bytes(b'recording')
  key = '00000001--1234567890--0/qlog.zst'
  request_uploads(tmp_path, [key])
  uploader = make_policy_uploader(tmp_path)

  def denied(*_):
    response = requests.Response()
    response.status_code = status
    response._content = b'{"error":"Cloud storage policy blocked this upload"}'
    response.raise_for_status()

  uploader._upload_recording = denied
  with pytest.raises(requests.HTTPError):
    uploader.do_upload(key, str(tmp_path / key))
  assert uploader.next_file_to_upload(False) is None
  assert upload_status(tmp_path / key)['uploadRequested']
  assert not upload_status(tmp_path / key)['uploaded']
  assert upload_progress.read_progress()['error'] == 'Cloud storage policy blocked this upload'
  assert upload_progress.read_progress()['path'] == 'routes/00000001--1234567890/0/qlog.zst'
  uploader.policy_retry_after = 0
  uploader.retry_after[str(tmp_path / key)] = 0
  assert uploader.next_file_to_upload(False)[1] == key


@pytest.mark.parametrize('name,codec', [('qcamera.mp4', 'libx264'), ('qcamera.mp4', 'libx265'), ('fcamera.mp4', 'libx265'),
                                       ('ecamera.mp4', 'libx265'), ('dcamera.mp4', 'libx265')])
def test_video_upload_uses_recording_time_and_keeps_temporary_files_outside_route(tmp_path, name, codec):
  import os
  from pathlib import Path
  from openpilot.system.loggerd.tests.media_fixture import make_video
  root = tmp_path / 'logs'
  segment = root / '00000001--1234567890--0'
  segment.mkdir(parents=True)
  source = segment / name
  make_video(source, 1, codec=codec)
  os.utime(source, (1700000000, 1700000000))
  directory_mtime = segment.stat().st_mtime_ns
  uploader = make_policy_uploader(root)
  uploader.private_key = ed25519.Ed25519PrivateKey.generate()
  uploader.owner = public_identity(uploader.private_key)
  uploader.sync_access = lambda: initial_state([])
  uploader.sync_summary = lambda route_id: None

  def upload_file(path, filename, **options):
    assert Path(filename).parent == tmp_path
    data = Path(filename).read_bytes()
    assert data[4:24] == b'uuidASIUS-INDEX-V001'
    media = json.loads(data[28:28+int.from_bytes(data[24:28],'big')])
    assert path == f'routes/00000001--1234567890/0/{name}'
    assert media['owner'] == uploader.owner
    assert media['alg'] == 'CENC-AES-CTR'
    assert media['codec'].startswith('avc1.' if codec == 'libx264' else ('hvc1.', 'hev1.'))
    unsigned = {key: value for key, value in media.items() if key != 'signature'}
    uploader.private_key.public_key().verify(decode64(media['signature']), canonical_json(unsigned).encode())
    return SimpleNamespace(status_code=200)

  uploader.client = SimpleNamespace(upload_file=upload_file)
  assert uploader.do_upload(f'00000001--1234567890--0/{name}', str(source)).status_code == 200
  assert segment.stat().st_mtime_ns == directory_mtime


def test_missing_summary_does_not_block_ready_routes(tmp_path):
  from openpilot.system.loggerd.route_summary import summary_path
  from openpilot.system.loggerd.data_upload_queue import upload_failure
  uploader = make_policy_uploader(tmp_path)
  ready = '00000001--1234567890'
  waiting = '00000002--1234567890'
  items = []
  for route in (waiting, ready):
    path = tmp_path / f'{route}--0' / 'qcamera.mp4'
    path.parent.mkdir()
    path.write_bytes(b'recording')
    items.append(('qcamera.mp4', f'{route}--0/qcamera.mp4', str(path)))
  uploader.list_upload_files = lambda _: iter(items)
  assert uploader.next_file_to_upload(False) == items[1]
  assert not upload_failure(items[0][2])
  summary_path(tmp_path, waiting).write_text('{}')
  assert uploader.next_file_to_upload(False) == items[0]

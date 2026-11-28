from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import random
import tempfile
from urllib.parse import quote
import threading
import time
from collections.abc import Iterator
from typing import Any

import requests

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware.hw import Paths
from openpilot.common.params import Params
from openpilot.common.realtime import set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import get_upload_stream
from openpilot.system.asius.identity import get_device_private_key, is_identity
from openpilot.system.asius.relayd import load_authorized_apps
from openpilot.system.asius.session import unb64_lenient
from openpilot.system.loggerd.storage_api import (
  StorageApiClient,
  access_document,
  b64url,
  can_wrap_for,
  encrypt_object,
  initial_state,
  recover_state,
  update_state,
)
from openpilot.system.loggerd.storage_media import derive_cenc_key, package_cenc_mp4, package_plain_mp4, embed_index
from openpilot.system.loggerd.storage_upload_queue import (
  AUTO_UPLOAD_FILES, MAX_FILE_BYTES, REQUEST_ATTR_NAME, ROUTE_FILES, ROUTE_SEGMENT_RE, UPLOAD_ATTR_NAME, attribute_set,
  clear_failure, clear_stale_locks, record_failure, upload_failure,
)
from openpilot.system.loggerd.route_summary import summary_path, storage_path, save_json, recover_counters
from openpilot.system.loggerd.uploader import NetworkType, Uploader, allow_sleep, force_wifi
from openpilot.system.loggerd.upload_progress import publish_error

STATE_PARAM = "StorageUploadState"
ENCRYPTION_PARAM = "StorageEncryption"
CENC_FILES = {"qcamera.mp4", "fcamera.mp4", "ecamera.mp4", "dcamera.mp4"}


class StorageUploader(Uploader):
  upload_attr_name = UPLOAD_ATTR_NAME
  max_file_size = MAX_FILE_BYTES

  def __init__(self, dongle_id: str, root: str, params: Params | None = None, client: StorageApiClient | None = None):
    self.private_key = get_device_private_key()
    self.owner = self.private_key.identity()
    if dongle_id != self.owner:
      raise ValueError("DongleId does not match the device identity")

    super().__init__(dongle_id, root)
    self.immediate_folders = []
    self.immediate_priority = {"qlog": 0, "qlog.zst": 0, "qcamera.mp4": 1}
    self.params = params or self.params
    self.client = client or StorageApiClient(self.params.get("StorageHost", return_default=True), self.private_key)
    self.retry_after: dict[str, float] = {}
    self.counter_conflict = None
    self.policy_retry_after = 0.0

  def defer_storage_policy(self, error: Exception, path: str = "") -> bool:
    if not isinstance(error, requests.HTTPError) or error.response is None or error.response.status_code not in (403, 507):
      return False
    # Keep requests queued. Recheck after a bounded delay so support approval or
    # deleting recordings resumes uploads without marking any file as complete.
    self.policy_retry_after = time.monotonic() + 60
    try:
      message = str(error.response.json().get("error", "Storage uploads are paused"))
    except (ValueError, AttributeError):
      message = "Storage uploads are paused"
    publish_error(path, message)
    return True

  def list_upload_files(self, metered: bool) -> Iterator[tuple[str, str, str]]:
    # Preview and qlog are automatic on network connections. Viewing a route
    # does not request any other files, even on an unmetered connection.
    for name, key, filename in super().list_upload_files(False):
      folder = key.split("/", 1)[0]
      route_id = folder.rsplit("--", 1)[0]
      if summary_path(self.root, route_id).with_suffix(".deleted").exists():
        continue
      if name not in ROUTE_FILES or not ROUTE_SEGMENT_RE.fullmatch(folder):
        continue
      if os.path.islink(filename) or os.path.islink(os.path.dirname(filename)):
        continue
      try:
        if name in AUTO_UPLOAD_FILES or attribute_set(filename, REQUEST_ATTR_NAME):
          yield name, key, filename
      except FileNotFoundError:
        continue

  def route_encrypted(self, route_id: str) -> bool:
    """Return the route encryption mode. The first summary upload fixes it with the local route state."""
    path = summary_path(self.root, route_id).with_suffix('.encryption')
    if path.exists():
      encrypted = json.loads(path.read_text()).get('encrypted')
      if not isinstance(encrypted, bool):
        raise ValueError("Invalid route encryption state")
      return encrypted
    # A route uploaded before this state existed is encrypted.
    encrypted = path.with_suffix('.uploaded').exists() or self.params.get(ENCRYPTION_PARAM, return_default=True) is not False
    save_json(path, {'encrypted': encrypted})
    return encrypted

  def sync_summary(self, route_id):
    path = summary_path(self.root, route_id)
    if not path.exists():
      raise RuntimeError("Waiting for local route summary")
    marker = path.with_suffix('.uploaded')
    content = path.read_bytes()
    fingerprint = hashlib.sha256(content).hexdigest()
    saved = json.loads(marker.read_text()) if marker.exists() else {}
    if saved.get('fingerprint') == fingerprint:
      return
    etag = {'etag': saved['etag']} if saved.get('etag') else {}
    state = self.sync_access()
    if self.route_encrypted(route_id):
      body = {'encrypted': b64url(encrypt_object(self.owner, storage_path(route_id), content, state)), **etag}
    else:
      body = {'summary': json.loads(content), **etag}
    for attempt in range(2):
      try:
        response = self.client.request('PUT', f'/v1/{self.owner}/routes/{route_id}', body)
        break
      except requests.HTTPError as error:
        if error.response.status_code == 410:
          path.with_suffix('.deleted').touch()
          return
        detail = error.response.json()
        if error.response.status_code == 409 and 'maxCounter' in detail:
          self.counter_conflict = (route_id, detail['maxCounter'])
        elif error.response.status_code == 409 and detail.get('encrypted') is True and 'summary' in body and attempt == 0:
          # Storage already has this route encrypted. Keep encryption; never downgrade a stored route.
          save_json(path.with_suffix('.encryption'), {'encrypted': True})
          body = {'encrypted': b64url(encrypt_object(self.owner, storage_path(route_id), content, self.sync_access())), **etag}
          continue
        elif error.response.status_code == 412 and detail.get('etag'):
          save_json(marker, {'etag': detail['etag']})
          if attempt == 0:
            body['etag'] = detail['etag']
            continue
        raise
    save_json(marker, {'fingerprint':fingerprint, 'etag':response.json()['etag']})

  def sync_summaries(self):
    journal = Path(self.root) / '.summaries' / 'rename-plan.json'
    if journal.exists() and not self.counter_conflict:
      self.counter_conflict = ('', -1)
    if self.counter_conflict:
      if recover_counters(self.root, self.params, *self.counter_conflict):
        self.counter_conflict = None
      else:
        return False
    for path in sorted((Path(self.root)/'.summaries').glob('*.json')):
      if path.stem == 'rename-plan' or path.with_suffix('.deleted').exists():
        continue
      self.sync_summary(path.stem)
    return True

  def next_file_to_upload(self, metered: bool) -> tuple[str, str, str] | None:
    if self.policy_retry_after > time.monotonic():
      return None
    automatic = None
    for item in self.list_upload_files(metered):
      route_id = item[1].split("/", 1)[0].rsplit("--", 1)[0]
      if not summary_path(self.root, route_id).exists():
        continue
      failure = upload_failure(item[2])
      if failure and (failure.get("permanent") or self.retry_after.get(item[2], 0) > time.monotonic()):
        continue
      try:
        if attribute_set(item[2], REQUEST_ATTR_NAME):
          return item
      except FileNotFoundError:
        continue
      if automatic is None:
        automatic = item
    return automatic

  def sync_access(self) -> dict:
    readers: list[dict[str, Any]] = []
    for key, app in sorted(load_authorized_apps().items()):
      reader: dict[str, Any] = {"publicKey": key, "storagePublicKey": app.get("storagePublicKey")}
      if not is_identity(key) or not can_wrap_for(reader):
        raise ValueError("Authorized app has no valid recording public key")
      readers.append(reader)
    authorized_apps = [reader['publicKey'] for reader in readers]
    if self.params.get_bool("ShareDrivingData"):
      config = self.client.get_config()
      reader = {"publicKey": config.get("sharingPublicKey"), "storagePublicKey": config.get("sharingStoragePublicKey")}
      if not is_identity(reader['publicKey']) or not can_wrap_for(reader):
        raise ValueError("Storage API returned an invalid sharing public key")
      readers.append(reader)
    readers = sorted({reader['publicKey']: reader for reader in readers}.values(), key=lambda reader: reader['publicKey'])
    state = self.params.get(STATE_PARAM)
    state = initial_state(readers, authorized_apps) if not isinstance(state, dict) else update_state(state, readers, authorized_apps)
    if state.get("publishedVersion") == state["version"] and state.get("publishedHost") == self.client.base_url:
      return state

    try:
      self.client.put_access(access_document(self.private_key, state, readers))
    except requests.HTTPError as error:
      if error.response is None or error.response.status_code != 409:
        raise
      state = recover_state(self.private_key, self.client.get_access(), readers, authorized_apps)
      self.client.put_access(access_document(self.private_key, state, readers))

    state["publishedVersion"] = state["version"]
    state["publishedHost"] = self.client.base_url
    self.params.put(STATE_PARAM, state, block=True)
    return state

  def do_upload(self, key: str, fn: str) -> requests.Response:
    try:
      response = self._upload_or_recover(key, fn)
      clear_failure(fn)
      return response
    except Exception as error:
      folder, name = key.split("/", 1)
      route, segment = folder.rsplit("--", 1)
      self.defer_storage_policy(error, f"routes/{route}/{segment}/{name}")
      if isinstance(error, requests.HTTPError) and error.response is not None and error.response.status_code == 410:
        route_id = key.split("/", 1)[0].rsplit("--", 1)[0]
        summary_path(self.root, route_id).with_suffix(".deleted").touch()
      record_failure(fn, isinstance(error, ValueError))
      self.retry_after[fn] = time.monotonic() + 60
      raise

  def _upload_or_recover(self, key: str, fn: str) -> requests.Response:
    try:
      return self._upload_recording(key, fn)
    except requests.HTTPError as error:
      if error.response is None or error.response.status_code != 409:
        raise
      # Ordinary uploads cannot replace recordings. A lost completion response (or a
      # missing local xattr) must not re-encrypt and overwrite an uploaded file.
      route_id = key.split("/", 1)[0].rsplit("--", 1)[0]
      files = []
      cursor = None
      while True:
        suffix = "?cursor=" + quote(cursor, safe="") if cursor else ""
        response = self.client.request("GET", f"/v1/{self.owner}/routes/{route_id}/files{suffix}")
        page = response.json()
        files.extend(page.get("files", []))
        cursor = page.get("cursor")
        if not cursor:
          break
      stream, _ = get_upload_stream(fn, key.endswith(".zst") and not fn.endswith(".zst"))
      try:
        source_checksum = b64url(hashlib.sha256(stream.read()).digest())
      finally:
        stream.close()
      object_path = "routes/" + key.rsplit("--", 1)[0] + "/" + key.rsplit("--", 1)[1]
      for file in files:
        if file.get("path") == object_path and file.get("uploadedAt") and file.get("sourceSha256") == source_checksum:
          return self.client.complete_upload(object_path, file["checksumSha256"])
      raise

  def _upload_recording(self, key: str, fn: str, *, replace_checksum: str | None = None) -> requests.Response:
    first_component = key.split("/", 1)[0]
    if not ROUTE_SEGMENT_RE.fullmatch(first_component):
      raise ValueError("storage uploader only accepts route segments")
    route_id, segment_text = first_component.rsplit("--", 1)
    self.sync_summary(route_id)
    if summary_path(self.root, route_id).with_suffix(".deleted").exists():
      raise RuntimeError("Route is deleted")
    state = self.sync_access()
    object_path = f"routes/{route_id}/{segment_text}/{os.path.basename(key)}"
    if not self.route_encrypted(route_id):
      return self._upload_plain(key, fn, object_path, replace_checksum)

    if os.path.basename(object_path) in CENC_FILES:
      active = next(entry for entry in state["keys"] if entry["id"] == state["activeKey"])
      folder_key = unb64_lenient(active["key"])
      content_key, kid = derive_cenc_key(folder_key, self.owner, active["id"], object_path)
      temporary_name = ""
      try:
        with tempfile.NamedTemporaryFile(prefix=".asius-cenc-", suffix=".mp4", dir=os.path.dirname(os.path.abspath(self.root)), delete=False) as temporary:
          temporary_name = temporary.name
        media = package_cenc_mp4(fn, temporary_name, content_key, kid)
        embed_index(temporary_name, media, self.owner, object_path, active["id"], self.private_key)
        checksum = b64url(hashlib.sha256(Path(temporary_name).read_bytes()).digest())
        response = self.client.upload_file(
          object_path,
          temporary_name,
          content_type="video/mp4",
          plaintext_length=os.path.getsize(fn),

          checksum=checksum,
          source_checksum=b64url(hashlib.sha256(Path(fn).read_bytes()).digest()),
          **({"replace_checksum": replace_checksum} if replace_checksum else {}),
        )
        cloudlog.event("storage_upload_cenc", key=key, path=object_path, encrypted_size=media["encryptedLength"], fragments=len(media["fragments"]))
        return response
      finally:
        if temporary_name:
          try:
            os.unlink(temporary_name)
          except FileNotFoundError:
            pass

    stream, _ = get_upload_stream(fn, key.endswith(".zst") and not fn.endswith(".zst"))
    try:
      plaintext = stream.read()
      encrypted = encrypt_object(self.owner, object_path, plaintext, state)
    finally:
      stream.close()

    response = self.client.upload(
      object_path,
      encrypted,
      plaintext_length=len(plaintext),
      source_checksum=b64url(hashlib.sha256(plaintext).digest()),
      **({"replace_checksum": replace_checksum} if replace_checksum else {}),
    )
    cloudlog.event("storage_upload_encrypted", key=key, path=object_path, encrypted_size=len(encrypted))
    return response

  def _upload_plain(self, key: str, fn: str, object_path: str, replace_checksum: str | None) -> requests.Response:
    """Upload without client-side encryption. Storage access checks stay the same."""
    replace = {"replace_checksum": replace_checksum} if replace_checksum else {}
    if os.path.basename(object_path) in CENC_FILES:
      temporary_name = ""
      try:
        with tempfile.NamedTemporaryFile(prefix=".asius-mp4-", suffix=".mp4", dir=os.path.dirname(os.path.abspath(self.root)), delete=False) as temporary:
          temporary_name = temporary.name
        # Keep the signed seek index and fragment hashes. Samples stay unencrypted.
        media = package_plain_mp4(fn, temporary_name)
        embed_index(temporary_name, media, self.owner, object_path, None, self.private_key)
        response = self.client.upload_file(
          object_path,
          temporary_name,
          content_type="video/mp4",
          plaintext_length=os.path.getsize(fn),
          checksum=b64url(hashlib.sha256(Path(temporary_name).read_bytes()).digest()),
          source_checksum=b64url(hashlib.sha256(Path(fn).read_bytes()).digest()),
          encryption=False,
          **replace,
        )
        cloudlog.event("storage_upload_plain", key=key, path=object_path, size=os.path.getsize(temporary_name), fragments=len(media["fragments"]))
        return response
      finally:
        if temporary_name:
          try:
            os.unlink(temporary_name)
          except FileNotFoundError:
            pass

    stream, _ = get_upload_stream(fn, key.endswith(".zst") and not fn.endswith(".zst"))
    try:
      plaintext = stream.read()
    finally:
      stream.close()
    checksum = b64url(hashlib.sha256(plaintext).digest())
    response = self.client.upload(object_path, plaintext, plaintext_length=len(plaintext), source_checksum=checksum, encryption=False, **replace)
    cloudlog.event("storage_upload_plain", key=key, path=object_path, size=len(plaintext))
    return response


def main(exit_event: threading.Event | None = None) -> None:
  exit_event = exit_event or threading.Event()
  try:
    set_core_affinity([0, 1, 2, 3])
  except Exception:
    cloudlog.exception("failed to set core affinity")

  params = Params()
  dongle_id = params.get("DongleId")
  if dongle_id is None:
    raise RuntimeError("storage uploader cannot start without DongleId")

  sm = messaging.SubMaster(["deviceState"])
  sm.update(1000)
  uploader = StorageUploader(dongle_id, Paths.log_root(), params=params)
  locks_cleared = False
  backoff = 0.1
  while not exit_event.is_set():
    host = params.get("StorageHost", return_default=True)
    if not params.get("StorageUploadEnabled", return_default=True):
      exit_event.wait(1)
      continue
    if host.rstrip("/") != uploader.client.base_url:
      uploader = StorageUploader(dongle_id, Paths.log_root(), params=params)
    sm.update(0)
    if sm["deviceState"].started:
      locks_cleared = False
    elif not locks_cleared and sm.valid["deviceState"]:
      try:
        locks_cleared = clear_stale_locks(uploader.root)
      except OSError:
        cloudlog.exception("failed to clear stale recording locks")
    network_type = sm["deviceState"].networkType.raw if not force_wifi else NetworkType.wifi
    if network_type == NetworkType.none:
      if allow_sleep:
        exit_event.wait(5)
      continue

    if uploader.policy_retry_after > time.monotonic():
      exit_event.wait(min(5, uploader.policy_retry_after - time.monotonic()))
      continue
    try:
      uploader.sync_access()
      if not uploader.sync_summaries():
        exit_event.wait(2)
        continue
    except Exception as error:
      uploader.defer_storage_policy(error)
      cloudlog.exception("storage access sync failed")
      if allow_sleep:
        exit_event.wait(backoff + random.uniform(0, backoff))
      backoff = min(backoff * 2, 120)
      continue

    success = uploader.step(network_type, sm["deviceState"].networkMetered)
    if success is None:
      backoff = 2
    elif success:
      backoff = 0.1
    else:
      cloudlog.info("storage upload backoff %r", backoff)
      backoff = min(backoff * 2, 120)
    if allow_sleep:
      exit_event.wait(backoff + random.uniform(0, backoff))


if __name__ == "__main__":
  main()

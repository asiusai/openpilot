#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import hashlib
import json
import os
import time
from typing import Any

import requests
from cryptography.hazmat.primitives import hashes
from openpilot.system.asius.pqc import RootIdentity
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from openpilot.system.loggerd.upload_progress import ProgressReader
from openpilot.system.asius.identity import bytes_to_identity
from openpilot.system.asius.recording_keys import decode_public, get_recording_key, public_text

PROTOCOL_VERSION = 2
ENCRYPTION_MAGIC = b"ASIUSDATA2\n"


def wall_time() -> float:
  return time.time()  # noqa: TID251


def b64url(value: bytes) -> str:
  return base64.urlsafe_b64encode(value).decode().rstrip("=")


def canonical_json(value: Any) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def public_identity(private_key: RootIdentity) -> str:
  return bytes_to_identity(private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def can_wrap_for(reader: dict[str, str]) -> bool:
  try:
    decode_public(reader['storagePublicKey'])
    return True
  except (KeyError, TypeError, ValueError):
    return False


def wrap_folder_key(owner: str, key_id: str, recipient: dict[str, str], folder_key: bytes) -> dict[str, Any]:
  shared, ciphertext = decode_public(recipient['storagePublicKey']).encapsulate()
  iv = os.urandom(12)
  aad = {"v": PROTOCOL_VERSION, "alg": "MLKEM768-HKDF-SHA384-A256GCM", "from": owner, "to": recipient['publicKey'], "owner": owner,
         "keyId": key_id, "kem": b64url(ciphertext), "iv": b64url(iv)}
  wrap_key = HKDF(algorithm=hashes.SHA384(), length=32, salt=f"asius-data-access-pq-v1:{owner}:{key_id}".encode(),
                  info=f"{owner}:{recipient['publicKey']}".encode()).derive(shared)
  ciphertext = AESGCM(wrap_key).encrypt(iv, folder_key, canonical_json(aad).encode())
  return {key: value for key, value in aad.items() if key != "owner"} | {"ciphertext": b64url(ciphertext)}


def access_document(private_key: RootIdentity, state: dict[str, Any], readers: list[dict[str, str]]) -> dict[str, Any]:
  owner = public_identity(private_key)
  device = {"publicKey": owner, "storagePublicKey": public_text(get_recording_key(private_key))}
  recipients = list({reader["publicKey"]: reader for reader in [*readers, device]}.values())
  keys = []
  for entry in state["keys"]:
    raw_key = base64.urlsafe_b64decode(entry["key"] + "=" * (-len(entry["key"]) % 4))
    keys.append({
      "id": entry["id"],
      "createdAt": entry["createdAt"],
      "grants": [wrap_folder_key(owner, entry["id"], recipient, raw_key) for recipient in recipients],
    })
  unsigned = {
    "v": PROTOCOL_VERSION,
    "owner": owner,
    "signingKey": b64url(private_key.public_key().public_bytes_raw()),
    "version": state["version"],
    "updatedAt": int(wall_time() * 1000),
    "activeKey": state["activeKey"],
    "authorizedApps": state.get("authorizedApps", []),
    "keys": keys,
  }
  return unsigned | {"signature": b64url(private_key.sign(canonical_json(unsigned).encode()))}


def recover_state(private_key: RootIdentity, document: dict[str, Any], readers: list[dict[str, str]],
                  authorized_apps: list[str] | None = None) -> dict[str, Any]:
  owner = public_identity(private_key)
  if document.get("v") != PROTOCOL_VERSION or document.get("owner") != owner or document.get("signingKey") != b64url(private_key.public_key().public_bytes_raw()):
    raise ValueError("access document owner mismatch")
  unsigned = {key: value for key, value in document.items() if key != "signature"}
  private_key.public_key().verify(base64.urlsafe_b64decode(document["signature"] + "=" * (-len(document["signature"]) % 4)), canonical_json(unsigned).encode())
  recovered = []
  for entry in document["keys"]:
    grant = next(grant for grant in entry["grants"] if grant["to"] == owner)
    if grant.get('v') != PROTOCOL_VERSION or grant.get('alg') != 'MLKEM768-HKDF-SHA384-A256GCM' or grant.get('from') != owner or grant.get('keyId') != entry['id']:
      raise ValueError('Invalid recording grant')
    shared = get_recording_key(private_key).decapsulate(base64.urlsafe_b64decode(grant['kem'] + '=' * (-len(grant['kem']) % 4)))
    wrap_key = HKDF(algorithm=hashes.SHA384(), length=32, salt=f"asius-data-access-pq-v1:{owner}:{entry['id']}".encode(),
                    info=f"{owner}:{owner}".encode()).derive(shared)
    aad = {key: value for key, value in grant.items() if key != 'ciphertext'} | {'owner': owner}
    iv = base64.urlsafe_b64decode(grant["iv"] + "=" * (-len(grant["iv"]) % 4))
    ciphertext = base64.urlsafe_b64decode(grant["ciphertext"] + "=" * (-len(grant["ciphertext"]) % 4))
    folder_key = AESGCM(wrap_key).decrypt(iv, ciphertext, canonical_json(aad).encode())
    recovered.append({"id": entry["id"], "createdAt": entry["createdAt"], "key": b64url(folder_key)})
  return {"version": document["version"] + 1, "activeKey": document["activeKey"], "readers": readers,
          "authorizedApps": sorted(set(authorized_apps or [])), "keys": recovered}


def initial_state(readers: list[dict[str, str]], authorized_apps: list[str] | None = None) -> dict[str, Any]:
  key_id = f"k_{b64url(os.urandom(18))}"
  return {
    "version": 1,
    "activeKey": key_id,
    "readers": readers,
    "authorizedApps": sorted(set(authorized_apps or [])),
    "keys": [{"id": key_id, "createdAt": int(wall_time() * 1000), "key": b64url(os.urandom(32))}],
  }


def update_state(state: dict[str, Any], readers: list[dict[str, str]], authorized_apps: list[str] | None = None) -> dict[str, Any]:
  previous = {(reader["publicKey"], reader["storagePublicKey"]) for reader in state.get("readers", [])}
  current = {(reader["publicKey"], reader["storagePublicKey"]) for reader in readers}
  apps = sorted(set(authorized_apps or []))
  if previous == current and state.get("authorizedApps") == apps:
    return state
  state = json.loads(json.dumps(state))
  state["version"] += 1
  state["readers"] = readers
  state["authorizedApps"] = apps
  if previous - current:
    key_id = f"k_{b64url(os.urandom(18))}"
    state["activeKey"] = key_id
    state["keys"].append({"id": key_id, "createdAt": int(wall_time() * 1000), "key": b64url(os.urandom(32))})
  return state


def encrypt_object(owner: str, path: str, plaintext: bytes, state: dict[str, Any]) -> bytes:
  active = next(entry for entry in state["keys"] if entry["id"] == state["activeKey"])
  folder_key = base64.urlsafe_b64decode(active["key"] + "=" * (-len(active["key"]) % 4))
  iv = os.urandom(12)
  header = {"v": PROTOCOL_VERSION, "alg": "HKDF-SHA256-A256GCM", "owner": owner, "path": path, "keyId": active["id"], "iv": b64url(iv), "plaintextLength": len(plaintext)}
  object_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=f"asius-data-object-v1:{owner}:{active['id']}".encode(), info=path.encode()).derive(folder_key)
  ciphertext = AESGCM(object_key).encrypt(iv, plaintext, canonical_json(header).encode())
  encoded_header = json.dumps(header, separators=(",", ":")).encode()
  return ENCRYPTION_MAGIC + len(encoded_header).to_bytes(4, "big") + encoded_header + ciphertext


class DataApiClient:
  def __init__(self, base_url: str, private_key: RootIdentity, session: requests.Session | None = None):
    self.base_url = base_url.rstrip("/")
    self.private_key = private_key
    self.owner = public_identity(private_key)
    self.session = session or requests.Session()
    self._config: dict[str, Any] | None = None

  def authorization(self, method: str, path: str, body: Any = None, *, timestamp: int | None = None) -> tuple[bytes, str]:
    encoded = b"" if body is None else canonical_json(body).encode()
    now = int(wall_time()) if timestamp is None else timestamp
    claims = {
      "v": PROTOCOL_VERSION,
      "identity": self.owner,
      "signingKey": b64url(self.private_key.public_key().public_bytes_raw()),
      "method": method,
      "path": path,
      "bodyHash": b64url(hashlib.sha256(encoded).digest()),
      "scope": "asius-storage-request-pq-v1",
      "timestamp": now,
    }
    signature = b64url(self.private_key.sign(canonical_json(claims).encode()))
    token = b64url(canonical_json(claims | {"signature": signature}).encode())
    return encoded, f"Data {token}"

  def request(self, method: str, path: str, body: Any = None) -> requests.Response:
    encoded, authorization = self.authorization(method, path, body)
    headers = {"Authorization": authorization, **({"Content-Type": "application/json"} if encoded else {})}
    response = self.session.request(method, self.base_url + path, data=encoded or None, headers=headers, timeout=30)
    response.raise_for_status()
    return response

  def put_access(self, document: dict[str, Any]) -> None:
    self.request("PUT", f"/v1/{self.owner}/access", document)

  def get_access(self) -> dict[str, Any]:
    return self.request("GET", f"/v1/{self.owner}/access").json()

  def get_config(self) -> dict[str, Any]:
    if self._config is None:
      response = self.session.get(f"{self.base_url}/v1/config", timeout=10)
      response.raise_for_status()
      self._config = response.json()
    return self._config

  def register_upload(self, path: str, *, content_type: str, content_length: int, plaintext_length: int,
                      checksum: str, source_checksum: str | None = None, replace_checksum: str | None = None) -> dict[str, Any]:
    request_path = f"/v1/{self.owner}/uploads"
    body = {
      "path": path,
      "contentType": content_type,
      "contentLength": content_length,
      "plaintextLength": plaintext_length,
      **({"sourceSha256": source_checksum} if source_checksum else {}),
      "checksumSha256": checksum,
      **({"replaceChecksumSha256": replace_checksum} if replace_checksum else {}),
    }
    return self.request("POST", request_path, body).json()

  def complete_upload(self, path: str, checksum: str) -> None:
    self.request("POST", f"/v1/{self.owner}/uploads/complete", {"path": path, "checksumSha256": checksum})

  def upload(self, path: str, encrypted: bytes, *, plaintext_length: int, source_checksum: str | None = None,
             replace_checksum: str | None = None) -> requests.Response:
    checksum = b64url(hashlib.sha256(encrypted).digest())
    signed = self.register_upload(path, content_type="application/octet-stream", content_length=len(encrypted),
                                  plaintext_length=plaintext_length, checksum=checksum, source_checksum=source_checksum, replace_checksum=replace_checksum)
    if signed.get("uploaded"):
      response = requests.Response()
      response.status_code = 200
      return response
    stream = ProgressReader(io.BytesIO(encrypted), path, len(encrypted))
    response = self.session.put(signed["url"], data=stream, headers=signed["headers"], timeout=120)
    response.raise_for_status()
    self.complete_upload(path, checksum)
    stream.complete()
    return response

  def upload_file(self, path: str, filename: str, *, content_type: str, plaintext_length: int,
                  checksum: str, source_checksum: str | None = None, replace_checksum: str | None = None) -> requests.Response:
    content_length = os.path.getsize(filename)
    signed = self.register_upload(path, content_type=content_type, content_length=content_length,
                                  plaintext_length=plaintext_length,
                                  checksum=checksum, source_checksum=source_checksum, replace_checksum=replace_checksum)
    if signed.get("uploaded"):
      response = requests.Response()
      response.status_code = 200
      return response
    with open(filename, "rb") as source:
      stream = ProgressReader(source, path, content_length)
      response = self.session.put(signed["url"], data=stream, headers=signed["headers"], timeout=120)
    response.raise_for_status()
    self.complete_upload(path, checksum)
    stream.complete()
    return response

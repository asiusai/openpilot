import base64
import fcntl
from contextlib import contextmanager
import json
import os
import ssl
import threading
import time

from openpilot.system.asius.pqc import verify_signature
from websocket import WebSocketException, create_connection

from openpilot.common.params import Params
from openpilot.system.asius import pairing_invitation
from openpilot.system.asius.recording_keys import decode_public
from openpilot.system.asius.access_policy import require_ignition_off
from openpilot.common.realtime import set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.system.athena.athenad import backoff
from openpilot.system.asius.identity import get_device_private_key, is_dongle_id


APP_AUTHORIZED_KEYS_PARAM = "AppAuthorizedKeys"
PAIR_TOKEN_SECONDS = 300

def base64url_encode(data: bytes) -> str:
  return base64.urlsafe_b64encode(data).decode().rstrip("=")


def base64url_decode(data: str) -> bytes:
  return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def wall_time() -> float:
  return time.time()  # noqa: TID251


def load_authorized_peers() -> dict[str, dict[str, str | int]]:
  try:
    raw_peers = Params().get(APP_AUTHORIZED_KEYS_PARAM) or {}
    peers = {}
    for public_key, peer in raw_peers.items():
      if not is_dongle_id(public_key):
        continue
      record: dict[str, str | int] = {"publicKey": public_key}
      if isinstance(peer, dict):
        if isinstance(peer.get("storagePublicKey"), str):
          decode_public(peer["storagePublicKey"])
          record["storagePublicKey"] = peer["storagePublicKey"]
        if isinstance(peer.get("label"), str):
          record["label"] = peer["label"]
        if isinstance(peer.get("createdAt"), (int, float)):
          record["createdAt"] = int(peer["createdAt"])
      peers[public_key] = record
    return peers
  except Exception:
    return {}


def save_authorized_peers(peers: dict[str, dict[str, str | int]]) -> None:
  Params().put(APP_AUTHORIZED_KEYS_PARAM, peers, block=True)


@contextmanager
def authorized_peers_lock():
  fd = os.open('/dev/shm/asius-authorized-peers.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
  with os.fdopen(fd, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    yield


def authorize_peer(public_key: str, storage_public_key: str, label: str | None = None) -> dict[str, str | int]:
  require_ignition_off()
  if not is_dongle_id(public_key):
    raise ValueError("invalid app peer key")
  decode_public(storage_public_key)
  if label is not None and (not isinstance(label, str) or len(label) > 80):
    raise ValueError("Invalid app label")
  with authorized_peers_lock():
    peers = load_authorized_peers()
    peer = peers.get(public_key, {"publicKey": public_key, "createdAt": int(wall_time())})
    peer['storagePublicKey'] = storage_public_key
    if label:
      peer["label"] = label
    peers[public_key] = peer
    save_authorized_peers(peers)
    return peer


def forget_peer(public_key: str) -> bool:
  with authorized_peers_lock():
    peers = load_authorized_peers()
    existed = public_key in peers
    peers.pop(public_key, None)
    if existed:
      save_authorized_peers(peers)
    return existed


def stable_json(value) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sign_jwt(payload: dict, expiry_seconds: int) -> str:
  now = int(wall_time())
  key = get_device_private_key()
  header = {"alg": "ML-DSA-65", "typ": "JWT", "key": base64url_encode(key.public_key().public_bytes_raw())}
  body = {**payload, "iat": now, "nbf": now, "exp": now + expiry_seconds}
  signed = f"{base64url_encode(stable_json(header).encode())}.{base64url_encode(stable_json(body).encode())}"
  return f"{signed}.{base64url_encode(key.sign(signed.encode()))}"


def pairing_token(recipient: str, expiry_seconds: int = PAIR_TOKEN_SECONDS) -> str:
  return sign_jwt({"type": "pair", "to": recipient, "jti": pairing_invitation.issue(recipient, expiry_seconds)}, expiry_seconds)


def pairing_url(recipient: str) -> str:
  return f"https://app.asius.ai/pair#token={pairing_token(recipient)}"


def verify_pair_token(token: str | None, recipient: str) -> bool:
  try:
    if not isinstance(token, str):
      return False
    if len(token) > 16384:
      return False
    header, payload, signature = token.split('.')
    algorithm = json.loads(base64url_decode(header))
    if algorithm.get('alg') != 'ML-DSA-65':
      return False
    verify_signature(recipient, base64url_decode(algorithm['key']), base64url_decode(signature), f"{header}.{payload}".encode())
    verified = json.loads(base64url_decode(payload))
    return (verified.get("type") == "pair" and verified.get("to") == recipient and
            pairing_invitation.consume(verified.get("jti"), recipient))
  except Exception:
    return False


def main(exit_event: threading.Event | None = None):
  from openpilot.system.asius import methods

  try:
    set_core_affinity([0, 1, 2, 3])
  except Exception:
    cloudlog.exception("failed to set core affinity")

  params = Params()
  dongle_id = params.get("DongleId")
  if not is_dongle_id(dongle_id):
    raise RuntimeError('Device identity is unavailable; relay connection cannot start')

  conn_start = None
  conn_retries = 0
  while exit_event is None or not exit_event.is_set():
    relay_host = params.get("RelayHost", return_default=True)
    if not params.get("RelayEnabled", return_default=True):
      if exit_event is not None:
        exit_event.wait(1)
      else:
        time.sleep(1)
      continue
    ws = None
    try:
      if conn_start is None:
        conn_start = time.monotonic()

      ws_uri = relay_host + "/room/v1/" + dongle_id + "/" + dongle_id
      cloudlog.event("relay.main.connecting_ws", ws_uri=ws_uri, retries=conn_retries)
      tls = ssl.create_default_context()
      ws = create_connection(ws_uri,
                             enable_multithread=True,
                             timeout=30.0,
                             sslopt={"context": tls})
      challenge = json.loads(ws.recv())
      if challenge.get("type") != "challenge" or not isinstance(challenge.get("challenge"), str):
        raise ValueError("relay did not send an authentication challenge")
      proof = f"asius-room-auth-pq-v1\n{dongle_id}\n{dongle_id}\n{challenge['challenge']}\n".encode()
      signature = base64url_encode(get_device_private_key().sign(proof))
      ws.send(json.dumps({"type": "authenticate", "signature": signature, "key": base64url_encode(get_device_private_key().public_key().public_bytes_raw())}))
      if json.loads(ws.recv()).get("type") != "ready":
        raise ValueError("relay authentication failed")
      cloudlog.event("relay.main.connected_ws", ws_uri=relay_host + "/room/v1/" + dongle_id + "/" + dongle_id, retries=conn_retries,
                     duration=time.monotonic() - conn_start)
      conn_start = None

      conn_retries = 0

      methods.handle_long_poll(ws, exit_event, relay_host)
    except (KeyboardInterrupt, SystemExit):
      break
    except (ConnectionError, TimeoutError, WebSocketException):
      cloudlog.exception("relayd.main.websocket_exception")
      conn_retries += 1
      params.remove("LastAthenaPingTime")
    except Exception:
      cloudlog.exception("relayd.main.exception")

      conn_retries += 1
      params.remove("LastAthenaPingTime")
    finally:
      if ws is not None:
        ws.close()

    time.sleep(backoff(conn_retries))


if __name__ == "__main__":
  main()

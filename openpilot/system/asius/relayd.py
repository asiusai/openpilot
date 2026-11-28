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
from openpilot.system.asius.storage_keys import decode_storage_public_key
from openpilot.system.asius.access_policy import require_ignition_off
from openpilot.common.realtime import set_core_affinity
from openpilot.common.swaglog import cloudlog
from openpilot.system.athena.athenad import backoff
from openpilot.system.asius.identity import get_device_private_key, is_identity
from openpilot.system.asius.service_hosts import relay_url
from openpilot.system.asius.session import b64, unb64_lenient


APP_AUTHORIZED_KEYS_PARAM = "AppAuthorizedKeys"
PAIR_TOKEN_SECONDS = 300

def wall_time() -> float:
  return time.time()  # noqa: TID251


def load_authorized_apps() -> dict[str, dict[str, str | int]]:
  try:
    raw_apps = Params().get(APP_AUTHORIZED_KEYS_PARAM) or {}
    apps = {}
    for public_key, app in raw_apps.items():
      if not is_identity(public_key):
        continue
      record: dict[str, str | int] = {"publicKey": public_key}
      if isinstance(app, dict):
        if isinstance(app.get("storagePublicKey"), str):
          decode_storage_public_key(app["storagePublicKey"])
          record["storagePublicKey"] = app["storagePublicKey"]
        if isinstance(app.get("label"), str):
          record["label"] = app["label"]
        if isinstance(app.get("createdAt"), (int, float)):
          record["createdAt"] = int(app["createdAt"])
      apps[public_key] = record
    return apps
  except Exception:
    return {}


def save_authorized_apps(apps: dict[str, dict[str, str | int]]) -> None:
  Params().put(APP_AUTHORIZED_KEYS_PARAM, apps, block=True)


@contextmanager
def authorized_apps_lock():
  fd = os.open('/dev/shm/asius-authorized-peers.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
  with os.fdopen(fd, 'w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    yield


def authorize_app(public_key: str, storage_public_key: str, label: str | None = None) -> dict[str, str | int]:
  require_ignition_off()
  if not is_identity(public_key):
    raise ValueError("invalid app peer key")
  decode_storage_public_key(storage_public_key)
  if label is not None and (not isinstance(label, str) or len(label) > 80):
    raise ValueError("Invalid app label")
  with authorized_apps_lock():
    apps = load_authorized_apps()
    app = apps.get(public_key, {"publicKey": public_key, "createdAt": int(wall_time())})
    app['storagePublicKey'] = storage_public_key
    if label:
      app["label"] = label
    apps[public_key] = app
    save_authorized_apps(apps)
    return app


def forget_app(public_key: str) -> bool:
  with authorized_apps_lock():
    apps = load_authorized_apps()
    existed = public_key in apps
    apps.pop(public_key, None)
    if existed:
      save_authorized_apps(apps)
    return existed


def stable_json(value) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sign_jwt(payload: dict, expiry_seconds: int) -> str:
  now = int(wall_time())
  key = get_device_private_key()
  header = {"alg": "ML-DSA-65", "typ": "JWT", "key": b64(key.public_key().public_bytes_raw())}
  body = {**payload, "iat": now, "nbf": now, "exp": now + expiry_seconds}
  signed = f"{b64(stable_json(header).encode())}.{b64(stable_json(body).encode())}"
  return f"{signed}.{b64(key.sign(signed.encode()))}"


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
    algorithm = json.loads(unb64_lenient(header))
    if algorithm.get('alg') != 'ML-DSA-65':
      return False
    verify_signature(recipient, unb64_lenient(algorithm['key']), unb64_lenient(signature), f"{header}.{payload}".encode())
    verified = json.loads(unb64_lenient(payload))
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
  if not is_identity(dongle_id):
    raise RuntimeError('Device identity is unavailable; relay connection cannot start')

  conn_start = None
  conn_retries = 0
  while exit_event is None or not exit_event.is_set():
    relay_host = params.get("ApiHost", return_default=True)
    if not params.get("ApiEnabled", return_default=True):
      if exit_event is not None:
        exit_event.wait(1)
      else:
        time.sleep(1)
      continue
    ws = None
    try:
      if conn_start is None:
        conn_start = time.monotonic()

      ws_uri = relay_url(relay_host) + "/room/v1/" + dongle_id + "/" + dongle_id
      cloudlog.event("relayd.main.connecting_ws", ws_uri=ws_uri, retries=conn_retries)
      tls = ssl.create_default_context()
      ws = create_connection(ws_uri,
                             enable_multithread=True,
                             timeout=30.0,
                             sslopt={"context": tls})
      challenge = json.loads(ws.recv())
      if challenge.get("type") != "challenge" or not isinstance(challenge.get("challenge"), str):
        raise ValueError("relay did not send an authentication challenge")
      proof = f"asius-room-auth-pq-v1\n{dongle_id}\n{dongle_id}\n{challenge['challenge']}\n".encode()
      signature = b64(get_device_private_key().sign(proof))
      ws.send(json.dumps({"type": "authenticate", "signature": signature, "key": b64(get_device_private_key().public_key().public_bytes_raw())}))
      if json.loads(ws.recv()).get("type") != "ready":
        raise ValueError("relay authentication failed")
      cloudlog.event("relayd.main.connected_ws", ws_uri=relay_url(relay_host) + "/room/v1/" + dongle_id + "/" + dongle_id, retries=conn_retries,
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

"""Explicit publication of individual uploaded files; never publish folder keys."""
import re


def publication_request(route_id: str, enabled: bool, files: list[dict]) -> dict:
  if not isinstance(route_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{3,160}", route_id):
    raise ValueError("Invalid route identifier")
  if type(enabled) is not bool or not isinstance(files, list) or len(files) > 4096:
    raise ValueError("Invalid route publication")
  if not enabled:
    return {"enabled": False, "files": []}
  if not files:
    raise ValueError("Upload route files before making the route public")
  checked = []
  names = r"(?:[qr]log\.zst|[qfed]camera\.(?:mp4|ts|hevc))"
  for entry in files:
    if not isinstance(entry, dict):
      raise ValueError("Invalid public file")
    path, checksum, key = (entry.get(field) for field in ("path", "checksumSha256", "key"))
    if not isinstance(path, str) or not re.fullmatch(rf"routes/{re.escape(route_id)}--\d{{1,5}}/{names}", path):
      raise ValueError("Only files belonging to this route can be published")
    if not isinstance(checksum, str) or not re.fullmatch(r"[A-Za-z0-9_-]{43}", checksum):
      raise ValueError("Invalid public file checksum")
    if not isinstance(key, str) or not re.fullmatch(r"(?:[A-Za-z0-9_-]{22}|[A-Za-z0-9_-]{43})", key):
      raise ValueError("Invalid public file key")
    checked.append({"path": path, "checksumSha256": checksum, "key": key})
  return {"enabled": True, "files": checked}


def authorize_publication(private_key, route_id: str, enabled: bool, files: list[dict], timestamp: int) -> dict:
  from openpilot.system.loggerd.data_api import DataApiClient
  if type(timestamp) is not int or not 0 < timestamp < 2**53:
    raise ValueError("Invalid request time")
  body = publication_request(route_id, enabled, files)
  client = DataApiClient("", private_key)
  path = f"/v1/{client.owner}/routes/{route_id}/publication"
  encoded, authorization = client.authorization("PUT", path, body, timestamp=timestamp)
  return {"path": path, "body": encoded.decode(), "authorization": authorization}

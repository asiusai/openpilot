"""Persistent upload requests attached to finished device recordings."""
import errno
import json
import os
import re
from pathlib import Path

UPLOAD_ATTR_NAME = "user.asius_upload"
REQUEST_ATTR_NAME = "user.asius_upload_requested"
ERROR_ATTR_NAME = "user.asius_upload_error"
ATTR_VALUE = b"1"
AUTO_UPLOAD_FILES = {"qlog", "qlog.zst", "qcamera.mp4", "qcamera.ts"}
ROUTE_FILES = AUTO_UPLOAD_FILES | {"rlog", "rlog.zst", "fcamera.mp4", "ecamera.mp4", "dcamera.mp4",
                                  "fcamera.hevc", "ecamera.hevc", "dcamera.hevc"}
MAX_FILE_BYTES = 128 * 1024 * 1024
ROUTE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]+--[0-9]+$")


def attribute_set(path: str | Path, name: str) -> bool:
  # Requests and completion are written by different processes. Do not cache.
  try:
    return os.getxattr(path, name, follow_symlinks=False) == ATTR_VALUE
  except OSError as error:
    if error.errno in (errno.ENODATA, getattr(errno, "ENOATTR", errno.ENODATA)):
      return False
    raise


def upload_failure(path: str | Path) -> dict:
  try:
    failure = json.loads(os.getxattr(path, ERROR_ATTR_NAME, follow_symlinks=False))
    stat = os.stat(path, follow_symlinks=False)
    return failure if failure.get("size") == stat.st_size and failure.get("mtimeNs") == stat.st_mtime_ns else {}
  except (OSError, ValueError, AttributeError):
    return {}


def clear_failure(path: str | Path) -> None:
  try:
    os.removexattr(path, ERROR_ATTR_NAME, follow_symlinks=False)
  except OSError as error:
    if error.errno not in (errno.ENODATA, getattr(errno, "ENOATTR", errno.ENODATA)):
      raise


def record_failure(path: str | Path, invalid_recording: bool) -> None:
  stat = os.stat(path, follow_symlinks=False)
  reason = "Recording is incomplete or invalid" if invalid_recording else "Upload failed; retry scheduled"
  failure = {"size": stat.st_size, "mtimeNs": stat.st_mtime_ns, "permanent": invalid_recording, "reason": reason}
  os.setxattr(path, ERROR_ATTR_NAME, json.dumps(failure).encode(), follow_symlinks=False)


def upload_status(path: str | Path) -> dict[str, bool | str]:
  uploaded = attribute_set(path, UPLOAD_ATTR_NAME)
  failure = upload_failure(path) if not uploaded else {}
  return {"uploaded": uploaded, "uploadRequested": not uploaded and attribute_set(path, REQUEST_ATTR_NAME),
          **({"uploadError": failure["reason"]} if failure else {})}


def request_uploads(root: str | Path, paths: list[str]) -> dict[str, list[str]]:
  if not isinstance(paths, list) or not 0 < len(paths) <= 256:
    raise ValueError("request between 1 and 256 recording files")
  root = Path(root).resolve()
  validated: dict[str, Path] = {}
  for relative in paths:
    if not isinstance(relative, str):
      raise ValueError("invalid recording path")
    parts = relative.split("/")
    if len(parts) != 2 or not ROUTE_SEGMENT_RE.fullmatch(parts[0]) or parts[1] not in ROUTE_FILES:
      raise ValueError("invalid recording path")
    path = root.joinpath(*parts)
    if path.parent.is_symlink() or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
      raise ValueError("recording is unavailable")
    if any(path.parent.glob("*.lock")):
      raise ValueError("recording is still being written")
    if not 0 < path.stat().st_size <= MAX_FILE_BYTES:
      raise ValueError("recording size is not supported")
    validated[relative] = path
  result: dict[str, list[str]] = {"queued": [], "uploaded": []}
  for relative, path in validated.items():
    if attribute_set(path, UPLOAD_ATTR_NAME):
      result["uploaded"].append(relative)
    else:
      clear_failure(path)
      os.setxattr(path, REQUEST_ATTR_NAME, ATTR_VALUE, follow_symlinks=False)
      result["queued"].append(relative)
  return result

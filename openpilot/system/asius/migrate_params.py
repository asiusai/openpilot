"""One-time parameter renames, before manager removes unregistered keys."""
import json
from pathlib import Path

from openpilot.common.params import Params


def migrate_cloud_params(params: Params) -> None:
  for old, new, decode in (
    ("DataUploadState", "CloudUploadState", json.loads),
    ("DataUploadEnabled", "CloudUploadEnabled", lambda value: value == "1"),
    ("DataApiHost", "CloudHost", lambda value: "https://cloud.asius.ai" if value.rstrip("/") == "https://storage.asius.ai" else value),
  ):
    path = Path(params.get_param_path(old))
    if path.is_file() and params.get(new) is None:
      params.put(new, decode(path.read_text()), block=True)

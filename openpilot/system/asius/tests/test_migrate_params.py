import json
from pathlib import Path

import pytest

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.system.asius.migrate_params import migrate_cloud_params


@pytest.mark.parametrize("host", ["https://storage.asius.ai", "https://custom.example"])
def test_cloud_rename_preserves_keys_and_disabled_uploads(tmp_path, host):
  params = Params(str(tmp_path))
  state = {"version": 7, "keys": {"recordings": "existing-encryption-key"}}
  for key, value in {"DataUploadState": json.dumps(state), "DataUploadEnabled": "0", "DataApiHost": host}.items():
    Path(params.get_param_path(key)).write_text(value)
  migrate_cloud_params(params)
  params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
  assert params.get("CloudUploadState") == state
  assert params.get("CloudUploadEnabled") is False
  assert params.get("CloudHost") == ("https://cloud.asius.ai" if host == "https://storage.asius.ai" else host)
  assert not Path(params.get_param_path("DataUploadState")).exists()
  migrate_cloud_params(params)
  assert params.get("CloudUploadState") == state


def test_cloud_rename_does_not_overwrite_new_state(tmp_path):
  params = Params(str(tmp_path))
  params.put("CloudUploadState", {"version": 8}, block=True)
  Path(params.get_param_path("DataUploadState")).write_text('{"version": 7}')
  migrate_cloud_params(params)
  assert params.get("CloudUploadState") == {"version": 8}

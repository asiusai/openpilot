from pathlib import Path
import os
import subprocess

from openpilot.common.basedir import BASEDIR

from openpilot.common.params import Params
from openpilot.system.updated.vamos_update import activate_vamos_update


def request_install(staging_root: str = "/data/safe_staging") -> None:
  finalized = Path(staging_root) / "finalized"
  if not (finalized / ".overlay_consistent").is_file():
    raise RuntimeError("Download a complete update before installing.")
  if (Path(staging_root) / "old_openpilot").exists():
    raise RuntimeError("A previous update backup still exists. Check and download the update again.")
  canary = Path(BASEDIR) / ".overlay_init"
  if not canary.is_file() or subprocess.check_output(
      ["find", str(Path(BASEDIR) / ".git"), "-newer", str(canary), "-print", "-quit"], encoding="utf8").strip():
    raise RuntimeError("Device checkout changed after staging. Download the update again.")
  if (finalized / ".vamos_update_pending").exists():
    activate_vamos_update()
  (finalized / ".install_requested").touch()
  os.sync()
  Params().put_bool("DoReboot", True, block=True)

"""Restrictions on app controls, independent of engagement and normal developer SSH."""
import threading
import time

from openpilot.cereal import log, messaging

IGNITION_RESTRICTED_PARAMS = {
  'SshEnabled', 'AdbEnabled', 'JoystickDebugMode', 'LongitudinalManeuverMode', 'LateralManeuverMode',
  'CameraDebugExpGain', 'CameraDebugExpTime', 'UpdaterTargetBranch', 'DoUninstall', 'FactoryReset', 'DoFactoryReset',
}

_lock = threading.Lock()
_state = None


def ignition_state() -> bool | None:
  """Unknown/stale Panda state must not grant ignition-off privileges."""
  global _state
  with _lock:
    first = _state is None
    if first:
      # RPCs and live snapshots poll at varying rates; freshness comes from message time.
      _state = messaging.SubMaster(['pandaStates'], ignore_avg_freq=['pandaStates'])
    _state.update(200 if first else 0)
    age = (time.monotonic_ns() - _state.logMonoTime['pandaStates']) / 1e9
    if not _state.all_checks(['pandaStates']) or not 0 <= age < 1:
      return None
    pandas = [p for p in _state['pandaStates'] if p.pandaType != log.PandaState.PandaType.unknown]
    return any(p.ignitionLine or p.ignitionCan for p in pandas) if pandas else None


def require_ignition_off() -> None:
  if ignition_state() is not False:
    raise PermissionError('Turn ignition off before using this control. Ignition must be confirmed off.')


def check_param_write(name: str) -> None:
  if name in IGNITION_RESTRICTED_PARAMS:
    require_ignition_off()

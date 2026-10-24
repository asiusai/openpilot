"""GPU status shared by the app's Bluetooth, relay and video snapshots."""
import time

from openpilot.selfdrive.modeld.helpers import chestnut_compiled
from openpilot.selfdrive.selfdrived.alertmanager import OFFROAD_ALERTS

CHESTNUT_ALERTS = tuple(k for k, a in OFFROAD_ALERTS.items() if k.startswith('Offroad_Chestnut') and a['severity'] >= 0)


def gpu_status(sm, params) -> str:
  now = time.monotonic()
  if not sm.seen['deviceState'] or not sm.valid['deviceState'] or now - sm.logMonoTime['deviceState'] / 1e9 >= 5:
    return 'unknown'
  device = sm['deviceState']
  if any(params.get(key) is not None for key in CHESTNUT_ALERTS):
    return 'failed'
  active = params.get('ChestnutActive')
  loading = params.get('ChestnutLoading') is True
  if not device.chestnutPresent:
    return 'failed' if device.started and active is not None else 'disconnected'
  if not chestnut_compiled():
    return 'uncompiled'
  if not device.started:
    return 'ready'
  if loading:
    return 'loading'
  if active is False:
    return 'failed'
  model_seen = sm.seen['modelV2']
  model_fresh = model_seen and sm.valid['modelV2'] and now - sm.logMonoTime['modelV2'] / 1e9 < 1
  if model_fresh and sm['modelV2'].big and active is True:
    return 'active'
  if active is True or model_seen:
    return 'failed'
  return 'loading'

"""Read installed openpilot messages before a parked GPU experiment."""
import json
import os
import time
from pathlib import Path
from openpilot.cereal import messaging

raw = Path('/sys/devices/soc0/serial_number').read_text().strip().lower()
serial = int(raw, 16) if raw.startswith('0x') or any(c in 'abcdef' for c in raw) else int(raw)
expected = int(os.environ['ALPAMAYO_EXPECTED_SOC_SERIAL'], 16)
if serial != expected:
  raise SystemExit(f'Wrong Dragon: {serial:08x}')
sm = messaging.SubMaster(['carState', 'selfdriveState', 'pandaStates', 'deviceState'])
deadline = time.monotonic() + 8
stable = 0
while time.monotonic() < deadline:
  sm.update(200)
  def fresh(s):
    return sm.valid[s] and sm.seen[s] and 0 <= time.monotonic() - sm.logMonoTime[s] / 1e9 < 1
  if not fresh('pandaStates'):
    stable = 0
    continue
  car, state, pandas = sm['carState'], sm['selfdriveState'], sm['pandaStates']
  parked = abs(car.vEgo) < .05 and str(car.gearShifter) == 'park'
  panda_disabled = len(pandas) > 0 and all(not p.controlsAllowed for p in pandas)
  offroad = os.environ.get('ALPAMAYO_CONFIRMED_PARKED') == '1' and fresh('deviceState') and not sm['deviceState'].started and panda_disabled and all(
    not p.ignitionLine and not p.ignitionCan for p in pandas)
  if fresh('carState') and not parked:
    raise SystemExit('Test blocked: park the car.')
  if fresh('selfdriveState') and (state.enabled or state.active):
    raise SystemExit('Test blocked: disengage controls.')
  if offroad:
    mode = 'ignition_off_user_confirmed_parked'
    parked, disabled = True, True
  elif fresh('carState') and fresh('selfdriveState'):
    mode = 'parked_vehicle_state'
    disabled = not state.enabled and not state.active and panda_disabled
  else:
    stable = 0
    continue
  if not parked or not disabled:
    raise SystemExit('Test blocked: park the car and disengage controls.')
  stable += 1
  if stable >= 5:
    print(json.dumps({'soc_serial': f'{serial:08x}', 'parked': True, 'controls_disabled': True, 'verification': mode, 'time': time.time()}))
    break
else:
  raise SystemExit('Test blocked: no stable, fresh parked/disengaged vehicle state.')

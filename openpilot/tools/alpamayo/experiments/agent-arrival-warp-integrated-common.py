"""Small source-bound arrival overlay; all numerical installers remain unchanged."""
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ARTIFACTS = WORK/'agent-arrival-warp-integrated-artifacts'
PROTOCOL = WORK/'agent-arrival-warp-integrated-protocol.json'
BASE = WORK/'agent-live-cpu-integrated-source'


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  value = importlib.util.module_from_spec(spec)
  sys.modules[name] = value
  spec.loader.exec_module(value)
  return value


C = load(BASE/'agent-live-cpu-integrated-common.py', 'arrival_integrated_selected')
sha, read, write = C.sha, C.read, C.write
PINS = {
  C.EXP/'agent-arrival-warp-cache.py': '4296046dd671feabe1055e595fa67cccfc7ac8f07c9235fce6e9bed6ba61c964',
  C.EXP/'agent-arrival-warp-check.py': '9a46bbd4d0d5f20b2cadfd55c93b68b0cac2651061a13fd8a45d0c828fff7211',
  WORK/'agent-arrival-warp-trial2/report.json': '2a614cef6a4647f3be91c7cfdeed2f43b047b998a42ba1cf9af00289a14c8c82',
  C.output('paced'): 'b22eb5c954f6908d8ffa3c279628a2edc89a32a513386a6409aa1f8d0d20cd25',
  C.output('consumer'): '002ea0312a4613e018731e0408bae92a673aebac8782df4d146683754ccc8102',
}


def configure():
  assert Path('/mnt/nvme-500').is_mount()
  C.ARTIFACTS = C.S.ARTIFACTS = C.N.ARTIFACTS = ARTIFACTS
  C.configure()


def foundations():
  for path, digest in PINS.items():
    assert sha(path) == digest, str(path)
  C.verify(C.PROTOCOL, sha(C.PROTOCOL))
  selected = read(C.output('consumer'))
  assert selected['predictions_qualified'] and selected['agent_live_cpu_integrated']['passed']
  assert selected['agent_live_cpu_integrated']['protocol_sha256'] == sha(C.PROTOCOL)
  C.parity(read(C.REFERENCE), selected, C.output('consumer'))
  assert read(C.output('paced'))['agent_live_cpu_integrated']['passed']
  assert read(WORK/'agent-arrival-warp-trial2/report.json')['complete']
  return dict(selected_consumer_sha256=PINS[C.output('consumer')],
    selected_continuous_sha256=PINS[C.output('paced')], selected_protocol_sha256=sha(C.PROTOCOL),
    original_full_startup=True, numerical_source_unchanged=True, source_frame_exclusion=False)


def files():
  paths = [*PINS, C.PROTOCOL, C.N.WRAPPER, *BASE.glob('agent-live-cpu-integrated-*.py'),
    *HERE.glob('agent-arrival-warp-integrated-*.py')]
  # The immutable base protocol recursively retains its existing source gates.
  return {str(path): sha(path) for path in paths}


def verify(path, digest, *, cpu_only=False):
  assert sha(path) == digest
  data = read(path)
  for filename, expected in data['files'].items():
    assert sha(filename) == expected, filename
  assert data['foundations'] == foundations()
  assert data['cpu_only'] is cpu_only
  assert data['seconds'] == 8 and data['start_seconds'] == 35 and data['hz'] == 10
  assert data['timeout_seconds'] == 900 and data['automatic_gpu'] is False
  if cpu_only:
    assert os.environ['DEV'] == 'CPU' and os.environ['ALLOW_DEVICE_USAGE'] == '0'
  else:
    proof = read(data['cpu_proof']['path'])
    assert sha(data['cpu_proof']['path']) == data['cpu_proof']['sha256']
    assert proof['complete'] and not proof['gpu_executed'] and proof['devices'] == []
  return data

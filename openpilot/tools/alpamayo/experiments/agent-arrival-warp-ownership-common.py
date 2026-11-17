"""Narrow coverage-ownership repair; preserve the failed arrival run and protocol."""
import importlib.util
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def original_load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec); sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


OLD = original_load(HERE/'agent-arrival-warp-integrated-common.py', 'arrival_ownership_original_common')
C, PINS = OLD.C, OLD.PINS
sha, read, write = OLD.sha, OLD.read, OLD.write
ARTIFACTS = WORK/'agent-arrival-warp-ownership-artifacts'
PROTOCOL = WORK/'agent-arrival-warp-ownership-protocol.json'
OLD_SHA = 'ed426c7f6e6c02ff6b8c3357eb4432ea755245f78643f75767f5067fd9cd8832'
FAILED = WORK/'agent-arrival-warp-integrated-paced1'
FAILED_PINS = {
  FAILED/'report.json': '5f97ec4f2e0198d8c30ba00c16b8dfeb91b2332bae627efafcff7ce6152ce77c',
  FAILED/'arrival-warp-integrated-status.json': '4694c1461e7a019516d60c5e0b2d20b10d148e72053bcef1f15c5454bd2328b2',
  FAILED/'wall.json': 'f929fd049c41e0a9b92991be1153e3bfd5545df92840c95dbe62f6c64c18f861',
}
ownership = []


def configure():
  # Reuse the exact compiler/runtime cache and sampler build from the failed run.
  # No compiler target, environment, numerical source, or cache key changes.
  OLD.configure()
  ARTIFACTS.mkdir(exist_ok=True)


def load(path, name):
  if Path(path).resolve() == HERE/'agent-arrival-warp-integrated-install.py':
    return original_load(HERE/'agent-arrival-warp-ownership-install.py', name)
  return original_load(path, name)


def files():
  paths = [OLD.PROTOCOL, *FAILED_PINS, FAILED/'run.log', *HERE.glob('agent-arrival-warp-ownership-*.py')]
  return {str(p):sha(p) for p in paths}


def foundation():
  OLD.verify(OLD.PROTOCOL, OLD_SHA)
  cachedb = Path(C.N.storage()['CACHEDB'])
  assert cachedb.is_file() and not cachedb.is_symlink() and cachedb.stat().st_uid == os.getuid()
  assert cachedb.resolve().is_relative_to(WORK)
  for p, digest in FAILED_PINS.items(): assert sha(p) == digest, str(p)
  status = read(FAILED/'arrival-warp-integrated-status.json')
  assert not status['passed'] and status['gpu_executed'] and status['hooks_restored'] and status['constructor_restored']
  assert status['failure'] == dict(type='ValueError', message='Unexpected outer hook after the canonical launcher restored its imports')
  assert status['arrival']['closed'] and status['arrival']['producer_binding_restored'] and status['arrival']['cache_cleanup_bytes'] == 0
  for event in status['storage_events']:
    if 'actual_environment' in event: assert event['actual_environment'] == C.N.storage()
  return dict(original_protocol_sha256=OLD_SHA, failed_run_stays_unqualified=True,
    numerical_recipe_unchanged=True, compiler_cache_reused=True, storage=C.N.storage())


def verify(path, digest, *, cpu_only=False):
  assert sha(path) == digest
  data = read(path)
  for p, expected in data['files'].items(): assert sha(p) == expected, p
  assert data['foundation'] == foundation() and data['cpu_only'] is cpu_only
  assert data['timeout_seconds'] == 900 and data['window'] == [35,43] and data['hz'] == 10
  if cpu_only: assert os.environ['DEV'] == 'CPU' and os.environ['ALLOW_DEVICE_USAGE'] == '0'
  else:
    proof = read(data['cpu_proof']['path'])
    assert sha(data['cpu_proof']['path']) == data['cpu_proof']['sha256']
    assert proof['complete'] and proof['devices'] == [] and not proof['gpu_executed']
    assert proof['production']['all_actual_super_coverage_executed']
  return data

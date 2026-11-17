"""CPU-only regression checks for mapped and unmapped tuning output poisoning."""
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# These checks must never discover or open AMD, even under a GPU trial environment.
os.environ.update(DEV='CPU', PROFILE='0', DEBUG='0', BEAM='0', PARALLEL='0')

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  spec = importlib.util.spec_from_file_location('agent_tune', Path(__file__).with_name('agent-tune.py'))
  tune = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = tune
  spec.loader.exec_module(tune)
  results = []
  for mapped in (True, False):
    for size in (4, 64, 65536, 65540, 131084):
      # Guarded subview tests both chunk tails and the offset inherited from a base.
      base = Buffer('CPU', size+16, initial_value=bytes([0x3c])*(size+16))
      original_storage = base.get_storage()
      if not mapped: base._storage = replace(original_storage, host=None)
      dest = base.view(size, 8).ensure_allocated()
      assert (dest.get_storage().host is not None) == mapped
      call = SimpleNamespace(body=SimpleNamespace(arg=SimpleNamespace(outs=(0,))), src=(None, UOp.from_buffer(dest, dtypes.uint8)))
      try:
        tune.poison_outputs(SimpleNamespace(src=(call,)), 'CPU')
        actual = bytes(original_storage.host[:])
        assert actual == bytes([0x3c])*8 + bytes([0xa5])*size + bytes([0x3c])*8
      finally:
        base._storage = original_storage
      results.append({'mapped': mapped, 'bytes': size, 'all_bytes_poisoned': True, 'guards_preserved': True})
  # A failed normal transfer must propagate; incomplete poisoning cannot pass.
  original_copy = Buffer.copy_from
  failing = Buffer('CPU', 64, initial_value=bytes(64))
  saved = failing.get_storage()
  failing._storage = replace(saved, host=None)
  def fail_copy(self, source): raise RuntimeError('injected transfer failure')
  Buffer.copy_from = fail_copy
  try:
    try: tune.poison_buffer(failing)
    except RuntimeError as exc: assert str(exc) == 'injected transfer failure'
    else: raise AssertionError('transfer failure was suppressed')
  finally:
    Buffer.copy_from = original_copy
    failing._storage = saved
  assert not any(d.startswith('AMD') for d in Device._opened_devices)
  result = {'passed': True, 'cases': results, 'transfer_failure_propagated': True,
            'opened_devices': sorted(Device._opened_devices), 'gpu_executed': False,
            'harness_sha256': tune.file_sha(Path(__file__).with_name('agent-tune.py'))}
  tune.atomic_json(args.output, result)
  print(json.dumps({'passed': True, 'cases': len(results), 'transfer_failure_propagated': True, 'gpu_executed': False}))


if __name__ == '__main__': main()

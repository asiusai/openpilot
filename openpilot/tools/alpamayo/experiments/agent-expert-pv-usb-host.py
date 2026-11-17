"""CPU semantic fixtures and strict runtime-observer rejection checks."""
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def module(name):
  path = Path(__file__).with_name(name+'.py')
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
  value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
  return value


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.environ.update(DEV='CPU', CCACHE='0', TC_OPT='2', BEAM='0', PROFILE='0', DEBUG='0')
  U, H = module('agent-expert-pv-usb-common'), module('agent-expert-pv-usb-check')
  B = U.old()
  U.verify_capture()
  from tinygrad import Device
  from tinygrad.helpers import Target
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import UOp
  ast, program = U.compile_program(U.renderers()['worker-cu'], B.SHAPE, 'original')
  description = B.describe(program, ast)
  registry = {program.key.hex(): (ast, description)}
  params = [UOp.param(i, p.dtype, p.numel(), device='AMD') for i, p in enumerate(program.kernel_params)]
  report, saves = {}, []
  with patch.object(hcq2, '_build_queues', lambda ctx: 'accepted'):
    original = hcq2._build_queues
    with H.observe(U, registry, report, lambda: saves.append(True)) as seen:
      assert hcq2._build_queues(SimpleNamespace(batch=[(program.call(*params), None, None)])) == 'accepted'
      assert seen == {program.key.hex(): 1} and len(saves) == 1
      changed = program.replace(arg=replace(program.arg, target=Target.parse('AMD:LLVM:gfx1200')))
      try: hcq2._build_queues(SimpleNamespace(batch=[(changed.call(*params), None, None)]))
      except AssertionError as exc: assert str(exc) == 'Unqualified runtime PV program'
      else: raise AssertionError('Runtime observer accepted a stripped USB target')
    assert hcq2._build_queues is original
  semantics = U.load('agent-expert-pv-check').cpu_checks()
  opened = sorted(Device._opened_devices)
  assert all(x.split(':')[0] in ('CPU', 'NPY') for x in opened), opened
  B.write(args.output, {'complete': True, 'gpu_executed': False, 'model_predictions': 0,
    'runtime_usb_program_observed_exactly': True, 'runtime_native_target_rejected': True,
    'observer_cleanup_exact': True, 'cpu_semantics': semantics, 'opened_devices': opened,
    'source_sha256': {str(p): U.sha(p.read_bytes()) for p in Path(__file__).parent.glob('agent-expert-pv-usb-*.py')}})


if __name__ == '__main__': main()

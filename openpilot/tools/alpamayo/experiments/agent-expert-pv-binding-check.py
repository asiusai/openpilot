"""CPU fake-capture, unchanged-gate, strict mutation, and direct-PROGRAM tests."""
from contextlib import contextmanager
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
  return value


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.environ.update(DEV='CPU', CCACHE='0', TC_OPT='2', PROFILE='0', BEAM='0', DEBUG='0')
  here = Path(__file__).parent
  B = load(here/'agent-expert-pv-binding-common.py', 'pv_binding_check_common')
  D = load(here/'agent-expert-pv-binding-diagnose.py', 'pv_binding_check_diagnostic')
  sys.path.insert(0, str(B.SNAP))
  import numpy as np
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.codegen import do_to_program
  from tinygrad.helpers import Context
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import KernelInfo, UOp
  H, C, E, T = (B.load(n) for n in ('agent-expert-pv-route-hook', 'agent-expert-pv-check',
    'agent-expert-gemm', 'agent-vision-tiles-transfer-check'))
  cpu = json.loads(B.CPU.read_text())
  renderer = B.renderers()['worker-cu']
  ast, program = B.compile_program(renderer, B.SHAPE, 'original')
  expected = cpu['compiler_cases']['expert']['worker-cu']['original']
  D.validate_program(B.describe(program), expected)
  rejected = []
  for key in ('name', 'lowered_ast_key', 'llvm_sha256', 'binary_sha256', 'global_size', 'local_size', 'target', 'params'):
    changed = copy.deepcopy(expected)
    changed[key] = 'mutated'
    try: D.validate_program(B.describe(program), changed)
    except AssertionError: rejected.append(key)
    else: raise AssertionError('Program mutation passed: '+key)
  report, saved, handles = {}, [], []
  class FakeDevice:
    DEFAULT = 'AMD'
    def __class_getitem__(cls, key): return SimpleNamespace(renderer=B.renderers()['parent-wgp'])
  def install(*args, **kwargs):
    handle = H.Installation()
    handle.candidate, handle.depth, handle.seen = False, 1, []
    handle.programs = {}
    handle.info = {'evidence': {'original_binary_sha256': cpu['compiler_cases']['expert']['parent-wgp']['original']['binary_sha256']},
                   'captures': [], 'runtime_programs': handle.programs}
    handle.dispatch = lambda *a, **kw: ast
    handle.module = SimpleNamespace(_matmul=handle.dispatch)
    handles.append(handle)
    return handle
  fake_h = SimpleNamespace(install=install, NAME=H.NAME, eligible=H.eligible, verify_evidence=H.verify_evidence)
  @contextmanager
  def interceptor(hook, candidate, evidence, seen):
    handle = hook.install(None, None, candidate=candidate)
    seen.append(handle)
    with handle.observe_programs(): yield
  fake_r = SimpleNamespace(interceptor=interceptor)
  real_load = B.load
  def routed_load(name):
    if name == 'agent-expert-pv-route-hook': return fake_h
    if name == 'agent-expert-pv-route-run': return fake_r
    return real_load(name)
  def execute(*args, **kw):
    handle = handles[-1]
    shapes = real_load('agent-expert-pv-compile').specs(B.SHAPE)
    values = [UOp.placeholder(shape, dtypes.bfloat16, i) for i, shape in enumerate(shapes)]
    for _ in range(84): handle.module._matmul(*values, shape=B.SHAPE)
    params = [UOp.param(i, dtypes.bfloat16, x.numel(), device='AMD') for i, x in enumerate(values)]
    hcq2._build_queues(SimpleNamespace(batch=[(program.call(*params), None, None)]))
    records = []
    for name, key in (('alpamayo_attention_values', program.key), ('alpamayo_attention_scores', b'q'*32),
                      ('alpamayo_attention_softmax', b's'*32)):
      handle.programs[key.hex()] = {'name': name, 'binary_sha256': B.sha(program.src[3].arg)}
      records.extend([('AMD', name, None, None, key, (2, 0, 1), ((0,), (1, 2)))]*84)
    call = SimpleNamespace(without_after=SimpleNamespace(arg=SimpleNamespace(aux=SimpleNamespace(kernels=records))))
    handle.inspect_capture(SimpleNamespace(captured=SimpleNamespace(_linear=SimpleNamespace(src=(call,)))))
  def save(): saved.append(copy.deepcopy(report))
  with patch.object(B, 'load', routed_load), patch('tinygrad.Device', FakeDevice), \
       patch.object(hcq2, '_build_queues', lambda ctx: None), patch.object(D.runpy, 'run_path', execute), Context(PARALLEL=2):
    observed = D.capture(B, cpu, report, save, ['fake.py'], args.output)
  assert observed is program and report['actual_reference_compiler_mode'] == 'worker-cu'
  assert not report['unchanged_old_binary_gate']['passed']
  assert report['capture_counts']['alpamayo_attention_values'] == 84
  assert any('actual_reference' in r and 'unchanged_old_binary_gate' not in r for r in saved)
  assert fake_h.install is install
  # Test direct precompiled PROGRAM calls, identity-preserving input mutations,
  # captured WAW replay, and restored bytes without a GPU device.
  def fill(out, a, b):
    r = UOp.range(out.numel(), 0)
    return out.flatten().index(r).store((a.flatten().index(0).float()+b.flatten().index(0).float()).cast(out.dtype)).end(r).sink(arg=KernelInfo())
  cpu_args = [UOp.placeholder(s, dtypes.bfloat16, i) for i, s in enumerate(((1, 1, 2, 3), (1,), (1,)))]
  compiled = do_to_program(fill(*cpu_args), Device['CPU'].renderer)
  transfer = T.InputTransfers(Device['CPU'])
  inputs = transfer.upload((E.bf16(np.array([2], np.float32)), E.bf16(np.array([3], np.float32))),
                           (dtypes.bfloat16, dtypes.bfloat16))
  jit = C.SerialPV((1, 1, 1, 2, 3, 1), lambda *_: compiled, 40)
  for _ in range(3): jit(*inputs)
  proof = E.prove_cpu_queue(jit.captured._linear.src)
  for a, b in ((2, 3), (7, 3), (7, -4), (2, 3)):
    transfer.replace(inputs, (E.bf16(np.array([a], np.float32)), E.bf16(np.array([b], np.float32))))
    np.testing.assert_array_equal(jit(*inputs)[0].bitcast(dtypes.uint16).numpy(), E.bf16(np.full((1, 1, 2, 3), a+b, np.float32)))
  assert all(name.startswith('CPU') for name in Device._opened_devices)
  B.write(args.output, {'complete': True, 'gpu_executed': False, 'strict_program_mutations_rejected': rejected,
    'unchanged_old_binary_gate_failed_as_expected': True, 'actual_program_saved_before_gate': True,
    'fake_capture_counts': report['capture_counts'], 'direct_program_same_buffer_mutation_restore_exact': True,
    'direct_program_serial_queue_proof': proof, 'new_source_sha256': {str(p): B.sha(p.read_bytes())
      for p in here.glob('agent-expert-pv-binding-*.py')}})


if __name__ == '__main__': main()

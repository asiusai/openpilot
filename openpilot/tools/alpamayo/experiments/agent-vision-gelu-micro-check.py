"""CPU-only compiler, queue, mutable-input, and failure preflight. Freeze one protocol."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def load():
  spec = importlib.util.spec_from_file_location('vision_gelu_micro', HERE/'agent-vision-gelu-micro-run.py')
  result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
  return result


@contextmanager
def patched(obj, **values):
  old = {key:getattr(obj, key) for key in values}
  for key, value in values.items(): setattr(obj, key, value)
  try: yield
  finally:
    for key, value in old.items(): setattr(obj, key, value)


def source_identity(micro):
  plan = json.loads((micro.PROFILE/'report.current-profile-plan.json').read_text())
  names = ['../model.py', '../amd_gemm.py', '../amd_attention.py', '../amd_decode.py', 'batched-vision.py',
           'quantize-reduction.py', 'tiled-gemm.py', 'wgp_compiler.py', 'prefill-kernels.py',
           'benchmark-pipeline.py', 'agent-gemm-compile.py']
  sources = {}
  for name in names:
    path = (HERE/name).resolve()
    expected = plan['source_sha256'][str(path)]
    assert micro.digest(path) == expected
    sources[str(path)] = expected
  previous = json.loads((HERE/'agent-vision-gelu-cpu.json').read_text())
  for path, expected in previous['source_sha256'].items():
    assert micro.digest(path) == expected, 'Previously inspected files must stay frozen'
    sources[path] = expected
  for name in ('agent-vision-gelu-cpu.json', 'agent-vision-gelu-inspect.json',
               'agent-vision-gelu-micro-run.py', 'agent-vision-gelu-micro-check.py'):
    sources[str(HERE/name)] = micro.digest(HERE/name)
  for name in ('report.current-profile-plan.json', 'report.current-profile-control-invocation.json'):
    sources[str(micro.PROFILE/name)] = micro.digest(micro.PROFILE/name)
  root = Path(plan['tinygrad']['path'])
  # Pin all Python implementation files, including compiler, scheduler, JIT, USB, and autogen interfaces.
  sources.update({str(p):micro.digest(p) for p in sorted((root/'tinygrad').rglob('*.py'))})
  for path, expected in plan['tinygrad']['source_sha256'].items(): assert sources[path] == expected
  return sources, plan['tinygrad']


def parameters(plan, device='AMD'):
  from tinygrad.uop.ops import UOp
  return [UOp.param(i, dt, row['nbytes']//dt.itemsize, device=device)
          for i, (row, dt) in enumerate(zip(plan['layout'], plan['dtypes'], strict=True))]


def queue_context(calls):
  return SimpleNamespace(batch=[(call, 'AMD', 'COMPUTE:0') for call in calls],
    calls={('AMD', 'COMPUTE:0'):list(range(len(calls)))}, peers={('AMD', 'COMPUTE:0'):set()},
    queues={'AMD':['COMPUTE:0']}, stamps=lambda *_:())


def queue_checks(micro, plan):
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  params = parameters(plan)
  calls = [program.call(*(params[i] for i in mapping)) for program, mapping in plan['steps']]*micro.CALLS
  ctx = queue_context(calls)
  proof = micro.ordered_queues(ctx, hcq2._build_queues(ctx), plan, micro.CALLS)
  assert proof['added_operation_barriers'] == 39
  # An independent output buffer or reordered program must fail the actual queue audit.
  bad = calls[:]
  program, mapping = plan['steps'][0]
  wrong = params[1].replace(arg=__import__('dataclasses').replace(params[1].arg, slot=8))
  bad[0] = program.call(*(wrong if i == 1 else params[i] for i in mapping))
  for changed in (bad, list(reversed(calls))):
    try:
      ctx = queue_context(changed)
      micro.ordered_queues(ctx, hcq2._build_queues(ctx), plan, micro.CALLS)
    except AssertionError: pass
    else: raise AssertionError('Invalid batch passed serial/source/buffer gate')
  assert all(call.body.op is Ops.PROGRAM for call in calls)
  return proof


def mutable_tensor(shape, dtype, raw=None):
  import math
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp, Ops
  size = math.prod(shape)*dtype.itemsize
  array = np.zeros(size, np.uint8) if raw is None else np.frombuffer(raw, np.uint8).copy()
  # NumPy owns this opaque storage. Avoid opening a device merely to free the wrapper.
  buffer = Buffer('NPY', size, opaque=array, allocator=SimpleNamespace(free=lambda *unused:None))
  tensor = Tensor(UOp.from_buffer(buffer, dtype)).reshape(shape)
  assert tensor.uop.base.op is Ops.BUFFER and tensor.uop.is_realized
  return tensor


def jit_checks(micro, plan):
  """Use the actual pinned TinyJit front end; mock execution and HCQ linking only."""
  from tinygrad import dtypes
  from tinygrad.engine import jit as jitmod
  from tinygrad import tensor as tensormod
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  import numpy as np
  scalar = mutable_tensor((), dtypes.float32, np.array(1., np.float32).tobytes())
  prepared = jitmod._prepare_jit_inputs((scalar,), {})
  assert len(prepared[0]) == 1 and prepared[0][0].op is Ops.BUFFER
  micro.write_buffer(scalar, np.array(2., np.float32).tobytes())
  assert np.frombuffer(bytes(scalar.uop.buffer.host.view(fmt='B')[:]), np.float32)[0] == 2
  inputs = tuple(mutable_tensor(row['shape'], dt) for row, dt in zip(plan['layout'], plan['dtypes'], strict=True))
  seen = []
  def compile_linear(linear, **kwargs):
    if linear.src:
      assert all(call.body.op is Ops.PROGRAM for call in linear.src)
      hcq2._build_queues(queue_context(linear.src))
    return linear
  def run_linear(linear, var_vals=None, input_uops=(), **kwargs):
    if input_uops:
      assert input_uops[0].op is Ops.BUFFER
      seen.append(bytes(input_uops[0].buffer.host.view(size=16, fmt='B')[:]))
  result = []
  with patched(jitmod, compile_linear=compile_linear, link_linear=lambda value:value, run_linear=run_linear), \
       patched(tensormod, run_linear=lambda *args, **kwargs:None):
    for count in (1, micro.CALLS):
      runner = micro.Runner(plan, count)
      if count == 1: runner(inputs)
      runner(inputs)
      assert runner.jit.captured is not None and len(runner.proofs) == 1
      for value in range(1, 7):
        micro.write_buffer(inputs[0], bytes([value])*inputs[0].nbytes())
        before = runner.jit.cnt
        runner(inputs)
        assert runner.jit.cnt == before+1
      assert len(set(seen[-6:])) == 6
      result.append(dict(operations=count, real_tinyjit_capture=True, changed_buffer_replay_observed=6,
                         numeric_execution_mocked=True, queue_proof=runner.proofs[0]))
  return dict(scalar_0d_is_mutable_buffer=True, records=result)


def fixture_checks(micro):
  import numpy as np
  rows, initial = [], micro.fixture(0)
  for state in range(8):
    data = micro.fixture(state)
    assert data.shape == micro.SHAPE and data.dtype == np.uint16
    if state in range(1, 7): assert not np.array_equal(data, initial)
    if state == 7: np.testing.assert_array_equal(data, initial)
    rows.append(dict(state=micro.STATES[state], sha256=micro.sha(data.tobytes())))
  assert len({row['sha256'] for row in rows}) == 7
  raw = dict(activation=b'\x00\x80', partial=b'1234', maximum=b'abcd', scale=b'efgh', fp8=b'\x80', packed=b'\x80', row_scale=b'ijkl')
  micro.exact(raw, raw)
  for key in raw:
    mutated = dict(raw, **{key:bytes([raw[key][0]^1])+raw[key][1:]})
    try: micro.exact(raw, mutated)
    except AssertionError: pass
    else: raise AssertionError(f'Byte mutation accepted: {key}')
  return dict(controls=rows, six_distinct_mutations=True, restoration_exact=True, each_boundary_byte_gate_rejects_mutation=True)


def finite_checks(micro):
  import numpy as np
  values = dict(activation=np.array([0, 0x8000, 1], np.uint16).tobytes(),
    partial=np.array([0, 1], np.float32).tobytes(), maximum=np.array([1, 2], np.float32).tobytes(),
    scale=np.array([1e-12, .5], np.float32).tobytes(), fp8=b'\x00\x80\x7e', packed=b'\x7e\x80\x00',
    row_scale=np.array([1e-12, .5], np.float32).tobytes())
  with patched(micro, read_buffer=lambda tensor, device:values[tensor]):
    micro.snapshot(micro.LABELS, None)
    for key in values:
      saved = values[key]
      values[key] = (np.array([0x7fc1], np.uint16).tobytes() if key == 'activation' else b'\x7f' if key in ('fp8', 'packed')
                     else np.array([np.nan], np.float32).tobytes())
      try: micro.snapshot(micro.LABELS, None)
      except AssertionError: pass
      else: raise AssertionError(f'Nonfinite {key} accepted')
      values[key] = saved
  return dict(all_seven_nonfinite_boundary_outputs_rejected=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--freeze', required=True, type=Path)
  args = parser.parse_args()
  micro = load()
  source_hashes, tinygrad = source_identity(micro)
  recorded = json.loads((micro.PROFILE/'report.current-profile-control-invocation.json').read_text())
  protocol = dict(schema=1, shape=list(micro.SHAPE), operations=micro.CALLS, controls=list(micro.STATES), timeout_seconds=300,
    source_sha256=source_hashes, tinygrad=tinygrad, environment=recorded['environment'], python=recorded['command'][0],
    native_gelu_equality_proven=False, diagnostic_only=True, model_hook_installed=False, adoption_allowed=False)
  configured = micro.environment(protocol, cpu=True)
  os.environ.clear(); os.environ.update(configured)
  from tinygrad import Device
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200', interface='USB'))
  renderer.compiler = micro.load('wgp_compiler').WGPCompiler('gfx1200')
  plans = {name:micro.build_plan(name, renderer) for name in ('original', 'candidate')}
  assert plans['original']['layout'] == plans['candidate']['layout']
  old = json.loads((HERE/'agent-vision-gelu-cpu.json').read_text())
  for name, plan in plans.items():
    for before, after in zip(old[name][1:], plan['records'], strict=True):
      assert all(before[key] == after[key] for key in ('source_sha256', 'binary_sha256', 'ast_key', 'resources'))
  report = dict(gpu_executed=False, native_gelu_equality_proven=False,
    frozen_inspected_kernel_sources_and_binaries_unchanged=True,
    queues={name:queue_checks(micro, plan) for name, plan in plans.items()},
    actual_tinyjit_frontend_with_execution_mocks={name:jit_checks(micro, plan) for name, plan in plans.items()},
    fixture_checks=fixture_checks(micro), finite_checks=finite_checks(micro), timing_not_run=True)
  assert not Device._opened_devices
  report['opened_devices'] = sorted(Device._opened_devices)
  protocol['plans'] = {name:dict(records=plan['records'], layout=plan['layout']) for name, plan in plans.items()}
  micro.verify(protocol)
  args.freeze.mkdir(parents=True, exist_ok=True)
  protocol_path = args.freeze/'agent-vision-gelu-micro-protocol.json'
  assert not protocol_path.exists(), 'Frozen protocol already exists; do not overwrite'
  for name, plan in plans.items():
    for i, (program, _) in enumerate(plan['steps']):
      (args.freeze/f'agent-vision-gelu-micro-{name}-{i}.source').write_text(program.src[2].arg)
      (args.freeze/f'agent-vision-gelu-micro-{name}-{i}.hsaco').write_bytes(program.src[3].arg)
  micro.save(protocol_path, protocol)
  report['protocol_sha256'] = micro.digest(protocol_path)
  micro.save(args.freeze/'agent-vision-gelu-micro-cpu.json', report)
  print(json.dumps(dict(gpu_executed=False, native_gelu_equality_proven=False,
    kernels={name:len(plan['steps']) for name, plan in plans.items()},
    full_boundary_operations=micro.CALLS, added_operation_barriers=39, mutations=6,
    mutable_0d_scalar_checked=True, protocol=str(protocol_path), opened_devices=report['opened_devices'])), flush=True)


if __name__ == '__main__': main()

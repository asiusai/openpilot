"""One source-bound, parent-only GELU boundary diagnostic. No model installation."""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
PROFILE = WORK/'current-profile-dynamic-2048'
SHAPE = (2, 720, 4304)
LABELS = ('input', 'activation', 'partial', 'maximum', 'scale', 'fp8', 'packed', 'row_scale')
STATES = ('initial', 'image0_tail', 'image1_tail', 'zero_subnormal', 'rescale', 'middle_outlier', 'swap_tail', 'restore')
CALLS = 40


def sha(value): return hashlib.sha256(value).hexdigest()
def digest(path): return sha(Path(path).read_bytes())
def save(path, value):
  path = Path(path)
  temporary = path.with_suffix(path.suffix+'.tmp')
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
  temporary.replace(path)


@lru_cache(None)
def load(name):
  spec = importlib.util.spec_from_file_location('vision_gelu_micro_'+name.replace('-', '_'), HERE/(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def verify(protocol):
  assert protocol['schema'] == 1 and protocol['shape'] == list(SHAPE) and protocol['operations'] == CALLS
  assert protocol['controls'] == list(STATES) and protocol['timeout_seconds'] == 300
  for path, expected in protocol['source_sha256'].items():
    assert digest(path) == expected, f'Source identity changed: {path}'
  root = protocol['tinygrad']['path']
  revision = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip()
  assert revision == protocol['tinygrad']['revision'] == '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
  plan = json.loads((PROFILE/'report.current-profile-plan.json').read_text())
  for path, expected in plan['tinygrad']['source_sha256'].items(): assert digest(path) == expected
  control = json.loads((PROFILE/'report.current-profile-control-invocation.json').read_text())
  assert '--llvm-wgp' in control['command'] and '--vision-calibration' not in control['command']
  assert '--grouped-quantize-reduction' in control['command'] and '--batched-vision' in control['command']
  assert control['environment']['DEV'] == 'USB+AMD:LLVM' and control['environment']['TC_OPT'] == '2'
  assert protocol['environment'] == control['environment']
  return control


def environment(protocol, cpu=False):
  result = dict(os.environ)
  for key, value in protocol['environment'].items():
    if value is None: result.pop(key, None)
    else: result[key] = value
  # Bound diagnostics and disable all tuning/profiling. Arithmetic settings stay pinned.
  result.update(PYTHONDONTWRITEBYTECODE='1', BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0',
                JIT='1', CAPTURING='1', LLVMOPT='1', CACHEDB=str(HERE/'agent-vision-gelu-micro-cache.db'))
  if cpu: result.update(DEV='CPU', ALLOW_DEVICE_USAGE='0')
  else: result['ALLOW_DEVICE_USAGE'] = '1'
  return result


def bf16(value):
  import numpy as np
  bits = np.asarray(value, np.float32).view(np.uint32)
  return ((bits+0x7fff+((bits >> 16) & 1)) >> 16).astype(np.uint16)


def fixture(state):
  import numpy as np
  assert state in range(len(STATES))
  rng = np.random.default_rng(90604304)
  data = np.clip(rng.standard_normal(SHAPE, dtype=np.float32), -4, 4)
  data[0] *= np.float32(.8)
  data[1] = data[1]*np.float32(1.7)+np.float32(.5)
  bits = bf16(data)
  witness = int(bf16([-3.46875])[0])
  probes = np.array([0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080,
                     witness-1, witness, witness+1, 0x3f80, 0x3f81], np.uint16)
  for image in range(2): bits[image].reshape(-1)[:len(probes)] = probes
  if state == 1: bits[0, -1, -1] = bf16([256])[0]
  if state == 2: bits[1, -1, -1] = bf16([512])[0]
  if state == 3: bits[0] = np.resize(probes[:8], bits[0].shape)
  if state == 4:
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    bits = bf16(values*np.array([.25, 4], np.float32)[:, None, None])
  if state == 5: bits[0, 360, 2152] = bf16([32])[0]
  if state == 6:
    bits = bits[::-1].copy()
    bits[0, -1, -1], bits[1, -1, -2] = bf16([128, 384])
  assert np.isfinite((bits.astype(np.uint32) << 16).view(np.float32)).all()
  return bits


def original_activation(x):
  """Invoke the canonical model method, stopping at its actual fc2 input."""
  from tinygrad import Tensor, dtypes
  captured = []
  class Reached(Exception): pass
  def linear(value, name, *args, **kwargs):
    if name.endswith('.mlp.fc1'): return x
    if name.endswith('.mlp.fc2'): captured.append(value); raise Reached()
    return Tensor.empty(2, 720, 1152, dtype=dtypes.bfloat16, device=x.device)
  model = SimpleNamespace(dtype=dtypes.bfloat16, weight_scales={}, norm=lambda value, *args: value,
                          linear=linear, attention=lambda q, k, v: q)
  try: load('../model').EdgeModel.vision_layer(model, Tensor.empty(2, 720, 1152, dtype=dtypes.bfloat16, device=x.device), 2)
  except Reached: pass
  assert len(captured) == 1
  return captured[0]


def original_quantize(x):
  """Use the canonical batched-vision closure and installed grouped reduction."""
  def unused(*args, **kwargs): raise AssertionError('Unexpected non-grouped path')
  model = SimpleNamespace(quantize=unused, linear=unused, norm=unused)
  load('quantize-reduction').install(model, grouped=True)
  encode = load('batched-vision').install(model)
  state = inspect.getclosurevars(encode).nonlocals['state']
  state['images'] = 2
  return model.quantize(x)


@contextmanager
def taps():
  """Retain existing intermediate tensors, without adding arithmetic or realizations."""
  from tinygrad import Tensor, dtypes
  reduction, candidate = load('quantize-reduction'), load('agent-vision-gelu-kernel')
  old_kernel, old_realize = Tensor.custom_kernel, Tensor.realize
  held, small = {}, []
  def custom(*values, fxn, **kwargs):
    output = old_kernel(*values, fxn=fxn, **kwargs)
    if getattr(fxn, 'func', None) is reduction.program:
      if fxn.keywords['final']: held['maximum'] = output[0]
      else: held.update(activation=values[1], partial=output[0])
    elif getattr(fxn, 'func', None) is candidate.kernel:
      held.update(activation=output[0], partial=output[1])
    return output
  def realize(self, *others, **kwargs):
    result = old_realize(self, *others, **kwargs)
    if self.dtype == dtypes.float32 and self.shape == (2, 1, 1): small.append(self)
    return result
  Tensor.custom_kernel, Tensor.realize = custom, realize
  try: yield held, small
  finally: Tensor.custom_kernel, Tensor.realize = old_kernel, old_realize


def boundary(variant, x):
  from amd_gemm import pack
  from tinygrad import dtypes
  assert x.shape == SHAPE and x.dtype == dtypes.bfloat16
  with taps() as (held, small):
    if variant == 'original': q, row_scale = original_quantize(original_activation(x))
    else:
      assert variant == 'candidate'
      q, row_scale = load('agent-vision-gelu-kernel').quantize(x, 2, load('quantize-reduction'))
    packed = pack(q).realize()
    row_scale.realize()
  assert len(small) == 2
  held.update(scale=small[-1], fp8=q, packed=packed, row_scale=row_scale)
  return tuple(held[name] for name in LABELS[1:])


def program_record(program):
  from tinygrad.uop.ops import Ops
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
  return dict(name=program.src[0].arg.function_name, program_key=program.key.hex(),
              source_sha256=sha(source.encode()), binary_sha256=sha(binary),
              global_size=list(program.arg.global_size), local_size=list(program.arg.local_size),
              outs=list(program.arg.outs), ins=list(program.arg.ins), globals=list(program.arg.globals),
              resources=load('agent-gemm-compile').metadata(binary))


def build_plan(variant, renderer):
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  x = Tensor.empty(*SHAPE, dtype=dtypes.bfloat16, device='CPU')
  with load('agent-vision-gelu-inspect').schedules() as calls: outputs = boundary(variant, x)
  tensors = [x, *outputs]
  # The packer materializes FP8 into an internal byte buffer, not the lazy q Tensor.
  # Retain that actual buffer without adding a separate q realization or changing fusion.
  known = {unwrap_lane(t.uop.base) for i, t in enumerate(tensors) if i != 5}
  internal = {unwrap_lane(arg):arg for call in calls for arg in call.src[1:] if unwrap_lane(arg) not in known}
  assert len(internal) == 1
  actual_q = next(iter(internal.values()))
  assert actual_q.dtype == dtypes.fp8e4m3 and actual_q.numel() == 2*720*4304
  tensors[5] = Tensor(actual_q).reshape(1440, 4304)
  spans = [unwrap_lane(t.uop.base) for t in tensors]
  assert len(set(spans)) == len(LABELS) and all(lane is None and offset == 0 for _, lane, offset in spans)
  steps, records = [], []
  for call in calls:
    program = to_program(call.body, renderer)
    mapping = [spans.index(unwrap_lane(arg)) for arg in call.src[1:]]
    for arg, slot in zip(call.src[1:], mapping, strict=True):
      assert arg.dtype == tensors[slot].dtype and arg.numel() == tensors[slot].numel()
    steps.append((program, mapping))
    records.append(dict(program_record(program), mapping=mapping, ast_key=call.body.key.hex()))
  assert len(steps) == (7 if variant == 'original' else 6)
  layout = [dict(name=name, shape=list(t.shape), dtype=str(t.dtype), nbytes=t.numel()*t.dtype.itemsize)
            for name, t in zip(LABELS, tensors, strict=True)]
  return dict(variant=variant, steps=steps, records=records, layout=layout, dtypes=[t.dtype for t in tensors])


def run_steps(plan, count, inputs):
  from tinygrad import Tensor
  values = list(inputs)
  for _ in range(count):
    for program, mapping in plan['steps']:
      args = [values[i] for i in mapping]
      result = Tensor.custom_kernel(*args, fxn=lambda *unused, p=program: p)
      changed = []
      for slot in program.arg.outs:
        values[mapping[slot]] = result[slot]
        changed.append(result[slot])
      assert changed
      Tensor.realize(*changed)
  return tuple(values[1:])


def ordered_queues(ctx, queues, plan, count):
  """Use real dependency queues; add explicit completion barriers between operations."""
  from tinygrad.uop.ops import Ops
  from tinygrad.runtime.support.hcq2 import ins, unwrap_lane
  expected = plan['steps']*count
  assert len(ctx.batch) == len(expected) and len(queues) == 1
  assert len({(dev, queue) for _, dev, queue in ctx.batch}) == 1
  key = next(iter(queues))
  assert key[1] == 'COMPUTE:0'
  mappings, identities = [], []
  for (call, _, _), (program, expected_mapping) in zip(ctx.batch, expected, strict=True):
    assert call.body.op is Ops.PROGRAM and program_record(call.body) == program_record(program)
    actual = []
    for arg in call.src[1:]:
      base, lane, offset = unwrap_lane(arg)
      assert base.op is Ops.PARAM and lane is None and offset == 0
      actual.append(base.arg.slot)
    assert actual == expected_mapping, 'Every operation must reuse the same input and output storage'
    mappings.append(actual)
    identities.append(call.body.key.hex())
  result, seen, inserted, barriers = [], 0, 0, []
  for command in queues[key]:
    if command.op is Ops.CALL and command.body.op is Ops.PROGRAM:
      if seen and seen % len(plan['steps']) == 0:
        # Full-operation serialization, including independent trailing row-scale writes.
        result.append(ins('exec_barrier')); inserted += 1
      if result and result[-1].op is Ops.CALL and result[-1].body.op is Ops.CUSTOM_FUNCTION and result[-1].body.arg.name == 'exec_barrier':
        barriers.append(seen)
      assert command == ctx.batch[seen][0]
      seen += 1
    result.append(command)
  assert seen == len(expected) and inserted == count-1
  assert all(i*len(plan['steps']) in barriers for i in range(1, count))
  queues[key] = result
  return dict(operations=count, kernels=seen, single_compute_queue=True, same_buffer_mapping=True,
              added_operation_barriers=inserted, barrier_before_kernel=barriers,
              active_program_keys=identities, mappings=mappings)


class Runner:
  def __init__(self, plan, count):
    from tinygrad import TinyJit
    self.plan, self.count, self.proofs = plan, count, []
    self.jit = TinyJit(lambda *inputs: run_steps(plan, count, inputs))
    # Capture immediately for the long batch; its programs already passed single replay.
    if count == CALLS: self.jit.cnt = 1

  def __call__(self, inputs):
    from tinygrad.runtime.support import hcq2
    original = hcq2._build_queues
    def build(ctx):
      queues = original(ctx)
      if len(ctx.batch) == self.count*len(self.plan['steps']):
        self.proofs.append(ordered_queues(ctx, queues, self.plan, self.count))
      return queues
    if self.jit.cnt == 1: hcq2._build_queues = build
    try: outputs = self.jit(*inputs)
    finally: hcq2._build_queues = original
    for output, storage in zip(outputs, inputs[1:], strict=True):
      a, b = output.uop.buffer, storage.uop.buffer
      assert a.base is b.base and a.offset == b.offset and a.nbytes == b.nbytes
    return outputs

  def identity(self):
    from tinygrad.engine.realize import HCQInfo
    assert len(self.proofs) == 1 and self.jit.captured is not None
    calls = self.jit.captured._linear.src
    assert len(calls) == 1 and isinstance(calls[0].without_after.arg.aux, HCQInfo)
    info = calls[0].without_after.arg.aux
    assert not info.skip_wait and len(info.kernels) == self.count*len(self.plan['steps'])
    assert [x[4].hex() for x in info.kernels] == self.proofs[0]['active_program_keys']
    return self.proofs[0]


def tensor_buffer(device, shape, dtype, data=None):
  """Always allocate real mutable storage, including shape=(). Never Tensor(np.scalar)."""
  from tinygrad import Tensor
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp, Ops
  import math
  size = math.prod(shape)*dtype.itemsize
  buffer = Buffer(device, size, options=BufferSpec(cpu_access=True)).ensure_allocated()
  if data is not None:
    assert len(data) == size
    buffer.host.view(size=size, fmt='B')[:] = data
  result = Tensor(UOp.from_buffer(buffer, dtype)).reshape(shape)
  assert result.uop.base.op is Ops.BUFFER and result.uop.is_realized and not result.uop.is_virtual
  return result


def write_buffer(tensor, raw):
  buffer = tensor.uop.buffer
  assert len(raw) == buffer.nbytes
  buffer.host.view(size=buffer.nbytes, fmt='B')[:] = raw


def read_buffer(tensor, device):
  # Canonical USB BAR traversal, with no Tensor cast/copy kernels in the comparison.
  buffer, chunks = tensor.uop.buffer, []
  offset, left = buffer.offset, tensor.numel()*tensor.dtype.itemsize
  for address, size in buffer.base.get_storage().meta.mapping.paddrs:
    if offset >= size: offset -= size; continue
    take = min(left, size-offset)
    for start in range(0, take, 65536):
      view = device.iface.pci_dev.map_bar(bar=device.iface.vram_bar, off=address+offset+start, size=min(take-start, 65536))
      chunks.append(bytes(view[:]))
    left -= take; offset = 0
    if not left: break
  assert left == 0
  return b''.join(chunks)


def snapshot(inputs, device):
  import numpy as np
  result = {key: read_buffer(value, device) for key, value in zip(LABELS[1:], inputs[1:], strict=True)}
  bf = np.frombuffer(result['activation'], np.uint16)
  assert np.isfinite((bf.astype(np.uint32) << 16).view(np.float32)).all(), 'Nonfinite original/current activation'
  for key in ('partial', 'maximum', 'scale', 'row_scale'):
    assert np.isfinite(np.frombuffer(result[key], np.float32)).all(), f'Nonfinite {key}'
  for key in ('fp8', 'packed'): assert ((np.frombuffer(result[key], np.uint8) & 0x7f) != 0x7f).all(), f'Nonfinite {key}'
  return result


def exact(left, right):
  assert left.keys() == right.keys()
  for key in left:
    if left[key] != right[key]:
      import numpy as np
      a, b = np.frombuffer(left[key], np.uint8), np.frombuffer(right[key], np.uint8)
      index = int(np.flatnonzero(a != b)[0])
      raise AssertionError(f'{key}: byte mismatch at {index}: {int(a[index])} != {int(b[index])}')


def checks(runners, inputs, device, report, path, phase, controls=range(8)):
  import numpy as np
  records, initial = [], None
  for state in controls:
    device.synchronize()
    bits = fixture(state)
    write_buffer(inputs[0], bits.tobytes())
    results = {}
    for label in ('original', 'candidate'):
      # Poison all seven outputs before replay; a missing write cannot pass on stale data.
      for tensor in inputs[1:]: write_buffer(tensor, bytes([0xa5])*tensor.nbytes())
      before = runners[label].jit.cnt
      runners[label](inputs)
      assert before >= 2 and runners[label].jit.cnt == before+1, 'Actual TinyJit replay required'
      device.synchronize()
      results[label] = snapshot(inputs, device)
    exact(results['original'], results['candidate'])
    current = results['original']
    if state == 0: initial = current
    if state == 7:
      assert initial is not None
      exact(initial, current)
    if state in range(1, 7):
      assert current['activation'] != initial['activation'] and current['scale'] != initial['scale'], 'Mutation did not affect outputs/scales'
    if state in (1, 2, 3, 5):
      unchanged = 1 if state in (1, 3, 5) else 0
      assert current['scale'][unchanged*4:unchanged*4+4] == initial['scale'][unchanged*4:unchanged*4+4]
    if state == 3: assert np.frombuffer(current['scale'], np.float32)[0] == np.float32(1e-12)
    records.append(dict(state=STATES[state], input_sha256=sha(bits.tobytes()), exact=True, all_finite=True,
      actual_changed_input_tinyjit_replay=True, output_sha256={k:sha(v) for k,v in current.items()},
      per_image_scale=np.frombuffer(current['scale'], np.float32).tolist()))
    report[phase] = records
    save(path, report)
  return records


def gpu(protocol, output):
  import numpy as np
  from tinygrad import Device
  from tinygrad.device import Compiled
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  Compiled.wait_timeout_ms = 10000
  report = dict(schema=1, diagnostic_only=True, completed=False, gpu_executed=False,
                native_gelu_equality_proven=False, model_hook_installed=False, adoption_allowed=False,
                source_sha256=protocol['source_sha256'], shape=list(SHAPE), operations=CALLS)
  save(output, report)
  try:
    verify(protocol)
    load('prefill-kernels').fix_dispatch_packet()
    device = Device[Device.DEFAULT]
    report['gpu_executed'] = True
    assert device.arch == 'gfx1200' and device.is_usb and isinstance(device.renderer, AMDLLVMRenderer)
    load('wgp_compiler').install(device)
    report['runtime'] = dict(device=device.device, arch=device.arch, usb=device.is_usb,
      renderer=type(device.renderer).__name__, compiler=type(device.renderer.compiler).__name__,
      environment={key:os.environ.get(key) for key in ('DEV', 'TC_OPT', 'LLVMOPT', 'BEAM', 'JITBEAM', 'PROFILE', 'GMMU')})
    plans = {label:build_plan(label, device.renderer) for label in ('original', 'candidate')}
    for label, plan in plans.items():
      assert plan['records'] == protocol['plans'][label]['records'], f'{label} active compilation differs from CPU preflight'
      assert plan['layout'] == protocol['plans'][label]['layout']
    assert plans['original']['layout'] == plans['candidate']['layout']
    artifacts = output.parent/'agent-vision-gelu-micro-active'
    artifacts.mkdir(exist_ok=True)
    for label, plan in plans.items():
      for i, (program, _) in enumerate(plan['steps']):
        (artifacts/f'{label}-{i}.source').write_text(program.src[2].arg)
        (artifacts/f'{label}-{i}.hsaco').write_bytes(program.src[3].arg)
    report['active_programs'] = {key:plan['records'] for key, plan in plans.items()}
    inputs = tuple(tensor_buffer(device.device, row['shape'], dt, fixture(0).tobytes() if i == 0 else None)
                   for i, (row, dt) in enumerate(zip(plans['original']['layout'], plans['original']['dtypes'], strict=True)))
    report['mutable_input_buffers'] = True
    single = {key:Runner(plan, 1) for key, plan in plans.items()}
    for runner in single.values():
      runner(inputs); runner(inputs)
    report['single_queue_proof'] = {key:runner.identity() for key, runner in single.items()}
    checks(single, inputs, device, report, output, 'single_controls')
    batch = {key:Runner(plan, CALLS) for key, plan in plans.items()}
    for runner in batch.values(): runner(inputs)
    report['batch_queue_proof'] = {key:runner.identity() for key, runner in batch.items()}
    checks(batch, inputs, device, report, output, 'batch_controls')
    verify(protocol)
    save(output, report)
    pairs = []
    for repeat in range(3):
      order = ('original', 'candidate') if repeat % 2 == 0 else ('candidate', 'original')
      row = dict(order=list(order), batch_ms={})
      for label in order:
        device.synchronize()
        start = time.perf_counter()
        batch[label](inputs)
        device.synchronize()  # One completion fence for all 40 full boundaries.
        row['batch_ms'][label] = (time.perf_counter()-start)*1000
      pairs.append(row)
    # Do not publish timing or exactness claims until replay remains exact after timing.
    checks(batch, inputs, device, report, output, 'post_timing_controls', controls=(0, 6, 7))
    verify(protocol)
    medians = {key:float(np.median([row['batch_ms'][key] for row in pairs]))/CALLS for key in plans}
    report.update(completed=True, native_gelu_equality_proven=True, native_equality_scope='Only this frozen finite-input boundary protocol',
      post_timing_exact=True, timings=dict(pairs=pairs, median_ms_per_boundary=medians,
        improvement_ms=medians['original']-medians['candidate'], operations_per_batch=CALLS,
        completion_fences_per_batch=1, full_boundary=True, transfers_and_checks_excluded=True,
        throughput_or_route_claim=False), no_variants_added=True)
    save(output, report)
    print(json.dumps(dict(completed=True, native_gelu_equality_proven=True, timings=report['timings'])), flush=True)
  except BaseException as exc:
    report.update(completed=False, native_gelu_equality_proven=False, failure=f'{type(exc).__name__}: {exc}')
    report.pop('timings', None)
    save(output, report)
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--protocol', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  parser.add_argument('--parent-gpu', action='store_true')
  parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
  args = parser.parse_args()
  assert args.parent_gpu, 'Parent-only GPU diagnostic; use micro-check.py for CPU work'
  protocol = json.loads(args.protocol.read_text())
  verify(protocol)
  if args.child:
    assert os.environ.get('AGENT_VISION_GELU_MICRO_CHILD') == '1'
    for item in protocol['environment']['PYTHONPATH'].split(os.pathsep): sys.path.insert(0, item)
    gpu(protocol, args.output)
    return
  args.output.parent.mkdir(parents=True, exist_ok=True)
  assert not args.output.exists(), 'Use a fresh output path'
  save(args.output, dict(completed=False, native_gelu_equality_proven=False, gpu_executed=False, diagnostic_only=True))
  env = environment(protocol)
  env['AGENT_VISION_GELU_MICRO_CHILD'] = '1'
  command = [protocol['python'], str(Path(__file__).resolve()), '--protocol', str(args.protocol.resolve()),
             '--output', str(args.output.resolve()), '--parent-gpu', '--child']
  process = subprocess.Popen(command, env=env, start_new_session=True)
  try: code = process.wait(timeout=295)
  except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=2)
    report = json.loads(args.output.read_text())
    report.update(completed=False, native_gelu_equality_proven=False, failure='Hard bounded parent diagnostic timeout')
    report.pop('timings', None)
    save(args.output, report)
    raise SystemExit(124)
  raise SystemExit(code)


if __name__ == '__main__': main()

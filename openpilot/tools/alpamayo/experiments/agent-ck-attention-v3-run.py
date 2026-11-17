"""Parent-only, <=300-second CK diagnostic. No model integration.

Default operation verifies CPU artifacts only. --parent-run explicitly opens the GPU.
Use the external timeout command in the report; this agent never runs that option.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

EXP = Path(__file__).parent
ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-attention-v3')
MANIFEST = ROOT/'prepared-final/protocol.json'
GUARD = 256


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), EXP/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def verify():
  plan = json.loads(MANIFEST.read_text())
  assert plan['schema'] == 1 and plan['timeout_seconds'] == 300
  for path, expected in plan['source_sha256'].items():
    assert digest(path) == expected, f'Source/binary changed: {path}'
  revision = subprocess.check_output(['git', '-C', plan['tinygrad'], 'rev-parse', 'HEAD'], text=True).strip()
  assert revision == plan['tinygrad_commit'] == '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
  oracle = json.loads((Path(plan['oracle'])/'manifest.json').read_text())
  assert oracle['limits'] == plan['limits'] and oracle['replay_order'] == plan['replay_order']
  for fixture in oracle['fixtures']:
    for file in fixture['files'].values(): assert digest(Path(plan['oracle'])/file['path']) == file['sha256']
  return plan, oracle


def metrics(actual, reference):
  import numpy as np
  delta = actual.astype(np.float64)-reference.astype(np.float64)
  assert np.isfinite(actual).all()
  rows = np.unique(np.concatenate([np.arange(max(0, r-1), min(2048, r+2)) for r in range(0, 2048, 64)]))
  return {'rmse': float(np.sqrt(np.mean(delta*delta))), 'max_abs': float(np.max(np.abs(delta))),
    'per_head_rmse': np.sqrt(np.mean(delta*delta, axis=(0, 2, 3))).tolist(),
    'causal_tile_edge_max_abs': float(np.max(np.abs(delta[:, :, rows])))}


def error_gate(candidate, baseline, limits):
  for name, field in [('output_rmse', 'rmse'), ('output_max_abs', 'max_abs')]:
    rule = limits[name]
    assert candidate[field] <= max(rule['absolute_floor'], rule['baseline_multiplier']*baseline[field]), (name, candidate, baseline)
  rule = limits['per_head_rmse']
  assert all(c <= max(rule['absolute_floor'], rule['baseline_multiplier']*b)
             for c, b in zip(candidate['per_head_rmse'], baseline['per_head_rmse'], strict=True))


def allocate(device, shape, dtype):
  from tinygrad import Tensor
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp
  buffer = Buffer(device, math.prod(shape)*dtype.itemsize, options=BufferSpec(cpu_access=True)).ensure_allocated()
  return Tensor(UOp.from_buffer(buffer, dtype)).reshape(shape)


def write(tensor, data):
  buffer = tensor.uop.buffer
  assert len(data) == buffer.nbytes and buffer.host is not None
  buffer.host.view(size=buffer.nbytes, fmt='B')[:] = data


def read(tensor, device):
  # Same USB BAR traversal used by the reviewed resident diagnostics.
  buffer, chunks = tensor.uop.buffer, []
  offset, left = buffer.offset, tensor.numel()*tensor.dtype.itemsize
  for address, size in buffer.base.get_storage().meta.mapping.paddrs:
    if offset >= size: offset -= size; continue
    take = min(left, size-offset)
    for start in range(0, take, 65536):
      view = device.iface.pci_dev.map_bar(bar=device.iface.vram_bar, off=address+offset+start, size=min(take-start, 65536))
      chunks.append(bytes(view[:]))
    left -= take
    offset = 0
    if not left: break
  assert left == 0
  return b''.join(chunks)


@contextmanager
def record_programs(report, folder):
  from tinygrad.runtime import ops_amd
  from tinygrad.uop.ops import UOp
  from tinygrad import dtypes
  old_build, old_packet = ops_amd.amd_build_program, ops_amd.dispatch_packet
  seen = {}
  def typed_packet(*args, **kwargs):
    return [UOp.const(x, dtypes.uint32) if isinstance(x, int) else x for x in old_packet(*args, **kwargs)]
  def build(dev, prg, devs):
    blob = prg.src[3].arg
    key = hashlib.sha256(blob).hexdigest()
    if key not in seen:
      data, _ = ops_amd._amd_program_image(dev, blob)
      assert ((data.rsrc1 >> 29) & 1) == 0, 'All programs must report actual CU descriptor mode'
      (folder/(key+'.hsaco')).write_bytes(blob)
      source = prg.src[2].arg
      (folder/(key+'.source')).write_text(source)
      seen[key] = {'name': prg.src[0].arg.name, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'descriptor': asdict(data), 'actual_mode': 'CU', 'global': list(prg.arg.global_size),
        'local': list(prg.arg.local_size), 'outs': list(prg.arg.outs), 'ins': list(prg.arg.ins)}
    return old_build(dev, prg, devs)
  ops_amd.amd_build_program, ops_amd.dispatch_packet = build, typed_packet
  try: yield
  finally:
    report['actual_programs'] = seen
    assert ops_amd.amd_build_program is build and ops_amd.dispatch_packet is typed_packet
    ops_amd.amd_build_program, ops_amd.dispatch_packet = old_build, old_packet


def run(plan, oracle, output):
  import numpy as np
  from tinygrad import Tensor, TinyJit, dtypes
  from tinygrad.device import Device, Compiled
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  adapter = load('agent-ck-attention-v2-adapter')
  baseline_blob = Path(plan['baseline']).read_bytes()
  baseline_source = Path(plan['baseline']).with_name('baseline.cpp').read_text()
  def baseline_program(out, q, k, v):
    value = q.flatten().index(0).load()+k.flatten().index(0).load()+v.flatten().index(0).load()
    sink = UOp.sink(out.flatten().index(0).store(value), UOp.special(256, 'gidx0'), UOp.special(256, 'lidx0'),
                    arg=KernelInfo(name='agent_attention_rowsoftmax'))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
      UOp(Ops.SOURCE, arg=baseline_source), UOp(Ops.BINARY, arg=baseline_blob)))
  def baseline(out, q, k, v):
    # Exact selected pack_values bit permutation; cost is inside both boundaries.
    packed = (v.bitcast(dtypes.uint16).reshape(1, 8, 128, 2, 8, 8, 16)
              .permute(0, 1, 5, 2, 3, 6, 4).contiguous().bitcast(dtypes.bfloat16))
    return Tensor.custom_kernel(out, q.contiguous(), k.contiguous(), packed, fxn=baseline_program)[0]
  report = {'schema': 1, 'completed': False, 'gpu_executed': False, 'model_integrated': False,
    'classification': plan['classification'], 'limits': plan['limits'], 'boundaries': plan['boundaries'],
    'boundary_B_limit': plan['boundary_B_limit'], 'protocol_sha256': digest(MANIFEST),
    'timing_kind': 'Synchronized wall time, including submit and full boundary GPU work; no isolated device-time claim.'}
  def save(): output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
  started = time.monotonic()
  def expired(*_): raise TimeoutError('290-second internal diagnostic deadline')
  signal.signal(signal.SIGALRM, expired)
  signal.alarm(290)
  Compiled.wait_timeout_ms = 10000
  programs = output.parent/(output.stem+'-programs')
  programs.mkdir(exist_ok=False)
  save()
  try:
    device = Device[Device.DEFAULT]
    assert device.arch == 'gfx1200' and device.is_usb and not device.prof_enabled
    report['gpu_executed'] = True
    report['runtime'] = {'device': device.device, 'arch': device.arch, 'renderer': type(device.renderer).__name__,
      'compiler': type(device.renderer.compiler).__name__, 'expected_actual_mode': 'CU',
      'environment': {k: os.environ.get(k) for k in ('DEV', 'PARALLEL', 'PROFILE', 'LLVMOPT', 'GMMU', 'CACHEDB')}}
    cases = {}
    for case in oracle['fixtures']:
      cases[case['name']] = {name: np.load(Path(plan['oracle'])/file['path'], allow_pickle=False)
                            for name, file in case['files'].items()}
    o_count, l_count = math.prod(adapter.QSHAPE), math.prod(adapter.LSHAPE)
    def state(boundary):
      shapes = [adapter.QSHAPE, adapter.KVSHAPE, adapter.KVSHAPE] if boundary == 'A' else [
        (1, 2048, 16, 128), (1, 8, 2304, 128), (1, 8, 2304, 128)]
      return [allocate(device.device, shape, dtypes.bfloat16) for shape in shapes]
    inputs = {b: state(b) for b in ('A', 'B')}
    outputs = {(b, arm): [allocate(device.device, (o_count+2*GUARD,), dtypes.bfloat16),
                            allocate(device.device, (l_count+2*GUARD,), dtypes.float32)]
               for b in ('A', 'B') for arm in ('candidate', 'baseline')}
    poison_o = np.full(o_count+2*GUARD, 0x7fc1, np.uint16)
    poison_o[:GUARD] = poison_o[-GUARD:] = 0x5aa5
    poison_l = np.full(l_count+2*GUARD, 0x7fc12345, np.uint32)
    poison_l[:GUARD] = poison_l[-GUARD:] = 0x5aa55aa5
    def upload(boundary, case):
      device.synchronize()
      for name, tensor in zip(('q', 'k', 'v'), inputs[boundary], strict=True):
        bits = cases[case][name]
        if boundary == 'B':
          if name == 'q': bits = bits.transpose(0, 2, 1, 3).copy()
          else:
            padded = np.full((1, 8, 2304, 128), 0x7fc1, np.uint16)
            padded[:, :, :2048] = bits
            bits = padded
        write(tensor, bits.tobytes())
    def poison(boundary, arm):
      device.synchronize()
      out, lse = outputs[boundary, arm]
      write(out, poison_o.tobytes())
      write(lse, poison_l.tobytes())
    def runner(boundary, arm):
      def invoke(q, k, v, out_storage, lse_storage):
        if boundary == 'B': q, k, v = q.transpose(1, 2), k[:, :, :2048], v[:, :, :2048]
        out = out_storage[GUARD:GUARD+o_count].reshape(adapter.QSHAPE)
        lse = lse_storage[GUARD:GUARD+l_count].reshape(adapter.LSHAPE)
        if arm == 'candidate': out, lse = adapter.attention(q, k, v, out=out, lse=lse)
        else: out = baseline(out, q, k, v)
        if boundary == 'B': out = out.transpose(1, 2).reshape(1, 2048, 2048).contiguous()
        # LSE is realized with O and retained through every replay.
        return (out.realize(lse), lse) if arm == 'candidate' else (out.realize(),)
      return TinyJit(invoke)
    runners = {(b, arm): runner(b, arm) for b in ('A', 'B') for arm in ('candidate', 'baseline')}
    def invoke(boundary, arm):
      return runners[boundary, arm](*inputs[boundary], *outputs[boundary, arm])
    def snapshot(boundary, arm, returned):
      device.synchronize()
      out, lse = outputs[boundary, arm]
      obits = np.frombuffer(read(out, device), np.uint16)
      assert np.array_equal(obits[:GUARD], poison_o[:GUARD]) and np.array_equal(obits[-GUARD:], poison_o[-GUARD:])
      obits = obits[GUARD:GUARD+o_count].reshape(adapter.QSHAPE).copy()
      value = (obits.astype(np.uint32) << 16).view(np.float32)
      assert np.isfinite(value).all(), 'O poison survived or output is nonfinite'
      result = {'bits': obits, 'value': value}
      if arm == 'candidate':
        lbits = np.frombuffer(read(lse, device), np.uint32)
        assert np.array_equal(lbits[:GUARD], poison_l[:GUARD]) and np.array_equal(lbits[-GUARD:], poison_l[-GUARD:])
        result['lse'] = lbits[GUARD:GUARD+l_count].view(np.float32).reshape(adapter.LSHAPE).copy()
        assert np.isfinite(result['lse']).all(), 'LSE poison survived or output is nonfinite'
      if boundary == 'B':
        common = np.frombuffer(read(returned[0], device), np.uint16).reshape(1, 2048, 16, 128).transpose(0, 2, 1, 3)
        assert np.array_equal(common, obits), 'Common output conversion changed BF16 bits'
      return result
    with adapter.abi_scope(), record_programs(report, programs):
      for boundary in ('A', 'B'):
        upload(boundary, 'a')
        for arm in ('baseline', 'candidate'):
          poison(boundary, arm)
          invoke(boundary, arm); invoke(boundary, arm)
          device.synchronize()
      report['cold_init_compile_upload_seconds'] = time.monotonic()-started
      references, checks = {}, []
      report['full_actual_shape_checks'] = checks
      for name in cases:
        upload('A', name)
        measured = {}
        for arm in ('baseline', 'candidate'):
          poison('A', arm)
          assert runners['A', arm].cnt >= 2
          measured[arm] = snapshot('A', arm, invoke('A', arm))
        errors = {arm: metrics(result['value'], cases[name]['output_fp32']) for arm, result in measured.items()}
        delta = measured['candidate']['lse'].astype(np.float64)-cases[name]['lse_fp32']
        lse_error = {'rmse': float(np.sqrt(np.mean(delta*delta))), 'max_abs': float(np.max(np.abs(delta)))}
        checks.append({'fixture': name, 'errors': errors, 'lse': lse_error, 'poison_redzones_finite': True})
        save()
        error_gate(errors['candidate'], errors['baseline'], plan['limits'])
        assert lse_error['rmse'] <= plan['limits']['lse_rmse_abs'] and lse_error['max_abs'] <= plan['limits']['lse_max_abs'], lse_error
        references[name] = measured
      for arm in ('baseline', 'candidate'):
        for name in ('mutate_q', 'mutate_k', 'mutate_v'):
          assert np.max(np.abs(references[name][arm]['value']-references['a'][arm]['value'])) > plan['limits']['mutation_min_output_max_abs_change']
        assert np.array_equal(references['a'][arm]['bits'][:, :, :1024], references['causal_future'][arm]['bits'][:, :, :1024])
        assert not np.array_equal(references['a'][arm]['bits'][:, :, 1024:], references['causal_future'][arm]['bits'][:, :, 1024:])
        assert all(np.array_equal(references['gqa'][arm]['bits'][:, h], references['gqa'][arm]['bits'][:, h+1]) for h in range(0,16,2))
      assert np.array_equal(references['a']['candidate']['lse'][:, :, :1024], references['causal_future']['candidate']['lse'][:, :, :1024])
      report['full_actual_shape_checks'] = checks
      report['causal_gqa_mutations_passed'] = True
      save()
      replay = []
      input_ids = [id(t.uop.buffer) for t in inputs['A']]
      for index, name in enumerate(plan['replay_order']):
        upload('A', name)
        poison('A', 'candidate')
        count = runners['A', 'candidate'].cnt
        result = snapshot('A', 'candidate', invoke('A', 'candidate'))
        assert runners['A', 'candidate'].cnt == count+1 and count >= 2
        assert [id(t.uop.buffer) for t in inputs['A']] == input_ids
        assert np.array_equal(result['bits'], references[name]['candidate']['bits'])
        assert np.array_equal(result['lse'], references[name]['candidate']['lse'])
        replay.append({'index': index, 'fixture': name, 'jit_count': count+1,
          'output_sha256': hashlib.sha256(result['bits'].tobytes()).hexdigest(),
          'lse_sha256': hashlib.sha256(result['lse'].tobytes()).hexdigest()})
      assert len(replay) == 40
      report['ordered_mutable_replay'] = replay
      save()
      timings = {}
      for boundary in ('A', 'B'):
        upload(boundary, 'a')
        for arm in ('baseline', 'candidate'):
          poison(boundary, arm)
          result = snapshot(boundary, arm, invoke(boundary, arm))
          assert np.array_equal(result['bits'], references['a'][arm]['bits'])
          if arm == 'candidate': assert np.array_equal(result['lse'], references['a'][arm]['lse'])
        samples = {'baseline': [], 'candidate': []}
        for repeat in range(12):
          for arm in (('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')):
            device.synchronize()
            begin = time.perf_counter()
            for _ in range(5): invoke(boundary, arm)
            device.synchronize()
            samples[arm].append((time.perf_counter()-begin)*1000/5)
        stats = {arm: {'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values,95)), 'samples_ms': values}
                 for arm, values in samples.items()}
        stats['median_ratio'] = stats['candidate']['median_ms']/stats['baseline']['median_ms']
        stats['p95_ratio'] = stats['candidate']['p95_ms']/stats['baseline']['p95_ms']
        timings[boundary] = stats
      report['timings'] = timings
      report['speed_screen_passed'] = all(x['median_ratio'] <= .95 and x['p95_ratio'] <= 1.05 for x in timings.values())
      report['numerical_micro_gates_passed'] = True
      report['completed'] = True
      device.synchronize()
  except BaseException as error:
    report['failure'] = {'type': type(error).__name__, 'message': str(error)}
    raise
  finally:
    signal.alarm(0)
    report['elapsed_seconds'] = time.monotonic()-started
    save()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--parent-run', action='store_true')
  parser.add_argument('--output', type=Path, default=ROOT/'parent-result.json')
  args = parser.parse_args()
  plan, oracle = verify()
  if not args.parent_run:
    print(json.dumps({'cpu_verification': 'passed', 'fixtures': len(oracle['fixtures']), 'gpu_executed': False}))
  else:
    assert args.output.parent.resolve() == ROOT.resolve() and not args.output.exists()
    os.environ.update(DEV='USB+AMD:LLVM', CPU='0', AMD='0', ALLOW_DEVICE_USAGE='1', PARALLEL='0', PROFILE='0',
      BEAM='0', JITBEAM='0', JIT='1', CAPTURING='1', LLVMOPT='1', GMMU='0', TC_OPT='2', DEBUG='0',
      PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(ROOT), CACHEDB=str(ROOT/'parent-cache.db'),
      OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    sys.path.insert(0, plan['tinygrad'])
    run(plan, oracle, args.output)

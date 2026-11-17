"""Bounded original-order FP8 tile study. CPU modes never open AMD devices."""
import argparse
from functools import partial
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = {'prefill-up': (2112, 9216, 2048), 'prefill-down': (2112, 2048, 9216),
         'vision-up': (720, 4304, 1152), 'vision-down': (720, 1152, 4304)}
TILES = ((3, 4), (4, 3), (3, 3), (2, 6), (6, 2), (3, 6))
SHORTLIST = {'prefill-up': ((3, 4), (4, 3), (3, 3), (3, 6)), 'prefill-down': ((3, 4), (3, 3), (6, 2)),
             'vision-up': ((3, 4), (4, 3), (3, 3)), 'vision-down': ((3, 4), (3, 3), (3, 6))}
CALLS = 40


def load(name):
  spec = importlib.util.spec_from_file_location('agent_matrix_tiles_'+name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


E = load('agent-expert-gemm')
T = load('tiled-gemm')


def functions(renderer, row_scale, tiles):
  from amd_gemm import _gemm
  result = {}
  if not row_scale: result['original'] = E.bind_renderer(_gemm, renderer)
  for tm, tn in ((4, 4), *tiles):
    label = 'current' if (tm, tn) == (4, 4) else f'{tm}x{tn}'
    result[label] = partial(E.bind_renderer(T.kernel, renderer), tile_m=tm, tile_n=tn,
      row_scale=row_scale, column_scale=False, local_m=1, local_n=1, swap=False, unroll_k=1, group_m=1)
  return result


def fixture(shape, state, integer, row_scale, nonuniform=False):
  data = E.fixture(shape, state, integer)
  if row_scale:
    data['sx'] = np.full(shape[0], data['sx'][0], np.float32)
    if nonuniform: data['sx'] *= np.exp2(np.arange(shape[0]) % 3 - 1).astype(np.float32)
  return data


def integer_reference(data, with_bias, row_scale):
  a, w = (E.decode_fp8(data[key+'_unpacked']).astype(np.int64) for key in ('a', 'w'))
  sums = a @ w.T
  assert np.max(np.abs(sums)) < 2**24
  scale = data['sx'][:, None] if row_scale else data['sx'][0]
  result = sums.astype(np.float32)*(scale*data['sw'][0])
  if with_bias: result += E.unpack_bf16(data['bias'])
  return E.bf16(result)


def cpu_checks():
  from tinygrad import Device, Tensor, dtypes
  from amd_gemm import pack
  checks = E.cpu_checks()
  for shape in ((17, 48, 32), (49, 80, 64)):
    for state in range(5):
      data = fixture(shape, state, True, True, nonuniform=True)
      for key in ('a', 'w'):
        np.testing.assert_array_equal(pack(Tensor(data[key+'_unpacked'], device='CPU')).numpy(), data[key])
      a, w = (Tensor(E.decode_fp8(data[key+'_unpacked']), device='CPU') for key in ('a', 'w'))
      scale = Tensor(data['sx'], device='CPU').reshape(-1, 1)*Tensor(data['sw'], device='CPU')
      for bias in (False, True):
        result = a.dot(w.T)*scale
        if bias: result += Tensor(E.unpack_bf16(data['bias']), device='CPU')
        np.testing.assert_array_equal(result.cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy(), integer_reference(data, bias, True))
      checks.append({'shape_mnk': shape, 'state': state, 'nonuniform_row_scales': True, 'integer_and_packing_exact': True})
  # A geometric proof covers every valid output and each ascending K16 block.
  geometry = []
  for name, (m, n, k) in CASES.items():
    for tm, tn in ((4, 4), *TILES):
      seen_m = [b*16*tm+i for b in range(math.ceil(m/(16*tm))) for i in range(16*tm) if b*16*tm+i < m]
      seen_n = [b*16*tn+i for b in range(math.ceil(n/(16*tn))) for i in range(16*tn) if b*16*tn+i < n]
      assert seen_m == list(range(m)) and seen_n == list(range(n))
      assert [base+j for base in range(0, k, 16) for j in range(16)] == list(range(k))
      geometry.append({'case': name, 'tile': [tm, tn], 'output_partition_exact': True, 'ascending_k16': True})
  assert not any(d.startswith('AMD') for d in Device._opened_devices)
  return {'semantic_checks': checks, 'geometry_checks': geometry, 'gpu_executed': False}


def compile_record(shape, bias, row_scale, fn, renderer, tile, artifact):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import UOp, Ops, AxisType
  m, n, k = shape
  specs = [((m, n), dtypes.bfloat16), ((math.ceil(m/16)*16, k), dtypes.fp8e4m3),
           ((math.ceil(n/16)*16, k), dtypes.fp8e4m3), ((m if row_scale else 1,), dtypes.float32), ((1,), dtypes.float32)]
  if bias: specs.append(((n,), dtypes.bfloat16))
  arguments = [UOp.placeholder(s, dt, i) for i, (s, dt) in enumerate(specs)]
  ast = fn(*arguments)
  reductions = [(u.axis_type.name, int(u.vmax+1)) for u in ast.toposort() if u.op is Ops.RANGE and u.axis_type in (AxisType.LOOP, AxisType.WEAK)]
  assert reductions == [('WEAK', k//16)], reductions
  program = to_program(ast, renderer)
  binary = program.src[3].arg
  llvm = program.src[2].arg
  assert isinstance(llvm, str)
  wmma = [line for line in llvm.splitlines() if 'call ' in line and '@llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8' in line]
  assert len(wmma) == math.prod(tile), (tile, len(wmma))
  notes = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
  keys = ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size',
          'group_segment_fixed_size', 'wavefront_size', 'workgroup_processor_mode')
  resources = {key: int(value) for key, value in re.findall(r'\.('+'|'.join(keys)+r'):\s*(\d+)', notes)}
  assert resources['wavefront_size'] == 32 and resources['workgroup_processor_mode'] == 1
  assert resources['private_segment_fixed_size'] == resources['group_segment_fixed_size'] == 0, resources
  for key in ('vgpr_spill_count', 'sgpr_spill_count'): assert resources.get(key, 0) == 0
  params = [UOp.param(slot, arg.dtype, arg.numel(), device='AMD') for slot, arg in zip((5, 0, 1, 2, 3, 4), arguments)]
  proof = E.prove_cpu_queue([program.call(*params)]*CALLS, device='AMD')
  if artifact:
    artifact.with_suffix('.ll').write_text(llvm)
    artifact.with_suffix('.hsaco').write_bytes(binary)
    artifact.with_suffix('.notes').write_text(notes)
  tm, tn = tile
  groups = math.ceil(m/(16*tm))*math.ceil(n/(16*tn))
  # Repeated per-workgroup operand requests, before cache reuse and invalid-tail suppression.
  operand_bytes = groups*16*(tm+tn)*k
  baseline_bytes = math.ceil(m/64)*math.ceil(n/64)*128*k
  return {'status': 'compiled', 'tile': tile, 'global_size': program.arg.global_size, 'local_size': program.arg.local_size,
    'resources': resources, 'binary_sha256': E.sha(binary), 'llvm_sha256': E.sha(llvm.encode()),
    'k16_reduction': reductions, 'wmma_instructions_per_iteration': len(wmma), 'operand_order': 'original fp8 A,B',
    'workgroups': groups, 'accumulator_vgprs_per_lane': 8*tm*tn,
    'nominal_operand_request_bytes': operand_bytes, 'operand_request_ratio_to_4x4': operand_bytes/baseline_bytes,
    'padded_output_elements': groups*256*tm*tn, 'padding_compute_ratio': groups*256*tm*tn/(m*n),
    'offline_serial_queue_proof': proof}


def run_case(name, shape, bias, row_scale, fns, resources, rounds, device, integer=False):
  record = {'case': name, 'shape_mnk': shape, 'bias': bias, 'row_scales': row_scale,
            'variants': {label: dict(info, checks=[]) for label, info in resources.items()}, 'timing': []}
  tensors = E.upload(fixture(shape, 0, integer, row_scale, nonuniform=integer and row_scale))
  singles = {}
  for label, fn in fns.items():
    if resources[label]['status'] != 'compiled': continue
    jit = E.make_jit(shape, bias, fn, 1)
    for _ in range(3): jit(*tensors)
    assert len(E.captured_kernels(jit)) == 1
    singles[label] = jit
  previous = None
  for state in range(4):
    data = fixture(shape, state, integer, row_scale, nonuniform=integer and row_scale)
    E.change_inputs(tensors, data, device)
    E.poison(singles['current'].captured.ret, device)
    expected = E.read_output(singles['current'](*tensors)[0], device)
    assert np.isfinite(E.unpack_bf16(expected)).all()
    if integer: np.testing.assert_array_equal(expected, integer_reference(data, bias, row_scale))
    if previous is not None: assert not np.array_equal(expected, previous)
    previous = expected.copy()
    if state == 0 and not (integer and row_scale):
      from amd_gemm import gemm, unpack
      a, w, sx, sw, b = tensors
      public = gemm(unpack(a)[:shape[0]], unpack(w)[:shape[1]], (sx[:1], sw), b if bias else None).realize()
      E.exact_output(public, expected, device)
      record['original_public_wrapper_exact'] = True
    for label in list(singles):
      E.poison(singles[label].captured.ret, device)
      try: E.exact_output(singles[label](*tensors)[0], expected, device)
      except AssertionError as exc:
        if label in ('current', 'original'): raise
        record['variants'][label].update(status='incorrect', failed_state=state, error=str(exc)[:1000])
        del singles[label]
        continue
      record['variants'][label]['checks'].append({'state': state, 'exact_bf16': True, 'independent_integer': integer,
                                                 'reference_sha256': E.sha(expected.tobytes())})
  if integer: return record
  batches = {}
  for label in singles:
    jit = E.make_jit(shape, bias, fns[label], CALLS)
    for _ in range(3): jit(*tensors)
    assert len(E.captured_kernels(jit)) == CALLS
    assert E.shared_output_slot(jit) == 5
    record['variants'][label]['serial_queue_proof'] = jit.proofs[0]
    batches[label] = jit
  data = fixture(shape, 4, False, row_scale)
  E.change_inputs(tensors, data, device)
  expected = E.read_output(singles['current'](*tensors)[0], device)
  assert not np.array_equal(expected, previous)
  for label in list(batches):
    E.poison(batches[label].captured.ret, device)
    try: E.exact_output(batches[label](*tensors)[0], expected, device)
    except AssertionError as exc:
      if label in ('current', 'original'): raise
      record['variants'][label].update(status='incorrect-batch-replay', error=str(exc)[:1000])
      del batches[label]
      continue
    record['variants'][label]['changed_all_inputs_after_capture_exact'] = True
  def measure(jit):
    device.synchronize()
    start = time.perf_counter()
    jit(*tensors)
    device.synchronize()
    return (time.perf_counter()-start)*1000
  for label in batches:
    if label == 'current': continue
    pairs = []
    for iteration in range(rounds):
      order = ('current', label) if iteration % 2 == 0 else (label, 'current')
      pairs.append({key: measure(batches[key]) for key in order})
    base, candidate = (statistics.median(pair[key] for pair in pairs)/CALLS for key in ('current', label))
    record['timing'].append({'variant': label, 'paired_batch_ms': pairs, 'current_amortized_ms': base,
      'candidate_amortized_ms': candidate, 'candidate_over_current': candidate/base, 'matrix_calls_per_batch': CALLS,
      'host_inclusive': True, 'device_only': False, 'warm_repeated_weights': True, 'completion_fences_per_batch': 1})
  return record


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=('cpu-check', 'compile', 'gpu'), default='compile')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--cases', default=','.join(CASES))
  parser.add_argument('--rounds', type=int, default=3)
  args = parser.parse_args()
  selected = args.cases.split(',')
  assert len(selected) == len(set(selected)) and all(case in CASES for case in selected)
  assert 3 <= args.rounds <= 5
  os.environ.update(BEAM='0', PROFILE='0', DEBUG='0', PARALLEL='0')
  if args.mode != 'gpu': os.environ['DEV'] = 'CPU'
  if args.mode == 'compile': os.environ['ALLOW_DEVICE_USAGE'] = '0'
  from tinygrad import Device
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  import tinygrad
  sources = [Path(__file__), HERE/'agent-expert-gemm.py', HERE/'tiled-gemm.py', HERE.parent/'amd_gemm.py', HERE/'wgp_compiler.py']
  report = {'schema': 1, 'mode': args.mode, 'gpu_executed': False, 'complete': False,
    'source_sha256': {str(p): E.sha(p.read_bytes()) for p in sources}, 'cases': [],
    'tinygrad_revision': subprocess.check_output(['git', '-C', str(Path(tinygrad.__file__).parent.parent), 'rev-parse', 'HEAD'], text=True).strip(),
    'runtime_environment': {k: os.environ.get(k) for k in ('DEV', 'CACHEDB', 'AM_POWER_LIMIT', 'AM_PERF_LEVEL', 'TC_OPT')},
    'compiler': 'gfx1200 AMD LLVM WGP (-cumode)', 'matrix_calls_per_batch': CALLS, 'rounds': args.rounds,
    'scope': 'Matrix-only, unchanged existing packed FP8 inputs/scales. No quantization, packing, uploads, or pipeline timing.',
    'timing': '40 ordered calls, same output buffer, actual HCQ queue proof, host wall time and one completion fence; warm weights.'}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  if args.mode == 'cpu-check':
    report.update(checks=cpu_checks(), complete=True)
    E.write_json(args.output, report)
    print(json.dumps({'complete': True, 'gpu_executed': False}))
    return
  if args.mode == 'compile':
    renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
    renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  else:
    assert os.environ.get('AM_POWER_LIMIT') == '144' and os.environ.get('AM_PERF_LEVEL') == 'auto'
    from tinygrad.device import Compiled
    Compiled.wait_timeout_ms = 10000
    load('prefill-kernels').fix_dispatch_packet()
    device = Device[Device.DEFAULT]
    assert device.arch == 'gfx1200' and device.is_usb and isinstance(device.renderer, AMDLLVMRenderer)
    load('wgp_compiler').install(device)
    renderer = device.renderer
    report['gpu_executed'] = True
  controls = [('control-scalar', (17, 48, 32), False, False), ('control-row-bias', (49, 80, 64), True, True)]
  actual = [(name, CASES[name], name.startswith('vision'), name.startswith('vision')) for name in selected]
  rejected = {}
  try:
    for name, shape, bias, row_scale in controls+actual:
      integer = name.startswith('control')
      tiles = TILES if integer or args.mode == 'compile' else SHORTLIST[name]
      fns = functions(renderer, row_scale, tiles)
      report['active_case'] = name
      E.write_json(args.output, report)
      resources = {}
      for label, fn in fns.items():
        print(json.dumps({'case': name, 'variant': label, 'phase': 'compile'}), flush=True)
        tile = (4, 4) if label in ('original', 'current') else tuple(map(int, label.split('x')))
        artifact = args.output.parent/f'agent-matrix-tiles-{name}-{label}' if args.mode == 'compile' else None
        resources[label] = compile_record(shape, bias, row_scale, fn, renderer, tile, artifact)
        if label in rejected: resources[label].update(status='control-rejected', failure=rejected[label])
      if args.mode == 'compile': row = {'case': name, 'shape_mnk': shape, 'bias': bias, 'row_scales': row_scale, 'variants': resources}
      else:
        row = run_case(name, shape, bias, row_scale, fns, resources, args.rounds, device, integer)
        if integer:
          for label, info in row['variants'].items():
            if info['status'] != 'compiled': rejected[label] = {'case': name, 'status': info['status'], 'error': info.get('error')}
      report['cases'].append(row)
      E.write_json(args.output, report)
    if args.mode == 'compile': assert not Device._opened_devices
    report.pop('active_case', None)
    report.update(complete=True, independent_control_rejections=rejected,
      summary={'cases': len(report['cases']), 'gpu_executed': report['gpu_executed'], 'pipeline_validated': False,
               'rejected_variants': sum(v['status'] != 'compiled' for row in report['cases'] for v in row['variants'].values())})
    E.write_json(args.output, report)
  except BaseException as exc:
    report.update(complete=False, failure=f'{type(exc).__name__}: {exc}')
    E.write_json(args.output, report)
    raise


if __name__ == '__main__': main()

"""CPU-only checks for transfer fixtures, case selectors, and retained GPU evidence."""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def rejected(fn):
  try: fn()
  except AssertionError: return
  raise AssertionError('Deliberately incomplete evidence passed')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--existing-report', type=Path, default=WORK/'agent-vision-tiles-kernels/report.json')
  parser.add_argument('--cache', type=Path, default=Path('/dev/shm/alpamayo-9060-fp8'))
  parser.add_argument('--calibration', type=Path, default=WORK/'vision-scale-calibration/report.vision-scales.json')
  args = parser.parse_args()
  os.environ.update(DEV='CPU', PROFILE='0', PYTHONDONTWRITEBYTECODE='1')
  from tinygrad import Device, dtypes
  H, C, T = (load(name) for name in ('agent-vision-tiles-check', 'agent-vision-tiles-transfer-check-coverage', 'agent-vision-tiles-transfer-check'))
  existing = json.loads(args.existing_report.read_text())
  old_checker = next(Path(path) for path in existing['sources'] if Path(path).name == 'agent-vision-tiles-check.py')
  manifest = C.source_manifest(HERE/'agent-vision-tiles-check.py')
  assert manifest == C.source_manifest(old_checker)
  assert len(C.selected_cases()) == 12 and len(C.ALL_CASES) == 16
  for quantization in ('dynamic', 'fixed'):
    selected = C.selected_cases('complete-linear', '2', 'fc2', quantization)
    assert selected == [f'complete-linear/batch2/fc2/{quantization}']
    empty = C.coverage({'gpu_executed': False}, selected)
    assert not empty['selected_gpu_cases_complete'] and not empty['full_kernel_suite_complete']
  assert C.selected_cases('matrix', '1', 'fc1') == ['matrix-control/batch1/fc1', 'matrix-actual/batch1/fc1']
  audited = C.audit([args.existing_report])
  assert len(audited['completed_cases']) == 14
  assert audited['missing_cases'] == ['complete-linear/batch2/fc2/dynamic', 'complete-linear/batch2/fc2/fixed']
  assert not audited['eligible_for_exact_route_pairs'] and not audited['route_qualified']
  # Removing a sample or a mutation proof must invalidate previously completed evidence.
  for field in ('sample', 'isolation', 'scale', 'queue'):
    bad = copy.deepcopy(existing)
    if field == 'sample': bad['full_linear'][0]['samples']['candidate'].pop()
    if field == 'isolation':
      next(row for row in bad['full_linear'] if row['images'] == 2)['checks'][1]['unchanged_other_image_exact'] = False
    if field == 'scale': bad['full_linear'][0]['checks'][2]['quantized_bytes_and_scale_bits_exact'] = False
    if field == 'queue':
      next(row for row in bad['matrix'] if row['timing'])['timing']['queue_proofs']['candidate'][0]['ordered_execution_barriers'] = 38
    rejected(lambda: C.completed_cases(bad))
  # Inspect the actual measurement functions: input copies must stay outside the timed scopes.
  tree = ast.parse((HERE/'agent-vision-tiles-check.py').read_text())
  definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
  matrix = definitions['matrix_check']
  measure = next(node for node in ast.walk(matrix) if isinstance(node, ast.FunctionDef) and node.name == 'measure')
  full = definitions['full_check']
  timed = next(node for node in full.body if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
    and ast.dump(node.iter) == ast.dump(ast.parse('range(CALLS)', mode='eval').body))
  for scope in (measure, timed):
    assert not any(isinstance(node, ast.Name) and node.id in ('transfers', 'values') for node in ast.walk(scope))
  # Exercise all five real input kinds and every original state on CPU allocations.
  # This checks transfer bytes and identity only; it does not substitute for GPU arithmetic.
  cases = []
  for images in (1, 2):
    for fc, shape in H.load('agent-vision-tiles').SHAPES.items():
      values, _ = H.full_values(images, shape, args.cache, fc, args.calibration, 0)
      tx = T.InputTransfers(Device['CPU'])
      tensors = tx.upload(values, (dtypes.bfloat16, dtypes.fp8e4m3, dtypes.float32, dtypes.float32, dtypes.bfloat16))
      buffers = tuple(t.uop.buffer for t in tensors)
      bytes_before = [v.tobytes() for v in values]
      expected_writes = [[False]*5, [True, False, False, False, False], [False, False, False, False, True],
        [True, True, True, True, False], [True]*5]
      all_write_bytes = sum(v.nbytes for v in values)*6  # Initial upload plus 0,1,2,3,0.
      for state, expected in zip((0, 1, 2, 3, 0), expected_writes, strict=True):
        values, _ = H.full_values(images, shape, args.cache, fc, args.calibration, state)
        event = tx.replace(tensors, values)
        assert event['writes'] == expected
        assert tuple(t.uop.buffer for t in tensors) == buffers
        actual_bytes = []
        for buffer, value in zip(buffers, values, strict=True):
          actual = bytes(buffer.host.view(size=buffer.nbytes, fmt='B')[:])
          assert actual == value.tobytes()
          actual_bytes.append(actual)
        if images == 2 and state == 1:
          assert actual_bytes[0][:values[0][0].nbytes] == bytes_before[0][:values[0][0].nbytes]
        bytes_before = actual_bytes
      cases.append({'images': images, 'fc': fc, 'states': [0, 1, 2, 3, 0], 'all_five_inputs_exact': True,
        'buffer_identities_unchanged': True, 'unconditional_upload_bytes': all_write_bytes, 'transfers': tx.report()})
  assert not any(name.startswith('AMD') for name in Device._opened_devices)
  result = {'gpu_executed': False, 'gpu_transfer_speed_qualified': False, 'selectors_checked': True,
    'partial_coverage_not_qualified': True, 'missing_sample_isolation_scale_queue_rejected': True,
    'timed_spans_exclude_input_uploads': True, 'numeric_sources_and_exact_gates_unchanged': True,
    'numeric_kernel_sources': manifest, 'retained_gpu_evidence': audited, 'actual_fixture_transfer_checks': cases}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps({'cpu_checks_passed': True, 'gpu_executed': False, 'fixture_cases': len(cases), 'output': str(args.output)}))


if __name__ == '__main__': main()

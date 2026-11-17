"""Read-only coverage audit across frozen vision-tile reports. No GPU access."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

SHAPES = {'fc1': (720, 4304, 1152), 'fc2': (720, 1152, 4304)}
TILES = {'fc1': (3, 3), 'fc2': (3, 6)}
NUMERIC_FILES = ('agent-vision-tiles.py', 'agent-vision-tiles-compile.py', 'image-gemm.py', 'tiled-gemm.py', 'batched-vision.py',
  'quantize-reduction.py', 'agent-spec-head-check.py', 'agent-expert-gemm.py', 'wgp_compiler.py', 'prefill-kernels.py')
NUMERIC_DEFS = ('fixture', 'MatrixBatch', 'independent_prepared', 'integer_reference', 'cached_weights',
  'full_values', 'full_operation', 'independent_full')


def sha(data): return hashlib.sha256(data).hexdigest()


def case_id(stage, batch, projection, quantization=None):
  return f'{stage}/batch{batch}/{projection}'+('/'+quantization if quantization else '')


def selected_cases(stage='all', batch='all', projection='all', quantization='dynamic'):
  batches = (1, 2) if batch == 'all' else (int(batch),)
  projections = ('fc1', 'fc2') if projection == 'all' else (projection,)
  cases = []
  if stage in ('all', 'matrix'):
    for images in batches:
      for kind in ('matrix-control', 'matrix-actual'):
        for fc in projections: cases.append(case_id(kind, images, fc))
  if stage in ('all', 'complete-linear'):
    for images in batches:
      for fc in projections:
        for mode in (('dynamic', 'fixed') if quantization == 'both' else (quantization,)):
          cases.append(case_id('complete-linear', images, fc, mode))
  return cases


ALL_CASES = selected_cases(quantization='both')


def source_manifest(checker):
  checker = Path(checker)
  tree = ast.parse(checker.read_text())
  definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
  constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'CALLS'}
  assert constants == {'CALLS': 40}
  result = {'schema': 1, 'calls': 40, 'source_kind': 'numeric source SHA256 and unchanged checker arithmetic AST',
    'files': {name: sha((checker.parent/name).read_bytes()) for name in NUMERIC_FILES},
    'checker_numeric_definitions': {name: sha(ast.dump(definitions[name], include_attributes=False).encode()) for name in NUMERIC_DEFS}}
  result['files']['amd_gemm.py'] = sha((checker.parent.parent/'amd_gemm.py').read_bytes())
  # Transfer calls may change. Preserve every arithmetic assertion and direct
  # exact-output check in the two GPU gates, as well as their original order.
  result['checker_exact_gates'] = {}
  for name in ('matrix_check', 'full_check'):
    checks = [node for node in ast.walk(definitions[name]) if isinstance(node, ast.Assert) or
      (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('assert_array_equal', 'exact_output'))]
    result['checker_exact_gates'][name] = sha('\n'.join(ast.dump(node, include_attributes=False) for node in checks).encode())
  return result


def _positive(value): return isinstance(value, (float, int)) and math.isfinite(value) and value > 0


def completed_cases(report):
  completed = []
  for row in report.get('matrix', []):
    batch, shape, tile = row['images'], tuple(row['shape_mnk']), tuple(row['tile'])
    assert batch in (1, 2)
    fc = next(name for name, value in TILES.items() if value == tile)
    control = shape == (65, 80, 64)
    assert control or shape == SHAPES[fc]
    checks = row['checks']
    assert [s['state'] for s in checks] == list(range(5))
    assert all(s['independent_original_images_exact'] and s['independent_integer'] == control for s in checks)
    if batch == 2: assert all(checks[state]['unchanged_other_image_exact'] for state in (1, 2))
    timing = row['timing']
    if control: assert timing is None
    else:
      assert timing['calls_per_batch'] == 40 and timing['profiling'] is False and timing['completion_fences_per_batch'] == 1
      assert 3 <= len(timing['paired_amortized_ms']) <= 5
      for pair in timing['paired_amortized_ms']: assert set(pair) == {'current', 'candidate'} and all(_positive(v) for v in pair.values())
      for variant in ('current', 'candidate'):
        proofs = timing['queue_proofs'][variant]
        assert len(proofs) == 1
        assert proofs[0]['captured_calls'] == 40 and proofs[0]['ordered_execution_barriers'] == 39
        assert proofs[0]['output_input_slot'] == 5 and proofs[0]['actual_queue_builder_observed'] is True
    completed.append(case_id('matrix-control' if control else 'matrix-actual', batch, fc))
  for row in report.get('full_linear', []):
    batch, fc, mode = row['images'], row['fc'], row['quantization']
    assert batch in (1, 2) and fc in SHAPES and mode in ('dynamic', 'fixed') and tuple(row['shape_mnk']) == SHAPES[fc]
    checks = row['checks']
    assert [s['state'] for s in checks] == list(range(4))
    assert all(s['independent_original_images_exact'] and s['quantized_bytes_and_scale_bits_exact'] for s in checks)
    if batch == 2: assert checks[1]['unchanged_other_image_exact']
    for variant in ('current', 'candidate'):
      samples = row['samples'][variant]
      assert len(samples) == row['timing'][variant]['calls'] == 40
      tile = TILES[fc]
      expected = f'agent_vision_tiles_{tile[0]}x{tile[1]}_b{batch}' if variant == 'candidate' else 'alpamayo_fp8_gemm_4_4'
      for sample in samples:
        assert _positive(sample['gpu_span_ms']) and _positive(sample['sum_kernel_ms'])
        assert sum(event['name'] == expected for event in sample['events']) == 1
    completed.append(case_id('complete-linear', batch, fc, mode))
  assert len(completed) == len(set(completed)), 'Duplicate case records in one report'
  return completed


def coverage(report, selected):
  completed = completed_cases(report)
  assert set(completed) <= set(selected)
  return {'required_full_suite': ALL_CASES, 'selected': selected, 'completed': completed,
    'selected_pending': [case for case in selected if case not in completed],
    'not_selected': [case for case in ALL_CASES if case not in selected],
    'full_suite_missing': [case for case in ALL_CASES if case not in completed],
    'selected_gpu_cases_complete': bool(report.get('gpu_executed')) and set(selected) == set(completed),
    'full_kernel_suite_complete': bool(report.get('gpu_executed')) and set(completed) == set(ALL_CASES), 'route_qualified': False}


def audit(paths):
  evidence, completed, manifest, provenance, weights = [], set(), None, None, {}
  for path in paths:
    path = Path(path)
    report = json.loads(path.read_text())
    assert report['gpu_executed'] is True, f'{path}: CPU results cannot satisfy GPU coverage'
    policy = report['environment']
    assert policy['AM_POWER_LIMIT'] == '144' and policy['AM_PERF_LEVEL'] == 'auto'
    assert report['device']['arch'] == 'gfx1200' and report['device']['renderer'] == 'AMDLLVMRenderer'
    current_provenance = {'environment': policy, 'device': report['device'], 'calibration_sha256': report['calibration']['sha256']}
    if provenance is None: provenance = current_provenance
    else: assert provenance == current_provenance, f'{path}: device, policy, or calibration differs'
    for row in report['full_linear']:
      if row['fc'] in weights: assert weights[row['fc']] == row['weights'], f'{path}: cached weight fixture differs'
      else: weights[row['fc']] = row['weights']
    sources = {Path(name).name: (Path(name), value) for name, value in report['sources'].items()}
    for source, expected in sources.values(): assert sha(source.read_bytes()) == expected, f'{path}: source changed: {source}'
    current = source_manifest(sources['agent-vision-tiles-check.py'][0])
    if 'numeric_kernel_sources' in report: assert report['numeric_kernel_sources'] == current
    if manifest is None: manifest = current
    else: assert manifest == current, f'{path}: numeric sources or exact gates differ'
    failure = report.get('failure')
    assert failure is None or failure['type'] in ('KeyboardInterrupt', 'TimeoutError'), f'{path}: unresolved failure: {failure}'
    cases = completed_cases(report)
    completed.update(cases)
    evidence.append({'report': str(path), 'sha256': sha(path.read_bytes()), 'completed_cases': cases,
      'interrupted': failure is not None, 'active_case_not_counted': report.get('active_case')})
  missing = [case for case in ALL_CASES if case not in completed]
  return {'numeric_sources_identical': True, 'numeric_kernel_sources': manifest, 'provenance': provenance, 'weights': weights, 'reports': evidence,
    'required_cases': ALL_CASES, 'completed_cases': [case for case in ALL_CASES if case in completed],
    'missing_cases': missing, 'full_kernel_suite_complete': not missing,
    'eligible_for_exact_route_pairs': not missing, 'route_qualified': False}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--reports', type=Path, nargs='+', required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  result = audit(args.reports)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps({'completed_cases': len(result['completed_cases']), 'missing_cases': result['missing_cases'],
    'eligible_for_exact_route_pairs': result['eligible_for_exact_route_pairs']}))


if __name__ == '__main__': main()

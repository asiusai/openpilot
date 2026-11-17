"""Read saved tile trials. No tinygrad import or device access."""
import argparse
import hashlib
import json
from pathlib import Path

SHAPES = ((2112, 9216, 2048), (2112, 2048, 9216), (720, 4304, 1152), (720, 1152, 4304))
REPORTS = ('alpamayo-gemm-tiles', 'alpamayo-gemm-groups', 'alpamayo-gemm-rectangular-unroll',
  'alpamayo-gemm-tile-order', 'alpamayo-gemm-tile-order-fixed', 'alpamayo-vector-gemm-check',
  'alpamayo-vector-gemm-fragment-order', 'alpamayo-shared-vector-gemm-check')


def audit():
  result = {'actual_shapes_mnk': SHAPES, 'reports': [], 'current_shape_records': [],
    'limitation': 'Audit covers these saved reports, not every historical command. Planned variants are not completed tests.'}
  for name in REPORTS:
    path = Path('/tmp/opencode')/(name+'.json')
    raw = path.read_bytes()
    records = json.loads(raw)
    assert isinstance(records, list)
    shapes = sorted({tuple(row['shape']) for row in records})
    tiles = sorted({tuple(row['tile']) for row in records if row.get('tile') is not None})
    result['reports'].append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'records': len(records),
      'shapes': shapes, 'explicit_tiles': tiles, 'errors': sum('error' in row for row in records),
      'current_shapes': [shape for shape in shapes if shape in SHAPES]})
    result['current_shape_records'] += [dict(report=name, **row) for row in records if tuple(row['shape']) in SHAPES]
    if name == 'alpamayo-gemm-rectangular-unroll': result['completed_rectangular_records'] = records
  sources = ('check-gemm-groups.py', 'check-gemm-tiles.py')
  result['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}
  result['timing_source_evidence'] = {
    'check-gemm-groups.py': 'Lines 50-52: one GEMM per call, synchronize every call, discard first two of twelve.',
    'check-gemm-tiles.py': 'Lines 28-32: one GEMM per call, synchronize every call, discard first two of seven.',
    'scales': 'Both scripts use scalar FP32 .125 activation and weight scales, BF16 output, no bias; no row-scale mutation test.'}
  assert not result['current_shape_records']
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  args.output.write_text(json.dumps(audit(), indent=2)+'\n')
  print(json.dumps({'audited_reports': len(REPORTS), 'matching_current_shape_records': 0, 'gpu_executed': False}))

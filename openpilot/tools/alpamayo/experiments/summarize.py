"""Summarize Harald benchmark files against the retained BF16 predictions."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
parser.add_argument('--expected', type=int, default=6)
args = parser.parse_args()
root = Path(__file__).resolve().parent
rows = []
for path in sorted(args.results.glob('*.json')):
  if path.name == 'preflight.json' or path.name == 'summary.json':
    continue
  data = json.loads(path.read_text())
  runs = data['runs']
  baseline = np.array(json.loads((root / 'references' / path.name).read_text())['xyz'])
  xyz = np.array([r['xyz'] for r in runs])
  assert xyz.shape[1:] == (1, 64, 3) and np.isfinite(xyz).all(), path
  times = np.array([r['timings_ms']['total'] for r in runs])
  reference = json.loads((root / 'references' / f'motion-{path.stem.split("-")[0]}.json').read_text())
  motion = np.array(reference['xyz'])
  error = np.linalg.norm(xyz[-1, 0, :, :2] - motion[:, :2], axis=-1)
  rows.append({'sample': path.stem, 'median_ms': float(np.median(times)),
               'measured_runs': len(runs), 'warmup_runs': len(data.get('warmup', [])),
               'steady_timing': len(data.get('warmup', [])) >= 2,
               'p95_ms': float(np.percentile(times, 95)), 'missed_250ms': int((times > 250).sum()),
                'max_repeat_position_delta_m': float(np.abs(xyz - xyz[0]).max()) if len(runs) > 1 else None,
               'max_bf16_position_delta_m': float(np.linalg.norm(xyz - baseline, axis=-1).max()),
               'approx_motion_ade_m': float(error.mean()),
               'approx_motion_error_1_3_6p4s_m': error[[9, 29, 63]].tolist(),
               'motion_time_source': reference['motion_time_source'],
               'arch': data['arch'], 'model_load_ms': data['model_load_ms'],
               'median_stage_ms': {k: float(np.median([r['timings_ms'][k] for r in runs])) for k in runs[0]['timings_ms']},
               'final_displacement_m': float(np.linalg.norm(xyz[-1, 0, -1, :2])),
               'reasoning': runs[-1]['reasoning']})
assert len(rows) == args.expected, f'Expected {args.expected} result files; found {len(rows)}'
(args.results / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))

"""Compare replay trajectories with recorded motion and an optional matched run."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('result', type=Path)
parser.add_argument('--baseline', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
run = json.loads(args.result.read_text())
baseline = {}
if args.baseline:
  previous = json.loads(args.baseline.read_text())
  baseline = {Path(d['sample']).name: d for d in previous['warmup']+previous['runs']}
entries = []
for d in run['warmup']+run['runs']:
  path = Path(d['sample'])
  seconds = path.name.removesuffix('.npz').removeprefix('sample-')
  expected = np.asarray(json.loads((path.parent/f'reference-{seconds}.json').read_text())['xyz'])
  actual = np.asarray(d['prediction']['xyz'])[0]
  assert actual.shape == expected.shape == (64, 3) and np.isfinite(actual).all()
  errors = np.linalg.norm(actual[:, :2]-expected[:, :2], axis=-1)
  entry = {'sample': str(path), 'ade_m': float(errors.mean()), 'fde_m': float(errors[-1]),
           'error_at_1_3_6_4s_m': errors[[9,29,63]].tolist(),
           'predicted_displacement_at_1_3_6_4s_m': np.linalg.norm(actual[[9,29,63], :2], axis=-1).tolist(),
           'recorded_displacement_at_1_3_6_4s_m': np.linalg.norm(expected[[9,29,63], :2], axis=-1).tolist(),
           'reasoning': d['prediction']['reasoning']}
  if path.name in baseline:
    other = baseline[path.name]['prediction']
    difference = np.linalg.norm(actual[:, :2]-np.asarray(other['xyz'])[0, :, :2], axis=-1)
    entry.update(baseline_mean_delta_m=float(difference.mean()), baseline_max_delta_m=float(difference.max()),
                 baseline_delta_at_1_3_6_4s_m=difference[[9,29,63]].tolist(),
                 same_reasoning=d['prediction']['reasoning'] == other['reasoning'])
  entries.append(entry)
report = {'count': len(entries), 'mean_ade_m': float(np.mean([d['ade_m'] for d in entries])),
          'unique_samples': len({d['sample'] for d in entries}),
          'mean_fde_m': float(np.mean([d['fde_m'] for d in entries])),
          'limitation': 'Overlapping observations from one route; integrated recorded motion is not a unique correct plan.',
          'entries': entries}
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'entries'}))

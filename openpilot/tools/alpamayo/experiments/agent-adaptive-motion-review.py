"""Compare completed adaptive-prefix predictions with the existing recorded-motion reference."""
import hashlib
import json
from pathlib import Path

import numpy as np

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
PINS = {
  'agent-adaptive-prefill-quality-reference/report.json': 'fbda536c6c9a541f2f27d0dc790d5628be3930b8c61356b6d0c02ba83a91f0b5',
  'agent-adaptive-coverage-quality-candidate/report.json': 'd94949b9261a354b06a71f1e4a074d10f375f3edabf64ad18185a786519395c6',
  'agent-ck-quality64-work/prepared/inputs.json': '55bd2fedd2ec7f48aa4526f2207e86431891fa198b435cc1e811e5825a5c6c93',
}


def load(name):
  data = (WORK / name).read_bytes()
  assert hashlib.sha256(data).hexdigest() == PINS[name], name
  return json.loads(data)


def motion(points):
  velocity = np.diff(np.vstack((np.zeros((1, 3)), points)), axis=0) / .1
  speed = np.linalg.norm(velocity[:, :2], axis=1)
  heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
  return {'speed_mps': speed.tolist(), 'acceleration_mps2': (np.diff(speed) / .1).tolist(),
          'heading_change_rad': (heading - heading[0]).tolist()}


def metrics(rows):
  values = {key: float(np.mean([r[key] for r in rows]))
            for key in ('baseline_ADE', 'candidate_ADE', 'baseline_FDE', 'candidate_FDE')}
  return {**values, 'count': len(rows),
          'within_five_percent': all(values['candidate_' + k] <= 1.05 * values['baseline_' + k] for k in ('ADE', 'FDE'))}


def main():
  baseline = load('agent-adaptive-prefill-quality-reference/report.json')
  candidate = load('agent-adaptive-coverage-quality-candidate/report.json')
  inputs = load('agent-ck-quality64-work/prepared/inputs.json')
  a, b = [r['warmup'] + r['runs'] for r in (baseline, candidate)]
  chosen = inputs[14:32]
  assert len(a) == len(b) == len(chosen) == 18
  rows = []
  for ar, br, item in zip(a, b, chosen, strict=True):
    assert ar['sample'] == br['sample'] == item['sample']
    for name, digest in item['files'].items():
      assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    points = {name: np.asarray(value, np.float64).reshape(64, 3) for name, value in (
      ('baseline', ar['prediction']['xyz']), ('candidate', br['prediction']['xyz']),
      ('recorded', item['recorded_motion_future_xyz']))}
    assert all(np.isfinite(value).all() for value in points.values())
    row = {'original64_index': item['original64_index'], 'sample': item['sample'], 'query_ns': item['query_ns'],
           'tokens_equal': ar['prediction']['generated_token_ids'] == br['prediction']['generated_token_ids'],
           'baseline_text': ar['prediction']['reasoning'], 'candidate_text': br['prediction']['reasoning']}
    for name, value in points.items():
      row[name + '_motion'] = motion(value)
      if name != 'recorded':
        error = np.linalg.norm((value - points['recorded'])[:, :2], axis=1)
        row[name + '_ADE'], row[name + '_FDE'] = float(error.mean()), float(error[-1])
    for field in ('speed_mps', 'acceleration_mps2', 'heading_change_rad'):
      row['maximum_change_' + field] = float(np.max(np.abs(np.asarray(row['candidate_motion'][field]) - row['baseline_motion'][field])))
    rows.append(row)
  aggregate = metrics(rows)
  windows = {f'{lo}-{hi}': metrics([r for r in rows if lo <= r['original64_index'] <= hi])
             for lo, hi in ((14, 15), (16, 31))}
  report = {'complete': True, 'gpu_executed': False, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_pins': PINS, 'aggregate': aggregate, 'covered_windows': windows, 'rows': rows,
            'subset_screen_passed': aggregate['within_five_percent'] and all(r['within_five_percent'] for r in windows.values()),
            'full64_qualified': False, 'manual_review_complete': False, 'throughput_qualified': False, 'adoption_allowed': False,
            'limit': 'Eighteen overlapping route samples. Recorded motion is an estimate. Only original indices14..31 are covered.'}
  output = WORK / 'agent-adaptive-coverage-quality-candidate/parent-motion-review.json'
  with output.open('x') as handle:
    json.dump(report, handle, indent=2, allow_nan=False)
    handle.write('\n')
  print(json.dumps({k: report[k] for k in ('aggregate', 'covered_windows', 'subset_screen_passed')}))
  for row in rows:
    short = {k: v for k, v in row.items() if k not in ('sample', 'query_ns') and not k.endswith('_motion')}
    short['final_speed_mps'] = {n: row[n + '_motion']['speed_mps'][-1] for n in points}
    short['final_heading_change_rad'] = {n: row[n + '_motion']['heading_change_rad'][-1] for n in points}
    print(json.dumps(short))


if __name__ == '__main__':
  main()

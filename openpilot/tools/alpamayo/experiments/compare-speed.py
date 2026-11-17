"""Compare optimization trials against saved trajectories and recorded motion."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
  return json.loads(path.read_text())


def entries(path):
  if path.suffix == '.jsonl':
    return [json.loads(line) for line in path.read_text().splitlines()]
  report = read(path)
  return report['warmup']+report['runs']


def evaluate(args):
  manifest = read(args.manifest)
  by_sample = {r['sample']: r for r in manifest['schedule']}
  by_name = {Path(r['sample']).name: r for r in manifest['schedule']}
  reference = {Path(e['sample']).name: e for e in entries(args.reference)}
  candidate = {Path(e['sample']).name: e for e in entries(args.report)}
  rows = []
  for name, entry in candidate.items():
    row = by_sample.get(entry['sample'], by_name.get(name))
    if row is None or name not in reference:
      continue
    pred, ref = entry['prediction'], reference[name]['prediction']
    a, b, actual = (np.asarray(x, dtype=np.float64).reshape(64, 3)
                    for x in (pred['xyz'], ref['xyz'], row['recorded_motion_future_xyz']))
    if not all(np.isfinite(v).all() for v in (a,b,actual)):
      raise ValueError('Nonfinite compared trajectory')
    error_a, error_b = (np.linalg.norm((v-actual)[:, :2], axis=-1) for v in (a,b))
    delta = np.linalg.norm(a-b, axis=-1)
    rows.append(dict(query_ns=row['query_ns'], time_s=row['display_time_s'],
                     reference_ade_m=float(error_b.mean()), candidate_ade_m=float(error_a.mean()),
                     reference_fde_m=float(error_b[-1]), candidate_fde_m=float(error_a[-1]),
                     mean_trajectory_change_m=float(delta.mean()), max_trajectory_change_m=float(delta.max()),
                     mean_coordinate_change_m=float(np.abs(a-b).mean()), max_coordinate_change_m=float(np.abs(a-b).max()),
                     tokens_equal=pred['generated_token_ids']==ref['generated_token_ids']))
  if not rows:
    raise ValueError('No matched timestamps')
  def summary(selected):
    if not selected:
      return None
    metrics = {key:float(np.mean([r[key] for r in selected])) for key in (
      'reference_ade_m','candidate_ade_m','reference_fde_m','candidate_fde_m','mean_trajectory_change_m')}
    return dict(samples=len(selected), **metrics,
                max_trajectory_change_m=max(r['max_trajectory_change_m'] for r in selected),
                identical_tokens=sum(r['tokens_equal'] for r in selected),
                arithmetic_parity=all(r['tokens_equal'] and r['mean_coordinate_change_m']<=.025 and
                                      r['max_coordinate_change_m']<=.15 for r in selected),
                within_five_percent_trajectory_error=(metrics['candidate_ade_m']<=metrics['reference_ade_m']*1.05 and
                                                     metrics['candidate_fde_m']<=metrics['reference_fde_m']*1.05))
  result = dict(reference=str(args.reference), report=str(args.report), manifest=str(args.manifest),
                reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                total_candidate_samples=len(candidate), matched=summary(rows), rows=rows,
                windows={f'{a:g}-{b:g}':summary([r for r in rows if a<=r['time_s']<b]) for a,b in args.window},
                limit='Paired finite-sample comparison against integrated recorded motion; not independent driving validation')
  args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
  print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__ == '__main__':
  parser=argparse.ArgumentParser(description=__doc__)
  for name in ('reference','report','manifest','output'):
    parser.add_argument('--'+name,type=Path,required=True)
  parser.add_argument('--window',type=float,nargs=2,action='append',default=[])
  evaluate(parser.parse_args())

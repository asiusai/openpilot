"""Measure camera-budget changes against saved official full-budget outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--candidates', type=Path, nargs='+', required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  baseline = json.loads((args.baseline/'report.json').read_text())
  baseline_rows = {row['sample_id']:row for row in baseline['samples']}
  results = []
  for candidate in args.candidates:
    report = json.loads((candidate/'report.json').read_text())
    for key in ('checkpoint','dtype','settings','noise_sha256','upstream_commit'):
      assert report[key] == baseline[key], f'Configuration differs: {key}'
    for row in report['samples']:
      name = row['sample_id']
      for key in ('input_sha256','nav_text'):
        assert row[key] == baseline_rows[name][key], f'Inputs differ: {name}/{key}'
      expected = load_file(args.baseline/name/'output.safetensors')
      actual = load_file(candidate/name/'output.safetensors')
      a, b = actual['pred_xyz'].reshape(64,3), expected['pred_xyz'].reshape(64,3)
      assert np.isfinite(a).all() and np.isfinite(b).all()
      delta = np.linalg.norm(a-b,axis=-1)
      ids = lambda data:data['generated_ids'].reshape(-1)[data['generated_ids'].reshape(-1)>=0]
      result = dict(candidate=str(candidate), sample=name, pixel_scale=report['pixel_scale'],
                    camera_pixel_scales=report.get('camera_pixel_scales'),
                    tokens_identical=np.array_equal(ids(actual),ids(expected)),
                    text=json.loads((candidate/name/'output.json').read_text())['text'],
                    baseline_text=json.loads((args.baseline/name/'output.json').read_text())['text'],
                    mean_position_delta_m=float(delta.mean()), max_position_delta_m=float(delta.max()),
                    position_delta_1_3_6_4s_m=delta[[9,29,63]].tolist(),
                    endpoint=a[-1].tolist(), baseline_endpoint=b[-1].tolist())
      results.append(result)
      print(json.dumps(result),flush=True)
  args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
  main()

"""Report measured offline inference throughput separately from replay cadence."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from camera_profiles import replay_module


def summarize(output, work):
  replay = replay_module()
  manifest = replay.read_json(output/'manifest.json')
  commands = [replay.read_json(p) for p in sorted(work.glob('command-timing-*.json'))]
  if not commands:
    raise ValueError('No command wall-time measurements')
  reports = []
  for path in sorted(work.glob('views-*.json')):
    value = replay.read_json(path)
    if 'runs' in value:
      reports.append((path.name, value))
  new_count = sum(c['new_predictions'] for c in commands)
  wall = sum(c['wall_seconds'] for c in commands)
  elapsed = (max(datetime.fromisoformat(c['finished_utc']) for c in commands)
             - min(datetime.fromisoformat(c['started_utc']) for c in commands)).total_seconds()
  measured = sum(r['summary']['measured_predictions'] for _, r in reports)
  measured_wall = sum(r['summary']['measured_wall_seconds'] for _, r in reports)
  predictions = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
  if len(predictions) != manifest['valid_samples']:
    raise ValueError('Incomplete route predictions')
  def distribution(values):
    return dict(zip(('median', 'p90', 'max'), map(float, np.percentile(values, [50, 90, 100])))) if values else None
  stages = {}
  for entry in predictions:
    for name, value in entry['prediction']['timings_ms'].items():
      stages.setdefault(name, []).append(value)
  result = {
    'requested_sampling_hz': manifest['sample_hz'], 'recording_seconds': manifest['duration_s'],
    'saved_predictions': len(predictions), 'newly_measured_predictions': new_count,
     'inference_wall_seconds': wall, 'overall_predictions_per_second': new_count/wall,
     'elapsed_since_first_attempt_seconds': elapsed,
     'between_attempts_seconds': max(0., elapsed-wall),
    'wall_time_over_recording_duration': wall/manifest['duration_s'],
    'completed_gpu_processes': len(reports), 'command_attempts': commands,
     'after_warmup': {'predictions': measured, 'wall_seconds': measured_wall,
                      'predictions_per_second': measured/measured_wall if measured_wall else None,
                      'boundary': 'Completed processes after configured initial warmup; includes any later graph compilation and CPU processing'},
    'model_stage_ms': {k: distribution(v) for k, v in stages.items()},
    'service_ms_after_warmup': distribution([e['service_ms'] for _, r in reports for e in r['runs'] if 'service_ms' in e]),
     'gpu_processes': [{'report': name, 'model_load_ms': r.get('model_load_ms'),
                       'vision_group_limit': r.get('vision_group_limit'),
                       'minimum_cache_capacity': r.get('minimum_cache_capacity', 0),
                       'gc_interval': r.get('gc_interval', 1),
                       'gc_ms_after_warmup': distribution([e['gc_ms'] for e in r['runs'] if 'gc_ms' in e]),
                       'validation': r.get('validation', []),
                      'startup_through_first_prediction_s': r.get('startup_through_first_prediction_s'),
                      'process_elapsed_s': r.get('process_elapsed_s'), **r['summary']} for name, r in reports],
    'boundary': 'Offline inference command, including CPU processing, loading, compilation, and process restarts; image preparation and rendering excluded',
    'live_limit': 'Inputs were reordered offline. A persistent process can change camera layouts, but this does not measure chronological live streaming or camera-to-prediction age.',
    'latency_note': 'Model stages and pipelined CPU work overlap. Throughput uses measured wall time, not reciprocal median latency.',
  }
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--work', type=Path, required=True)
  parser.add_argument('--recording-uuid')
  args = parser.parse_args()
  replay = replay_module()
  replay.check_mount(args.output, args.recording_uuid)
  result = summarize(args.output, args.work)
  replay.write_json(args.output/'timing-summary.json', result)
  print(json.dumps({k: v for k, v in result.items() if k not in ('gpu_processes', 'command_attempts', 'model_stage_ms')}, indent=2))


if __name__ == '__main__':
  main()

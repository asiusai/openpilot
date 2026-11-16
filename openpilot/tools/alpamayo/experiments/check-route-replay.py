"""Check restart recovery, exact sample coverage, and command timing without a GPU."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from camera_profiles import replay_module


def main():
  replay = replay_module()
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    output, work = root/'output', root/'work'
    output.mkdir()
    work.mkdir()
    rows = [{'tick': i, 'query_ns': i*10**9, 'available_views': [1, 6] if i%2 else [0, 1, 2, 3, 5, 6],
             'sample': str(root/f'sample-{i}')} for i in range(7)]
    replay.write_json(output/'manifest.json', {'schedule': rows, 'navigation': 'Follow the road.'})
    noise = np.zeros((1, 64, 2), dtype=np.float32)
    np.save(root/'noise.npy', noise)
    config = {'tinygrad_revision': 'test-revision', 'initial_noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest(),
              'vocabulary_sha256': hashlib.sha256((Path(__file__).parent/'coc_vocab.json').read_bytes()).hexdigest(),
              'environment': {}, 'arguments': ['--warmup', '0', '--runs', '1']}
    args = SimpleNamespace(output=output, work=work, tinygrad=root, checkpoint=root, noise=root/'noise.npy',
                           fp8_cache=root, cache=root/'cache.db', group_prompts=False, batch_size=2,
                           collect_garbage=True, limit=0, recording_uuid=None)
    original_read = replay.read_json
    calls = []
    interrupt = True

    def read(path):
      return config if Path(path).name == 'best-config.json' else original_read(path)

    def run(command, **kwargs):
      nonlocal interrupt
      paths = command[command.index('--samples')+1:command.index('--initial-noise')]
      calls.append(paths)
      report = Path(command[command.index('--output')+1])
      entries = [{'sample': p, 'prediction': {'reasoning': 'Fixture', 'xyz': np.full((1, 64, 3), int(Path(p).name.split('-')[-1])).tolist()}}
                 for p in paths]
      if interrupt:
        interrupt = False
        report.with_suffix('.jsonl').write_text(json.dumps(entries[0])+'\n')
        raise RuntimeError('Simulated interrupted GPU process')
      report.with_suffix('.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in entries))
      warmup = int(command[command.index('--warmup')+1])
      replay.write_json(report, {'warmup': entries[:warmup], 'runs': entries[warmup:], 'summary': {'saved': len(entries)}})

    with patch.object(replay, 'read_json', side_effect=read), \
         patch.object(replay.subprocess, 'check_output', return_value='test-revision\n'), \
         patch.object(replay.subprocess, 'run', side_effect=run):
      try:
        replay.timed_infer(args)
      except RuntimeError as error:
        assert str(error) == 'Simulated interrupted GPU process'
      else:
        raise AssertionError('Expected interrupted inference')
      first_saved = calls[0][0]
      replay.timed_infer(args)
      completed_calls = len(calls)
      replay.timed_infer(args)
      assert len(calls) == completed_calls, 'Completed samples ran again'
      assert all(first_saved not in batch for batch in calls[1:]), 'Saved partial prediction ran again'
      predictions = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
      assert [p['sample'] for p in predictions] == [r['sample'] for r in rows], 'Chronological output coverage changed'
      assert all(len(batch) <= 2 for batch in calls), 'Batch limit exceeded'
      timings = [original_read(p) for p in work.glob('command-timing-*.json')]
      assert sorted(t['new_predictions'] for t in timings) == [0, 1, 6]
      assert sum(t['succeeded'] for t in timings) == 2
      assert all(t['wall_seconds'] > 0 for t in timings)
      duplicate = work/'views-0-1-2-3-5-6-duplicate.jsonl'
      duplicate.write_text(json.dumps(predictions[0])+'\n')
      try:
        replay.infer(args)
      except ValueError as error:
        assert 'Duplicate or unexpected' in str(error)
      else:
        raise AssertionError('Duplicate prediction accepted')
      args.work = root/'persistent'
      args.persistent_runtime = True
      config['arguments'] += ['--warmup-graphs', '--vision-group-limit', '8']
      calls.clear()
      interrupt = True
      try:
        replay.timed_infer(args)
      except RuntimeError:
        pass
      replay.timed_infer(args)
      assert [len(batch) for batch in calls] == [7, 6], 'Persistent mode restarted per layout or batch'
      assert calls[0][0] not in calls[1], 'Persistent resume repeated a saved sample'
      predictions = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
      assert [p['sample'] for p in predictions] == [r['sample'] for r in rows]
  print('Resume recovery, batch coverage, duplicate rejection, and command timing passed')


if __name__ == '__main__':
  main()

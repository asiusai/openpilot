"""Replay advancing inputs, save predictions, and check timing and numerical regression."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from openpilot.selfdrive.alpamayo.compare import (FIELDS, compare_replays, load_reference, performance_gate, qualification,
                                                trajectory_config, validate_reference)
from openpilot.selfdrive.alpamayo.inference import Runtime, sha
from openpilot.selfdrive.alpamayo.runtime import private_output, run_prediction, storage_ok


def repeat_decoder(runtime, inputs, expected, repeats):
  """Time advancing six-token blocks from the actual prompt. Do not feed reference tokens back."""
  from openpilot.selfdrive.alpamayo.selected.transport import download
  dec = runtime.decoder
  if dec.statistics['lookup_rounds'] or dec.statistics['serial_tokens'] != len(expected) or len(expected) % 6:
    raise ValueError('Choose a prediction generated entirely in full six-token serial blocks')
  history, prefill, first, cos, sin, prompt = inputs
  lengths = [*sorted({6, min(12, len(expected)), len(expected)}), 'full_generate']
  timings = {str(n): [] for n in lengths}
  for trial in range(repeats+2):  # Prime both initial-logit and returned-logit graph bindings.
    for n in lengths if trial % 2 == 0 else reversed(lengths):
      logits, tokens = first, []
      runtime.synchronize()
      begin = time.monotonic_ns()
      if n == 'full_generate':
        dec.history = list(history)
        tokens = dec.generate(prefill, first, cos, sin, prompt)
      else:
        for offset in range(0, n, 6):
          logits, block = dec.decode(logits, cos, sin, prompt+offset, offset, 6)
          tokens.extend(download(block).reshape(-1).astype(int).tolist())
      runtime.synchronize()
      elapsed = (time.monotonic_ns()-begin)/1e6
      if tokens != (expected if n == 'full_generate' else expected[:n]):
        raise ValueError(f'Resident decoder changed greedy tokens for {n}')
      if trial >= 2:
        timings[str(n)].append(elapsed)
  return timings


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True, help='Prepared schedule with sample directories and frame references')
  parser.add_argument('--checkpoint', type=Path, required=True)
  parser.add_argument('--initial-noise', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True, help='New output directory')
  parser.add_argument('--variant', choices=('default', 'selected'), default='default')
  parser.add_argument('--fp8-cache', type=Path)
  parser.add_argument('--prefill-calibration', type=Path)
  parser.add_argument('--cache', type=Path, help='Persistent private compiler and firmware cache')
  parser.add_argument('--start-index', type=int, default=0, help='First manifest row, including any warmup')
  parser.add_argument('--warmup', type=int, default=5, help='Exclude startup graph capture and image-history priming')
  parser.add_argument('--runs', type=int, required=True)
  parser.add_argument('--period-ms', type=int, choices=(100, 250, 500), default=500,
                      help='Recorded query spacing and measured deadline; initial car cadence is 500 ms')
  parser.add_argument('--reference', type=Path)
  parser.add_argument('--thresholds', type=Path, help='Optional explicit regression limits')
  parser.add_argument('--no-vision-cache', action='store_true')
  parser.add_argument('--decoder-repeats', type=int, default=0, help='Repeat serial decoder prefixes on one selected prediction')
  args = parser.parse_args(argv)
  if args.start_index < 0 or args.warmup < 0 or args.runs < 1:
    parser.error('Require nonnegative start and warmup, and positive measured count')
  if args.variant == 'selected' and (not all((args.fp8_cache, args.prefill_calibration, args.cache)) or args.no_vision_cache):
    parser.error('Selected requires --fp8-cache, --prefill-calibration, --cache, and its complete vision cache')
  if not 0 <= args.decoder_repeats <= 100 or (args.decoder_repeats and (args.variant != 'selected' or args.runs != 1)):
    parser.error('--decoder-repeats requires --variant selected, --runs 1, and at most 100 repeats')
  os.environ.setdefault('DEV', 'USB+AMD:LLVM')
  os.environ.setdefault('GMMU', '0')
  os.environ.setdefault('TC_OPT', '2')
  os.environ.setdefault('AM_POWER_LIMIT', '100')
  schedule = json.loads(args.manifest.read_text())['schedule'][args.start_index:args.start_index+args.warmup+args.runs]
  if len(schedule) != args.warmup+args.runs:
    raise ValueError('Insufficient advancing samples')
  samples = {}
  for row in schedule:
    path = Path(row['sample'])
    if not path.is_absolute():
      path = args.manifest.parent/path
    row['sample'] = str(path.resolve())
    samples[row['sample']] = {'sha256': {key: sha(path/f'{key}.npy') for key in FIELDS},
                             'frame_references': {k: row[k] for k in ('query_ns', 'frames', 'vehicle_history_audit') if k in row}}
  if len(samples) != len(schedule) or any(b['query_ns'] <= a['query_ns'] for a, b in zip(schedule, schedule[1:], strict=False)):
    raise ValueError('Use distinct advancing queries, including warmup')
  intervals = [(b['query_ns']-a['query_ns'])/1e6 for a, b in zip(schedule, schedule[1:], strict=False)]
  if any(interval != args.period_ms for interval in intervals):
    raise ValueError('Recorded query spacing must match --period-ms; retain the complete camera histories')
  expected = load_reference(args.reference) if args.reference else None
  if expected is not None:
    validate_reference(expected[0])
  thresholds = json.loads(args.thresholds.read_text()) if args.thresholds else None
  os.umask(0o077)
  args.output = private_output(args.output)
  args.output.mkdir(parents=True, exist_ok=False)
  initialization_start = time.monotonic_ns()
  if args.variant == 'selected':
    from openpilot.selfdrive.alpamayo.selected import Config, Runtime as SelectedRuntime
    runtime = SelectedRuntime(Config(args.checkpoint, args.fp8_cache, args.prefill_calibration, args.cache, args.initial_noise))
  else:
    if args.cache:
      os.environ.update(XDG_CACHE_HOME=str(args.cache.resolve()), CACHEDB=str(args.cache.resolve()/'tinygrad.db'))
    runtime = Runtime(args.checkpoint, args.initial_noise, cache_vision=not args.no_vision_cache)
  initialization_ms = (time.monotonic_ns()-initialization_start)/1e6
  rows = []
  try:
    config = {'kind': 'configuration', 'variant': args.variant, 'warmup': args.warmup, 'runs': args.runs, 'runtime': runtime.manifest,
              'initialization_ms': initialization_ms, 'period_ms': args.period_ms, 'start_index': args.start_index,
              'trajectory': trajectory_config(json.loads((args.checkpoint/'config.json').read_text())),
              'identities': dict(runtime.manifest['identities'], samples=samples)}
    with (args.output/'predictions.jsonl').open('x') as output:
      written = 0
      def save(row):
        nonlocal written
        raw = json.dumps(row, allow_nan=False)+'\n'
        size = len(raw.encode())
        if written+size > 256*1024**2 or not storage_ok(args.output, size):
          raise RuntimeError('Replay reached its output limit or recording disk reserve')
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
        written += size
      save(config)
      class Adapter:
        def predict(self, arrays, meta):
          prediction = runtime.predict(arrays)
          if args.variant == 'selected':
            prediction['selected_metrics'] = asdict(runtime.metrics)
          return {'query_ns': meta['query_ns'], 'prediction': prediction}
        def synchronize(self):
          runtime.synchronize()
      decoder_inputs = []
      for index, row in enumerate(schedule):
        begin = time.monotonic_ns()
        if index == args.warmup:
          measured_start = begin
        data = {key: np.load(Path(row['sample'])/f'{key}.npy', allow_pickle=False) for key in FIELDS}
        data['nav_text'] = [str(data['nav_text'].item())]
        meta = dict(row, navigation_text=data['nav_text'][0])
        capture = args.decoder_repeats and index == args.warmup
        if capture:
          def capture_prompt(*inputs, generate=runtime.decoder.generate):
            decoder_inputs.append(tuple(runtime.decoder.history))
            decoder_inputs.extend(inputs)
            return generate(*inputs)
          runtime.decoder.generate = capture_prompt
        try:
          result = run_prediction(Adapter(), data, meta)
        finally:
          if capture:
            del runtime.decoder.generate
        result.update(kind='prediction', index=index, warmup=index < args.warmup, sample=row['sample'],
                      full_pipeline_ms=(time.monotonic_ns()-begin)/1e6)
        rows.append(result)
        save(result)
        print(json.dumps({'manifest_index': args.start_index+index, 'warmup': result['warmup'], 'inference_ms': result['inference_ms'],
                          'full_pipeline_ms': result['full_pipeline_ms'],
                          'timings_ms': result['prediction']['timings_ms'],
                          'vision_misses': result['prediction'].get('selected_metrics', {}).get('vision_misses')}), flush=True)
      elapsed = (time.monotonic_ns()-measured_start)/1e9
      performance = performance_gate(rows, elapsed, args.period_ms)
      regression = compare_replays(config, rows, *expected, thresholds) if expected else None
      status = qualification(performance, regression)
      summary = {'kind': 'summary', 'gpu_executed': not runtime.device.startswith(('CPU', 'NPY', 'PYTHON', 'NULL')),
                 **status, 'performance': performance, 'goal_100ms': performance_gate(rows, elapsed, 100), 'regression': regression,
                  'recorded_query_intervals_ms': sorted(set(intervals)),
                  'startup': {'initialization_ms': initialization_ms, 'first_prediction_ms': rows[0]['full_pipeline_ms'],
                              'through_warmup_ms': (measured_start-initialization_start)/1e6},
                  'timing_scope': ('Prepared sample reads, token/position preprocessing, transfers, inference, and trajectory decode; '+
                                   'excludes capture, ISP, video decode, projection, and ego-history construction; not live camera age')}
      if decoder_inputs:
        tokens = rows[-1]['prediction']['generated_token_ids']
        summary['resident_decoder_ms_by_token_count'] = repeat_decoder(runtime, decoder_inputs, tokens, args.decoder_repeats)
      save(summary)
    print(json.dumps(summary, allow_nan=False))
    return int(summary['result'] != 'PASS')
  finally:
    runtime.close()


if __name__ == '__main__':
  raise SystemExit(main())

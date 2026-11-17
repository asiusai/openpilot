"""CPU comparison on actual chronological arrivals and previously selected queries.

Recorded dispatch times stay fixed. This is an overlap feasibility measurement,
not continuous model throughput or a capture-to-output latency benchmark.
"""
import argparse
from collections import deque
from contextlib import ExitStack
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
HERE = Path(__file__).resolve().parent
SOURCE = WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def sha(path):
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  assert Path('/mnt/nvme-500').is_mount() and args.output.resolve().is_relative_to(WORK)
  assert not args.output.exists()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  for part in ('tmp', 'cache'):
    (args.output.parent/part).mkdir(exist_ok=True)
  os.environ.update(DEV='CPU', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(args.output.parent/'tmp'), XDG_CACHE_HOME=str(args.output.parent/'cache'),
    CACHEDB=str(args.output.parent/'cache/compiler.db'))
  sys.path[:0] = [str(TINY), str(SOURCE/'experiments'), str(SOURCE), '/tmp/opencode/openpilot-alpamayo',
    '/home/john/asius/openpilot/opendbc_repo']
  import numpy as np
  from tinygrad import Device
  from tinygrad.helpers import Context
  def no_network(*a, **kw):
    raise AssertionError('Network is outside this CPU experiment')
  with ExitStack() as stack:
    stack.enter_context(Context(ALLOW_DEVICE_USAGE=0))
    stack.enter_context(patch.object(socket.socket, 'connect', no_network))
    paced = load(SOURCE/'experiments/agent-paced-replay.py', 'arrival_paced')
    bridge = paced.bridge
    trace = paced.Trace(paced.SOURCE/'manifest.json', paced.LOGS)
    report_path = WORK/'agent-live-cpu-integrated-paced/report.json'
    assert sha(report_path) == 'b22eb5c954f6908d8ffa3c279628a2edc89a32a513386a6409aa1f8d0d20cd25'
    jsonl = report_path.with_suffix('.jsonl')
    queries = [r for r in map(json.loads, jsonl.read_text().splitlines()) if r.get('kind') == 'model_completion']
    sampler_root = WORK/'warp-fast-pathfix-source'
    sample = load(sampler_root/'agent-warp-fast-sampler.py', 'arrival_sampler')
    sample.ARTIFACTS = args.output.parent/'sampler'
    cfg = json.loads((sampler_root/'agent-warp-fast-validated-portable-v6.json').read_text())['variants'][0]
    samplers = [sample.Sampler(cfg['kind'], cfg['workers'], Path(cfg['provenance']['command'][0]).name,
      cfg['provenance']['native'], cfg['max_maps'], cfg['max_bytes']) for _ in range(2)]
    for sampler in samplers:
      assert sampler.lib.provenance['library_sha256'] == cfg['provenance']['library_sha256']
      stack.callback(sampler.close)
    source, target = trace.manifest['source_camera_geometry'], trace.manifest['target_camera_geometry']
    cached = load(HERE/'agent-arrival-warp-cache.py', 'arrival_warp').ArrivalWarp(
      samplers[1].calibrated_image, bridge, source, target)
    stack.callback(cached.close)
    preparers = [bridge.Preparer(source, target, warp=warp) for warp in (samplers[0].calibrated_image, cached)]
    store = bridge.Store()
    rgb = paced.RGB(WORK/'agent-paced-replay-cache-35-43/rgb', trace.frames, verify=True)
    index, rows, last, currents = 0, [], 0, {}
    arrivals = []
    for query in queries:
      # Replay only observations available before this recorded dispatch.
      cutoff = query['input']['selected_ns']
      while index < len(trace.events) and trace.events[index].release_ns <= cutoff:
        event = trace.events[index]
        if event.kind == 'frame':
          ref = event.value
          pixels = rgb(ref)
          frame = bridge.Frame(ref['view'], ref['frame_id'], ref['capture_ns'], ref['eof_ns'],
            event.release_ns, event.release_ns, pixels)
          assert store.add_frame(frame)
          before = time.perf_counter_ns()
          warmed = cached.received(store, frame, event.release_ns)
          arrivals.append(dict(view=frame.camera, capture_ns=frame.capture_ns, release_ns=event.release_ns,
            primed=event.release_ns <= trace.start, prepared=warmed, wall_ms=(time.perf_counter_ns()-before)/1e6))
        else:
          store.add_state(event.kind, event.stamp, event.release_ns, event.value, event.valid)
        index += 1
      # Keep the recorded query. Publication-only arrival times can expose a
      # newer road frame than the measured producer admitted at that instant.
      # This fixture tests exact preparation, not a replacement scheduler.
      query_store = copy.copy(store)
      query_store.frames = {**store.frames, 6: deque(f for f in store.frames[6]
        if f.capture_ns <= query['input']['query_ns'])}
      newer = len(store.frames[6])-len(query_store.frames[6])
      selection = query_store.select(cutoff, last, currents, 250_000_000, False)
      # Every original conditioning frame must still match canonical FLOOR.
      expected = [(f['view'], f['frame_id'], f['capture_ns']) for f in query['input']['frames']]
      actual = [(f.camera, f.frame_id, f.capture_ns) for f in selection.frames]
      if actual != expected:
        raise ValueError('Original query conditioning differs under publication-only arrival simulation')
      outputs, durations = {}, {}
      for arm in ((0, 1) if len(rows) % 2 == 0 else (1, 0)):
        begin = time.perf_counter_ns()
        outputs[arm] = preparers[arm].prepare(selection)
        durations[str(arm)] = (time.perf_counter_ns()-begin)/1e6
      for key in outputs[0][0]:
        a, b = outputs[0][0][key], outputs[1][0][key]
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
      skip = {'history_build_ms', 'color_warp_stack_ms'}
      for key in outputs[0][1].keys()-skip:
        assert outputs[0][1][key] == outputs[1][1][key], key
      rows.append(dict(query_ns=selection.query_ns, exact=True, prepare_ms=durations,
        cache=cached.state(), warm=len(rows) > 0, newer_published_road_frames_excluded_from_query=newer))
      last = selection.query_ns
      currents = {c: selection.frames[i*5+4].capture_ns for i, c in enumerate(bridge.CAMERAS)}
    # Changed calibration and replacement pixels must not reuse an earlier output.
    frame = selection.frames[-1]
    states = bridge.records(selection.states['extrinsicsCalibration'], selection.query_ns)
    cal, _ = bridge.required(states, frame.capture_ns, bridge.STATE_LIMITS['extrinsicsCalibration'])
    controls = []
    for label, array, calibration in [('unchanged', frame.pixels, cal),
      ('calibration changed', frame.pixels, {**cal, 'rpyCalib': [cal['rpyCalib'][0], cal['rpyCalib'][1]+.01, cal['rpyCalib'][2]]}),
      ('new capture storage', frame.pixels.copy(), cal)]:
      array.flags.writeable = False
      wanted = samplers[0].calibrated_image(array, frame.camera, source, target, calibration)
      got = cached(array, frame.camera, source, target, calibration)
      assert wanted.tobytes() == got.tobytes(), label
      controls.append(dict(case=label, exact=True))
    # Calibration that arrives after early preparation must change the key.
    late_cal = copy.deepcopy(cal)
    late_cal['rpyCalib'][1] += .01
    late_state = bridge.State(selection.query_ns-1, selection.selected_ns, late_cal, True)
    late_states = dict(selection.states, extrinsicsCalibration=tuple(sorted(
      (*selection.states['extrinsicsCalibration'], late_state), key=lambda r: r.stamp)))
    late_selection = replace(selection, states=late_states)
    changed = [p.prepare(late_selection) for p in preparers]
    for key in changed[0][0]:
      assert changed[0][0][key].tobytes() == changed[1][0][key].tobytes(), key
    assert changed[0][0]['image_frames'].tobytes() != outputs[0][0]['image_frames'].tobytes()
    controls.append(dict(case='late calibration recomputes actual preparation', exact=True))
    invalid_states = dict(late_states, extrinsicsCalibration=tuple(sorted(
      (*late_states['extrinsicsCalibration'], bridge.State(selection.query_ns, selection.selected_ns, None, False)),
      key=lambda r: r.stamp)))
    for p in preparers:
      try:
        p.prepare(replace(selection, states=invalid_states))
      except bridge.Unavailable as error:
        assert str(error) == 'invalid_state_update'
      else:
        raise AssertionError('Invalid calibration reused an older cached output')
    controls.append(dict(case='late invalid calibration rejected in both arms', rejected=True))
    try:
      cached.received(store, frame, frame.ready_ns-1)
    except ValueError:
      controls.append(dict(case='future frame rejected', rejected=True))
    else:
      raise AssertionError('Future frame accepted')
    cache_state = cached.state()
    cached.close()
    assert cached.state()['bytes'] == 0 and not Device._opened_devices
    timing = {str(arm): dict(median_ms=float(np.median([r['prepare_ms'][str(arm)] for r in rows if r['warm']])),
      p95_ms=float(np.percentile([r['prepare_ms'][str(arm)] for r in rows if r['warm']], 95))) for arm in (0, 1)}
    result = dict(complete=True, cpu_only=True, gpu_executed=False, source_sha256={str(p): sha(p) for p in
      (Path(__file__), HERE/'agent-arrival-warp-cache.py', SOURCE/'experiments/agent-live-bridge.py',
       sampler_root/'agent-warp-fast-sampler.py', report_path, jsonl)}, queries=rows, arrivals=arrivals,
      prepare_timing=timing, arrival_prepare_total_ms=sum(r['wall_ms'] for r in arrivals), cache=cache_state,
      controls=controls, hooks_restored=True, timing_limit='Fixed recorded dispatches; work and producer contention must be included in a later paced run. No live Hz or capture-age claim.',
      admission_limit='Publication-only arrivals with original recorded query timestamps; newer road frames are excluded only for fixed-query comparison. Every conditioning frame must match canonical FLOOR. Producer contention and actual dispatch selection are unmeasured.',
      no_hidden_future_images=True, no_model_features_computed=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True, queries=len(rows), timing=timing, arrival_prepare_total_ms=result['arrival_prepare_total_ms'])))


if __name__ == '__main__':
  main()

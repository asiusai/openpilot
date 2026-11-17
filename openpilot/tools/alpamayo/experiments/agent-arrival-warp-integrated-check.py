"""Actual Producer/Preparer concurrency and actual factory/Consumer CPU boundaries."""
import argparse
from contextlib import ExitStack
import copy
from dataclasses import replace
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import threading
import time
from unittest.mock import patch
import weakref

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('arrival_integrated_check_common', HERE/'agent-arrival-warp-integrated-common.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
C = A.C


def blocked(*args, **kwargs):
  raise AssertionError('Device/network forbidden in this CPU check')


def arrays_equal(a, b):
  assert a.keys() == b.keys()
  for key in a:
    x, y = a[key], b[key]
    assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), key


def prepare_equal(a, b):
  arrays_equal(a[0], b[0])
  skip = {'history_build_ms', 'color_warp_stack_ms'}
  assert a[1].keys() == b[1].keys()
  for key in a[1].keys()-skip:
    assert a[1][key] == b[1][key], key


def adapter_digest(value):
  import numpy as np
  import torch
  if isinstance(value, torch.Tensor):
    assert value.device.type == 'cpu'
    value = value.detach().numpy()
  if isinstance(value, np.ndarray):
    return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
  if isinstance(value, dict):
    return {k: adapter_digest(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [adapter_digest(v) for v in value]
  assert value is None or isinstance(value, (str, bool, int, float)), type(value)
  return value


def production(root, data):
  from tinygrad import Device
  K = A.load(C.EXP/'agent-live-cpu-integrated-check.py', 'arrival_integrated_boundary')
  K.C = C
  boundary = K.Boundary('paced')
  protocol = root/'cpu-protocol.json'
  A.write(protocol, dict(data, cpu_only=True), new=True)
  output = root/'entry/report.json'
  args = [str(HERE/'agent-arrival-warp-integrated-run.py'), '--protocol', str(protocol),
    '--protocol-sha', A.sha(protocol), '--output', str(output), '--cpu-preflight']
  previous = importlib.util.spec_from_file_location
  modules = {name:sys.modules.get(name) for name in ('model','benchmark','inputs','native_warp','stream_inputs')}
  targets = []
  def profile(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == 'launch' and 'path' in frame.f_locals:
      targets.append(str(frame.f_locals['path']))
  error = None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ)); stack.enter_context(patch.object(sys, 'path', sys.path[:]))
      stack.enter_context(patch.object(sys, 'argv', args))
      stack.enter_context(patch.object(type(Device), '__getitem__', blocked))
      stack.enter_context(patch.object(socket.socket, 'connect', blocked))
      stack.enter_context(patch.object(importlib.util, 'spec_from_file_location', boundary.observe(previous, stack)))
      old = sys.getprofile(); sys.setprofile(profile)
      try:
        runpy.run_path(args[0], run_name='__main__')
      except K.ConstructedBoundary as exc:
        error = exc
      finally:
        sys.setprofile(old)
  finally:
    for name, old in modules.items():
      if old is None: sys.modules.pop(name, None)
      else: sys.modules[name] = old
  assert error is not None and targets == [str(C.N.LIVE/'experiments/benchmark-pipeline.py')]
  assert importlib.util.spec_from_file_location is previous and not Device._opened_devices
  assert boundary.owner_errors == ['duplicate owner rejected','active query rejected']
  status = A.read(output.parent/'arrival-warp-integrated-status.json')
  assert not status['passed'] and not status['gpu_executed'] and status['hooks_restored'] and status['constructor_restored']
  arrival = status['arrival']
  assert arrival['installed'] and arrival['closed'] and arrival['producer_binding_restored']
  # Original startup obtains its current input from canonical past-only prime.
  assert arrival['cache_cleanup_bytes'] == 0 and arrival['prime']['frames'] == 130
  assert arrival['active']['frames'] == 0 and all(row['primed'] for row in arrival['arrivals'])
  assert arrival['cache_before_cleanup']['counts']['hits'] > 0
  assert boundary.hooks.arrival.preparer.warp is not boundary.hooks.arrival.cache
  assert boundary.hooks.sampler.pool is None
  assert boundary.consumer._closed and not boundary.consumer._lock.locked()
  assert status['cpu_hook']['cleanup_exact'] and status['cpu_hook']['shadow_after_cleanup'] == 0
  events = status['storage_events']
  assert [e['path'] for e in events if e['event'] == 'cache_open'] == [str(C.N.WEIGHTS/'index.json')]
  return dict(complete=True, actual_runpy_targets=targets, factory_index_read=True, owner_rejections=boundary.owner_errors,
    actual_consumer_constructed=True, actual_overlay_installed=True, hooks_restored=True,
    source_clock_started=False, actual_past_only_prime_frames=arrival['prime']['frames'],
    mock_boundary='Only GPU model completion/allocation/startup computation'), boundary.components.adapter, boundary.consumer


def cpu_arrivals(root, adapter, consumer):
  import numpy as np
  import torch
  install = A.load(HERE/'agent-arrival-warp-integrated-install.py', 'arrival_integrated_cpu_install')
  paced = A.load(C.N.LIVE/'experiments/agent-paced-replay.py', 'arrival_integrated_cpu_paced')
  bridge = paced.bridge
  trace = paced.Trace(paced.SOURCE/'manifest.json', paced.LOGS)
  source, target = trace.manifest['source_camera_geometry'], trace.manifest['target_camera_geometry']
  sampler_root = A.WORK/'warp-fast-pathfix-source'
  sample = A.load(sampler_root/'agent-warp-fast-sampler.py', 'arrival_integrated_cpu_sampler')
  sample.ARTIFACTS = root/'sampler'
  cfg = A.read(sampler_root/'agent-warp-fast-validated-portable-v6.json')['variants'][0]
  rows, adapters, controls = [], [], []
  with ExitStack() as stack:
    samplers = [sample.Sampler(cfg['kind'], cfg['workers'], Path(cfg['provenance']['command'][0]).name,
      cfg['provenance']['native'], cfg['max_maps'], cfg['max_bytes']) for _ in range(2)]
    for sampler in samplers:
      assert sampler.lib.provenance['library_sha256'] == cfg['provenance']['library_sha256']
      stack.callback(sampler.close)
    preparers = [bridge.Preparer(source, target, warp=s.calibrated_image) for s in samplers]
    binding_stack = stack.enter_context(ExitStack())
    integration = install.Integration(A, binding_stack)
    integration.attach_preparer(preparers[1], bridge)
    stack.callback(integration.close)
    rgb = paced.RGB(A.WORK/'agent-paced-replay-cache-35-43/rgb', trace.frames, verify=True)
    stores = [bridge.Store(frame_limit=64, byte_limit=384*1024**2) for _ in range(2)]
    host = [0]
    clocks = [paced.Clock(trace.start, host=lambda:host[0]) for _ in range(2)]
    producers = [paced.Producer(trace, store, rgb, clock) for store, clock in zip(stores, clocks, strict=True)]
    integration.attach_producer(producers[1])
    for producer in producers: producer.prime_history()
    for clock in clocks: clock.start()
    queries = [r for r in map(json.loads, C.output('paced').with_suffix('.jsonl').read_text().splitlines())
      if r.get('kind') == 'model_completion']
    last, currents = 0, {}
    for index, query in enumerate(queries):
      cutoff = query['input']['selected_ns']; host[0] = cutoff-trace.start
      for producer in producers: producer.pump()
      selections = [store.select(cutoff, last, currents, 250_000_000, False) for store in stores]
      assert [f.reference() for f in selections[0].frames] == [f.reference() for f in selections[1].frames]
      values = [preparer.prepare(selection) for preparer, selection in zip(preparers, selections, strict=True)]
      prepare_equal(*values)
      native = []
      for arrays, detail in values:
        nav = trace.nav(selections[0].query_ns)
        live_arrays = dict(arrays, nav_text=[nav])
        meta = dict(detail, query_ns=selections[0].query_ns, selected_ns=cutoff, prepared_ns=cutoff, navigation_text=nav)
        data = consumer._data(live_arrays, meta, cutoff)
        if index in (0, len(queries)-1):
          _, _, prepared = adapter.prepare(data)
          native.append(dict(prepared=adapter_digest(prepared), native=adapter_digest(adapter.native_inputs(prepared))))
      if native:
        assert native[0] == native[1]
        adapters.append(dict(query_ns=selections[0].query_ns, native_input_digest=native[0], exact=True))
      rows.append(dict(query_ns=selections[0].query_ns, ten_frames=[f.reference() for f in selections[0].frames],
        arrays_exact=True, all_metadata_except_timers_exact=True,
        newest_query_differs_from_old_timing=selections[0].query_ns != query['input']['query_ns']))
      last = selections[0].query_ns
      currents = {c:selections[0].frames[i*5+4].capture_ns for i,c in enumerate(bridge.CAMERAS)}
    assert producers[0].arrivals == producers[1].arrivals
    assert producers[0].index == producers[1].index
    for label in ('navigation', 'frame_conditioning', 'unknown_input'):
      bad_arrays, bad_meta = dict(live_arrays), copy.deepcopy(meta)
      if label == 'navigation': bad_arrays['nav_text'] = ['Changed navigation.']
      elif label == 'frame_conditioning': bad_meta['frames'][0]['requested_ns'] += 1
      else: bad_arrays['unexpected'] = 1
      try: consumer._data(bad_arrays, bad_meta, cutoff)
      except ValueError: controls.append('actual Consumer rejected '+label)
      else: raise AssertionError('Actual Consumer accepted '+label)
    selection = selections[1]; frame = selection.frames[-1]
    cal, _ = bridge.required(bridge.records(selection.states['extrinsicsCalibration'], selection.query_ns),
      frame.capture_ns, bridge.STATE_LIMITS['extrinsicsCalibration'])
    changed = copy.deepcopy(cal); changed['rpyCalib'][1] += .01
    late = replace(selection, states=dict(selection.states, extrinsicsCalibration=tuple(sorted(
      (*selection.states['extrinsicsCalibration'], bridge.State(selection.query_ns-1, selection.selected_ns, changed, True)), key=lambda r:r.stamp))))
    # Each Preparer retains frames from its own Store. Preserve those identities.
    late_values = [p.prepare(replace(sel, states=late.states)) for p, sel in zip(preparers, selections, strict=True)]
    prepare_equal(*late_values)
    assert late_values[0][0]['image_frames'].tobytes() != values[0][0]['image_frames'].tobytes()
    controls.append('late calibration recomputed exactly')
    invalid = replace(late, states=dict(late.states, extrinsicsCalibration=tuple(sorted(
      (*late.states['extrinsicsCalibration'], bridge.State(selection.query_ns, selection.selected_ns, None, False)), key=lambda r:r.stamp))))
    for preparer in preparers:
      try: preparer.prepare(invalid)
      except bridge.Unavailable as error: assert str(error) == 'invalid_state_update'
      else: raise AssertionError('Invalid calibration accepted')
    controls.append('late calibration tombstone rejected by both actual Preparers')
    empty = bridge.Store()
    assert integration.cache.received(empty, frame, frame.ready_ns) is False
    assert integration.cache.received(empty, replace(frame, format='nv12'), frame.ready_ns) is False
    try: integration.cache.received(stores[1], frame, frame.ready_ns-1)
    except ValueError: controls.append('future frame rejected')
    else: raise AssertionError('Future frame accepted')
    before = len(integration.rows)
    assert producers[1].store.add_frame(frame) is False
    assert len(integration.rows) == before
    controls.append('duplicate Store rejection does not prewarp')
    # Cache retains outputs only. The Store and original Preparer retain their own bounded raw frames.
    small = install.Integration(A, stack.enter_context(ExitStack()))
    p = bridge.Preparer(source, target, warp=samplers[1].calibrated_image)
    small.attach_preparer(p, bridge)
    small.cache.max_entries, small.cache.max_bytes = 2, 2*3*432*768
    refs = []
    for i in range(8):
      raw = frame.pixels.copy(); raw[0, 0, 0] ^= i; raw.flags.writeable = False
      refs.append(weakref.ref(raw)); p.warp(raw, frame.camera, source, target, cal)
      del raw
      assert small.cache.state()['bytes'] <= small.cache.max_bytes and small.cache.state()['entries'] <= 2
    gc.collect(); assert all(ref() is None for ref in refs)
    small.close(); assert small.cache.state()['bytes'] == 0
    controls.append('eight raw arrays released; output cache byte and entry eviction bounds exact')
    integration.close(); producers[0].close()
    fixed = integration.report()
  assert fixed['cache_cleanup_bytes'] == 0 and integration.info['producer_binding_restored']
  # True concurrent native calls: canonical source thread plus actual main-thread preparation.
  # Compare each exact Selection in both Preparers; never compare different receipt schedules.
  concurrent_trace = copy.copy(trace); concurrent_trace.end = trace.start+1_000_000_000
  with ExitStack() as stack:
    samplers = [sample.Sampler(cfg['kind'], cfg['workers'], Path(cfg['provenance']['command'][0]).name,
      cfg['provenance']['native'], cfg['max_maps'], cfg['max_bytes']) for _ in range(2)]
    for sampler in samplers: stack.callback(sampler.close)
    preparers = [bridge.Preparer(source, target, warp=s.calibrated_image) for s in samplers]
    bindings = stack.enter_context(ExitStack()); integration = install.Integration(A, bindings)
    # Observe BELOW the production serialization wrapper. This probe adds no lock.
    # The native sampler releases the GIL, so concurrent entry remains observable.
    underlying = preparers[1].warp; activity = [0,0]
    def observed(*args, **kwargs):
      activity[0] += 1; activity[1] = max(activity[1], activity[0])
      try: return underlying(*args, **kwargs)
      finally: activity[0] -= 1
    preparers[1].warp = observed
    integration.attach_preparer(preparers[1], bridge); stack.callback(integration.close)
    cache = integration.cache
    store = bridge.Store(frame_limit=64, byte_limit=384*1024**2)
    clock = paced.Clock(trace.start)
    producer = paced.Producer(concurrent_trace, store, rgb, clock)
    integration.attach_producer(producer); producer.prime_history()
    clock.start(); producer.start(); last, currents, concurrent_rows = 0, {}, []
    while clock() < concurrent_trace.end:
      producer.check()
      try: selection = store.select(clock(), last, currents, 250_000_000, False)
      except bridge.Unavailable:
        producer.stop.wait(.001); continue
      values = [p.prepare(selection) for p in preparers]; prepare_equal(*values)
      f = selection.frames[-1]
      c, _ = bridge.required(bridge.records(selection.states['extrinsicsCalibration'], selection.query_ns), f.capture_ns, 1_000_000_000)
      mutable = f.pixels.copy()
      assert cache(mutable, f.camera, source, target, c).tobytes() == samplers[0].calibrated_image(mutable, f.camera, source, target, c).tobytes()
      last = selection.query_ns
      currents = {c:selection.frames[i*5+4].capture_ns for i,c in enumerate(bridge.CAMERAS)}
      concurrent_rows.append(last)
    producer.close(); integration.close()
    assert len(concurrent_rows) >= 5 and activity == [0,1]
    concurrency = dict(queries=len(concurrent_rows), arrays_exact=True, writable_fallback_exact=True,
      sampler_max_active=activity[1], source_thread_joined=not producer.thread.is_alive(),
      observation_wall_ms=(clock()-trace.start)/1e6, arrival=integration.report())
  return dict(queries=rows, actual_adapter_native_inputs=adapters, fixed_arrivals=fixed,
    concurrency=concurrency, controls=controls, frame_exclusions=0, cpu_timing_is_not_model_throughput=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True); p.add_argument('--admit', action='store_true')
  args = p.parse_args(); assert not args.output.exists() and args.output.resolve().is_relative_to(A.WORK)
  A.configure()
  if args.admit:
    K = A.load(C.EXP/'agent-live-cpu-integrated-check.py', 'arrival_integrated_admission'); K.C = C
    result = K.admit()
  else:
    root = A.ARTIFACTS/args.output.stem; assert not root.exists(); root.mkdir()
    data = dict(files=A.files(), foundations=A.foundations(), cpu_only=True,
      start_seconds=35, seconds=8, hz=10, timeout_seconds=900, automatic_gpu=False)
    from tinygrad import Device
    from tinygrad.helpers import Context
    with Context(ALLOW_DEVICE_USAGE=0), patch.object(socket.socket, 'connect', blocked), patch.object(type(Device), '__getitem__', blocked):
      preflight, adapter, consumer = production(root, data)
      controls = cpu_arrivals(root, adapter, consumer)
    assert not Device._opened_devices
    assert A.files() == data['files']
    result = dict(complete=True, gpu_executed=False, devices=[], production=preflight,
      arrival=controls, source_before_after_exact=True, files=data['files'])
  A.write(args.output, result, new=True)
  print(dict(complete=True, gpu_executed=False, admit_only=args.admit), flush=True)


if __name__ == '__main__': main()

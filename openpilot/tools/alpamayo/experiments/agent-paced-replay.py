"""RECORDED REPLAY: source-paced native frames through the frozen live bridge.

Import is CPU-only. Model execution is a separate parent operation, gated by an
actual Consumer parity report. Nothing publishes controls or asserts device health.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SOURCE = Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six')
LOGS = Path('/mnt/nvme-256/footage/f9ad9a96/realdata')
RAW = Path('/mnt/nvme-500/tmp/alpamayo-route-4hz-20261116/native-frames')
LABEL = 'RECORDED REPLAY - PC host; not device capture latency or Asius performance'
GIB = 1024**3


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


sys.path.insert(0, str(ROOT))
bridge = local('agent-live-bridge')
capture = local('agent-live-capture')


def sha(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_hash(value):
  return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Event:
  ordinal: int
  release_ns: int
  published_ns: int
  kind: str
  stamp: int
  value: object
  valid: bool = True


class Trace:
  """Bounded metadata only. Preserve filtered rlog order and real source stamps."""
  def __init__(self, manifest, logs, start_s=35., duration_s=8.):
    begin = time.monotonic_ns()
    if not 0 < duration_s <= 8 or start_s < 3.3:
      raise ValueError('Require 3.3 seconds of past history and a source window of at most eight seconds')
    self.manifest_path = Path(manifest)
    source = json.loads(self.manifest_path.read_text())
    self.navigation = sorted((r['query_ns'], r['navigation_text']) for r in source['schedule'] if r.get('sample'))
    self.manifest = {key: source[key] for key in ('camera_input_profile', 'camera_subset', 'capture_start_ns', 'export',
      'rlog_sha256', 'source_camera_geometry', 'target_camera_geometry', 'camera_manifest', 'navigation_source')}
    del source  # Do not retain recorded future trajectories or small-model predictions.
    if self.manifest['camera_input_profile'] != 'calibrated_vehicle_history' or self.manifest['camera_subset'] != [1, 6]:
      raise ValueError('Use the calibrated two-camera SOURCE manifest')
    self.origin = self.manifest['capture_start_ns']
    self.start = self.origin+round(start_s*1e9)
    self.end = self.start+round(duration_s*1e9)
    self.prime_start = self.start-3_300_000_000
    export = Path(self.manifest['export'])
    native_path = export/'clips/native-timestamps.json'
    native = json.loads(native_path.read_text())['views']
    maps = {c: {r['frame_id']: dict(r, view=c, video_index=i) for i, r in enumerate(native[str(c)])
                if self.prime_start <= r['capture_ns'] < self.end} for c in (1, 6)}
    self.events, self.frames, self.log_hashes = [], [], {}
    release = ordinal = reordered = 0
    self.max_release_delay_ns = 0
    from openpilot.cereal import log
    import zstandard
    camera_kinds = {'wideRoadCameraState': 1, 'narrowRoadCameraState': 6}
    for relative, expected in sorted(self.manifest['rlog_sha256'].items(), key=lambda item: int(Path(item[0]).parent.name.rsplit('--', 1)[1])):
      path = Path(logs)/relative
      actual = sha(path)
      if actual != expected:
        raise ValueError(f'Original rlog changed: {relative}')
      self.log_hashes[relative] = actual
      with path.open('rb') as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
        raw = decoder.read(256*1024**2+1)
      if len(raw) > 256*1024**2:
        raise ValueError('One uncompressed rlog exceeds the 256 MiB metadata-parser bound')
      for message in log.Event.read_multiple_bytes(raw):
        ordinal += 1
        kind, published = message.which(), int(message.logMonoTime)
        if kind not in bridge.STATE_LIMITS and kind not in camera_kinds:
          continue
        # A monotonic lower bound preserves ORIGINAL log order across services.
        # It is a replay release rule, never an asserted subscriber receipt time.
        release = max(release, published)
        if release >= self.end:
          continue
        msg = getattr(message, kind)
        if kind in camera_kinds:
          view = camera_kinds[kind]
          ref = maps[view].get(int(msg.frameId))
          if ref is None:
            continue
          if (not message.valid or int(msg.timestampSof) != ref['capture_ns'] or
              int(msg.timestampEof) != ref['eof_ns'] or not ref['capture_ns'] <= ref['eof_ns'] <= published):
            raise ValueError('Native camera metadata or SOF/EOF/publication order differs from the original recording')
          value = dict(ref, published_ns=published, release_ns=release)
          event = Event(ordinal, release, published, 'frame', ref['capture_ns'], value)
          self.frames.append(value)
        else:
          stamp = int(msg.timestamp) if kind == 'deviceMotion' else published
          horizon = 121_000_000_000 if kind == 'carParams' else 3_300_000_000
          if stamp < self.start-horizon or stamp <= 0:
            continue
          if stamp > published:
            raise ValueError('A state source timestamp exceeds its real publication timestamp')
          value, valid = capture.state_value(kind, msg)
          event = Event(ordinal, release, published, kind, stamp, value, bool(message.valid and valid))
        reordered += int(release > published)
        self.max_release_delay_ns = max(self.max_release_delay_ns, release-published)
        self.events.append(event)
        if len(self.events) > 10000 or len(self.frames) > 480:
          raise ValueError('Source window exceeds bounded metadata/frame counts')
      del raw
    if any(b.release_ns < a.release_ns or b.ordinal <= a.ordinal for a, b in zip(self.events, self.events[1:])):
      raise ValueError('Recorded event order was not preserved')
    for view in (1, 6):
      frames = [r for r in self.frames if r['view'] == view]
      expected = [r for r in maps[view].values() if r['eof_ns'] < self.end-10_000_000]
      if not frames or len({r['frame_id'] for r in frames}) != len(frames):
        raise ValueError('Missing or duplicate native front-camera publication')
      if {r['frame_id'] for r in expected}-{r['frame_id'] for r in frames}:
        raise ValueError('Native camera publication metadata is incomplete inside the source window')
      stamps = np.array([r['capture_ns'] for r in frames])
      if not np.all(np.diff(stamps) > 0) or not 45_000_000 <= np.median(np.diff(stamps)) <= 55_000_000:
        raise ValueError('Keep the original approximately 20 Hz front streams')
    self.navigation = [(q, text) for q, text in self.navigation if q <= self.end]
    if not self.navigation or self.navigation[0][0] > self.start:
      raise ValueError('Missing retrospective navigation protocol at source start')
    self.provenance = dict(label=LABEL, manifest_sha256=sha(manifest), native_metadata_sha256=sha(native_path),
      rlog_sha256=self.log_hashes, events=len(self.events), frames=len(self.frames),
      source_start_ns=self.start, source_end_ns=self.end, history_prime_start_ns=self.prime_start,
      metadata_load_ms=(time.monotonic_ns()-begin)/1e6, read_ahead='bounded metadata only; no pixels or model features',
      release_policy='original filtered rlog order; prefix maximum of real publication times',
      reordered_publication_records=reordered, maximum_release_delay_ns=self.max_release_delay_ns,
      receipt_limit='Subscriber receipt times were not logged. Actual replay admission/IO timestamps are measured separately.',
      navigation_source=self.manifest['navigation_source'], native_health_assertions=False)
    self.frame_digest = json_hash(self.frames)

  def nav(self, query):
    from bisect import bisect_right
    i = bisect_right([q for q, _ in self.navigation], query)-1
    if i < 0:
      raise ValueError('No causal annotation in the retrospective navigation protocol')
    return self.navigation[i][1]


class Clock:
  """Map elapsed host monotonic time into the original source timestamp domain."""
  def __init__(self, source_start, host=time.monotonic_ns):
    self.source_start, self.host, self.anchor = source_start, host, None

  def __call__(self):
    return self.source_start if self.anchor is None else self.source_start+self.host()-self.anchor

  def start(self):
    if self.anchor is not None:
      raise RuntimeError('Replay clock cannot restart or pause around inference')
    self.anchor = self.host()


class RGB:
  """One on-demand cached RGB load. No mmap, decoder, future pixels, or read-ahead."""
  def __init__(self, root, frames, *, verify=True):
    self.root, self.reads, self.bytes, self.io_ns, self.max_io_ns = Path(root).resolve(), 0, 0, 0, 0
    self.lock = threading.Lock()
    if not Path('/mnt/nvme-500').is_mount() or not self.root.is_relative_to(Path('/mnt/nvme-500')):
      raise ValueError('Raw RGB must reside on the 500 GB SSD')
    missing = [self.path(r) for r in frames if not self.path(r).is_file()]
    if missing:
      raise ValueError(f'Incomplete native 20 Hz RGB cache: {len(missing)} missing; first {missing[0]}')
    self.inventory = None
    if verify:
      path = self.root.parent/'paced-cache.json'
      inventory = json.loads(path.read_text())
      if inventory['frame_metadata_sha256'] != json_hash(frames):
        raise ValueError('Raw-cache source window differs')
      begin = time.monotonic_ns()
      for key, expected in inventory['files'].items():
        if sha(self.root/key) != expected:
          raise ValueError(f'Raw RGB checksum changed: {key}')
      if len(inventory['files']) != len(frames):
        raise ValueError('Raw-cache inventory is incomplete')
      self.inventory = dict(path=str(path), sha256=sha(path), verify_ms=(time.monotonic_ns()-begin)/1e6,
        page_cache_note='Full-file integrity reads occur before replay and can populate the OS page cache; cold storage latency is not claimed')

  def path(self, ref):
    return self.root/str(ref['view'])/f'{ref["video_index"]}.npy'

  def __call__(self, ref):
    begin = time.monotonic_ns()
    pixels = np.ascontiguousarray(np.load(self.path(ref), allow_pickle=False))
    if pixels.dtype != np.uint8 or pixels.shape != (3, 760, 1344):
      raise ValueError('Cached native RGB must retain 1344x760 resolution')
    pixels.flags.writeable = False
    elapsed = time.monotonic_ns()-begin
    with self.lock:
      self.reads += 1
      self.bytes += pixels.nbytes
      self.io_ns += elapsed
      self.max_io_ns = max(self.max_io_ns, elapsed)
    return pixels

  def report(self):
    with self.lock:
      return dict(rgb_reads=self.reads, rgb_bytes=self.bytes, rgb_load_ms=self.io_ns/1e6, maximum_rgb_load_ms=self.max_io_ns/1e6)


class Producer:
  def __init__(self, trace, store, rgb, clock):
    self.trace, self.store, self.rgb, self.clock = trace, store, rgb, clock
    self.index, self.error, self.thread = 0, None, None
    self.stop = threading.Event()
    self.arrivals = []  # <=480 compact camera records; never image arrays.
    self.state_counts = Counter()
    self.prime = None

  def pump(self, cutoff=None):
    cutoff = min(self.clock() if cutoff is None else cutoff, self.trace.end-1)
    while self.index < len(self.trace.events) and not self.stop.is_set():
      event = self.trace.events[self.index]
      if event.release_ns > cutoff:
        break
      received = self.clock()
      if event.release_ns > received:
        raise ValueError('Future observation reached the bridge')
      if event.kind == 'frame':
        ref = event.value
        pixels = self.rgb(ref)
        ready = self.clock()
        frame = bridge.Frame(ref['view'], ref['frame_id'], ref['capture_ns'], ref['eof_ns'], received, ready, pixels)
        if not self.store.add_frame(frame):
          raise ValueError('Recorded camera arrival became duplicate or reordered')
        self.arrivals.append(dict(view=ref['view'], frame_id=ref['frame_id'], capture_ns=ref['capture_ns'],
          published_ns=event.published_ns, release_ns=event.release_ns, received_ns=received, ready_ns=ready,
          ordinal=event.ordinal, primed=self.clock.anchor is None))
      else:
        self.store.add_state(event.kind, event.stamp, received, event.value, event.valid)
        self.state_counts[event.kind] += 1
      self.index += 1

  def prime_history(self):
    if self.clock.anchor is not None or self.index:
      raise RuntimeError('Prime only the past, once, before the replay clock starts')
    begin = time.monotonic_ns()
    self.pump(self.trace.start)
    self.prime = dict(wall_ms=(time.monotonic_ns()-begin)/1e6, events=self.index,
                      arrays=self.rgb.report(), latest_release_ns=self.trace.events[self.index-1].release_ns)

  def start(self):
    if self.clock.anchor is None:
      raise RuntimeError('Start the source clock after initialization and past-only priming')
    def run():
      try:
        while not self.stop.is_set() and self.clock() < self.trace.end:
          self.pump()
          if self.index == len(self.trace.events):
            break
          delay = max(0, self.trace.events[self.index].release_ns-self.clock())/1e9
          self.stop.wait(min(delay, .002))
      except BaseException as error:
        self.error = error
        self.stop.set()
    self.thread = threading.Thread(target=run, daemon=True, name='recorded-source-arrivals')
    self.thread.start()

  def check(self):
    if self.error is not None:
      raise RuntimeError('Recorded source ingestion failed') from self.error

  def close(self):
    self.stop.set()
    if self.thread is not None:
      self.thread.join(timeout=1)
      if self.thread.is_alive():
        raise RuntimeError('RGB reader did not stop; discard the run')
    self.check()


class Navigation:
  """Use the current query annotation, without replacing the canonical preparer."""
  def __init__(self, preparer, trace):
    self.preparer, self.trace, self.runner = preparer, trace, None

  def prepare(self, selection):
    self.runner.navigation = self.trace.nav(selection.query_ns)
    return self.preparer.prepare(selection)


class Monitor:
  """Wrap an already initialized Consumer; keep its actual predictions unchanged."""
  def __init__(self, consumer, components, producer, clock, end, budget_ns, emit):
    self.consumer, self.components, self.producer, self.clock = consumer, components, producer, clock
    self.end, self.budget_end, self.emit = end, clock.source_start+budget_ns, emit
    self.record = None

  def predict(self, arrays, meta):
    self.producer.check()
    now = self.clock()
    if now >= self.end:
      raise StopIteration('Source window ended during preparation; no new model inference admitted')
    if now >= self.budget_end:
      raise TimeoutError('Inference budget exhausted')
    meta['navigation_source'] = self.producer.trace.manifest['navigation_source']
    meta['measurement_label'] = LABEL
    self.record = dict(query_ns=meta['query_ns'], consumer_started_ns=now)
    self.emit(dict(kind='dispatch', **self.record, label=LABEL))
    return self.consumer.predict(arrays, meta)

  def synchronize(self):
    self.consumer.synchronize()
    if self.clock() >= self.budget_end:
      raise TimeoutError('Inference exceeded its hard completion budget')
    self.producer.check()
    cache, inputs = self.components.vision_cache, self.consumer.inputs
    self.record.update(consumer_completed_ns=self.clock(), vision_hits=cache.hits, vision_misses=cache.misses,
      uploaded_pixel_bytes=inputs.uploaded_pixel_bytes,
      speculative_record=(self.components.decoder.prefill_draft_records[-1]
                          if hasattr(self.components.decoder, 'prefill_draft_records') else None))


def observe(trace, producer, runner, consumer, components, *, clock, budget_seconds=180., emit=lambda row: None,
            idle=None, before_tick=None, collect=None):
  if not 0 < budget_seconds <= 180 or clock.anchor is None:
    raise ValueError('Start the source clock once and retain a maximum 180-second inference budget')
  if getattr(consumer, 'clock', clock) is not clock or getattr(consumer, 'last_query_ns', 0) >= trace.start:
    raise ValueError('Inject the same replay clock and a Consumer whose query history precedes this source window')
  monitor = Monitor(consumer, components, producer, clock, trace.end, round(budget_seconds*1e9), emit)
  rows, selected_ids = [], set()
  # Initial history is eligible conditioning, never a new prediction query.
  runner.last_query = trace.start-1
  try:
    while clock() < trace.end:
      producer.check()
      if before_tick is not None:
        before_tick()
      row = runner.run_once(monitor)
      if row is not None:
        if len(rows) >= 160 or row['input']['frames'][-1]['frame_id'] in selected_ids:
          raise RuntimeError('Output bound or fresh-query identity failed')
        selected_ids.add(row['input']['frames'][-1]['frame_id'])
        row['label'] = LABEL
        row['paced_consumer'] = dict(monitor.record)
        row['output_source_clock_ns'] = row['completed_ns']
        row['source_elapsed_seconds'] = (row['completed_ns']-trace.start)/1e9
        row['raw_ingestion'] = producer.rgb.report()
        emit(row)
        rows.append(dict(completed_ns=row['completed_ns'], frame_id=row['input']['frames'][-1]['frame_id'], **row['paced_consumer']))
        if collect is not None:
          collect(len(rows))
      elif idle is not None:
        idle()
      else:
        producer.stop.wait(.001)
  except StopIteration:
    stop_reason = 'source_window_ended_during_preparation'
  else:
    stop_reason = 'source_window_elapsed'
  producer.close()
  wall = (clock()-trace.start)/1e9
  arrivals = [a for a in producer.arrivals if a['view'] == 6 and trace.start <= a['capture_ns'] < trace.end]
  last = max((r['query_ns'] for r in rows), default=trace.start-1)
  obsolete = sum(a['capture_ns'] < last and a['frame_id'] not in selected_ids for a in arrivals)
  tail = sum(a['capture_ns'] > last and a['frame_id'] not in selected_ids for a in arrivals)
  expected_arrivals = [r for r in trace.frames if r['view'] == 6 and trace.start <= r['capture_ns'] < trace.end]
  unadmitted = len(expected_arrivals)-len(arrivals)
  if len(arrivals) != len(rows)+obsolete+tail:
    raise AssertionError('Source admissions, unique completions, obsolete queries, and tail do not reconcile')
  return dict(kind='summary', label=LABEL, stop_reason=stop_reason, requested_hz=1e9/runner.period_ns,
    completed_queries=len(rows), recorded_replay_completion_hz=len(rows)/wall if wall else 0.,
    source_window_s=(trace.end-trace.start)/1e9, observation_with_final_completion_s=wall,
    unqueued_obsolete_road_queries=obsolete, unselected_arrived_tail=tail, admitted_road_queries=len(arrivals),
    native_road_queries_in_window=len(expected_arrivals), unadmitted_road_queries=unadmitted,
    source_frame_id_gaps=producer.store.counts['source_frame_id_gaps'], queued_prediction_snapshots=0,
    maximum_active_inference=1, source_start_ns=trace.start, source_end_ns=trace.end,
    final_completion_ns=rows[-1]['completed_ns'] if rows else None, bridge_counts=dict(runner.counts),
    bounds=producer.store.bounds(), raw_ingestion=producer.rgb.report(), rows=rows,
    inference_budget_seconds=budget_seconds, small_model_guard='not asserted; PC recorded replay',
    late_completion_policy='Admit only before source end; finish one in-flight inference within the hard budget; never drain obsolete queries')


@contextmanager
def hard_budget(seconds):
  """Default kernel SIGALRM terminates even a blocked Python/GPU call."""
  if not 0 < seconds <= 180 or signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
    raise ValueError('Require an unused process alarm and at most 180 seconds')
  previous = signal.signal(signal.SIGALRM, signal.SIG_DFL)
  signal.setitimer(signal.ITIMER_REAL, seconds)
  try:
    yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def prepare_cache(trace, directory):
  """Parent-only CPU preparation, using the canonical raw-video extractor."""
  from camera_profiles import extract
  directory = Path(directory).resolve()
  if not Path('/mnt/nvme-500').is_mount() or not directory.is_relative_to(Path('/mnt/nvme-500')) or directory.exists():
    raise ValueError('Use a new cache directory on the 500 GB SSD')
  anticipated = len(trace.frames)*(3*760*1344+256)+64*1024**2
  usage = shutil.disk_usage(directory.parent)
  if anticipated > 1536*1024**2 or not bridge.storage_ok(usage.total, usage.free, anticipated):
    raise ValueError('Raw cache exceeds its 1.5 GiB bound or storage reserve')
  directory.mkdir()
  source = directory/'source'
  source.mkdir()
  manifest = dict(trace.manifest, schedule=[dict(tick=0, frames=trace.frames)])
  (source/'manifest.json').write_text(json.dumps(manifest)+'\n')
  begin = time.monotonic_ns()
  output = directory/'rgb'
  extract(SimpleNamespace(source=source, output=output, recording_uuid=None, ticks=None))
  decode_ns = time.monotonic_ns()-begin
  files = {f'{r["view"]}/{r["video_index"]}.npy': sha(output/str(r['view'])/f'{r["video_index"]}.npy') for r in trace.frames}
  report = dict(label=LABEL, frame_metadata_sha256=trace.frame_digest, frames=len(trace.frames), files=files,
    trace=trace.provenance, decode_and_save_ms=decode_ns/1e6, integrity_read_ms=(time.monotonic_ns()-begin-decode_ns)/1e6,
    source_reads='Canonical video decoding before replay; entire required native frame interval cached, without calibration or model features',
    source_extract_sha256=sha(ROOT/'camera_profiles.py'))
  (directory/'paced-cache.json').write_text(json.dumps(report, indent=2)+'\n')
  return report


def model_run(cli, trace):
  """Existing launcher, factory, and binder. No duplicated model initialization."""
  replay, runtime = local('agent-live-replay'), local('agent-live-runtime')
  gate = json.loads(cli.consumer_parity_report.read_text())
  if (gate.get('mode') != 'consumer' or not gate.get('gpu_executed') or not gate.get('predictions_qualified') or
      gate.get('comparison', {}).get('exact') is not True or gate['comparison']['queries'] != 64):
    raise ValueError('A completed exact real-GPU Consumer parity report is required first')
  for name, expected in gate['initialized']['source_sha256'].items():
    path = ROOT.parent/'model.py' if name == 'model.py' else ROOT/name
    if sha(path) != expected:
      raise ValueError(f'Parity-qualified source changed: {name}')
  recipe = SimpleNamespace(samples=gate['protocol']['replay_order'], provenance=gate['protocol']['dataset'])
  normalized, args, protocol = replay.normalize(cli.invocation, recipe, local('benchmark-pipeline'), cli.output.with_suffix('.launcher.json'))
  if protocol != gate['protocol']:
    raise ValueError('Initialization recipe differs from the real-GPU Consumer parity gate')
  if args.verify_vision_warmup:
    raise ValueError('Iteration-limited warmup verification has no paced startup phase; qualify a continuous-verification recipe separately')
  rgb = RGB(cli.raw_frames, trace.frames)
  store = bridge.Store(frame_limit=64, byte_limit=384*1024**2)
  clock = Clock(trace.start)
  producer = Producer(trace, store, rgb, clock)
  warp = bridge.calibrated_image
  if cli.warp == 'native':
    from native_warp import calibrated_image, library
    library()  # Parent initialization only; no feature/model warming.
    warp = calibrated_image
  preparer = bridge.Preparer(trace.manifest['source_camera_geometry'], trace.manifest['target_camera_geometry'], warp=warp, clock=clock)
  nav = Navigation(preparer, trace)
  runner = bridge.Bridge(store, nav, hz=cli.hz, clock=clock, require_small_model=False)
  nav.runner = runner
  output = runtime.Output(cli.output.with_suffix('.jsonl'), bridge.storage_ok)
  output.save(dict(kind='protocol', label=LABEL, trace=trace.provenance, rgb=rgb.inventory, initialization=protocol,
    actual_consumer_parity_report_sha256=sha(cli.consumer_parity_report), active_startup_warming=False,
    requested_hz=cli.hz, source_window_seconds=cli.seconds, inference_budget_seconds=cli.inference_budget_seconds,
    hard_timeout='SIGALRM terminates the child; dispatch records without completion never count as predictions',
    initial_priming='only raw frames and states released at/before source start; no feature warming',
    device_health='not asserted', source_sha256=runtime.source_hashes()))
  invocation = cli.output.with_suffix('.normalized-invocation.json')
  replay.save(invocation, normalized)
  launcher = local('agent-speculation-run')
  initialized = []
  summary = {}
  def launch(path, run_name):
    if Path(path).resolve() != ROOT/'benchmark-pipeline.py' or run_name != '__main__':
      raise ValueError('Unexpected canonical launcher target')
    pipeline = local('benchmark-pipeline')
    effective = pipeline.parse_args()
    begin = time.monotonic_ns()
    sources, source_dir = pipeline.snapshot_sources(effective.output)
    components = pipeline.initialize(effective, sources=sources, source_dir=source_dir)
    initialized.append(components)
    policy = local('agent-speculation-policy') if gate['decoder'] == 'speculative' else None
    consumer = runtime.bind_components(components, replay.binding_args(effective), clock=clock,
      speculation_policy=policy, speculation_draft=policy.CHECKPOINT_DRAFT_IDS if policy else None)
    output.save(dict(kind='initialized', label=LABEL, initialization_ms=(time.monotonic_ns()-begin)/1e6,
                     consumer_provenance=consumer.provenance, source_clock_started=False))
    producer.prime_history()
    output.save(dict(kind='past_only_priming', label=LABEL, **producer.prime))
    output.save(dict(kind='budget_armed', label=LABEL, hard_seconds=cli.inference_budget_seconds))
    collect = None
    if effective.collect_garbage:
      import gc
      collect = lambda completed: gc.collect() if completed % effective.gc_interval == 0 else None
    with hard_budget(cli.inference_budget_seconds):
      clock.start()
      producer.start()
      try:
        summary.update(observe(trace, producer, runner, consumer, components, clock=clock,
                               budget_seconds=cli.inference_budget_seconds, emit=output.save, collect=collect))
      finally:
        producer.close()
    consumer.close()
    output.save(summary)
    return {}
  original_spec = importlib.util.spec_from_file_location
  argv = [str(ROOT/'agent-speculation-run.py'), '--invocation', str(invocation), '--output', str(args.output),
    '--mode', 'candidate' if gate['decoder'] == 'speculative' else 'reference', '--warmup', str(args.warmup),
    '--runs', str(args.runs), '--pad-prompt-to', str(args.pad_prompt_to), '--prefill-attention', gate['prefill_attention']]
  try:
    with patch.dict(os.environ), patch.object(sys, 'argv', argv), patch.object(sys, 'path', list(sys.path)), \
         patch.object(importlib.util, 'spec_from_file_location', original_spec), patch.object(runpy, 'run_path', launch):
      sys.path[:0] = [p for p in (normalized['environment'].get('PYTHONPATH') or '').split(os.pathsep) if p]
      launcher.main()
  finally:
    output.close()
  if len(initialized) != 1:
    raise RuntimeError('Exactly one initialized model owner is required')
  return summary


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=('preflight', 'prepare-cache', 'run'), default='preflight')
  parser.add_argument('--manifest', type=Path, default=SOURCE/'manifest.json')
  parser.add_argument('--logs', type=Path, default=LOGS)
  parser.add_argument('--raw-frames', type=Path, default=RAW)
  parser.add_argument('--cache-output', type=Path)
  parser.add_argument('--start-seconds', type=float, default=35.)
  parser.add_argument('--seconds', type=float, default=8.)
  parser.add_argument('--hz', type=float, choices=(5., 10.), default=5.)
  parser.add_argument('--inference-budget-seconds', type=float, default=180.)
  parser.add_argument('--warp', choices=('native', 'scipy'), default='native')
  parser.add_argument('--invocation', type=Path)
  parser.add_argument('--consumer-parity-report', type=Path)
  parser.add_argument('--output', type=Path)
  cli = parser.parse_args(argv)
  if not 0 < cli.inference_budget_seconds <= 180:
    parser.error('Inference budget must not exceed 180 seconds')
  os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  sys.dont_write_bytecode = True
  trace = Trace(cli.manifest, cli.logs, cli.start_seconds, cli.seconds)
  if cli.mode == 'prepare-cache':
    if cli.cache_output is None:
      parser.error('--prepare-cache requires --cache-output')
    result = prepare_cache(trace, cli.cache_output)
  elif cli.mode == 'run':
    if any(x is None for x in (cli.output, cli.invocation, cli.consumer_parity_report)):
      parser.error('Run requires --output, --invocation, and --consumer-parity-report')
    if any(cli.output.parent.glob(cli.output.stem+'*')):
      raise FileExistsError('Use a new output stem; retain all existing artifacts')
    result = model_run(cli, trace)
  else:
    missing = [str(cli.raw_frames/str(r['view'])/f'{r["video_index"]}.npy') for r in trace.frames
               if not (cli.raw_frames/str(r['view'])/f'{r["video_index"]}.npy').is_file()]
    result = dict(label=LABEL, model_executed=False, trace=trace.provenance, missing_native_rgb_files=missing,
                  frame_metadata_sha256=trace.frame_digest, native_receipt_timing_available=False)
  if cli.output is not None:
    with cli.output.open('x') as stream:
      json.dump(result, stream, indent=2, allow_nan=False)
      stream.write('\n')
  print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'rows', 'missing_native_rgb_files')}, indent=2))


if __name__ == '__main__':
  main()

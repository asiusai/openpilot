"""Deferred native shadow runtime using the benchmark's single initializer.

Importing this module opens no device or connection. The CLI is an explicit local
device operation, not a deployment command. It publishes no messages or controls.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), Path(__file__).with_name(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def source_hashes():
  root = Path(__file__).parent
  return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [*root.glob('*.py'), *root.glob('*.c'), root.parent/'model.py']}


def parse_pipeline_options(pipeline, options, output):
  """Reuse every model flag. Replay arguments are explicit unused parser markers."""
  reserved = {'--samples', '--output', '--runs', '--warmup', '--native-manifest', '--native-calibrations',
              '--native-frame-cache', '--validation-reference'}
  if not isinstance(options, list) or not all(isinstance(value, str) for value in options):
    raise ValueError('Pipeline options must be a JSON array of argument strings')
  if any(value.split('=', 1)[0] in reserved for value in options):
    raise ValueError('Pipeline options must contain model options only, without replay arguments')
  args = pipeline.parse_args([*options, '--samples', '<in-memory-live-arrays>', '--output', str(output), '--warmup', '0', '--runs', '0'])
  validate_live_args(args)
  return args


def validate_live_args(args):
  # Do not silently ignore flags which change verification, warming, or timing.
  excluded = ('warmup_graphs', 'warm_current_shapes', 'freeze_warmup_gc', 'verify_vision_warmup',
              'pipeline_inputs', 'late_inputs', 'mmap_inputs', 'profile_kernels', 'profile_host',
              'capture_prefill_scales', 'capture_vision_scales', 'gpu_metrics', 'validation_reference',
              'native_manifest', 'native_calibrations', 'native_frame_cache')
  if any(getattr(args, field) for field in excluded) or args.warmup or args.runs:
    raise ValueError('Live runtime excludes replay, overlap, startup warming, and profiling flags; select them explicitly elsewhere')
  required = ('allow_missing_cameras', 'compact_inputs', 'byte_pixels', 'selective_pixels', 'vision_cache', 'cache_resize', 'direct_byte_inputs')
  if not all(getattr(args, field) for field in required) or args.camera_profile != 'nvidia':
    raise ValueError('Use the established calibrated two-front byte-input/cache configuration')
  if (args.steps != 3 or args.trajectory_solver != 'euler' or args.vision_batch != 1 or not args.pad_prompt_to or
      args.pixel_scale != 1 or args.camera_pixel_scales is not None or args.front_history_indices is not None or
      args.history_pixel_scale != 1 or args.history_feature_scale != 1 or args.narrow_crop != 1 or args.wide_equidistant_focal):
    raise ValueError('Preserve full histories, full pixel/feature grids, fixed padding, and three Euler expert steps')
  if args.initial_noise is None or args.prefill_calibration is None or not args.packed_quantize:
    raise ValueError('Declare the selected existing noise and static prefill calibration')


def bind_components(components, args, *, consumer_module=None, speculation_policy=None, speculation_draft=None,
                    clock=time.monotonic_ns):
  """Bind factory output, including already installed candidate hooks, exactly once."""
  validate_live_args(args)
  if getattr(components, '_live_bound', False):
    raise RuntimeError('Initialized components already have a live owner')
  if any(components.startup_active) or (components.startup is not None and components.startup.active()):
    raise RuntimeError('Active startup warming cannot be handed to the live consumer')
  decoder = getattr(components.runner, '__self__', None)
  if decoder is None or (components.decoder is not None and decoder is not components.decoder):
    raise ValueError('Factory runner does not belong to its prepared decoder')
  if components.config is not components.adapter.config or components.tokenizer is not components.adapter.tokenizer:
    raise ValueError('Factory adapter configuration/tokenizer identity changed')
  if components.vision_cache is None or components.vision_cache.verify != args.verify_vision_cache:
    raise ValueError('Prepared cache verification differs from the requested verification mode')
  if components.pad_inputs is None or components.settings.diffusion_steps != args.steps:
    raise ValueError('Incomplete or incompatible prepared components')
  expected = dict(max_tokens=args.max_tokens, diffusion_steps=args.steps, vision_batch=args.vision_batch, greedy=True,
                  byte_pixels=args.byte_pixels, cache_rotary_inputs=args.cache_rotary_inputs, coarse_timing=args.coarse_timing)
  if any(getattr(components.settings, name, None) != value for name, value in expected.items()):
    raise ValueError('Prepared settings differ from the selected model options')
  if getattr(components, 'profiler', None) is not None:
    raise ValueError('A profiled runtime cannot report live throughput')
  implementation = consumer_module or local('agent-live-consumer')
  provenance = dict(factory=components.provenance, checkpoint=components.vision_provenance,
                    scheduling='serial live preparation; zero queued queries; no startup warming')
  consumer = implementation.Consumer(adapter=components.adapter, model=components.model, runner=components.runner,
    settings=components.settings, config=components.config, input_class=components.input_class,
    vision_cache=components.vision_cache, pad_inputs=components.pad_inputs, pad_length=args.pad_prompt_to,
    initializer_provenance=provenance, decoder=decoder, tokenizer=components.tokenizer, inputs=components.inputs,
    speculation_policy=speculation_policy, speculation_draft=speculation_draft,
    startup_helpers=() if components.startup is None else (components.startup,), clock=clock)
  components._live_bound = True
  return consumer


def initialize_consumer(pipeline, args, *, sources=None, speculation_policy=None, speculation_draft=None,
                        consumer_module=None, clock=time.monotonic_ns):
  validate_live_args(args)  # Reject unsupported modes before model allocation.
  components = pipeline.initialize(args, sources=source_hashes() if sources is None else sources,
                                   source_dir='<live source hashes; no snapshot files>')
  consumer = bind_components(components, args, consumer_module=consumer_module, speculation_policy=speculation_policy,
                             speculation_draft=speculation_draft, clock=clock)
  return components, consumer


class Output:
  """New bounded JSONL only. Storage failures never delete old records."""
  limit = 64*1024**2

  def __init__(self, path, storage_ok):
    self.path, self.storage_ok, self.written = path, storage_ok, 0
    usage = shutil.disk_usage(path.parent)
    if not storage_ok(usage.total, usage.free, self.limit):
      raise RuntimeError('Output would cross the 10% or 5 GiB reserve')
    self.stream = path.open('x')

  def save(self, value):
    data = json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n'
    size = len(data.encode())
    usage = shutil.disk_usage(self.path.parent)
    if self.written+size > self.limit or not self.storage_ok(usage.total, usage.free, size):
      raise RuntimeError('Output limit or storage reserve reached')
    self.stream.write(data)
    self.stream.flush()
    self.written += size

  def close(self):
    self.stream.close()


class Guard:
  """Monitor the existing small model while initialization or inference blocks.

  A failure stops admission of further work. It cannot preempt an in-flight GPU
  call; the selected backend timeout and native process recovery remain separate.
  """
  def __init__(self, capture, producer, args, *, clock=time.monotonic_ns):
    self.capture, self.producer, self.args, self.clock = capture, producer, args, clock
    self.identity = capture.verify_small_model(args.model_pid, args.manager_pid)
    self.stop = threading.Event()
    self.thread, self.error = None, None
    self.initializing = True
    self.storage_paths = {args.output.parent}
    for key in ('ALPAMAYO_FP8_CACHE', 'CACHEDB'):
      if os.environ.get(key):
        path = Path(os.environ[key])
        self.storage_paths.add(path if path.is_dir() else path.parent)

  def health_ready(self):
    now, health = self.clock(), self.producer.health
    return (health is not None and health['model_alive'] and health['device_valid'] and
            0 <= health['model_age_ns'] < 250_000_000 and 0 <= now-health['time_ns'] < 500_000_000 and
            health['thermal_status'] not in ('red', 'danger'))

  def resources(self, *, require_health=True):
    if self.producer.error is not None:
      raise RuntimeError('Native producer failed') from self.producer.error
    self.capture.verify_small_model(self.args.model_pid, self.args.manager_pid, self.identity)
    if require_health and not self.health_ready():
      raise RuntimeError('Small-model or device health guard failed')
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(mem['MemAvailable'].split()[0]) < 768*1024:
      raise RuntimeError('Host memory reserve reached')
    reserve = self.args.initialization_reserve_mib*1024**2 if self.initializing else 0
    for path in self.storage_paths:
      usage = shutil.disk_usage(path)
      if not self.capture.bridge.storage_ok(usage.total, usage.free, reserve):
        raise RuntimeError('Storage reserve reached during model initialization or inference')

  def check(self):
    if self.error is not None:
      raise RuntimeError('Background resource guard failed') from self.error
    self.resources()

  def start(self):
    def monitor():
      try:
        while not self.stop.wait(.25):
          self.resources()
      except Exception as error:
        self.error = error
        self.producer.stop.set()
        self.stop.set()
    self.check()
    self.thread = threading.Thread(target=monitor, daemon=True)
    self.thread.start()

  def close(self):
    self.stop.set()
    if self.thread is not None:
      self.thread.join(timeout=2)
      if self.thread.is_alive():
        raise RuntimeError('Resource monitor did not stop')


def observe(bridge, consumer, producer, guard, output, seconds, *, clock=time.monotonic, collect=None):
  started, completed = clock(), 0
  while clock()-started < seconds:
    guard.check()
    row = bridge.run_once(consumer)
    guard.check()
    if row is not None:
      row['runtime_guard_passed'] = True
      output.save(row)
      completed += 1
      if collect is not None:
        collect(completed)
    producer.stop.wait(.002)
  return dict(kind='summary', **bridge.summary(), limits=bridge.store.bounds(),
              observation_seconds=clock()-started, initialization_excluded=True, cold_graph_stalls_included=True)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--live-model', action='store_true', required=True)
  parser.add_argument('--serial', required=True)
  parser.add_argument('--model-pid', type=int, required=True)
  parser.add_argument('--manager-pid', type=int, required=True)
  parser.add_argument('--geometry-manifest', type=Path, required=True)
  parser.add_argument('--pipeline-options', type=Path, required=True, help='JSON model argument list; no replay/scheduling arguments')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--seconds', type=float, default=60)
  parser.add_argument('--hz', type=float, choices=(5., 10.), default=5.)
  parser.add_argument('--warp', choices=('native', 'scipy'), default='native')
  parser.add_argument('--navigation-text', required=True)
  parser.add_argument('--initialization-reserve-mib', type=int, default=1024)
  args = parser.parse_args(argv)
  if os.getuid() == 0 or not 0 < args.seconds <= 600 or args.initialization_reserve_mib < 64:
    parser.error('Run as comma for at most 600 seconds, with at least 64 MiB initialization reserve')
  if not args.navigation_text.strip() or len(args.navigation_text) > 1024:
    parser.error('Explicit navigation must contain 1-1024 characters')
  if Path('/sys/devices/soc0/serial_number').read_text().strip() != args.serial:
    raise RuntimeError('Wrong SoC serial')
  pipeline = local('benchmark-pipeline')
  model_args = parse_pipeline_options(pipeline, json.loads(args.pipeline_options.read_text()), args.output)
  manifest = json.loads(args.geometry_manifest.read_text())
  if manifest.get('camera_input_profile') != 'calibrated_vehicle_history' or manifest.get('camera_subset') != [1, 6]:
    raise ValueError('Use the calibrated two-front geometry manifest')
  capture = local('agent-live-capture')
  store = capture.bridge.Store()
  producer = capture.NativeProducer(store)
  guard = Guard(capture, producer, args)
  guard.resources(require_health=False)
  os.umask(0o077)
  output = Output(args.output, capture.bridge.storage_ok)
  consumer = None
  error = None
  try:
    producer.start()
    deadline = time.monotonic()+5
    while not guard.health_ready() and time.monotonic() < deadline:
      guard.resources(require_health=False)
      producer.stop.wait(.02)
    guard.start()
    warp = capture.bridge.calibrated_image
    if args.warp == 'native':
      from native_warp import calibrated_image as warp, library
      library()
    prep = capture.bridge.Preparer(manifest['source_camera_geometry'], manifest['target_camera_geometry'], warp=warp)
    bridge = capture.bridge.Bridge(store, prep, hz=args.hz, navigation=args.navigation_text, require_small_model=True)
    output.save(dict(kind='manifest', model_predictions=True, deployed=False, control_publishers=False,
      geometry_manifest_sha256=hashlib.sha256(args.geometry_manifest.read_bytes()).hexdigest(),
      pipeline_options_sha256=hashlib.sha256(args.pipeline_options.read_bytes()).hexdigest(),
      serial=args.serial, normal_model_pid=args.model_pid, source_sof='v0 EOF minus sensor readout estimate',
      camera_hz='native; no temporal decimation', source_sha256=source_hashes(),
      scope='native inputs and causal history through completed decoded trajectory', startup_warming=False))
    init_started = time.monotonic()
    components, consumer = initialize_consumer(pipeline, model_args)
    guard.check()
    guard.initializing = False
    output.save(dict(kind='initialized', seconds=time.monotonic()-init_started, provenance=consumer.provenance))
    collect = None
    if model_args.collect_garbage:
      import gc
      collect = lambda completed: gc.collect() if completed % model_args.gc_interval == 0 else None
    output.save(observe(bridge, consumer, producer, guard, output, args.seconds, collect=collect))
  except BaseException as exc:
    error = exc
    raise
  finally:
    # Preserve the primary failure; do not mask it with a cleanup/storage error.
    failures = []
    for cleanup in (guard.close, producer.close, *(() if consumer is None else (consumer.close,)), output.close):
      try:
        cleanup()
      except BaseException as exc:
        failures.append(exc)
    if error is not None:
      for failure in failures:
        error.add_note(f'Cleanup also failed: {type(failure).__name__}: {failure}')
    elif failures:
      raise failures[0]


if __name__ == '__main__':
  main()

"""Offline qualification of the original benchmark versus an injected live Consumer.

Default: CPU preflight only. --execute-gpu is a separate parent operation.
No model implementation, weight loader, native receiver, or publisher lives here.
Recorded capture times are provenance; the replay clock is deliberately synthetic.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SOURCE = Path('/mnt/nvme-256/footage/f9ad9a96/exports/0000008e--321b2ddc03/alpamayo-4hz-calibrated-two-six')
SAMPLES_SHA256 = '9f93d391d31c134b2106f401812e6a61ad95db0a17ef3c5b7ee0abdaf495b440'
MANIFEST_SHA256 = '5100176f7e48eaeeaf314e4295eccaabbde5dc1674028e8fcbe2faaf746c24af'
DISABLED = ('--pipeline-inputs', '--late-inputs', '--warmup-graphs', '--warm-current-shapes', '--freeze-warmup-gc')
ARRAY_KEYS = {'image_frames', 'camera_indices', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot', 'nav_text'}
TIMING = 'Offline instrumented parity validation; synthetic protocol clock; no native acquisition latency or throughput claim'


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def sha(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def tree(value):
  """Read-only fingerprints, including actual CPU tensors and their exact bytes."""
  import torch
  if isinstance(value, torch.Tensor):
    if value.device.type != 'cpu':
      raise ValueError('Audit must not read additional GPU tensors')
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
  if isinstance(value, np.ndarray):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())
  if isinstance(value, dict):
    return {str(k): tree(v) for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))}
  if isinstance(value, (tuple, list)):
    return [tree(v) for v in value]
  if isinstance(value, bytes):
    return dict(bytes_sha256=hashlib.sha256(value).hexdigest())
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if value is None or isinstance(value, (str, bool, int, float)):
    return value
  raise TypeError(f'Unsupported audit object: {type(value).__name__}')


def digest(value):
  return hashlib.sha256(json.dumps(tree(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
  with Path(path).open('x') as stream:
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write('\n')


class ReplayClock:
  """Protocol clock only. Values must never be presented as measured input age."""
  def __init__(self):
    self.now = 0

  def __call__(self):
    self.now += 1_000
    return self.now

  def metadata(self, row):
    q = row['query_ns']  # Never infer nanoseconds from a filename or rounded array.
    if q <= self.now:
      raise ValueError('Recorded queries must strictly advance the replay clock')
    self.now = q
    return dict(query_ns=q, selected_ns=q, prepared_ns=q, frames=copy.deepcopy(row['frames']),
                navigation_text=row['navigation_text'], clock_kind='synthetic offline protocol clock',
                native_acquisition_age_ms=None)


class Dataset:
  """The pinned 64 original snapshots; one array set at a time, no ring replacement."""
  def __init__(self, samples_path, manifest_path):
    if sha(samples_path) != SAMPLES_SHA256 or sha(manifest_path) != MANIFEST_SHA256:
      raise ValueError('Use the exact original quality list and SOURCE calibrated-two-camera manifest')
    self.samples = json.loads(Path(samples_path).read_text())
    if len(self.samples) != 64 or len(set(self.samples)) != 64:
      raise ValueError('Exactly 64 distinct original samples are required')
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['camera_input_profile'] != 'calibrated_vehicle_history' or manifest['camera_subset'] != [1, 6]:
      raise ValueError('Wrong camera/history profile')
    selected = set(self.samples)
    by_path = {row['sample']: row for row in manifest['schedule'] if row.get('sample') in selected}
    if set(by_path) != selected:
      raise ValueError('Missing authoritative native query metadata')
    self.rows = []
    for sample in self.samples:
      source = by_path[sample]
      # Exclude recorded future motion and small-model predictions from inputs.
      self.rows.append({key: copy.deepcopy(source[key]) for key in (
        'sample', 'query_ns', 'frames', 'navigation_text', 'ego_history_xyz', 'ego_history_rot', 'vehicle_history_audit')})
    calibration_path = Path(manifest_path).with_name('render-calibrations.json')
    calibrations = json.loads(calibration_path.read_text())
    calibration_records = (np.asarray([r['ns'] for r in calibrations], np.int64), calibrations)
    from calibration_trials import asof
    from camera_profiles import replay_module
    export = Path(manifest['export'])
    _, cameras = replay_module().camera_records(export)
    native = {c: (np.asarray([r['capture_ns'] for r in cameras[c]], np.int64), cameras[c]) for c in (1, 6)}
    self.inputs = []
    live = local('agent-live-consumer')
    previous = 0
    for row in self.rows:
      self.validate_native(row, native, calibration_records, asof)
      arrays = self.load(row)
      meta = ReplayClock().metadata(row)
      live.Consumer._data(SimpleNamespace(last_query_ns=previous), arrays, meta, row['query_ns'])
      previous = row['query_ns']
      for key in ('ego_history_xyz', 'ego_history_rot'):
        expected = np.asarray(row[key], dtype=np.float32)
        if arrays[key].shape != expected.shape or arrays[key].tobytes() != expected.tobytes():
          raise ValueError(f'Original calibrated history differs from manifest: {key}')
      self.inputs.append(tree(arrays))
      del arrays
    self.provenance = dict(samples_sha256=sha(samples_path), manifest_sha256=sha(manifest_path),
      calibration_sha256=sha(calibration_path), native_timestamps_sha256=sha(export/'clips/native-timestamps.json'),
      camera_geometry_sha256=digest((manifest['source_camera_geometry'], manifest['target_camera_geometry'])),
      inputs_sha256=digest(self.inputs), metadata_sha256=digest(self.rows), count=64,
      history='five complete full-resolution frames per front camera; canonical causal floor; calibrated learned vehicle history',
      native_acquisition_age_ms=None, live_state_age_validation=False,
      limits='Recorded learned-history audit retained; no live CAN replay, camera acquisition, or image re-warp is timed')

  @staticmethod
  def validate_native(row, native, calibrations, asof):
    if not isinstance(row['query_ns'], int) or len(row['frames']) != 10:
      raise ValueError('Missing native query/frame metadata')
    for ref in row['frames']:
      records = native[ref['view']]
      native_row, capture = asof(records, ref['requested_ns'], 150_000_000)
      _, calibration = asof(calibrations, capture, 1_000_000_000)
      if (capture != ref['capture_ns'] or native_row['frame_id'] != ref['frame_id'] or
          records[1][ref['video_index']] is not native_row or calibration != ref['image_calibration_ns']):
        raise ValueError('Native frame/calibration is not the canonical causal floor')
    audit = row['vehicle_history_audit']
    if not 0 <= audit['max_car_state_age_ms'] <= 150 or not 0 <= audit['max_learned_parameters_age_ms'] <= 1000:
      raise ValueError('Recorded vehicle-history state age exceeds the canonical limits')

  @staticmethod
  def load(row):
    files = {p.stem: p for p in Path(row['sample']).glob('*.npy')}
    if set(files) != ARRAY_KEYS:
      raise ValueError('Require the exact six original sample arrays')
    return {key: [str(np.load(path, allow_pickle=False).item())] if key == 'nav_text'
            else np.load(path, allow_pickle=False, mmap_mode='c') for key, path in files.items()}


def binding_args(args):
  """Only replay loader/iteration metadata differs from the existing live binder."""
  result = copy.copy(args)
  result.warmup = result.runs = 0
  result.mmap_inputs = False  # Replay file loading remains outside Consumer.
  result.verify_vision_warmup = False  # Implement its original per-query schedule below.
  return result


def normalize(invocation, dataset, pipeline, output):
  recorded = json.loads(Path(invocation).read_text())
  command = recorded['command']
  if len(command) < 2 or Path(command[1]).resolve() != ROOT/'benchmark-pipeline.py':
    raise ValueError('Invocation must select this adjacent current benchmark factory')
  before = pipeline.parse_args(command[2:])
  if before.samples != dataset.samples or before.warmup < 0 or before.runs < 1 or before.warmup+before.runs != 64:
    raise ValueError('Preserve the exact original 64 samples and one ordered pass, including warmup labels')
  options = [v for v in command[2:] if v not in DISABLED]
  differences = [dict(flag=flag, before=True, after=False, reason='serial parity; no active startup warming')
                 for flag in DISABLED if flag in command]
  options[options.index('--output')+1] = str(output)
  args = pipeline.parse_args(options)
  local('agent-live-runtime').validate_live_args(binding_args(args))
  if not args.lookup_draft or args.vision_calibration is None or args.reduced_vocab is None:
    raise ValueError('Preserve the selected lookup runner, reduced vocabulary, and fixed vision calibration')
  if recorded['environment'].get('PROFILE') not in (None, '0'):
    raise ValueError('Existing launcher forces PROFILE=0; supply an unprofiled invocation explicitly')
  # Check exactly which parser values changed. Output location is provenance only.
  changed = {key for key in vars(args) if getattr(before, key) != getattr(args, key)}-{'output'}
  if changed-set(flag[2:].replace('-', '_') for flag in DISABLED):
    raise AssertionError('Normalization changed inference settings')
  normalized = dict(recorded, command=[command[0], command[1], *options])
  files = {key: dict(path=str(getattr(args, key)), sha256=sha(getattr(args, key)))
           for key in ('initial_noise', 'prefill_calibration', 'vision_calibration', 'reduced_vocab')}
  checkpoint = Path(args.checkpoint)
  files['checkpoint_metadata'] = {p.name: sha(p) for p in checkpoint.iterdir() if p.is_file() and p.suffix == '.json'}
  protocol = dict(dataset=dataset.provenance, flags=tree({k: v for k, v in vars(args).items() if k != 'output'}),
                  environment=recorded['environment'], files=files, disabled_flags=differences,
                  replay_order=dataset.samples, seed=0)
  return normalized, args, protocol


def history(decoder):
  return tree({name: getattr(decoder, name, None) for name in ('history', 'prefill_draft_old_history')})


def cpu_buffers(inputs):
  result = []
  for value in inputs.cpu:
    dtype = value.dtype.name if isinstance(value.dtype, np.dtype) else np.dtype(value.dtype.fmt).name
    data = value._buffer().as_memoryview(allow_zero_copy=True)
    result.append(dict(shape=list(value.shape), dtype=dtype, sha256=hashlib.sha256(data).hexdigest()))
  return result


class Audit:
  """Read-only runner observation. Finalize only after successful completion sync."""
  def __init__(self, dataset):
    self.dataset, self.rows, self.pending = dataset, [], None
    self.prepared_index = -1

  def install(self, components):
    self.components = components
    decoder = components.runner.__self__
    original_prepare, original_run = components.adapter.prepare, components.runner
    owner = self
    def prepare(data):
      index = owner.prepared_index+1
      arrays = {k: list(v) if k == 'nav_text' else v.numpy() for k, v in data.items()}
      if index >= len(owner.dataset.rows) or tree(arrays) != owner.dataset.inputs[index]:
        raise ValueError('Current input arrays differ from the exact original ordered snapshots')
      if owner.pending is not None or index != len(owner.rows):
        raise RuntimeError('Previous replay query has no successful completion barrier')
      owner.prepared_index = index
      return original_prepare(data)
    def run(bound, model, settings, config, tokenizer, prepared, packed, **kwargs):
      if bound is not decoder or owner.pending is not None:
        raise RuntimeError('Unexpected runner owner or duplicate inference')
      index = owner.prepared_index
      if index != len(owner.rows):
        raise RuntimeError('Runner has no current prepared query')
      inputs = kwargs['inputs']
      owner.pending = dict(index=index, sample=owner.dataset.samples[index], query_ns=owner.dataset.rows[index]['query_ns'],
        completed=False, prepared=tree(prepared), packed=tree(packed), cpu_buffers=cpu_buffers(inputs),
        input_signature=tree(inputs._signature), noise=tree(settings.initial_noise), seed=kwargs['seed'],
        vision_keys=tree(components.vision_cache.keys), history_before=history(decoder))
      prediction = original_run(model, settings, config, tokenizer, prepared, packed, **kwargs)
      owner.pending.update(prediction=copy.deepcopy(prediction), history_after=history(decoder),
        decoder_statistics=copy.deepcopy(decoder.statistics[-1]),
        speculation=copy.deepcopy(decoder.prefill_draft_records[-1]) if hasattr(decoder, 'prefill_draft_records') else None,
        uploaded_pixel_bytes=inputs.uploaded_pixel_bytes)
      return prediction  # Canonical object and fields are untouched.
    components.adapter.prepare = prepare
    decoder.run = MethodType(run, decoder)
    components.runner = decoder.run

  def complete(self, *, consumer_record=None):
    row = self.pending
    if row is None or 'prediction' not in row:
      raise RuntimeError('No completed runner output')
    if consumer_record is not None:
      if not consumer_record['completed'] or consumer_record['query_ns'] != row['query_ns']:
        raise RuntimeError('Consumer completion failed or belongs to another query')
      # Deliberately omit synthetic age/duration values from the qualification report.
      row['consumer_protocol'] = {key: copy.deepcopy(consumer_record[key]) for key in (
        'input_reused', 'input_signature_sha256', 'upload_array_bytes', 'source_capture_timestamps_ns',
        'actual_prompt_tokens', 'padded_prompt_tokens', 'cache_entries', 'owned_input_objects')}
    row['completed'] = True
    self.rows.append(row)
    self.pending = None


def replay_consumer(components, args, dataset, audit, *, consumer, clock):
  """Explicit injected Consumer; sample files never enter Consumer.predict."""
  for index, row in enumerate(dataset.rows):
    if args.collect_garbage and index % args.gc_interval == 0:
      gc.collect()
    arrays = dataset.load(row)
    metadata = clock.metadata(row)
    components.vision_cache.verify = args.verify_vision_cache or (args.verify_vision_warmup and index < max(1, args.warmup-4))
    output = consumer.predict(arrays, metadata)
    consumer.synchronize()  # An exception leaves no accepted audit result.
    if output['query_ns'] != row['query_ns']:
      raise ValueError('Consumer returned a stale plan')
    audit.complete(consumer_record=output['live_consumer'])
    del arrays, output, metadata
  consumer.close()


def execute_existing_launcher(invocation, args, *, mode, decoder_mode, dataset, prefill_attention='baseline'):
  """Keep the canonical launcher's tokenizer validation and paired import hooks."""
  launcher = local('agent-speculation-run')
  audit = Audit(dataset)
  result = {}
  original_spec = importlib.util.spec_from_file_location
  def run_pipeline(path, run_name):
    if Path(path).resolve() != ROOT/'benchmark-pipeline.py' or run_name != '__main__':
      raise ValueError('Unexpected launcher target')
    pipeline = local('benchmark-pipeline')  # Import while actual optional hooks are installed.
    original_initialize = pipeline.initialize
    def initialize(*values, **kwargs):
      components = original_initialize(*values, **kwargs)
      if components.startup is not None or any(components.startup_active):
        raise ValueError('Active startup warming is forbidden for this qualification')
      audit.install(components)
      result['initialized'] = dict(settings=tree(vars(components.settings)), config_sha256=digest(components.config.to_dict()),
        source_sha256=components.report['source_sha256'], checkpoint=tree(components.vision_provenance),
        runtime_environment=components.report['runtime_environment'])
      return components
    pipeline.initialize = initialize
    if mode == 'reference':
      # The ORIGINAL adjacent benchmark owns every replay/decoder invocation.
      def initialize_reference(*values, **kwargs):
        components = initialize(*values, **kwargs)
        audited = components.runner
        def completed(bound, *a, **kw):
          prediction = audited(*a, **kw)
          components.model.synchronize()
          audit.complete()
          return prediction
        components.runner = MethodType(completed, components.decoder)
        return components
      pipeline.initialize = initialize_reference
      pipeline.main()
    else:
      effective = pipeline.parse_args()
      sources, source_dir = pipeline.snapshot_sources(effective.output)
      components = initialize(effective, sources=sources, source_dir=source_dir)
      clock = ReplayClock()
      policy = local('agent-speculation-policy') if decoder_mode == 'speculative' else None
      # Launcher already validated tokenizer-derived IDs and installed both hooks.
      draft = policy.CHECKPOINT_DRAFT_IDS if policy is not None else None
      consumer = local('agent-live-runtime').bind_components(components, binding_args(effective),
        consumer_module=local('agent-live-consumer'), speculation_policy=policy, speculation_draft=draft, clock=clock)
      result['consumer_provenance'] = consumer.provenance
      replay_consumer(components, effective, dataset, audit, consumer=consumer, clock=clock)
    return {}
  launch_args = [str(ROOT/'agent-speculation-run.py'), '--invocation', str(invocation), '--output', str(args.output),
    '--mode', 'candidate' if decoder_mode == 'speculative' else 'reference', '--warmup', str(args.warmup),
    '--runs', str(args.runs), '--pad-prompt-to', str(args.pad_prompt_to), '--prefill-attention', prefill_attention]
  # The canonical launcher mutates these globals; confine them to this one run.
  with patch.dict(os.environ), patch.object(sys, 'argv', launch_args), patch.object(sys, 'path', list(sys.path)), \
       patch.object(importlib.util, 'spec_from_file_location', original_spec), patch.object(runpy, 'run_path', run_pipeline):
    saved = json.loads(Path(invocation).read_text())
    sys.path[:0] = [p for p in (saved['environment'].get('PYTHONPATH') or '').split(os.pathsep) if p]
    launcher.main()
  if audit.pending is not None or len(audit.rows) != 64:
    raise RuntimeError('Qualification requires 64 successfully synchronized predictions')
  result['rows'] = audit.rows
  return result


def compare(reference, candidate):
  if reference['mode'] != 'reference' or candidate['mode'] != 'consumer':
    raise ValueError('Compare the adjacent benchmark reference with its live Consumer')
  for key in ('protocol', 'decoder', 'prefill_attention', 'initialized'):
    if reference[key] != candidate[key]:
      raise ValueError(f'Parity provenance differs: {key}')
  if len(reference['rows']) != 64 or len(candidate['rows']) != 64:
    raise ValueError('Parity requires every original query, including the warmup labels')
  differences = []
  for index, (before, after) in enumerate(zip(reference['rows'], candidate['rows'], strict=True)):
    if not before['completed'] or not after['completed']:
      raise ValueError('An unsynchronized output cannot qualify')
    keys = set(before)-{'consumer_protocol'}
    if keys != set(after)-{'consumer_protocol'}:
      raise ValueError('Audit schemas differ')
    changed = []
    for key in sorted(keys):
      a, b = before[key], after[key]
      if key == 'prediction':
        # Compare every canonical output except measured timing values.
        a, b = ({k: v for k, v in p.items() if k != 'timings_ms'} for p in (a, b))
        for p in (a, b):
          if not np.isfinite(p['xyz']).all() or not np.isfinite(p['rot']).all():
            raise ValueError('Nonfinite trajectory')
      if digest(a) != digest(b):
        changed.append(key)
    if changed:
      differences.append(dict(index=index, query_ns=after['query_ns'], fields=changed))
  return dict(exact=not differences, queries=64, differences=differences,
              scope='all canonical non-timing prediction fields, input buffers, history, statistics, and acceptance/fallback records')


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--invocation', type=Path, required=True)
  parser.add_argument('--mode', choices=('reference', 'consumer'), required=True)
  parser.add_argument('--decoder', choices=('ordinary', 'speculative'), default='ordinary')
  parser.add_argument('--prefill-attention', choices=('baseline', 'shuffle-fma'), default='baseline')
  parser.add_argument('--samples-json', type=Path, default=WORK/'quality-samples.json')
  parser.add_argument('--source-manifest', type=Path, default=SOURCE/'manifest.json')
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--compare-reference', type=Path)
  parser.add_argument('--execute-gpu', action='store_true', help='Parent-only explicit GPU execution; otherwise CPU preflight')
  cli = parser.parse_args(argv)
  os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  sys.dont_write_bytecode = True
  sys.path.insert(0, str(ROOT))
  dataset = Dataset(cli.samples_json, cli.source_manifest)
  raw_output = cli.output.with_suffix('.benchmark.json')
  normalized, args, protocol = normalize(cli.invocation, dataset, local('benchmark-pipeline'), raw_output)
  paths = [cli.output, cli.output.with_suffix('.normalized-invocation.json'), raw_output,
           raw_output.with_suffix('.speculation.json'), raw_output.with_suffix('.jsonl'),
           raw_output.parent/(raw_output.stem+'-sources')]
  if any(p.exists() for p in paths):
    raise FileExistsError('Use new output paths; existing qualification artifacts are immutable')
  cli.output.parent.mkdir(parents=True, exist_ok=True)
  usage = shutil.disk_usage(cli.output.parent)
  if usage.free-64*1024**2 < max(usage.total*.1, 5*1024**3):
    raise RuntimeError('Output would cross the 10% or 5 GiB storage reserve')
  report = dict(schema=1, mode=cli.mode, decoder=cli.decoder, prefill_attention=cli.prefill_attention,
    timing_kind=TIMING, timings_valid_for_throughput=False, live_capture_age_ms=None, protocol=protocol,
    source_sha256=local('agent-live-runtime').source_hashes(), protocol_sha256=digest(protocol),
    original_invocation_sha256=sha(cli.invocation), gpu_executed=False, predictions_qualified=False,
    native_device_operations=False, publishers=False)
  if cli.execute_gpu:
    invocation = cli.output.with_suffix('.normalized-invocation.json')
    save(invocation, normalized)
    report.update(execute_existing_launcher(invocation, args, mode=cli.mode, decoder_mode=cli.decoder,
                                           dataset=dataset, prefill_attention=cli.prefill_attention))
    report['gpu_executed'] = True
    if cli.compare_reference:
      report['comparison'] = compare(json.loads(cli.compare_reference.read_text()), report)
      report['predictions_qualified'] = report['comparison']['exact']
  elif cli.compare_reference:
    raise ValueError('CPU preflight cannot compare uncomputed GPU predictions')
  save(cli.output, report)
  print(json.dumps(dict(output=str(cli.output), gpu_executed=report['gpu_executed'],
                        predictions_qualified=report['predictions_qualified'], queries=64, timing_kind=TIMING)))
  if report.get('comparison', {}).get('exact') is False:
    raise SystemExit('Exact Consumer parity failed; inspect the comparison before any live use')


if __name__ == '__main__':
  main()

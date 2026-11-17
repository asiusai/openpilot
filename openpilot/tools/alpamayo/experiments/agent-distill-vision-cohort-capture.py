"""CPU preflight by default. Parent-only, bounded capture of all original vision groups."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
ROOT = WORK/'agent-distill-vision-cohort-artifacts'
COVERAGE = WORK/'agent-distill-coverage-artifacts/metadata-v1/agent-distill-coverage-manifest.json'
COVERAGE_SHA = '965fa3875e5b26e1138a05ae08c64ab9c0b64cd982f6e25d4111ee847a53ee78'
PLAN = WORK/'agent-distill-command-artifacts/capture-v1/agent-distill-command-capture-plan.json'
PLAN_SHA = '5cb4a9baa5c448f507b64b250686d8cae16e43405a7d990b2c48c5212bf9871c'
SOURCES = {
  'agent-distill-joint-parent.py': '9782636463911ab41d142c9c473d4a5f0422dc1000fb5e6e7ea331e0ec1977ef',
  'agent-distill-joint-runtime.py': '4f098eb5ddda5ae4c5cfe0b9e01956f5dc05177aa4994fb9f8c83b86dcec939e',
  'agent-distill-command-capture.py': '6b42a39b7ed3c615dda36176b4d27cec5bfd69ef5d61a6f490a91b5f1cf35c39',
  'agent-distill-command-common.py': 'f134a165d882d177a9671e1154f494ee5c788a7d5de10b85c1a3467c76598210',
  'agent-distill-vision-runtime-hook.py': '6cd53a8acdcb0c97a87abdf609ff526e42612ee1c5a3116789dacb88e31fce27'}
LEGACY = {
  'train-command-0': (WORK/'agent-distill-joint-teacher-v1/report.json', '5b796b1490bf234cf874a007be575612cb03599d61ba4cde041302e606b4052c'),
  'validation-command-0': (WORK/'agent-distill-joint-validation-v1/report.json', 'fdfc6a47355e9f164bc63df1ec240062931be43a0e05a56794593e6ade234d7c')}
GROUPS = tuple(tuple(range(i, min(i+4, 27))) for i in range(0, 27, 4))
SETTINGS = ('prefill_calibration_sha256', 'vision_calibration_sha256', 'initial_noise_sha256', 'settings', 'runtime_environment')
EXCLUDED = ('0000008e--321b2ddc03', '00000090--1cf3f7d926', 'sealed_test')
INPUT_NAMES = {'image_frames.npy', 'camera_indices.npy', 'absolute_timestamps.npy', 'ego_history_xyz.npy', 'ego_history_rot.npy', 'nav_text.npy'}
BYTES_PER_SCENE = 10*(14*720*1152*2 + 180*2048*2)


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def module(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
  return value


def checked_plan():
  if sha(PLAN) != PLAN_SHA or sha(COVERAGE) != COVERAGE_SHA: raise ValueError('Changed closed recipe or cohort selection')
  plan = read(PLAN)
  for name, digest in SOURCES.items():
    if sha(HERE/name) != digest: raise ValueError('Changed capture helper '+name)
  for name, digest in plan['runtime_files_sha256'].items():
    if sha(Path(plan['runtime_root'])/name) != digest: raise ValueError('Changed original runtime '+name)
  return plan


def validate_manifest(manifest, ids, *, verify_files=True):
  """Metadata admission precedes any selected numerical input read. Never discovers a pending manifest."""
  if sha(COVERAGE) != COVERAGE_SHA: raise ValueError('Changed closed cohort')
  anchors = read(COVERAGE)['anchors']
  if (not manifest.get('admissible_for_internal_layer_capture') or manifest.get('admissible_for_driving_quality_validation')
      or manifest.get('completed_native_frames') != 240 or manifest.get('completed_conditioned_inputs') != 24
      or manifest.get('independent_drives') != 2 or manifest.get('protected_and_sealed_drives_opened') is not False):
    raise ValueError('Need the completed 16-train/8-validation input manifest')
  if manifest['protocol']['coverage'] != dict(path=str(COVERAGE), sha256=COVERAGE_SHA): raise ValueError('Changed scene selection')
  rows = manifest['samples']
  if [r['id'] for r in rows] != [r['id'] for r in anchors]: raise ValueError('Cohort order, completeness, or uniqueness changed')
  if not 1 <= len(ids) <= 2 or len(ids) != len(set(ids)): raise ValueError('Select one or two distinct scenes per bounded job')
  selected = []
  for row, anchor in zip(rows, anchors, strict=True):
    if any(word in json.dumps(row) for word in EXCLUDED): raise ValueError('Excluded route')
    if any(row[k] != anchor[k] for k in ('id', 'split', 'route_id', 'drive_id', 'query_ns', 'conditioning')):
      raise ValueError('Scene identity or command differs from closed metadata')
    if (row['command_index'] != 0 or not 1800 < row['actual_prompt_tokens'] <= 2042
        or not row['complete_input'] or not row['official_selected_tokens_and_image_normalization_exact']):
      raise ValueError('Incomplete original input')
    directory = Path(row['sample']).resolve()
    if (not directory.is_relative_to(WORK/'agent-distill-cohort-artifacts/inputs-v1/samples')
        or directory.name != row['id'] or set(row['input_file_sha256']) != INPUT_NAMES): raise ValueError('Unexpected input files')
    if row['id'] in ids:
      if verify_files:
        for name, digest in row['input_file_sha256'].items():
          if sha(directory/name) != digest: raise ValueError('Changed input '+name)
        if {p.name for p in directory.glob('*.npy')} != INPUT_NAMES: raise ValueError('Unexpected extra numerical inputs')
      selected.append(row)
  if len(selected) != len(ids): raise ValueError('Unknown scene')
  return selected  # Canonical order, independent of command-line argument order.


def make_chunks(cache, batches):
  """Same grouping as the selected VisionCache.encode, before its language-embedding stitch."""
  if (not cache.grouped or cache.batch_encoder is None or cache.group_limit != 2 or cache.verify
      or cache.warmup_graphs or cache.cache): raise ValueError('Need fresh original grouped vision, limit2, without warmup/verification')
  if len(batches) != len(cache.keys) or len(batches) != 10: raise ValueError('Expected ten ordered historical images')
  missing = {}
  for i, (key, (begin, end, count)) in enumerate(zip(cache.keys, batches, strict=True)):
    if tuple(key[0]) != (1,20,36) or (begin,end,count) != (i*720,(i+1)*720,1): raise ValueError('Changed full720 image layout')
    missing.setdefault(key, (begin, end))
  groups = {}
  for item in missing.items(): groups.setdefault(item[0][0], []).append(item)
  return [group[offset:offset+2] for group in groups.values() for offset in range(0,len(group),2)]


def vision_only(cache, pixels, positions, batches):
  from tinygrad import Tensor, TinyJit, dtypes
  if pixels.shape != (7200,768) or positions.shape != (7200,1152): raise ValueError('Changed original pixel or position geometry')
  if pixels.dtype != dtypes.bfloat16 or positions.dtype != dtypes.bfloat16: raise ValueError('Expected original BF16 inputs')
  chunks = make_chunks(cache, batches); encoded = {}; feature_index = 0
  for batch, chunk in enumerate(chunks):
    pairs = [(cache.image_view(t,a,b) if cache.views else t[a:b].unsqueeze(0).clone())
             for _, (a,b) in chunk for t in (pixels, positions)]
    Tensor.realize(*pairs)
    shape = tuple(t.shape for t in pairs)
    if shape not in cache.encoders: cache.encoders[shape] = TinyJit(cache.batch_encoder)
    values = [t.clone() for t in cache.encoders[shape](*pairs)]
    Tensor.realize(*values)
    if len(values) != len(chunk) or any(t.shape != (180,2048) for t in values): raise ValueError('Changed projector features')
    cache.cache.update(zip((key for key,_ in chunk), values, strict=True))
    for member, (key, _) in enumerate(chunk):
      encoded[key] = dict(batch=batch, member=member, projector=feature_index); feature_index += 1
  cache.misses, cache.hits = len(encoded), 0
  mapping = [dict(logical_image=i, camera=1 if i < 5 else 6, history_index=i%5,
                  grid=[int(n) for n in key[0]], pixel_sha256=key[1].hex(), **encoded[key]) for i,key in enumerate(cache.keys)]
  return mapping


@contextmanager
def vision_boundaries(model, save):
  """Use the original teacher hooks; fail immediately if any language/expert execution occurs."""
  runtime = module('vision_cohort_teacher_hooks', HERE/'agent-distill-joint-runtime.py')
  names = ('text_layer', 'expert_step', 'prefill', 'trajectory', 'greedy_decode', 'decode')
  old = {name: getattr(model,name) for name in names if hasattr(model,name)}
  def forbidden(*args, **kwargs): raise ValueError('Vision-only capture reached language or expert execution')
  for name in old: setattr(model, name, forbidden)
  try:
    with runtime.teacher_targets(model, save, groups=GROUPS) as state: yield state
  finally:
    for name, value in old.items(): setattr(model, name, value)


def save_bits(directory, name, tensor):
  import numpy as np
  from tinygrad import dtypes
  if tensor.dtype != dtypes.bfloat16: raise ValueError('Expected unchanged BF16 vision target '+name)
  array = tensor.contiguous().bitcast(dtypes.uint16).numpy().copy()
  if np.any((array & 0x7f80) == 0x7f80): raise ValueError('Nonfinite BF16 target '+name)
  path = directory/(name+'.npy')
  with path.open('xb') as stream: np.save(stream, array, allow_pickle=False)
  return dict(path=str(path), shape=list(array.shape), dtype='uint16', bf16_bits=True, sha256=sha(path))


def validate_targets(state, mapping):
  if state['language_calls'] or state['language_orders'] or state['expert_steps']: raise ValueError('Non-vision targets present')
  if [r['logical_image'] for r in mapping] != list(range(10)): raise ValueError('Lost historical image positions')
  expected = set(); locations = set(); feature = 0
  for batch, row in enumerate(state['vision_batches']):
    if row['patches'] != 720 or row['images'] not in (1,2) or row['layer_order'] != list(range(27)):
      raise ValueError('Incomplete original27 vision execution')
    for group in GROUPS:
      for edge in ('x','y'):
        name = f'vision{batch}.group{group[0]}.{edge}'; expected.add(name)
        target = state['tensors'][name]
        if target['shape'] != [row['images'],720,1152] or target['dtype'] != 'uint16' or not target['bf16_bits']:
          raise ValueError('Changed boundary geometry or dtype')
    for left, right in zip(GROUPS, GROUPS[1:]):
      if state['tensors'][f'vision{batch}.group{left[0]}.y']['sha256'] != state['tensors'][f'vision{batch}.group{right[0]}.x']['sha256']:
        raise ValueError('Adjacent original layer boundaries differ')
    for member in range(row['images']):
      locations.add((batch,member,feature)); name = f'projector{feature}.features'; expected.add(name)
      target = state['tensors'][name]
      if target['shape'] != [180,2048] or target['dtype'] != 'uint16' or not target['bf16_bits']: raise ValueError('Changed final projector')
      feature += 1
  if not 1 <= feature <= 10 or state.get('projector_calls') != feature or set(state['tensors']) != expected:
    raise ValueError('Missing or extra targets')
  if {(r['batch'],r['member'],r['projector']) for r in mapping} != locations: raise ValueError('Invalid image-to-target mapping')
  for i, row in enumerate(mapping):
    if row['camera'] != (1 if i < 5 else 6) or row['history_index'] != i%5 or row['grid'] != [1,20,36]: raise ValueError('Changed camera/history order')
  return True


def compare_legacy(entry, settings, sample_id):
  """Parent may run one excluded-from-cohort control, then demand bit-exact old group12 targets."""
  path, digest = LEGACY[sample_id]
  if sha(path) != digest: raise ValueError('Changed completed legacy report')
  report = read(path); saved = next(r for r in report['rows'] if r['id'] == sample_id)
  if not report['complete'] or not report['gpu_executed'] or report['settings'] != settings: raise ValueError('Legacy execution settings differ')
  for key in ('id','split','route_id','query_ns','input_file_sha256','actual_prompt_tokens','vision_input_keys'):
    if saved[key] != entry[key]: raise ValueError('Legacy input differs '+key)
  names = [f'vision{i}.group12.{edge}' for i in range(5) for edge in ('x','y')]
  for name in names:
    old, new = saved['targets']['tensors'][name], entry['targets']['tensors'][name]
    if old['sha256'] != sha(old['path']) or new['sha256'] != old['sha256']: raise ValueError('Legacy group12 bits differ '+name)
  return dict(report_sha256=digest, exact_group12_arrays=len(names), admitted_to_cohort=False)


def capture_ready(options, plan, ready, *, initialize=None, cpu_mock=False):
  from tinygrad.helpers import Context
  P = module('vision_cohort_parent', HERE/'agent-distill-joint-parent.py')
  H = module('vision_cohort_reset', HERE/'agent-distill-vision-runtime-hook.py')
  outcome = dict(complete=False, gpu_executed=not cpu_mock, cpu_integration_mock=cpu_mock, mode='vision-target-only',
    quality_qualified=False, throughput_qualified=False, groups=[list(g) for g in GROUPS], rows=[],
    inputs_manifest_sha256=getattr(options,'inputs_sha256',None), coverage_sha256=COVERAGE_SHA,
    legacy_control=getattr(options,'legacy_control',None), source_sha256={str(HERE/n):sha(HERE/n) for n in SOURCES})
  outcome['source_sha256'][str(Path(__file__))] = sha(__file__)
  def pipeline(value, args):
    components = (initialize or value.initialize)(args, sources=outcome['source_sha256'], source_dir=Path(plan['runtime_root'])/'experiments')
    model, cache = components.model, components.vision_cache
    if not cpu_mock and components.Device[components.Device.DEFAULT].arch != 'gfx1200': raise ValueError('Expected original Chestnut teacher target')
    outcome['settings'] = {k:components.report[k] for k in SETTINGS}
    for row, prepared, packed in ready:
      padded, packed = components.pad_inputs(prepared, packed, 2048)
      if padded['actual_prompt_length'] != row['actual_prompt_tokens']: raise ValueError('Changed prompt length')
      entry = {k:row[k] for k in ('id','split','route_id','query_ns','input_file_sha256','actual_prompt_tokens')}
      directory = options.output.parent/row['id']; directory.mkdir()
      with Context(JIT=0):
        H.reset_state(model, components.decoder, cache)
        try:
          cache.prepare(padded)
          entry['vision_input_keys'] = [dict(grid=[int(n) for n in key[0]], pixel_sha256=key[1].hex()) for key in cache.keys]
          inputs = components.input_class(components.settings, components.config, padded, packed)
          if args.selective_pixels: inputs.selective_cache = cache
          uploaded = inputs.upload(components.settings.initial_noise)
          with vision_boundaries(model, lambda name,t: save_bits(directory,name,t)) as state:
            entry['images'] = vision_only(cache, uploaded[0], uploaded[1], inputs.batches)
          validate_targets(state, entry['images']); entry['targets'] = state
          if model.cache is not None or components.decoder.history: raise ValueError('Language state changed')
          if getattr(options,'legacy_control',None): entry['legacy_comparison'] = compare_legacy(entry,outcome['settings'],row['id'])
          entry['complete'] = True
        finally: H.reset_state(model, components.decoder, cache)
      outcome['rows'].append(entry)
      options.output.write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')
    outcome['complete'] = True
  try: P.launch(plan, options.output, [r['sample'] for r,_,_ in ready], pipeline)
  except BaseException as error:
    outcome['failure'] = dict(type=type(error).__name__,message=str(error)); raise
  finally: options.output.write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')
  return outcome


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--inputs-manifest', type=Path)
  parser.add_argument('--inputs-sha256')
  parser.add_argument('--sample-id', action='append')
  parser.add_argument('--legacy-control', choices=tuple(LEGACY))
  parser.add_argument('--execute-gpu', action='store_true')
  args = parser.parse_args(); args.output = args.output.resolve(); root = args.output.parent
  if (args.output.name != 'report.json' or not root.is_relative_to(ROOT) or not root.is_dir()
      or any(p.name not in {'usb.acl','invocation.json','run.log'} for p in root.iterdir())): raise ValueError('Need fresh owned output directory')
  if subprocess.check_output(['findmnt','-n','-o','TARGET','-T',str(root)], text=True,timeout=5).strip() != '/mnt/nvme-500':
    raise ValueError('500GB workspace is not mounted')
  if shutil.disk_usage(root).free < 5*1024**3+2*BYTES_PER_SCENE+1024**3: raise ValueError('Preserve5GiB plus targets and1GiB cache allowance')
  if bool(args.legacy_control) == bool(args.inputs_manifest) or (not args.legacy_control and (not args.inputs_sha256 or not args.sample_id)):
    raise ValueError('Supply a completed cohort manifest/hash/scenes, or one explicit legacy control')
  if args.legacy_control and (args.sample_id or args.inputs_sha256): raise ValueError('Legacy control cannot be cohort data')
  if args.execute_gpu:
    parent = Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    if b'/tmp/opencode/run-alpamayo-speed-trial.py' not in parent or b'--program' not in parent or os.fsencode(__file__) not in parent:
      raise ValueError('Only the normal parent GPU wrapper may launch capture')
    with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
      try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError: pass
      else: raise ValueError('Normal parent GPU wrapper must hold the global lock')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('300-second capture limit'))); signal.alarm(300)
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
                    CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  plan = checked_plan()
  if args.execute_gpu:
    for path,digest in plan['checkpoint_shards_sha256'].items():
      if sha(path) != digest: raise ValueError('Changed original BF16 checkpoint')
    cache = plan['fp8_cache']
    if sha(Path(cache['path'])/'index.json') != cache['index_sha256'] or sha(Path(cache['path'])/'weights.bin') != cache['weights_sha256']:
      raise ValueError('Changed original selected FP8 cache')
    seed = plan['compiler_cache_seed']
    if sha(seed['path']) != seed['sha256']: raise ValueError('Changed original compiler cache')
    shutil.copyfile(seed['path'],root/'vision-kernels.db'); (root/'tmp').mkdir()
    invocation = read(plan['invocation_path']); invocation['environment'].update(CACHEDB=str(root/'vision-kernels.db'),TMPDIR=str(root/'tmp'))
    local = root/'vision-invocation.json'; local.write_text(json.dumps(invocation)+'\n'); plan['invocation_path'] = str(local)
    for key,value in invocation['environment'].items():
      if value is None: os.environ.pop(key,None)
      else: os.environ[key] = value
  else: os.environ['DEV'] = 'CPU'
  sys.path.insert(0,str(BASE/'tinygrad'))
  P = module('vision_cohort_preparation_parent',HERE/'agent-distill-joint-parent.py')
  if args.legacy_control:
    if sha(P.INPUTS) != plan['inputs_manifest']['sha256']: raise ValueError('Changed completed legacy inputs')
    rows = P.selected_rows(P.INPUTS,[args.legacy_control])
  else:
    if sha(args.inputs_manifest) != args.inputs_sha256: raise ValueError('Changed completed input manifest')
    rows = validate_manifest(read(args.inputs_manifest),args.sample_id)
  C = module('vision_cohort_original_prepare',HERE/'agent-distill-command-capture.py')
  ready = C.prepared_inputs(plan,dict(samples=rows))
  if not args.execute_gpu:
    P.no_devices(); args.output.write_text(json.dumps(dict(complete=True,preflight_only=True,gpu_executed=False,
      scenes=[r['id'] for r in rows],inputs_manifest_sha256=args.inputs_sha256,maximum_target_bytes=len(rows)*BYTES_PER_SCENE,
      groups=[list(g) for g in GROUPS],legacy_control=args.legacy_control),indent=2)+'\n'); return
  capture_ready(args,plan,ready)


if __name__ == '__main__': main()

"""Opt-in 24-block vision inference. No model defaults, scales, or text depths change."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType
import zipfile

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
PILOT = WORK/'agent-distill-vision-artifacts/pilot-v1'
PLAN_SHA = '47c8d9c8b7224d0524185cf6a7d640a81d536518c6dc7fcd39ede644ae4734a3'
REPORT_SHA = 'ad720f1bc28578ff2009871cf7fa93dc67e69ab64a32fe53f7ad66442f8f5746'
BEST_SHA = '304490b4596b34cdada0f1cfd23eb8aeb617d4dd59632227e262d61082cd3261'
LATEST_SHA = '7a73b635573ee8b11aa03da71cf2a7daaeab30cedafd50c0f4ede252dc9b3faf'
PREFIX = 'vlm.model.visual.encoder.layers.'
# Storage slot27 is outside the original 0..26 namespace. It has no FP8 scales or packed-weight aliases.
STUDENT_INDEX = 27
SOURCES = (*range(12), STUDENT_INDEX, *range(16, 27))


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def module(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  result = importlib.util.module_from_spec(spec); sys.modules[name] = result; spec.loader.exec_module(result)
  return result


def shapes():
  return module('vision_runtime_block_shapes', HERE/'agent-distill-vision-block.py').shapes()


def validate_weights(values):
  import numpy as np
  expected = shapes()
  if set(values) != set(expected): raise ValueError('Unexpected vision student parameter names')
  for name, value in values.items():
    if value.shape != expected[name] or value.dtype != np.float32 or not np.isfinite(value).all():
      raise ValueError('Invalid FP32 student parameter '+name)
  return expected


def load_selected():
  """Only the parent-confirmed completed pilot is admitted. No fallback or active-output discovery."""
  import numpy as np
  if sha(PILOT/'report.json') != REPORT_SHA: raise ValueError('Changed completed pilot report')
  report = read(PILOT/'report.json')
  if not report['complete'] or report['mode'] != 'pilot' or not report['gpu_executed'] or report['committed_step'] != 25:
    raise ValueError('Vision training is incomplete')
  if report['plan_sha256'] != PLAN_SHA or report['real_optimizer_updates']+report['starting_checkpoint_step'] != 25:
    raise ValueError('Unexpected vision pilot lineage')
  wall = read(PILOT/'wall.json')
  if wall['returncode'] != 0 or wall['timed_out']: raise ValueError('Vision training process did not finish successfully')
  if read(PILOT/'journal.json') != dict(committed=25, intended=None): raise ValueError('Uncommitted vision update')
  if sha(PILOT/'latest.json') != LATEST_SHA: raise ValueError('Changed committed checkpoint pointer')
  pointer, selected = read(PILOT/'latest.json'), read(PILOT/'best.json')
  metadata = pointer['metadata']; checks = metadata['validation']
  if metadata['step'] != 25 or checks != report['validation'] or [x['step'] for x in checks] != [0, 25]:
    raise ValueError('Invalid validation selection history')
  winner = min(checks, key=lambda row: row['mean'])
  if selected != dict(file='best.npz', sha256=BEST_SHA, selected_step=winner['step'], validation_mean=winner['mean']):
    raise ValueError('Changed selected artifact')
  if (metadata['best_step'], metadata['best_validation']) != (winner['step'], winner['mean']): raise ValueError('Selection mismatch')
  path = PILOT/'best.npz'
  if path.stat().st_size > 80*1024**2 or sha(path) != BEST_SHA: raise ValueError('Changed selected FP32 weights')
  with zipfile.ZipFile(path) as archive:
    if sum(x.file_size for x in archive.infolist()) > 80*1024**2: raise ValueError('Oversized student payload')
  with np.load(path, allow_pickle=False) as archive: arrays = {name: archive[name].copy() for name in archive.files}
  validate_weights(arrays)
  return arrays, dict(kind='completed_pilot', report_sha256=REPORT_SHA, checkpoint_sha256=BEST_SHA,
                       selected_step=winner['step'], completed_updates=25, vision_trained=winner['step'] > 0,
                       master_dtype='float32', quality_qualified=False)


def load_seed():
  """Explicit numerical control only. This never substitutes for a missing trained checkpoint."""
  C = module('vision_runtime_original_seed', HERE/'agent-distill-vision-common.py')
  arrays, entries = C.seeds()
  plan = WORK/'agent-distill-vision-artifacts/plan.json'
  if sha(plan) != PLAN_SHA or entries != read(plan)['seed']: raise ValueError('Original BF16 seed changed')
  validate_weights(arrays)
  return arrays, dict(kind='original_seed_control', selected_step=0, completed_updates=0,
                       vision_trained=False, master_dtype='float32', quality_qualified=False)


def mapping():
  return [dict(slot=slot, source_layer=source, teacher_group=[12, 13, 14, 15] if slot == 12 else [source],
               student=slot == 12, seed_layer=12 if slot == 12 else source)
          for slot, source in enumerate(SOURCES)]


def reset_state(model, decoder=None, vision_cache=None):
  """Discard captured execution and values; restored architectures must start with fresh caches."""
  R = module('vision_runtime_joint_reset', HERE/'agent-distill-joint-runtime.py')
  R.reset_state(model, decoder, vision_cache)
  model.encode_jits.clear()
  prefill = getattr(model, 'padded_prefill', None)
  for owner in (prefill, getattr(prefill, 'baseline', None)):
    if owner is not None:
      for name in ('last_targets', 'last_prompt', 'prefix_audit'):
        if hasattr(owner, name): setattr(owner, name, None)
  if decoder is not None:
    for name in ('jits', 'verifiers', 'select_jits'):
      value = getattr(decoder, name, None)
      if value is not None: value.clear()
  if vision_cache is not None:
    for value in getattr(vision_cache, 'reference_encoders', {}).values():
      if hasattr(value, 'reset'): value.reset()
    vision_cache.reference_encoders.clear(); vision_cache.reference_model = None
    vision_cache.keys = (); vision_cache.output_grids = None
    vision_cache.hits = vision_cache.misses = vision_cache.reference_checks = 0


@contextmanager
def vision_pilot(model, weights, *, selection, decoder=None, vision_cache=None, allow_untrained_seed=False,
                 allow_short_cpu_fixture=False):
  """Install after the normal native/batched/packed-projector hooks, before first encoder execution."""
  from tinygrad import Tensor, dtypes
  expected = validate_weights(weights)
  if selection.get('kind') == 'original_seed_control':
    if not allow_untrained_seed: raise ValueError('Untrained seed requires explicit control opt-in')
  elif (selection.get('kind'), selection.get('checkpoint_sha256'), selection.get('report_sha256')) != ('completed_pilot', BEST_SHA, REPORT_SHA):
    raise ValueError('Use the verified completed student selection')
  if hasattr(model, 'vision_depth_mapping'): raise ValueError('Vision student already installed')
  if model.config['vlm_config']['vision_config']['num_hidden_layers'] != 27: raise ValueError('Expected original 27 vision blocks')
  if model.dtype not in (dtypes.float32, dtypes.bfloat16): raise ValueError('Expected native FP32 or BF16 residuals')
  original_prefix = PREFIX+'12.'; student_prefix = PREFIX+str(STUDENT_INDEX)+'.'
  for name, shape in expected.items():
    if model.weights[original_prefix+name].shape != shape: raise ValueError('Original vision geometry changed')
  if any(name.startswith(student_prefix) for name in (*model.weights, *model.weight_scales)):
    raise ValueError('Student storage namespace already exists')
  device = model.weights[original_prefix+'layer_norm1.weight'].device
  added = {student_prefix+name: Tensor(value.copy(), device='CPU').to(device).cast(model.dtype).realize()
           for name, value in weights.items()}
  old_config, old_layer = model.config, model.vision_layer
  text_depth = old_config['vlm_config']['text_config']['num_hidden_layers']
  expert_depth = old_config['expert_config']['llm_config']['num_hidden_layers']
  reset_state(model, decoder, vision_cache)  # Before architecture, dispatch, or weights change.
  model.weights.update(added)
  model.config = deepcopy(old_config)
  model.config['vlm_config']['vision_config']['num_hidden_layers'] = 24
  model.vision_depth_mapping = mapping()
  receipt = dict(mapping=mapping(), selection=dict(selection), runtime_dtype=str(model.dtype),
                 student_namespace=student_prefix, new_quantization_scales=0,
                 language_depth=text_depth, expert_depth=expert_depth, quality_qualified=False)
  def layer(self, x, index):
    if not isinstance(index, int) or not 0 <= index < 24: raise ValueError('Unmapped vision loop position')
    if x.ndim != 3 or x.shape[-1] != 1152 or x.dtype != self.dtype: raise ValueError('Changed vision residual geometry')
    if x.shape[1] != 720 and not (allow_short_cpu_fixture and x.device == 'CPU'):
      raise ValueError('Student inference requires all720 patches')
    if any(name.startswith(student_prefix) for name in self.weight_scales): raise ValueError('Student must remain unscaled FP32/BF16')
    return old_layer(x, SOURCES[index])
  model.vision_layer = MethodType(layer, model)
  try:
    yield receipt
  finally:
    # Reset captured student graphs before removing their weights, then restore the enclosing context exactly.
    try: reset_state(model, decoder, vision_cache)
    finally:
      model.vision_layer, model.config = old_layer, old_config
      for name in added: del model.weights[name]
      del model.vision_depth_mapping

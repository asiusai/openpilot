"""CPU-only navigation-absence checks against the actual official and selected adapters."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'): os.environ[key] = '1'
for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'): os.environ[key] = ''
EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec); sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def rejected(fn):
  try: fn()
  except (ValueError, TypeError, KeyError, IndexError) as error:
    return dict(rejected=True, type=type(error).__name__, message=str(error))
  raise AssertionError('Absence unexpectedly accepted; review the supported input contract')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(); output = args.output.resolve()
  assert output.parent.is_relative_to(WORK) and output.name.startswith('agent-distill-data-')
  output.parent.mkdir(parents=True, exist_ok=True)
  # Import Transformers before exposing local kernels.py as a top-level module.
  # This matches the benchmark's namespace guard and avoids its optional Hub
  # package detector importing an unrelated tinygrad module.
  sys.path.insert(0, str(BASE/'upstream/src'))
  import numpy as np
  import torch
  torch.set_num_threads(2)
  from alpamayo2_edge.sample import CAMERA_FRAME_COUNTS, CAMERA_PIXEL_BOUNDS, validate_sample
  from alpamayo2_edge.chat_template import build_conversation
  from alpamayo2_edge.helper import create_messages, prepare_model_inputs
  from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeModel  # noqa: F401
  sys.path[:0] = [str(EXP), str(EXP.parent)]
  fast = load('agent_distill_data_fast_inputs', EXP/'fast-inputs.py')
  live = load('agent_distill_data_live_consumer', EXP/'agent-live-consumer.py')
  # Bind the tested functions to the source used by the current successful paced launch.
  frozen = WORK/'agent-exact-live-paced/report.launcher-sources'
  for name in ('fast-inputs.py', 'agent-live-consumer.py'):
    source = EXP/name
    assert sha(source) == sha(frozen/name), f'Current/frozen source mismatch: {name}'
  core_inputs = WORK/'agent-fused-mlp-2048-protocol-v2/alpamayo/inputs.py'
  assert sha(EXP.parent/'inputs.py') == sha(core_inputs)
  adapter = fast.FastInputs(BASE/'model', camera_profile='nvidia', allow_missing_cameras=True,
    direct_byte_inputs=True, cache_resize=True)
  def sample(front=False):
    ids = [c for c, n in CAMERA_FRAME_COUNTS.items() for _ in range(n) if not front or c in (1, 6)]
    times = []
    for c in sorted(set(ids)):
      offsets = (-2_000_000, -1_000_000, -500_000, -100_000, 0) if c in (1, 6) else (-1_000_000, -100_000, 0)
      times.extend(10_000_000+offset for offset in offsets)
    return dict(image_frames=torch.zeros(len(ids), 1, 3, 32, 32, dtype=torch.uint8),
      camera_indices=torch.tensor(ids), absolute_timestamps=torch.tensor(times),
      ego_history_xyz=torch.zeros(1, 20, 3),
      ego_history_rot=torch.eye(3).expand(1, 20, 3, 3).contiguous(), nav_text=['CPU control instruction'])
  official, front = sample(), sample(True)
  validate_sample(official); fast.validate_available_sample(front)
  prepare_model_inputs(official, adapter.config, adapter.tokenizer)
  _, _, positive = adapter.prepare(front)
  assert positive['tokenized_data']['input_ids'].ndim == 2
  q = 10_000_000_000
  arrays = {k: v.numpy().copy() for k, v in front.items() if k != 'nav_text'}
  arrays['image_frames'] = np.zeros((10, 1, 3, 432, 768), dtype=np.uint8)
  arrays['nav_text'] = front['nav_text']
  frames = [dict(view=c, capture_ns=int(t)*1000, requested_ns=int(t)*1000,
    image_calibration_ns=int(t)*1000) for c, t in zip(arrays['camera_indices'], arrays['absolute_timestamps'], strict=True)]
  meta = dict(query_ns=q, selected_ns=q, prepared_ns=q, frames=frames, navigation_text=front['nav_text'][0])
  # Call the original method, without constructing or importing a GPU model.
  live.Consumer._data(SimpleNamespace(last_query_ns=0), arrays, meta, q)
  cases = {}
  for label, nav in [('missing', ...), ('None', None), ('empty-list', []), ('None-entry', [None]),
                     ('empty-text', ['']), ('whitespace', [' '])]:
    d, f = dict(official), dict(front)
    if nav is ...:
      d.pop('nav_text'); f.pop('nav_text')
    else: d['nav_text'] = f['nav_text'] = nav
    row = dict(official_validator=rejected(lambda: validate_sample(d)),
      official_create_messages=rejected(lambda: create_messages(d, adapter.config)),
      official_prepare=rejected(lambda: prepare_model_inputs(d, adapter.config, adapter.tokenizer)),
      selected_available_validator=rejected(lambda: fast.validate_available_sample(f)),
      selected_prepare=rejected(lambda: adapter.prepare(f)))
    a = dict(arrays)
    if nav is ...: a.pop('nav_text')
    else: a['nav_text'] = nav
    # Live rejects actual absence. Whitespace is caught later by selected_prepare.
    if label != 'whitespace':
      row['original_live_data'] = rejected(lambda: live.Consumer._data(SimpleNamespace(last_query_ns=0), a, meta, q))
    cases[label] = row
  # Diagnostic only: bypass validation to compare an empty labelled field with
  # deleting that field. Neither result is admitted as a driving sample.
  empty = dict(front, nav_text=[''], camera_pixel_bounds=CAMERA_PIXEL_BOUNDS)
  labelled = build_conversation(empty)
  omitted = copy.deepcopy(labelled)
  omitted[1]['content'] = [item for item in omitted[1]['content']
    if not (item['type'] == 'text' and item['text'] == 'Navigation input is: ')]
  def text_ids(messages):
    text = adapter.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
      add_vision_id=False, continue_final_message=False, enable_thinking=False)
    ids = adapter.tokenizer(text, add_special_tokens=False)['input_ids']
    return text, ids
  text_a, ids_a = text_ids(labelled); text_b, ids_b = text_ids(omitted)
  assert ids_a != ids_b and 'Navigation input is: ' in text_a and 'Navigation input is: ' not in text_b
  # Official/current adapter share the actual function object, not a rewritten template.
  assert fast.reference.build_conversation is build_conversation
  assert not torch.cuda.is_initialized()
  bad_fds = []
  for fd in Path('/proc/self/fd').iterdir():
    try: target = os.readlink(fd)
    except OSError: continue
    if target.startswith(('/dev/dri', '/dev/kfd', '/dev/nvidia', '/dev/bus/usb')): bad_fds.append(target)
  assert not bad_fds
  sources = [Path(__file__), EXP/'fast-inputs.py', EXP/'agent-live-consumer.py', EXP.parent/'inputs.py',
    BASE/'upstream/src/alpamayo2_edge/chat_template.py', BASE/'upstream/src/alpamayo2_edge/sample.py',
    BASE/'upstream/src/alpamayo2_edge/helper.py', BASE/'model/config.json', BASE/'model/tokenizer.json',
    BASE/'model/tokenizer_config.json', BASE/'model/chat_template.jinja', BASE/'model/processor_config.json']
  result = dict(schema=1, status='native absent navigation unsupported by both tested driving paths',
    cases=cases, synthetic_controls_only=True, owned_recordings_opened=False,
    official_full_camera_positive_control=True, selected_full_front_history_positive_control=True,
    original_live_data_positive_control=True, official_selected_template_same_function=True,
    selected_positive_input_ids_shape=list(positive['tokenized_data']['input_ids'].shape),
    empty_vs_deleted_field=dict(equal=False, labelled_token_count=len(ids_a), deleted_field_token_count=len(ids_b),
      labelled_ids_sha256=hashlib.sha256(np.asarray(ids_a, dtype=np.int64).tobytes()).hexdigest(),
      deleted_ids_sha256=hashlib.sha256(np.asarray(ids_b, dtype=np.int64).tobytes()).hexdigest(),
      limit='Text/token diagnostic only. Deleting navigation is an unsupported changed prompt, not an inference sample.'),
    training_executed=False, cuda_initialized=False, gpu_usb_fds=bad_fds,
    source_sha256={str(p): sha(p) for p in sources},
    boundary='No navigation was inferred from future motion, model outputs, or a synthetic Follow road instruction.')
  with output.open('x') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
  print(json.dumps(dict(output=str(output), status=result['status'], absence_cases=len(cases),
    token_comparison=result['empty_vs_deleted_field']), indent=2))


if __name__ == '__main__': main()

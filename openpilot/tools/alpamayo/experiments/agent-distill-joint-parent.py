"""CPU preflight by default. Parent may opt into teacher capture or paired smoke comparison."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
OLD = WORK/'agent-distill-command-artifacts/capture-v1'
PLAN = OLD/'agent-distill-command-capture-plan.json'
INPUTS = WORK/'agent-distill-command-artifacts/inputs-v1/agent-distill-command-inputs.json'
PILOT = WORK/'agent-distill-continue-artifacts/pilot-100-v1/best.npz'
ALLOWED = {'train': '00000071--858232e52d', 'validation': '00000073--41474e402a'}


def module(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
  return value


def no_devices():
  for path in Path('/proc/self/fd').iterdir():
    try: target = str(path.readlink())
    except FileNotFoundError: continue
    if target.startswith(('/dev/dri/', '/dev/kfd', '/dev/nvidia', '/dev/bus/usb/')): raise RuntimeError(target)


def selected_rows(path, ids):
  manifest = json.loads(Path(path).read_text())
  # Headers can share a manifest. Never open unselected or sealed numerical data.
  by_id = {row['id']: row for row in manifest['samples'] if row['split'] in ALLOWED}
  if len(ids) != len(set(ids)) or not 1 <= len(ids) <= 6: raise ValueError('Select one to six explicit examples')
  rows = []
  for sample_id in ids:
    row = by_id[sample_id]
    if row['route_id'] != ALLOWED[row['split']]: raise ValueError('Unapproved training/validation route')
    if any(word in str(row) for word in ('0000008e--321b2ddc03', '00000090--1cf3f7d926', 'sealed_test')):
      raise ValueError('Protected or sealed input')
    rows.append(row)
  return rows


def prepare(plan, rows):
  import numpy as np
  import torch
  torch.set_num_threads(4); torch.set_default_device('cpu')
  sys.path.insert(0, str(BASE/'upstream/src'))
  # Import Transformers before native kernels.py can shadow its optional package.
  from alpamayo2_edge import helper  # noqa: F401
  from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeModel  # noqa: F401
  root = Path(plan['runtime_root'])
  sys.path[:0] = [str(root/'experiments'), str(root)]
  fast = module('agent_distill_joint_fast', root/'experiments/fast-inputs.py')
  adapter = fast.FastInputs(BASE/'model', camera_profile='nvidia', allow_missing_cameras=True,
                           direct_byte_inputs=True, cache_resize=True)
  result = []
  for row in rows:
    for name, digest in row['input_file_sha256'].items():
      if hashlib.sha256((Path(row['sample'])/name).read_bytes()).hexdigest() != digest: raise ValueError('Changed input '+name)
    data = {p.stem: [str(np.load(p, allow_pickle=False).item())] if p.stem == 'nav_text'
            else torch.from_numpy(np.load(p, allow_pickle=False)) for p in sorted(Path(row['sample']).glob('*.npy'))}
    _, _, prepared = adapter.prepare(data); packed = adapter.native_inputs(prepared)
    grids = prepared['tokenized_data']['image_grid_thw'].numpy()
    if grids.shape != (10, 3) or not np.all(grids.prod(axis=1) == 720): raise ValueError('Expected ten full 720-patch images')
    if packed[0].shape[1] != row['actual_prompt_tokens']: raise ValueError('Prompt length changed')
    result.append((row, prepared, packed))
  no_devices()
  return result


def launch(plan, output, samples, callback):
  """Enter the actual selected launcher, including its installed speculation hooks."""
  root = Path(plan['runtime_root']); exps = root/'experiments'
  sample_file = output.parent/'agent-distill-joint-samples.json'
  sample_file.write_text(json.dumps(samples)+'\n')
  old_run, old_spec, old_argv = runpy.run_path, importlib.util.spec_from_file_location, sys.argv[:]
  def pipeline(path, run_name='__main__'):
    if Path(path).resolve() != exps/'benchmark-pipeline.py': raise ValueError('Unexpected runner')
    value = module('agent_distill_joint_pipeline', path)
    args = value.parse_args(sys.argv[1:])
    assert args.steps == 3 and args.pad_prompt_to == 2048 and args.vision_calibration is None
    assert args.history_pixel_scale == args.history_feature_scale == args.pixel_scale == 1
    assert args.front_history_indices is None and args.trajectory_solver == 'euler'
    return callback(value, args)
  try:
    runpy.run_path = pipeline
    sys.argv = [str(exps/'agent-speculation-run.py'), '--output', str(output), '--invocation', plan['invocation_path'],
                '--mode', 'candidate', '--prefill-attention', 'shuffle-fma', '--pad-prompt-to', '2048',
                '--warmup', '0', '--runs', str(len(samples)), '--samples-json', str(sample_file)]
    namespace = dict(__file__=sys.argv[0], __name__='__main__')
    exec(compile(Path(sys.argv[0]).read_text(), sys.argv[0], 'exec'), namespace)
  finally:
    runpy.run_path, importlib.util.spec_from_file_location, sys.argv = old_run, old_spec, old_argv


def run(args, plan, ready, runtime):
  import numpy as np
  from tinygrad.helpers import Context
  outcome = dict(complete=False, gpu_executed=args.execute_gpu, quality_qualified=False,
                 timings_valid_for_throughput=False, mode=args.mode, rows=[])
  root = args.output_dir.resolve(); output = root/'report.json'
  outcome['source_sha256'] = {str(p): runtime.sha(p) for p in HERE.glob('agent-distill-joint-*.py')}
  outcome['source_sha256'][str(Path(plan['runtime_root'])/'model.py')] = runtime.sha(Path(plan['runtime_root'])/'model.py')
  if args.mode == 'compare': outcome['language_checkpoint_sha256'] = runtime.sha(args.language_checkpoint)
  def pipeline(value, options):
    if not args.execute_gpu:
      outcome.update(complete=True, actual_launcher_and_parser_checked=True, prepared_samples=[r[0]['id'] for r in ready])
      no_devices(); return
    os.environ['CACHEDB'] = str(root/'agent-distill-joint-kernels.db')
    components = value.initialize(options, sources=outcome['source_sha256'], source_dir=Path(plan['runtime_root'])/'experiments')
    model, settings = components.model, components.settings
    assert components.Device[components.Device.DEFAULT].arch == 'gfx1200'
    outcome['settings'] = {k: v for k, v in components.report.items() if k in
      ('prefill_calibration_sha256', 'vision_calibration_sha256', 'initial_noise_sha256', 'settings', 'runtime_environment')}
    if args.mode == 'compare':
      language = runtime.load_pilot(args.language_checkpoint)
      expert = runtime.seed_block(BASE/'model', runtime.EXPERT+'12.')
    for row, prepared, packed in ready:
      prepared, packed = components.pad_inputs(prepared, packed, 2048)
      entry = dict(id=row['id'], split=row['split'], route_id=row['route_id'], query_ns=row['query_ns'],
                   input_file_sha256=row['input_file_sha256'], actual_prompt_tokens=row['actual_prompt_tokens'])
      def predict():
        components.vision_cache.prepare(prepared)
        inputs = components.input_class(settings, components.config, prepared, packed)
        if options.selective_pixels: inputs.selective_cache = components.vision_cache
        return components.runner(model, settings, components.config, components.tokenizer, prepared, packed, seed=0, inputs=inputs)
      runtime.reset_state(model, components.decoder, components.vision_cache)
      # Disable JIT capture only for observations. Keep all original arithmetic and inputs.
      # These complete predictions are diagnostic and never count as throughput evidence.
      with Context(JIT=0):
        if args.mode == 'capture':
          directory = root/row['id']; directory.mkdir()
          def save(name, tensor):
            from tinygrad import dtypes
            bf16 = tensor.dtype == dtypes.bfloat16
            array = tensor.contiguous().bitcast(dtypes.uint16).numpy().copy() if bf16 else tensor.numpy().copy()
            numeric = (array.astype(np.uint32)<<16).view(np.float32) if bf16 else array
            if name.endswith('.mask'):
              if not np.all((numeric == 0) | np.isneginf(numeric)): raise ValueError('Invalid mask')
            elif not np.isfinite(numeric).all(): raise ValueError('Nonfinite teacher target '+name)
            path = directory/(name+'.npy'); np.save(path, array, allow_pickle=False)
            return dict(path=str(path), shape=list(array.shape), dtype=str(array.dtype), bf16_bits=bf16, sha256=runtime.sha(path))
          with runtime.teacher_targets(model, save) as captured:
            entry['teacher'] = predict()
          runtime.validate_targets(captured)
          entry['vision_input_keys'] = [dict(grid=[int(n) for n in key[0]], pixel_sha256=key[1].hex())
                                        for key in components.vision_cache.keys]
          for i in range(3):
            t = np.load(captured['tensors'][f'expert{i}.timestep']['path'], allow_pickle=False)
            np.testing.assert_allclose(t, [i/3], atol=1e-7, rtol=0)
          entry['targets'] = captured
        else:
          entry['teacher'] = predict()
          with runtime.joint_pilot(model, language, expert, allow_untrained_expert=True,
                                   decoder=components.decoder, vision_cache=components.vision_cache) as receipt:
            entry['candidate'] = predict(); entry['adapter'] = receipt
            assert len(model.cache) == 25
          a, b = (np.asarray(entry[k]['xyz']) for k in ('teacher', 'candidate'))
          entry['comparison'] = dict(tokens_equal=entry['teacher']['generated_token_ids'] == entry['candidate']['generated_token_ids'],
                                     max_xyz_delta_m=float(np.abs(a-b).max()), mean_xyz_delta_m=float(np.abs(a-b).mean()))
      outcome['rows'].append(entry)
      output.write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')
    outcome['complete'] = True
  try: launch(plan, output, [r['sample'] for r, _, _ in ready], pipeline)
  finally: output.write_text(json.dumps(outcome, indent=2, allow_nan=False)+'\n')
  return outcome


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--inputs-manifest', type=Path, default=INPUTS)
  parser.add_argument('--sample-id', action='append', required=True)
  parser.add_argument('--mode', choices=('capture', 'compare'), default='capture')
  parser.add_argument('--language-checkpoint', type=Path, default=PILOT)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--execute-gpu', action='store_true')
  args = parser.parse_args()
  if args.output_dir.exists(): raise ValueError('Output directory must be new')
  if not args.output_dir.resolve().is_relative_to(WORK): raise ValueError('Keep artifacts on the 500 GB workspace')
  os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                    CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
  runtime = module('agent_distill_joint_runtime', HERE/'agent-distill-joint-runtime.py')
  plan = json.loads(PLAN.read_text())
  rows = selected_rows(args.inputs_manifest, args.sample_id)
  ready = prepare(plan, rows)
  args.output_dir.mkdir(parents=True)
  # The parent owns all actual GPU execution through the shared lock.
  context = Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') if args.execute_gpu else nullcontext()
  with context as lock:
    if args.execute_gpu:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      shutil.copyfile(plan['compiler_cache_seed']['path'], args.output_dir/'agent-distill-joint-kernels.db')
      for name, value in plan['environment'].items():
        if value is None: os.environ.pop(name, None)
        else: os.environ[name] = value
    else: os.environ['DEV'] = 'CPU'
    sys.path.insert(0, str(BASE/'tinygrad'))
    result = run(args, plan, ready, runtime)
  print(json.dumps(dict(output=str(args.output_dir/'report.json'), complete=result['complete'], gpu_executed=args.execute_gpu)))


if __name__ == '__main__': main()

"""Preflight by default; parent-only bounded selected-runtime internal-layer capture."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from types import MethodType

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('agent_distill_command_common_capture', HERE/'agent-distill-command-common.py')
c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = c; spec.loader.exec_module(c)
DEFAULT = c.ROOT/'capture-v1/agent-distill-command-capture-plan.json'


def preflight(path):
  plan = json.loads(Path(path).read_text())
  assert plan['protocol'] == 'experimenter-command-layer-reconstruction-v1' and plan['parent_timeout_seconds'] == 300
  assert plan['capture_layers'] == [12, 15] and plan['expected_examples'] == 9 and plan['max_model_prefills'] == 9
  for filename, digest in plan['source_sha256'].items(): assert c.sha(filename) == digest, filename
  pin = plan['inputs_manifest']; assert c.sha(pin['path']) == pin['sha256']
  manifest = json.loads(Path(pin['path']).read_text())
  assert manifest['admissible_for_internal_layer_capture'] and not manifest['admissible_for_driving_quality_validation']
  assert manifest['protocol'] == c.protocol() and len(manifest['samples']) == 9
  for filename, digest in manifest['source_sha256'].items(): assert c.sha(filename) == digest, filename
  for row in manifest['samples']:
    assert c.PROTECTED not in json.dumps(row)
    assert row['conditioning']['basis'] == 'experimental command' and row['conditioning']['recorded_available_ns'] is None
    assert row['conditioning']['text'] == c.COMMANDS[row['command_index']]
    assert row['actual_prompt_tokens'] in (2027, 2030)
    for name, digest in row['input_file_sha256'].items(): assert c.sha(Path(row['sample'])/name) == digest
  cache = plan['fp8_cache']
  assert c.sha(Path(cache['path'])/'index.json') == cache['index_sha256']
  assert c.sha(Path(cache['path'])/'weights.bin') == cache['weights_sha256']
  assert (Path(cache['path'])/'weights.bin').stat().st_size == cache['nbytes']
  for filename, digest in plan['checkpoint_shards_sha256'].items(): assert c.sha(filename) == digest
  c.assert_no_devices()
  return plan, manifest


def prepared_inputs(plan, manifest):
  """Real CPU adapter, before any GPU initialization. All nine inputs must match."""
  torch, _ = c.cpu_imports()
  import numpy as np
  root = Path(plan['runtime_root'])
  sys.path[:0] = [str(root/'experiments'), str(root)]
  fast = c.module('distill_command_frozen_fast', root/'experiments/fast-inputs.py')
  assert Path(sys.modules['inputs'].__file__).resolve() == root/'inputs.py'
  adapter = fast.FastInputs(c.BASE/'model', camera_profile='nvidia', allow_missing_cameras=True,
    direct_byte_inputs=True, cache_resize=True)
  result = []
  for row in manifest['samples']:
    data = {p.stem: [str(np.load(p, allow_pickle=False).item())] if p.stem == 'nav_text'
            else torch.from_numpy(np.load(p, allow_pickle=False)) for p in sorted(Path(row['sample']).glob('*.npy'))}
    config, tokenizer, prepared = adapter.prepare(data); packed = adapter.native_inputs(prepared)
    assert c.fingerprint(prepared) == row['prepared'] and c.fingerprint(packed) == row['packed'], row['id']
    result.append((row, prepared, packed))
  c.assert_no_devices()
  return result


class Captured(Exception): pass


@contextmanager
def capture_boundary(model, save_bits):
  """Wrap the final installed layer once. Execute its exact arithmetic without replacements."""
  original = model.text_layer
  blocked_names = ('decode', 'greedy_decode', 'trajectory')
  blocked = {name: getattr(model, name) for name in blocked_names}
  state = dict(order=[], payload={}, calls=0)
  def layer(self, x, *args, **kwargs):
    prefix = args[2] if len(args) >= 3 else kwargs['prefix']
    assert prefix.startswith('vlm.model.language_model.layers.')
    index = int(prefix.rsplit('.', 1)[1]); assert index == len(state['order']) and index <= 15
    state['order'].append(index)
    assert x.shape == (1, 2048, 2048) and kwargs.get('cache_pos') == 0
    if index == 12:
      state['payload']['x'] = save_bits(x)
      state['payload']['cos'] = save_bits(args[0][:, 0])
      state['payload']['sin'] = save_bits(args[1][:, 0])
    y = original(x, *args, **kwargs); state['calls'] += 1
    if index == 15:
      state['payload']['y'] = save_bits(y)
      # Original text_layer wrote post-RoPE K and unchanged V into this actual cache.
      cache = model.cache[12]
      state['payload']['k'] = save_bits(cache[0, :, :, :2048])
      state['payload']['v'] = save_bits(cache[1, :, :, :2048])
      assert state['order'] == list(range(16)) and state['calls'] == 16
      raise Captured()
    return y
  def forbidden(*args, **kwargs): raise AssertionError('Driving/reasoning output is outside internal-layer capture')
  model.text_layer = MethodType(layer, model)
  for name in blocked_names: setattr(model, name, forbidden)
  try: yield state
  finally:
    model.text_layer = original
    for name, value in blocked.items(): setattr(model, name, value)


def find_quantizer(method):
  pending, seen = [method], set()
  while pending:
    fn = pending.pop(); fn = fn.__func__ if inspect.ismethod(fn) else fn
    if not inspect.isfunction(fn) or id(fn) in seen: continue
    seen.add(id(fn))
    if Path(fn.__code__.co_filename).name == 'quantize-packed.py' and fn.__name__ == 'quantize_input': return fn
    pending.extend(inspect.getclosurevars(fn).nonlocals.values())
  raise ValueError('Original packed fixed-scale prefill quantizer is absent')


def execute(plan, manifest, ready, output, plan_path):
  import numpy as np
  from tinygrad import dtypes
  root = Path(plan['runtime_root']); exps = root/'experiments'
  old_spec, old_run, old_argv = importlib.util.spec_from_file_location, runpy.run_path, sys.argv[:]
  outcome = dict(schema=1, protocol=plan['protocol'], complete=False, gpu_executed=True,
    diagnostic_only=True, timings_valid_for_throughput=False, prefills=0, completed_examples=[],
    driving_outputs_executed=False, training_executed=False, targets_sealed=True,
    capture_plan=dict(path=str(plan_path), sha256=c.sha(plan_path)), inputs_manifest=plan['inputs_manifest'])
  targets = output.parent/(output.stem+'.agent-distill-command-targets'); targets.mkdir(exist_ok=False)
  for split in ('train', 'validation', 'sealed_test'): (targets/split).mkdir()
  class Completed(Exception): pass
  def pipeline(path, run_name='__main__'):
    assert Path(path).resolve() == exps/'benchmark-pipeline.py'
    module = c.module('distill_command_frozen_pipeline_gpu', path)
    args = module.parse_args(sys.argv[1:])
    assert args.steps == 3 and args.pad_prompt_to == 2048 and args.vision_calibration is None
    assert args.history_pixel_scale == args.history_feature_scale == args.pixel_scale == 1 and args.front_history_indices is None
    components = module.initialize(args, sources=plan['runtime_files_sha256'], source_dir=exps)
    model, settings = components.model, components.settings
    assert model.dtype == dtypes.bfloat16 and components.Device[components.Device.DEFAULT].arch == 'gfx1200'
    outcome['actual_settings'] = dict(diffusion_steps=settings.diffusion_steps, max_tokens=settings.max_tokens,
      camera_profile=components.report['camera_profile'], pad_prompt_to=components.report['pad_prompt_to'],
      vision_calibration_sha256=components.report['vision_calibration_sha256'],
      prefill_calibration_sha256=components.report['prefill_calibration_sha256'],
      runtime_environment=components.report['runtime_environment'])
    quantizer = find_quantizer(model.quantize)
    fixed = inspect.getclosurevars(quantizer).nonlocals['fixed']
    expected = json.loads(Path(args.prefill_calibration).read_text())['scales']
    assert set(fixed) == set(expected)
    for key, value in fixed.items():
      assert value.numpy().tobytes() == np.float32(expected[key]).tobytes(), key
    def bits(tensor):
      assert tensor.dtype == dtypes.bfloat16
      value = tensor.contiguous().bitcast(dtypes.uint16).numpy().copy()
      assert not np.any((value & 0x7f80) == 0x7f80), 'Nonfinite teacher internal target/input'
      return value
    for row, prepared, packed in ready:
      prepared, packed = components.pad_inputs(prepared, packed, 2048)
      assert prepared['actual_prompt_length'] == row['actual_prompt_tokens'] and prepared['prefill_draft_eligible']
      components.vision_cache.prepare(prepared)
      inputs = components.input_class(settings, components.config, prepared, packed)
      if args.selective_pixels: inputs.selective_cache = components.vision_cache
      tensors = inputs.upload(settings.initial_noise)
      pixels, positions, ids, cos, sin = tensors[:5]
      embedding = model.encode(pixels, positions, ids, inputs.batches, inputs.spans)
      # Always execute the original warm-call path for diagnostic Python hooks.
      # No previously captured graph can skip a requested boundary.
      model.padded_prefill.jit.reset(); model.padded_prefill.shape = None
      with capture_boundary(model, bits) as state:
        try: model.padded_prefill(embedding, cos, sin, settings.max_tokens, row['actual_prompt_tokens'])
        except Captured: pass
        else: raise AssertionError('Internal capture did not stop at the original layer15 boundary')
      payload = state['payload']
      shapes = dict(x=(1,2048,2048), y=(1,2048,2048), k=(1,8,2048,128), v=(1,8,2048,128), cos=(1,2048,128), sin=(1,2048,128))
      assert set(payload) == set(shapes)
      for name, value in payload.items(): assert value.dtype == np.uint16 and value.shape == shapes[name]
      assert sum(value.nbytes for value in payload.values()) == 25*1024**2
      destination = targets/row['split']/(row['id']+'.npz')
      np.savez(destination, **payload)
      record = dict(id=row['id'], split=row['split'], drive_id=row['drive_id'], route_id=row['route_id'],
        actual_prompt_tokens=row['actual_prompt_tokens'], query_ns=row['query_ns'],
        conditioning=row['conditioning'], input_file_sha256=row['input_file_sha256'],
        capture_path=str(destination), capture_sha256=c.sha(destination),
        capture_shapes={name: list(shape) for name, shape in shapes.items()},
        layer_order=state['order'], original_layer_calls=state['calls'], finite_bf16_bits=True,
        vision_cache_hits=components.vision_cache.hits, vision_cache_misses=components.vision_cache.misses,
        target_semantics=plan['target_semantics'], fit_rows='actual prompt rows only; exclude all draft and END rows')
      outcome['completed_examples'].append(record); outcome['prefills'] += 1
      # Partial results are explicit diagnostics, never a completed capture protocol.
      output.write_text(json.dumps(outcome, indent=2)+'\n')
      print(f'Captured internal layers for {row["id"]}', flush=True)
    assert outcome['prefills'] == 9
    training = [r for r in outcome['completed_examples'] if r['split'] in ('train', 'validation')]
    sealed = [r for r in outcome['completed_examples'] if r['split'] == 'sealed_test']
    assert len(training) == 6 and len(sealed) == 3
    common = dict(schema=1, protocol=plan['protocol'], plan_sha256=c.sha(plan_path),
      inputs_manifest_sha256=plan['inputs_manifest']['sha256'], target_semantics=plan['target_semantics'])
    c.write(targets/'agent-distill-command-train-validation.json', dict(common, samples=training, sealed_test_targets_included=False))
    c.write(targets/'sealed_test/agent-distill-command-sealed-test.json', dict(common, samples=sealed, forbidden_for_fit_and_early_stopping=True))
    # Global status has only sealed headers/checksums, never numerical tensors or losses.
    outcome.update(complete=True, runtime_admission='Original target-architecture FP8 parity and full upload-byte verification retained.',
      train_validation_manifest=str(targets/'agent-distill-command-train-validation.json'),
      sealed_manifest=str(targets/'sealed_test/agent-distill-command-sealed-test.json'))
    raise Completed()
  try:
    runpy.run_path = pipeline
    sys.argv = [str(exps/'agent-speculation-run.py'), '--output', str(output), '--invocation', plan['invocation_path'],
      '--mode', 'candidate', '--prefill-attention', 'shuffle-fma', '--pad-prompt-to', '2048', '--warmup', '0', '--runs', '9']
    namespace = dict(__file__=str(exps/'agent-speculation-run.py'), __name__='__main__')
    try: exec(compile((exps/'agent-speculation-run.py').read_text(), namespace['__file__'], 'exec'), namespace)
    except Completed: pass
    else: raise AssertionError('Canonical launcher failed to enter bounded capture')
  except BaseException as error:
    outcome['failure'] = dict(type=type(error).__name__, message=str(error)); raise
  finally:
    importlib.util.spec_from_file_location, runpy.run_path, sys.argv = old_spec, old_run, old_argv
    outcome['launcher_hooks_restored'] = importlib.util.spec_from_file_location is old_spec and runpy.run_path is old_run
    output.write_text(json.dumps(outcome, indent=2)+'\n')
  return outcome


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan', type=Path, default=DEFAULT)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--execute-gpu', action='store_true')
  parser.add_argument('--check-imports', action='store_true')
  args = parser.parse_args(); output = args.output.resolve()
  assert output.is_relative_to(c.WORK) and not output.exists()
  output.parent.mkdir(parents=True, exist_ok=True)
  assert not (args.execute_gpu and args.check_imports)
  if 'tinygrad' in sys.modules: raise RuntimeError('Use a fresh process before any tinygrad imports')
  plan, manifest = preflight(args.plan)
  ready = prepared_inputs(plan, manifest)
  if not args.execute_gpu:
    imports = []
    if args.check_imports:
      os.environ['DEV'] = 'CPU'
      sys.path.insert(0, str(c.BASE/'tinygrad'))
      root = Path(plan['runtime_root'])
      for name in ('benchmark-pipeline', 'benchmark-fast', 'compact-inputs', 'quantize-packed', 'padded-prefill', 'agent-speculation-prefill'):
        value = c.module('distill_command_import_'+name.replace('-', '_'), root/'experiments'/(name+'.py'))
        imports.append(dict(name=name, source=str(Path(value.__file__).resolve())))
      from tinygrad import Device
      assert not Device._opened_devices
    c.assert_no_devices()
    c.write(output, dict(schema=1, preflight_passed=True, prepared_inputs=9, gpu_executed=False, training_executed=False,
      imported_actual_modules=imports, model_initialized=False, plan_sha256=c.sha(args.plan)))
    print(json.dumps(dict(output=str(output), preflight_passed=True, gpu_executed=False), indent=2)); return
  # Parent-only branch begins after every CPU/source/input/cache gate.
  seed = plan['compiler_cache_seed']
  assert c.sha(seed['path']) == seed['sha256']
  owned_cache = output.with_suffix('.agent-distill-command-kernel-cache.db')
  assert not owned_cache.exists(); shutil.copyfile(seed['path'], owned_cache)
  for key, value in plan['environment'].items():
    if value is None: os.environ.pop(key, None)
    else: os.environ[key] = value
  os.environ['CACHEDB'] = str(owned_cache)
  sys.path[:0] = [str(Path(plan['runtime_root'])/'experiments'), plan['runtime_root'], str(c.BASE/'tinygrad')]
  execute(plan, manifest, ready, output, args.plan)


if __name__ == '__main__': main()

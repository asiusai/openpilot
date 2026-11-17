#!/usr/bin/env python3
"""Strict observer adapter and explicit overlay on an immutable passing baseline.

No device opens at import, plan preparation, or verification. Parent alone runs
the overlay. The old controls, reference, source tree, and plan stay unchanged.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import CodeType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OLD = WORK/'activation-sparsity-devicefix-source'
SCHEMA = 'agent-activation-sparsity-observer-overlay-v1'
APPROVED = {
  'quantize-packed.py': 'c57368db69b04d2c05c2140712b3a4f337c04717a165004d94693aedccbc141d',
  'batched-vision.py': '868977217720d0a5c607e179a1e14a810d189bd4c4800c8178f86a5bc1674804',
}
CORE_SHA = '2c27e281db702575975f267111ad173ef6c910de973f69e814131442951142eb'


def sha(path):
  value = hashlib.sha256()
  with Path(path).open('rb') as stream:
    while chunk := stream.read(8*1024*1024): value.update(chunk)
  return value.hexdigest()


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def identity(value): return hashlib.sha256(canonical(value)).hexdigest()


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def code_named(code, qualname):
  matches = []
  if code.co_qualname == qualname: matches.append(code)
  for child in code.co_consts:
    if isinstance(child, CodeType): matches.extend(code_named(child, qualname))
  return matches


def approved_function(function, root, filename, qualname, model=None):
  """Check source bytes AND executable code/globals/binding, never names alone."""
  if model is not None:
    if not inspect.ismethod(function) or function.__self__ is not model:
      raise ValueError('Quantizer/layer/linear method is not bound to the current model')
    function = function.__func__
  path = (Path(root)/filename).resolve()
  module_name = filename.removesuffix('.py').replace('-', '_')
  if (not inspect.isfunction(function) or sha(path) != APPROVED[filename]
      or Path(function.__code__.co_filename).resolve() != path
      or Path(function.__globals__.get('__file__', '')).resolve() != path
      or function.__module__ != module_name or function.__globals__.get('__name__') != module_name
      or function.__qualname__ != qualname):
    raise ValueError('Unsupported selected function/source/binding: '+filename+'/'+qualname)
  expected = code_named(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True,
                                optimize=sys.flags.optimize), qualname)
  if len(expected) != 1 or function.__code__ != expected[0]:
    raise ValueError('Selected callable code differs from the approved source: '+qualname)
  return inspect.getclosurevars(function).nonlocals


class Binding:
  def __init__(self, model, root):
    self.model, self.root, self.outer = model, Path(root), model.quantize
    vision = approved_function(self.outer, root, 'batched-vision.py', 'install.<locals>.quantize', model)
    if vision.get('model') is not model or vision.get('state') != {'images': 0}:
      raise ValueError('Expected idle current-model batched-vision wrapper')
    self.vision_state, self.inner = vision['state'], vision['old_quantize']
    packed = approved_function(self.inner, root, 'quantize-packed.py', 'install.<locals>.quantize_input', model)
    if packed.get('model') is not model: raise ValueError('Packed quantizer closure belongs to another model')
    self.state = packed.get('state')
    if self.state != dict(prefill=False, prefix=None, index=0) or set(self.state) != {'prefill', 'prefix', 'index'}:
      raise ValueError('Expected idle packed-prefill quantizer state')
    self.fixed = packed.get('fixed')
    if not isinstance(self.fixed, dict) or not self.fixed:
      raise ValueError('This bounded observer requires the selected fixed prefill scales')
    outer_linear = approved_function(model.linear, root, 'batched-vision.py', 'install.<locals>.linear', model)
    inner_linear = approved_function(outer_linear['old_linear'], root, 'quantize-packed.py', 'install.<locals>.linear', model)
    layer = approved_function(model.text_layer, root, 'quantize-packed.py', 'install.<locals>.text_layer', model)
    if (outer_linear['state'] is not self.vision_state or inner_linear['state'] is not self.state
        or layer['state'] is not self.state):
      raise ValueError('Quantizer, layer, and linear wrappers do not share the actual selected states')
    self.quantize_function = self.inner.__func__.__globals__['quantize']
    approved_function(self.quantize_function, root, 'quantize-packed.py', 'quantize')
    self.packed_class = self.inner.__func__.__globals__['PackedActivation']
    approved_function(self.packed_class.__init__, root, 'quantize-packed.py', 'PackedActivation.__init__')
    if (outer_linear['old_linear'].__func__.__globals__ is not self.inner.__func__.__globals__
        or outer_linear['old_linear'].__func__.__globals__['PackedActivation'] is not self.packed_class):
      raise ValueError('PackedActivation class identity differs across selected call sites')
    self.codes = (self.outer.__func__.__code__, self.inner.__func__.__code__, self.quantize_function.__code__)
    self.evidence = dict(outer_module=self.outer.__func__.__module__, outer_function=self.outer.__func__.__qualname__,
      inner_module=self.inner.__func__.__module__, inner_function=self.inner.__func__.__qualname__,
      outer_state_keys=sorted(self.vision_state), inner_state_keys=sorted(self.state),
      source_sha256=APPROVED, code_objects_match_approved_source=True, shared_layer_linear_quantizer_states=True)

  def quantize_once(self, x):
    outer = self.model.quantize
    if (not inspect.ismethod(outer) or outer.__self__ is not self.model or outer.__func__ is not self.outer.__func__
        or self.vision_state != {'images': 0}
        or tuple(f.__code__ for f in (self.outer.__func__, self.inner.__func__, self.quantize_function)) != self.codes
        or self.inner.__func__.__globals__['quantize'] is not self.quantize_function
        or self.inner.__func__.__globals__['PackedActivation'] is not self.packed_class):
      raise ValueError('Selected quantizer binding/code changed or vision context is active')
    if not self.state['prefill'] or self.state['index'] != 3:
      raise ValueError('Observer must enter the actual language prefill slot3')
    key = self.state['prefix']+'/3'
    if key not in self.fixed: raise ValueError('Missing actual fc2 calibration key')
    result = self.outer(x)  # Exactly one original OUTER call; it delegates once to the original inner quantizer.
    if (not isinstance(result, tuple) or len(result) != 2 or type(result[0]) is not self.packed_class
        or result[1] is not self.fixed[key] or self.state['index'] != 4):
      raise ValueError('Actual slot3 PackedActivation/scale identity or single-delegation contract failed')
    return result


def make_observer(model, core, root, cpu=False):
  if sha(core.__file__) != CORE_SHA: raise ValueError('The GPU-tested statistics/oracle source changed')
  binding = Binding(model, root)
  # The unchanged constructor inspects only its quantizer's closure. Present the
  # VERIFIED inner bound method during construction, with no quantization calls.
  # Inference always sees the original outer method and delegates through it.
  model.quantize = binding.inner
  try: observer = core.Observer(model, cpu=cpu)
  finally: model.quantize = binding.outer
  observer.quantize = binding.quantize_once
  observer.quantizer_binding = binding.evidence
  return observer


def verify_baseline(plan_path, controls_path, reference_path):
  plan = json.loads(Path(plan_path).read_text())
  root = Path(plan['pipeline']).parent
  runner = load(root/'agent-activation-sparsity-run.py', 'sparsity_immutable_runner')
  runner.verify(plan)  # All old sources, inputs, checkpoint/cache, noise, and calibration bytes.
  controls = json.loads(Path(controls_path).read_text())
  if controls.get('cpu_only') is not False or not controls.get('complete') or not controls.get('all_pass'):
    raise ValueError('Require the original completed actual-GPU controls')
  for name, expected in controls['sources'].items():
    path = root/name
    if plan['sources'].get(str(path)) != expected or sha(path) != expected:
      raise ValueError('Original tested source changed: '+name)
  if controls['sources'].get('agent-activation-sparsity.py') != CORE_SHA:
    raise ValueError('GPU report does not qualify the unchanged statistics engine')
  for path, expected in controls['quantizer_dependencies'].items():
    if plan['sources'].get(path) != expected or sha(path) != expected:
      raise ValueError('Original selected quantizer/GEMM dependency changed: '+path)
  reference = json.loads(Path(reference_path).read_text())
  if (reference.get('mode') != 'reference' or not reference.get('complete')
      or reference.get('plan_id') != plan['identity_sha256'] or reference.get('plan') != plan
      or len(reference.get('queries', [])) != 10):
    raise ValueError('Require the completed original two-warmup/eight-query reference and identical plan')
  samples = [row['path'] for row in plan['selected_inputs']]
  if [(q['index'], q['sample'], q['phase']) for q in reference['queries']] != [
      (i, samples[i % 8], 'warmup' if i < 2 else 'diagnostic') for i in range(10)]:
    raise ValueError('Reference query schedule changed')
  for q in reference['queries']:
    if q['prediction']['sha256'] != identity(q['prediction']['values']): raise ValueError('Reference prediction identity mismatch')
  return runner, plan, root


def prepare(args):
  _, plan, root = verify_baseline(args.baseline_plan, args.kernel_report, args.compare_reference)
  cpu = json.loads(args.cpu_report.read_text())
  if not cpu.get('cpu_only') or not cpu.get('complete') or not cpu.get('all_pass'):
    raise ValueError('Require completed actual-wrapper CPU regression report')
  files = {p.name: sha(p) for p in HERE.glob('agent-activation-sparsity-closure*.py')}
  if cpu['overlay_sources'] != files or cpu['baseline_sources'] != {name: sha(root/name) for name in APPROVED}:
    raise ValueError('Actual-wrapper CPU qualification differs from the frozen adapter/source bytes')
  manifest = dict(schema=SCHEMA, baseline_plan=dict(path=str(args.baseline_plan.resolve()), sha256=sha(args.baseline_plan),
    identity=plan['identity_sha256']), kernel_report=dict(path=str(args.kernel_report.resolve()), sha256=sha(args.kernel_report)),
    reference=dict(path=str(args.compare_reference.resolve()), sha256=sha(args.compare_reference)),
    cpu_report=dict(path=str(args.cpu_report.resolve()), sha256=sha(args.cpu_report)),
    overlay_sources={str(HERE/name): value for name, value in files.items()}, numeric_source_sha256=CORE_SHA,
    original_plan_and_evidence_immutable=True, unchanged_gpu_controls_reused=True,
    overlay_gpu_validated=False, bitexact_current_input_reference_gate_required=True)
  manifest['identity_sha256'] = identity(manifest)
  return manifest


def verify_overlay(manifest):
  if manifest['schema'] != SCHEMA or identity({k:v for k,v in manifest.items() if k != 'identity_sha256'}) != manifest['identity_sha256']:
    raise ValueError('Invalid observer-overlay plan')
  for key in ('baseline_plan', 'kernel_report', 'reference', 'cpu_report'):
    if sha(manifest[key]['path']) != manifest[key]['sha256']: raise ValueError('Overlay evidence changed: '+key)
  for path, expected in manifest['overlay_sources'].items():
    if sha(path) != expected: raise ValueError('Observer overlay source changed: '+path)
  if str(Path(__file__).resolve()) not in manifest['overlay_sources']:
    raise ValueError('Run the exact frozen overlay entry point recorded in its plan')
  return verify_baseline(manifest['baseline_plan']['path'], manifest['kernel_report']['path'], manifest['reference']['path'])


def execute(args, manifest):
  if args.output.exists(): raise FileExistsError('Use a new observer-overlay output; preserve all existing evidence')
  runner, plan, root = verify_overlay(manifest)
  original_local = runner.local
  core = original_local('agent-activation-sparsity')
  observers = []
  def attach(model):
    observer = make_observer(model, core, root)
    observers.append(observer)
    return observer
  def local(name):
    return SimpleNamespace(Observer=attach) if name == 'agent-activation-sparsity' else original_local(name)
  options = SimpleNamespace(mode='observe', output=args.output,
    kernel_report=Path(manifest['kernel_report']['path']), compare_reference=Path(manifest['reference']['path']))
  try:
    with patch.object(runner, 'local', local): runner.execute(options, plan)
  finally:
    if args.output.exists():
      report = json.loads(args.output.read_text())
      unchanged_controls = sha(manifest['kernel_report']['path']) == manifest['kernel_report']['sha256']
      unchanged_reference = sha(manifest['reference']['path']) == manifest['reference']['sha256']
      if not unchanged_controls or not unchanged_reference:
        report['complete'] = False
        report['error'] = dict(type='EvidenceChanged', message='Original GPU controls/reference changed during observation')
      report['observer_overlay'] = dict(plan=manifest, bindings=[o.quantizer_binding for o in observers],
        overlay_executed=True, execution_completed=bool(report.get('complete') and len(observers) == 1),
        original_gpu_report_modified=not unchanged_controls, original_reference_modified=not unchanged_reference)
      args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--plan', type=Path)
  parser.add_argument('--prepare', action='store_true')
  parser.add_argument('--preflight', action='store_true')
  parser.add_argument('--baseline-plan', type=Path, default=OLD/'plan.json')
  parser.add_argument('--kernel-report', type=Path, default=WORK/'activation-sparsity-devicefix-controls/report.json')
  parser.add_argument('--compare-reference', type=Path, default=WORK/'activation-sparsity-devicefix-reference/report.json')
  parser.add_argument('--cpu-report', type=Path, default=HERE/'agent-activation-sparsity-closure-cpu.json')
  args = parser.parse_args()
  if args.prepare:
    if args.output.exists(): raise FileExistsError('Use a new observer overlay plan')
    manifest = prepare(args);args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(cpu_only=True, baseline_and_evidence_verified=True, identity=manifest['identity_sha256'])));return
  if args.plan is None: parser.error('--plan is required')
  manifest = json.loads(args.plan.read_text())
  if args.preflight:
    verify_overlay(manifest);print(json.dumps(dict(cpu_only=True, all_provenance_verified=True)));return
  execute(args, manifest)


if __name__ == '__main__': main()

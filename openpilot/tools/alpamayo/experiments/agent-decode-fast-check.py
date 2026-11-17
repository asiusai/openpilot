"""CPU admission/fallback mocks and offline gfx1200 compilation. Never opens AMD."""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  return module


def rejects(fn):
  try: fn()
  except (ValueError, KeyError, AssertionError): return
  raise AssertionError('Invalid evidence accepted')


def gates(hook, check, report):
  environment = report['environment']
  current = check.provenance()
  choices = {'16,2048,2048': '1x2'}
  validate = lambda r: hook.validate(r, choices, current, environment)
  assert validate(report) == {(16, 2048, 2048): '1x2'}
  assert hook.validate(report, {'16,2048,2048': '1x1-accum'}, current, environment)
  rejects(lambda: hook.validate(report, {'16,2048,2048': '1x1'}, current, environment))
  rejects(lambda: hook.validate(report, {'32,2048,2048': '1x2'}, current, environment))
  def put(path, value):
    r = deepcopy(report); dest = r
    for key in path[:-1]: dest = dest[key]
    dest[path[-1]] = value
    rejects(lambda: validate(r))
  invalid = [(['complete'], False), (['strict_equality_required'], False), (['gpu_executed'], False),
    (['provenance', 'tinygrad_revision'], 'changed'), (['environment', 'TC_OPT'], '1'),
    (['controls'], report['controls'][:-1]),
    (['controls', 0, 'variants', '1x2', 'status'], 'rejected'),
    (['controls', 0, 'variants', '1x2', 'checks', 2, 'exact_bf16'], False),
    (['controls', 0, 'checks', 2, 'independent_oracle'], False),
    (['cases', 0, 'variants', '1x2', 'status'], 'rejected'),
    (['cases', 0, 'variants', '1x4', 'status'], 'rejected'),
    (['cases', 0, 'variants', '1x2', 'failures'], [{'stage': 'post_timing'}]),
    (['cases', 0, 'variants', '1x2', 'checks'], report['cases'][0]['variants']['1x2']['checks'][:-1]),
    (['cases', 0, 'variants', '1x2', 'checks', 0, 'output_sha256'], '0'*64),
    (['cases', 0, 'checks', 2, 'isolated_row_scale'], False),
    (['cases', 0, 'serial_proofs', '1x2', 'ordered_execution_barriers'], 38),
    (['cases', 0, 'serial_proofs', '1x2', 'output_span_bytes'], 32),
    (['cases', 0, 'serial_proofs', '1x2', 'actual_queue_builder'], False),
    (['cases', 0, 'compiled', '1x2', 'lowered_accumulator_contract', 'running_wmma_accumulator'], False),
    (['cases', 0, 'model_identity', 'all_selected_fp8_finite'], False),
    (['cases', 0, 'timings', 'calls_per_batch'], 39),
    (['cases', 0, 'timings', 'median_ms_per_call', '1x2'], .00001)]
  for path, value in invalid: put(path, value)
  old = load('agent-decode-gemm-hook')
  rejects(lambda: old.validate(report, {'16,2048,2048': 2}, current, True, environment['TC_OPT']))
  assert report['cases'][0]['variants']['1x1']['failures'][0]['mismatched_bf16'] == 13
  assert '1x1' not in report['cases'][0]['timings']['median_ms_per_call']
  return dict(real_report_admitted=['1x2', '1x1-accum'], negative_admission_tests=len(invalid)+3,
    original_1x1_rejection_preserved=True, legacy_gate_still_closed=True)


def dispatch_checks(hook):
  calls = []
  sentinel = object()
  original = SimpleNamespace(__file__=str(HERE/'tiled-gemm.py'), gemm=lambda *a, **kw: calls.append((a, kw)) or sentinel)
  methods = [object() for _ in range(5)]
  decoder = SimpleNamespace(tile=(1, 4), verifiers={}, gemm=original,
    model=SimpleNamespace(quantize=methods[0], linear=methods[1], attention=methods[2]),
    generate=methods[3], decode=methods[4])
  def tensor(shape): return SimpleNamespace(shape=shape)
  args = tensor((16, 2048)), tensor((2048, 2048)), (tensor((16, 1)), tensor((1,)))
  with hook.scoped(decoder, {(16, 2048, 2048): '1x2'}) as stats:
    assert decoder.gemm.gemm(*args, tile=(1, 4), row_scale=True) is sentinel
    assert calls[-1][0] == args and calls[-1][1]['tile'] == (1, 2)
    for shape in ((32, 2048), (48, 2048), (64, 2048)):
      decoder.gemm.gemm(tensor(shape), args[1], (tensor((shape[0], 1)), args[2][1]), tile=(1, 4), row_scale=True)
      assert calls[-1][1]['tile'] == (1, 4)
    bias = object()
    decoder.gemm.gemm(*args, tile=(1, 4), row_scale=True, bias=bias)
    assert calls[-1][1]['tile'] == (1, 4) and calls[-1][1]['bias'] is bias
    require_unchanged = lambda: [decoder.model.quantize, decoder.model.linear, decoder.model.attention, decoder.generate, decoder.decode] == methods
    assert require_unchanged()
    reset = []
    decoder.verifiers['mock'] = SimpleNamespace(reset=lambda: reset.append(True))
  assert decoder.gemm is original and decoder.verifiers == {} and reset == [True] and stats['restored'] and require_unchanged()
  try:
    with hook.scoped(decoder, {(16, 2048, 2048): '1x2'}): raise RuntimeError('mock failure')
  except RuntimeError: pass
  assert decoder.gemm is original
  decoder.verifiers[0] = object()
  rejects(lambda: hook.scoped(decoder, {(16, 2048, 2048): '1x2'}).__enter__())
  return dict(original_argument_identity=True, shape_and_bias_fallback=True, quantizers_attention_generate_decode_untouched=True,
              restoration_on_exception=True, preexisting_verifiers_rejected=True)


def compile_integration(hook, check, report):
  from tinygrad import Device, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.uop.ops import UOp, Ops
  import tinygrad.codegen as codegen
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = check.local('wgp_compiler').WGPCompiler('gfx1200')
  old_matchers = codegen.pm_reduce_local, codegen.pm_expand_broadcast
  compiled = []
  for m in (16, 32, 48):
    for n in (2048, 1024):
      shape = (m, n, 2048)
      for label in ('1x2', '1x1-accum'):
        module = load('tiled-gemm')
        module.kernel = check.bind_renderer(module.kernel, renderer)
        captured, packs = [], []
        class Tensor:
          @staticmethod
          def empty(*shape, **kwargs): return SimpleNamespace(shape=shape)
          @staticmethod
          def custom_kernel(*args, fxn): captured.append((args, fxn)); return ('output',)
        def pack(value): packs.append(value); return value
        module.gemm = FunctionType(module.gemm.__code__, dict(module.gemm.__globals__, Tensor=Tensor, pack=pack),
                                  module.gemm.__name__, module.gemm.__defaults__)
        decoder = SimpleNamespace(gemm=module, tile=(1, 4), verifiers={})
        x, w, sx, sw = [SimpleNamespace(shape=s, device='CPU') for s in ((m, 2048), (n, 2048), (m, 1), (1,))]
        with hook.scoped(decoder, {shape: label}, renderer=renderer):
          assert decoder.gemm.gemm(x, w, (sx, sw), tile=(1, 4), row_scale=True) == 'output'
        assert packs == [x, w] and captured[0][0][1:] == (x, w, sx, sw)
        args = [UOp.placeholder(s, dt, i) for i, (s, dt) in enumerate([
          ((m, n), dtypes.bfloat16), ((m, 2048), dtypes.fp8e4m3), ((n, 2048), dtypes.fp8e4m3),
          ((m, 1), dtypes.float32), ((1,), dtypes.float32)])]
        ast = captured[0][1](*args)
        program = ast if ast.op is Ops.PROGRAM else to_program(ast, renderer)
        binary = check.sha(program.src[3].arg)
        source = check.sha(next(u.arg for u in program.src if u.op is Ops.SOURCE).encode())
        reference = check.compile_one(shape, False, check.kernels(renderer)[label], renderer)
        assert binary == reference['binary_sha256'] and source == reference['source_sha256']
        if shape == (16, 2048, 2048):
          assert binary == report['cases'][0]['compiled'][label]['binary_sha256']
          assert source == report['cases'][0]['compiled'][label]['source_sha256']
        compiled.append(dict(shape=list(shape), variant=label, binary_sha256=binary, source_sha256=source,
                             wrapper_matches_frozen_harness=True))
  assert old_matchers == (codegen.pm_reduce_local, codegen.pm_expand_broadcast)
  assert not any('AMD' in d for d in Device._opened_devices)
  return compiled


def policy_checks(route):
  chain = route.inspect(WORK/'dynamic-vision-full-route/invocation.json')
  assert len(chain) == 2
  rejects(lambda: route.inspect(WORK/'combined-prefill-quality/invocation.json'))
  return dict(recursive_dynamic_vision_accepted=True, actual_fixed_vision_quality_rejected=True)


def audit_checks(route):
  import numpy as np
  from tinygrad.uop.ops import UOp
  class Array:
    def __init__(self, value): self.value = np.asarray(value, np.float32); self.shape = self.value.shape
    def float(self): return self
    def contiguous(self): return self
    def realize(self): return self
    def __getitem__(self, index): return Array(self.value[index])
  result = (Array(np.arange(8).reshape(1, 2, 4)), Array([[3, 3]]))
  cache = Array(np.arange(96).reshape(2, 1, 2, 12, 2))
  records = []
  decoder = SimpleNamespace(readback=lambda x: x.value, model=SimpleNamespace(cache=[cache]))
  proxy = route.AuditJit(lambda *_: result, decoder, records)
  cos, sin, tokens = Array(np.ones((3, 1, 12, 2))), Array(np.zeros((3, 1, 12, 2))), Array([11, 12])
  position, step = UOp.variable('fast_test_position', 0, 10).bind(3), UOp.variable('fast_test_step', 0, 10).bind(1)
  assert proxy(tokens, cos, sin, position, step) is result
  assert records[0]['position'] == 3 and records[0]['step'] == 1 and records[0]['written_kv'][0]['shape'] == [2, 1, 2, 2, 2]
  old = deepcopy(records)
  proxy(tokens, cos, sin, position, step)
  assert records == old+old
  cache.value[:, :, :, 3] += 1
  proxy(tokens, cos, sin, position, step)
  assert records[2]['written_kv'] != old[0]['written_kv']
  row = dict(sample='mock', draft=dict(lookup_rounds=1, verified_tokens=2, serial_tokens=6,
    prefill_draft=dict(accepted=False, final_cache_length=11, action_rotary_offset=8)),
    prediction=dict(generated_token_ids=[1, 2], reasoning='mock', xyz=[[0., 1., 2.]], rot=[[1., 0., 0.]]))
  report = dict(warmup=[], runs=[row], agent_decode_fast=dict(decoders=[dict(audit=old)]))
  protocol = dict(samples=['mock'], audit=True)
  assert route.compare(report, deepcopy(report), protocol)['all_exact']
  for field in ('generated_token_ids', 'reasoning', 'xyz', 'rot'):
    changed = deepcopy(report)
    changed['runs'][0]['prediction'][field] = {'generated_token_ids': [1, 3], 'reasoning': 'different',
      'xyz': [[0., 1., 2.0000000000000004]], 'rot': [[1., -0., 0.]]}[field]
    rejects(lambda: route.compare(report, changed, protocol))
  changed = deepcopy(report); changed['runs'][0]['draft']['serial_tokens'] += 1
  rejects(lambda: route.compare(report, changed, protocol))
  changed = deepcopy(report); changed['agent_decode_fast']['decoders'][0]['audit'] = records[-1:]
  rejects(lambda: route.compare(report, changed, protocol))
  return dict(bound_mrope_variables=True, actual_call_return_unchanged=True, written_cache_mutation_detected=True,
              tokens_text_xyz_rotation_fallback_and_verifier_changes_rejected=True)


def interceptor_checks(route, hook):
  import tinygrad
  original_device = tinygrad.Device
  original_spec = importlib.util.spec_from_file_location
  original = SimpleNamespace(__file__=str(HERE/'tiled-gemm.py'), gemm=lambda *a, **kw: kw['tile'])
  calls, records = [], []
  class Devices:
    DEFAULT = 'mock'
    def __getitem__(self, _):
      return SimpleNamespace(arch='gfx1200', is_usb=True, renderer=SimpleNamespace(compiler=type('WGPCompiler', (), {})()))
  try:
    with route.intercept(HERE/'lookup-decode.py', hook, {(16, 2048, 2048): '1x2'}, 'candidate', False, records):
      spec = importlib.util.spec_from_file_location('fast_lookup_mock', HERE/'lookup-decode.py')
      module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
      cls = module.LookupDecoder
      # Replace only this freshly loaded mock base method, never a shared module.
      def run(self):
        calls.append(True)
        return self.gemm.gemm(SimpleNamespace(shape=(16, 2048)), SimpleNamespace(shape=(2048, 2048)),
          (SimpleNamespace(shape=(16, 1)), SimpleNamespace(shape=(1,))), tile=(1, 4), row_scale=True)
      cls.__mro__[1].run = run
      decoder = object.__new__(cls)
      decoder.tile, decoder.verifiers, decoder.gemm = (1, 4), {}, original
      tinygrad.Device = Devices()
      assert decoder.run() == decoder.run() == (1, 2)
      assert len(records) == 1 and len(calls) == 2
      assert decoder.gemm is not original
    assert decoder.gemm is original and records[0]['hook']['restored']
  finally: tinygrad.Device = original_device
  assert importlib.util.spec_from_file_location is original_spec
  return dict(delayed_install_before_first_run=True, installs_once=True, import_interceptor_and_decoder_restored=True,
              mocked_device_only=True)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--evidence', type=Path, default=WORK/'agent-decode-gemm-rounding-retry/report.json')
  args = p.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', TC_OPT='2', BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0')
  os.environ.pop('LLVMOPT', None)
  hook, check, route = load('agent-decode-fast-hook'), load('agent-decode-gemm'), load('agent-decode-fast-route')
  report = json.loads(args.evidence.read_text())
  result = dict(gpu_executed=False, evidence_sha256=hook.sha(args.evidence.read_bytes()),
    admission=gates(hook, check, report), dispatch=dispatch_checks(hook), policy=policy_checks(route),
    audit=audit_checks(route), interceptor=interceptor_checks(route, hook),
    compilation=compile_integration(hook, check, report), route_and_timing_unverified=True)
  from tinygrad import Device
  assert not any('AMD' in d for d in Device._opened_devices)
  result['opened_devices'] = sorted(Device._opened_devices)
  args.output.write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps(dict(gpu_executed=False, compilations=len(result['compilation']), admission=result['admission'])))


if __name__ == '__main__': main()

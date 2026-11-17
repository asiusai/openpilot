"""Offline checks for the observation hooks, metadata gates, and bounded plan."""
import argparse
import builtins
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace, FunctionType

HERE = Path(__file__).resolve().parent


def rejects(f):
  try: f()
  except (AssertionError, ValueError): return True
  raise AssertionError('A negative control was accepted')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  assert not args.output.exists()
  inv = importlib.import_module('agent-compiler-mode-inventory')
  proof = importlib.import_module('agent-compiler-mode-proof')
  fixed = importlib.import_module('agent-compiler-mode-renderer')
  from tinygrad import Device, dtypes
  from tinygrad.codegen import do_to_program, to_program_context
  from tinygrad.engine import realize, jit
  from tinygrad.helpers import ALLOW_DEVICE_USAGE, Context, Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer as Base
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  assert ALLOW_DEVICE_USAGE.value == 0 and not Device._opened_devices
  target = Target('AMD', 'LLVM', 'gfx1200', 'USB')
  renderer, cu_renderer = fixed.AMDLLVMRenderer(target), Base(target)
  ast = proof.ast_for(259)
  wgp, cu = (do_to_program(ast, r) for r in (renderer, cu_renderer))
  assert inv.metadata(wgp.src[-1].arg)['workgroup_processor_mode'] == 1
  assert inv.metadata(cu.src[-1].arg)['workgroup_processor_mode'] == 0
  records = inv.Inventory(HERE/'agent-compiler-mode-check-artifacts')
  records.record(wgp)
  # Construct queue metadata without allocating or opening any device.
  arguments = [UOp.param(p.arg.slot, p.dtype, p.max_numel(), device='AMD') for p in wgp.kernel_params]
  call = wgp.call(*arguments)
  batch = object.__new__(hcq2.BatchCtx)
  batch.batch, batch.profile = [(call, 'AMD', 'COMPUTE:0')]*2, False
  batch.queues, batch.calls, batch.peers = {'AMD': ['COMPUTE:0']}, {('AMD', 'COMPUTE:0'): [0, 1]}, {('AMD', 'COMPUTE:0'): set()}
  batch.slots = {'AMD': UOp.alloc((4,), dtypes.uint64, device='AMD', name='compiler_mode_slots')}
  original_queue = hcq2._build_queues(batch)
  observed_queue = records.build(hcq2._build_queues, batch)
  assert original_queue == observed_queue
  linear = UOp(Ops.LINEAR, src=(call,))
  assert records.before_after_compile(realize.lower_and_compile, linear) is linear
  row = ('AMD', wgp.src[0].arg.function_name, None, (), wgp.key, (0, 1), ((0,), (1,)))
  copy_row = ('AMD', 'copy', None, (), None, (0, 1), ((0,), (1,)))
  host_row = ('CPU', 'host', None, (), bytes(32), (0,), ((0,), ()))
  capture = UOp.sink(arg=KernelInfo()).call(aux=hcq2.HCQInfo(('AMD',), kernels=(row, copy_row, host_row)))
  fake_jit = SimpleNamespace(captured=SimpleNamespace(_linear=UOp(Ops.LINEAR, src=(capture,))), fxn=lambda: None)
  records.captured(fake_jit)
  records.captured(fake_jit)
  assert len(records.captures) == 1 and len(records.captures[0]['kernels']) == 1
  bad = UOp.sink(arg=KernelInfo()).call(aux=hcq2.HCQInfo(('AMD',), kernels=(('AMD', *row[1:4], bytes(32), *row[5:]),)))
  rejects(lambda: records.captured(SimpleNamespace(captured=SimpleNamespace(_linear=UOp(Ops.LINEAR, src=(bad,))), fxn=lambda: None)))
  old_hooks = (realize.lower_and_compile, hcq2.lower_and_compile, hcq2._build_queues, jit._TinyJit.__call__)
  try:
    with records.observe(): raise RuntimeError('exercise restoration')
  except RuntimeError: pass
  assert old_hooks == (realize.lower_and_compile, hcq2.lower_and_compile, hcq2._build_queues, jit._TinyJit.__call__)
  # Test the stopping boundary with a synthetic main() whose filename is pinned.
  filename = HERE/'agent-compiler-mode-synthetic-boundary.py'
  saved = []
  def save(value): saved.append(copy.deepcopy(value))
  emit = inv.stop_after_query(lambda *a, **kw: None, filename, save)
  namespace = {'emit': emit}
  code = "def main():\n  report={'warmup':[{},{}], 'runs':[]}\n  iteration=1\n  emit('Iteration 1: done')\n  raise AssertionError('stop missed')\n"
  exec(compile(code, str(filename), 'exec'), namespace)
  try: namespace['main']()
  except inv.TwoQueriesComplete: pass
  else: raise AssertionError('Boundary did not stop')
  assert saved == [{'warmup': [{}, {}], 'runs': []}]
  # One actual packed FC2 AST, with unchanged frozen code in a private namespace.
  gemm_path = inv.SNAP/'amd_gemm.py'
  gemm = inv.load(gemm_path, 'agent_compiler_mode_frozen_gemm')
  class Devices:
    DEFAULT = 'AMD'
    def __getitem__(self, name): return SimpleNamespace(renderer=renderer)
  function = FunctionType(gemm._gemm.__code__, dict(gemm._gemm.__globals__, Device=Devices()), gemm._gemm.__name__)
  function.__kwdefaults__ = gemm._gemm.__kwdefaults__
  specs = (((2048, 2048), dtypes.bfloat16), ((2048, 9216), dtypes.fp8e4m3),
           ((2048, 9216), dtypes.fp8e4m3), ((1,), dtypes.float32), ((1,), dtypes.float32))
  fc2_ast = function(*(UOp.placeholder(shape, dtype, i) for i, (shape, dtype) in enumerate(specs)))
  legacy = Base(target); legacy.compiler = fixed.WGPCompiler('gfx1200')
  fc2_wgp, fc2_cu, fc2_legacy = [do_to_program(fc2_ast, r) for r in (renderer, cu_renderer, legacy)]
  assert proof.describe(fc2_wgp) == proof.describe(fc2_legacy)
  for i in (0, 1, 2): assert fc2_wgp.src[i] is fc2_cu.src[i]
  assert fc2_wgp.arg == fc2_cu.arg and fc2_wgp.src[3].arg != fc2_cu.src[3].arg
  fc2_context = {v.key: v.value for v in to_program_context}
  from tinygrad.engine import worker
  with Context(PARALLEL=2):
    try:
      fc2_spawned = next(worker.get_worker_pool().imap_unordered(proof.worker_task, [(0, (fc2_ast, renderer), fc2_context)]))
      assert fc2_spawned['program'] == proof.describe(fc2_wgp)
      assert fc2_spawned['renderer'] == proof.identity(renderer) and fc2_spawned['context'] == fc2_context
    finally: worker.terminate_worker_pool()
  fc2_rows = {}
  for name, program in (('wgp', fc2_wgp), ('cu', fc2_cu)):
    key = records.record(program)
    fc2_rows[name] = records.programs[key]
  assert fc2_rows['wgp']['resources']['workgroup_processor_mode'] == 1
  assert fc2_rows['cu']['resources']['workgroup_processor_mode'] == 0
  def comparison_fixture(name):
    row = fc2_rows[name]
    return {'inventory': {'programs': {row['key']: row}, 'compile_records': [
      {'input_ast_key': fc2_ast.key.hex(), 'origin': 'renderer_compilation', 'program_key': row['key'],
       'context': fc2_context, 'parent_renderer': {'target': repr(target)}}]}}
  control, candidate = comparison_fixture('cu'), comparison_fixture('wgp')
  assert len(inv.compare(control, candidate)['changed_programs']) == 1
  for field in ('lowered_ast', 'linear', 'source_sha256', 'program_info', 'params'):
    bad_candidate = copy.deepcopy(candidate)
    next(iter(bad_candidate['inventory']['programs'].values()))[field] = 'changed'
    rejects(lambda: inv.compare(control, bad_candidate))
  # Use the original strict comparator. Signed-zero, token, and field mutations fail.
  route = inv.load(inv.SNAP/'experiments/agent-expert-pv-route-run.py', 'agent_compiler_mode_comparison_test')
  base = {'sample': 'one', 'prediction': {'xyz': [0.0], 'rot': [1.0], 'generated_token_ids': [3], 'timings_ms': {}}}
  for mutated in (dict(base, sample='two'), dict(base, prediction=dict(base['prediction'], xyz=[-0.0])),
                  dict(base, prediction=dict(base['prediction'], generated_token_ids=[4])),
                  dict(base, prediction=dict(base['prediction'], xyz=[float('nan')]))):
    rejects(lambda m=mutated: (_ for _ in ()).throw(AssertionError()) if route.canonical_row(base) != route.canonical_row(m) else None)
  # Saved actual evidence is read only. Do not rerun the separate PV binding task.
  evidence_path = inv.WORK/'agent-expert-pv-binding-kernels-pathfix/report.json'
  evidence = json.loads(evidence_path.read_text())
  actual = evidence['actual_reference']
  assert actual['target'] == repr(target)
  assert actual['binary_sha256'] == '66dd48e6cd81db1e0397f4cba8268e3da6037a8a81c02f6c1325d896d1ef49b4'
  assert actual['resources']['workgroup_processor_mode'] == 0
  assert evidence['capture_counts']['alpamayo_attention_values'] == 84
  assert evidence['runtime']['compiler_cachekey'] == renderer.compiler.cachekey
  assert not Device._opened_devices
  result = {'complete': True, 'cpu_only': True, 'gpu_executed': False, 'opened_devices': [],
    'actual_build_queues_unchanged': True, 'already_complete_program_identity_unchanged': True,
    'captured_amd_binary_coverage_and_missing_key_rejection': True, 'hooks_restore_on_failure': True,
    'exact_two_query_stop_boundary': True, 'finite_signedzero_token_field_negative_controls': True,
    'metadata_and_hardware_descriptor_agree': {'wgp': inv.metadata(wgp.src[-1].arg), 'cu': inv.metadata(cu.src[-1].arg)},
    'fc2_cpu_only': {'shape_m_n_k': [2048, 2048, 9216], 'input_ast_key': fc2_ast.key.hex(),
      'fixed_class_equals_independent_original_wgp': True, 'cu_wgp_same_ast_source_arguments': True,
      'spawned_actual_compile_task_matches': fc2_spawned, 'programs': fc2_rows,
      'program_comparison_mutations_rejected': True, 'not_an_actual_route_capture': True},
    'supporting_actual_capture': {'report': str(evidence_path), 'sha256': inv.sha(evidence_path.read_bytes()),
      'program_key': actual['program_key'], 'target': actual['target'], 'binary_sha256': actual['binary_sha256'],
      'parent_compiler_cachekey': evidence['runtime']['compiler_cachekey'], 'parallel': evidence['runtime']['parallel'],
      'captured_pv_calls': 84, 'diagnostic_complete': evidence['complete'], 'not_quality_evidence': True},
    'source_sha256': {str(p): inv.sha(p.read_bytes()) for p in (Path(__file__), Path(inv.__file__), Path(proof.__file__), Path(fixed.__file__), evidence_path, gemm_path)}}
  inv.write(args.output, result)
  print(json.dumps({'complete': True, 'gpu_executed': False, 'checks': len(result)-5}))


if __name__ == '__main__': main()

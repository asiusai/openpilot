"""CPU-only dispatch, original-body, binary, mutation, capture, and route-gate checks."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def load(name):
  spec = importlib.util.spec_from_file_location('pv_route_check_'+name.replace('-', '_'), HERE/(name+'.py'))
  value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


def reject(fn):
  try: fn()
  except (AssertionError, KeyError, ValueError): return
  raise AssertionError('Invalid case passed')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--evidence', type=Path, default=WORK/'agent-expert-pv-kernels/report.json')
  args = parser.parse_args()
  os.environ.update(DEV='CPU', PROFILE='0', PYTHONDONTWRITEBYTECODE='1')
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.uop.ops import KernelInfo, UOp
  from model import EdgeModel
  import amd_attention
  H, R, E = (load(n) for n in ('agent-expert-pv-route-hook', 'agent-expert-pv-route-run', 'agent-expert-gemm'))
  evidence = H.verify_evidence(args.evidence)
  def make(candidate):
    model = EdgeModel.__new__(EdgeModel)
    model.fp8_wmma, model.dtype, model.weights = True, dtypes.bfloat16, {}
    model.config = {'expert_config': {'llm_config': {'num_hidden_layers': 28}}}
    model.trajectory_jits = {}
    model.cache = [SimpleNamespace(shape=(2, 1, 8, 2304, 128))]*28
    expert = load('expert-kernels'); expert.install(model)
    before = (model.linear, model.quantize, model.text_layer, expert.attention, expert._pack, expert._softmax)
    handle = H.install(model, expert, candidate=candidate, evidence=args.evidence)
    assert before == (model.linear, model.quantize, model.text_layer, expert.attention, expert._pack, expert._softmax)
    return model, expert, handle
  shapes = ((1, 16, 64, 128), (1, 16, 64, 2368), (1, 8, 8, 148, 4, 16, 4))
  buffers = [UOp.placeholder(s, dtypes.bfloat16, i) for i, s in enumerate(shapes)]
  checks = []
  for candidate in (False, True):
    model, expert, handle = make(candidate)
    original, numeric = handle.original_matmul, handle.numeric.kernel
    calls = []
    handle.original_matmul = lambda *a, **kw: calls.append(('original', a, kw)) or 'original'
    handle.numeric.kernel = lambda *a, **kw: calls.append(('candidate', a, kw)) or 'candidate'
    assert expert._matmul(*buffers, shape=H.SHAPE) == 'original'
    handle.depth = 1
    assert expert._matmul(*buffers, shape=H.SHAPE) == ('candidate' if candidate else 'original')
    for shape, causal, scaled in ((H.QK_SHAPE, False, True), (H.SHAPE, True, False), (H.SHAPE, False, True),
      ((2, 16, 8, 64, 128, 2368), False, False), ((1, 16, 8, 32, 128, 2368), False, False),
      ((1, 16, 8, 64, 128, 2112), False, False)):
      assert expert._matmul(*buffers, shape=shape, causal=causal, scaled=scaled) == 'original'
    for index in range(3):
      changed = buffers.copy(); changed[index] = UOp.placeholder(shapes[index], dtypes.float16, index)
      assert expert._matmul(*changed, shape=H.SHAPE) == 'original'
      changed[index] = UOp.placeholder((1,), dtypes.bfloat16, index)
      assert expert._matmul(*changed, shape=H.SHAPE) == 'original'
    handle.depth = 0
    # Reuse the installed trace wrapper, with a CPU stand-in for model work.
    def trace(*a, **kw):
      for _ in range(3):
        for _ in range(28):
          expert._matmul(*buffers, shape=H.QK_SHAPE, scaled=True)
          expert._matmul(*buffers, shape=H.SHAPE)
      return 'trajectory'
    handle.originals['_trajectory'] = trace
    assert model._trajectory(None, None, None, 2033, 3, 6) == 'trajectory'
    assert handle.info['traces'] == [{'qk': 84, 'pv': 84}]
    model.cache[0] = SimpleNamespace(shape=(2, 1, 8, 2048, 128))
    assert model._trajectory(None, None, None, 2033, 3, 6) == 'trajectory'
    assert handle.info['unqualified_trajectory_calls'] == 1
    model.cache[0] = SimpleNamespace(shape=(2, 1, 8, 2304, 128))
    def fail(*a, **kw): raise RuntimeError('injected')
    handle.originals['_trajectory'] = fail
    try: model._trajectory(None, None, None, 2033, 3, 6)
    except RuntimeError: pass
    assert handle.depth == 0
    handle.original_matmul, handle.numeric.kernel = original, numeric
    handle.close()
    assert expert._matmul is amd_attention._matmul and not hasattr(model, 'agent_expert_pv')
    checks.append({'candidate': candidate, 'scope_shapes_dtype_flags_and_original_body': True, 'trace_84_qk_pv': True,
                   'unqualified_trajectory_original': True, 'exception_cleanup': True})
  # Compile through the actual installed dispatch, with a private CPU renderer.
  model, expert, handle = make(True)
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  numeric = handle.numeric.kernel
  handle.numeric.kernel = E.bind_renderer(numeric, renderer)
  handle.depth = 1
  program = to_program(expert._matmul(*buffers, shape=H.SHAPE), renderer)
  assert H.sha(program.src[3].arg) == H.BINARY and H.sha(program.src[2].arg.encode()) == H.LLVM
  handle.depth = 0
  # Mutate both operands after CPU capture through the same dispatch boundary.
  def standin(out, a, b, **kwargs):
    r = UOp.range(out.numel(), 0)
    return out.flatten().index(r).store((a.flatten().index(0).float()+b.flatten().index(0).float()).cast(out.dtype)).end(r).sink(arg=KernelInfo())
  handle.numeric.kernel = standin
  def execute(a, b, out):
    handle.depth = 1
    try: return Tensor.custom_kernel(out, a, b, fxn=lambda *v: expert._matmul(*v, shape=H.SHAPE))[0].realize()
    finally: handle.depth = 0
  import numpy as np
  transfer = load('agent-vision-tiles-transfer-check').InputTransfers(Device['CPU'])
  values = [np.zeros(s, np.uint16) for s in shapes[1:]]
  inputs = transfer.upload(values, (dtypes.bfloat16, dtypes.bfloat16))
  out = Tensor.empty(shapes[0], device='CPU', dtype=dtypes.bfloat16).realize()
  jit = TinyJit(execute)
  for _ in range(3): jit(*inputs, out)
  for av, bv in ((2, 3), (7, 3), (7, -4), (2, 3)):
    values[0].flat[0], values[1].flat[0] = E.bf16(np.float32(av)), E.bf16(np.float32(bv))
    transfer.replace(inputs, values)
    result = jit(*inputs, out).bitcast(dtypes.uint16).numpy()
    np.testing.assert_array_equal(result, E.bf16(np.full(shapes[0], av+bv, np.float32)))
  handle.numeric.kernel = numeric
  # Captured metadata proves exactly 84 operations, with observed binary keys.
  pv_key, qk_key, sm_key = (bytes([i])*32 for i in (1, 2, 3))
  handle.programs.update({key.hex(): {'name': name, 'binary_sha256': digest} for key, name, digest in
    ((pv_key, H.NAME, H.BINARY), (qk_key, 'alpamayo_attention_scores', 'qk'), (sm_key, 'alpamayo_attention_softmax', 'softmax'))})
  def capture(count=84, name=H.NAME):
    records = [(None, n, None, None, key) for n, key, c in ((name, pv_key, count), ('alpamayo_attention_scores', qk_key, 84),
               ('alpamayo_attention_softmax', sm_key, 84)) for _ in range(c)]
    return SimpleNamespace(captured=SimpleNamespace(_linear=SimpleNamespace(src=[SimpleNamespace(without_after=SimpleNamespace(
      arg=SimpleNamespace(aux=SimpleNamespace(kernels=records))))])))
  handle.inspect_capture(capture())
  reject(lambda: handle.inspect_capture(capture(83)))
  reject(lambda: handle.inspect_capture(capture(85)))
  reject(lambda: handle.inspect_capture(capture(name='alpamayo_attention_values')))
  from tinygrad.runtime.support import hcq2
  before = hcq2._build_queues
  try:
    with handle.observe_programs(): raise RuntimeError('injected')
  except RuntimeError: pass
  assert hcq2._build_queues is before
  handle.close()
  # Existing exact route pair supplies realistic canonical schemas, never GPU work.
  a = json.loads((WORK/'agent-expert-tiles-quality-reference/report.json').read_text())
  b = json.loads((WORK/'agent-expert-tiles-quality-candidate/report.json').read_text())
  for r in (a, b): r.pop('agent_expert_tiles')
  spec_a = json.loads((WORK/'agent-expert-tiles-quality-reference/report.speculation.json').read_text())
  spec_b = json.loads((WORK/'agent-expert-tiles-quality-candidate/report.speculation.json').read_text())
  planned = {'warmup': 2, 'runs': 62, 'scheduled_samples': [r['sample'] for r in a['warmup']+a['runs']]}
  assert R.compare_reports(a, b, planned, spec_a, spec_b)['samples'] == 64
  rejected = []
  for field in ('token', 'text', 'xyz_zero', 'rotation_ulp', 'timestamp', 'cache_length', 'target6', 'finite_flag',
                'history_upload', 'new_prediction', 'missing_row', 'speculation', 'padding', 'fixed_vision'):
    changed, spec = copy.deepcopy(a), copy.deepcopy(spec_a)
    row = changed['runs'][0]
    if field == 'token': row['prediction']['generated_token_ids'][0] += 1
    elif field == 'text': row['prediction']['reasoning'] += 'x'
    elif field == 'xyz_zero': row['prediction']['xyz'][0][0][2] = -0.
    elif field == 'rotation_ulp': row['prediction']['rot'][0][0][0][0] = math.nextafter(row['prediction']['rot'][0][0][0][0], 2.)
    elif field == 'timestamp': row['input_latest_timestamp_us'] += 1
    elif field == 'cache_length': row['draft']['prefill_draft']['final_cache_length'] += 1
    elif field == 'target6': row['draft']['prefill_draft']['targets'][5] += 1
    elif field == 'finite_flag': row['draft']['prefill_draft']['all_targets_finite'] = False
    elif field == 'history_upload': row['uploaded_pixel_bytes'] -= 1
    elif field == 'new_prediction': row['prediction']['unrecognized_new_field'] = 1
    elif field == 'missing_row': changed['runs'].pop()
    elif field == 'speculation': spec['records'][0]['action_rotary_offset'] += 1
    elif field == 'padding': changed['pad_prompt_to'] = 2112
    else: changed['vision_calibration_sha256'] = 'rejected'
    reject(lambda: R.compare_reports(a, changed, planned, spec_a, spec))
    rejected.append(field)
  reject(lambda: H.checked('expert-kernels.py', HERE/'agent-expert-pv-route-hook.py'))
  assert not any(d != 'CPU' for d in Device._opened_devices)
  report = {'complete': True, 'gpu_executed': False, 'evidence': evidence, 'dispatch_checks': checks,
    'installed_dispatch_binary_matches_gpu': True, 'cpu_replay_both_operands_restore': True,
    'capture_84_and_negative_controls': True, 'queue_observer_cleanup': True,
    'canonical_64_existing_pair_exact': True, 'canonical_mutations_rejected': rejected,
    'sources': {str(p): H.sha(p.read_bytes()) for p in HERE.glob('agent-expert-pv-route-*.py')}}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'complete': True, 'gpu_executed': False, 'canonical_negative_cases': len(rejected)}))


if __name__ == '__main__': main()

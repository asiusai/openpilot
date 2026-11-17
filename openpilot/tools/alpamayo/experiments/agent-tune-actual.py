"""Capture exact runtime optimizer ASTs and tune bounded elementwise kernels.

Parent owns GPU capture/tuning/replay. CPU modes never open AMD. Unsupported
ASTs are recorded and skipped; their precision or correctness gates never relax.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import statistics
import sys
import tempfile
import time

import numpy as np

DEFAULT_ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/kernel-followup-source/alpamayo')
BINDING = ('schema', 'tinygrad_commit', 'tinygrad_files', 'tinygrad_diff_sha256', 'llvm_version', 'llvmopt',
           'power_policy', 'target', 'wgp', 'wgp_compiler_sha256', 'codegen_context')
SUPPORTED = {'PARAM', 'CONST', 'RANGE', 'INDEX', 'LOAD', 'STORE', 'END', 'SINK', 'CAST', 'ADD', 'SUB', 'MUL',
             'RECIP', 'NEG', 'MAX', 'CMPLT', 'CMPEQ', 'CMPNE', 'WHERE', 'DIV', 'MOD', 'TRUNCDIV', 'TRUNCMOD',
             'SHL', 'SHR', 'AND', 'OR', 'XOR'}


def load_base(root):
  path = Path(root)/'experiments/agent-tune.py'
  spec = importlib.util.spec_from_file_location('agent_tune_actual_base', path)
  m = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = m
  spec.loader.exec_module(m)
  return m


def params(ast):
  from tinygrad.uop.ops import Ops
  return sorted((u for u in ast.toposort() if u.op is Ops.PARAM), key=lambda u: u.arg.slot)


def describe(ast, max_elements=2097152):
  from tinygrad import dtypes
  from tinygrad.uop.ops import Ops
  nodes, ps = list(ast.toposort()), params(ast)
  detail = {'ast_key': ast.key.hex(), 'ops': dict(Counter(u.op.name for u in nodes)), 'name': ast.arg.name,
            'parameters': [{'slot': p.arg.slot, 'dtype': str(p.dtype), 'elements': p.max_numel(), 'device': p.device} for p in ps]}
  try:
    assert len(nodes) <= 1200, 'AST node bound'
    assert not (unsupported := set(detail['ops'])-SUPPORTED), 'Unsupported operations: '+','.join(sorted(unsupported))
    assert 2 <= len(ps) <= 8 and [p.arg.slot for p in ps] == list(range(len(ps))), 'Noncontiguous or excessive parameter slots'
    assert all(p.dtype in (dtypes.bfloat16, dtypes.fp8e4m3, dtypes.float32, dtypes.float16, dtypes.uint8,
                          dtypes.uint16, dtypes.uint32, dtypes.int32, dtypes.int64, dtypes.uint64) for p in ps), 'Unsupported parameter dtype'
    assert sum(p.max_numel()*p.dtype.itemsize for p in ps) <= 64*1024*1024, 'Buffer byte bound'
    ranges = [u for u in nodes if u.op is Ops.RANGE]
    assert ranges and all(u.src[0].op is Ops.CONST and isinstance(u.src[0].arg, int) for u in ranges), 'Dynamic loop bounds'
    count = math.prod(u.src[0].arg for u in ranges)
    assert 0 < count <= max_elements, 'Oracle iteration bound'
    stores = [u for u in nodes if u.op is Ops.STORE]
    assert len(stores) == 1 and stores[0].src[0].op is Ops.INDEX, 'Require one direct output store'
    index = stores[0].src[0]
    assert index.src[0].op is Ops.PARAM and len(index.src) == 2, 'Unsupported output view'
    output = index.src[0].arg.slot
    assert not any(u.op is Ops.INDEX and u.src[0] is ps[output] for u in stores[0].src[1].toposort()), 'Read/write alias output'
    assert all(u.op is not Ops.INDEX or (u.src[0].op is Ops.PARAM and len(u.src) == 2) for u in nodes), 'Indirect or viewed memory'
    detail.update(eligible=True, output_slot=output, iterations=count)
  except AssertionError as exc: detail.update(eligible=False, reason=str(exc))
  return detail


def storage_dtype(dt):
  from tinygrad import dtypes
  if dt == dtypes.bfloat16: return np.dtype('uint16')
  if dt == dtypes.fp8e4m3: return np.dtype('uint8')
  mapping = {dtypes.float32: 'float32', dtypes.float16: 'float16', dtypes.bool: 'bool', dtypes.weakint: 'int64', dtypes.weakfloat: 'float64',
             dtypes.int32: 'int32', dtypes.int64: 'int64', dtypes.uint8: 'uint8', dtypes.uint16: 'uint16',
             dtypes.uint32: 'uint32', dtypes.uint64: 'uint64'}
  return np.dtype(mapping[dt])


def decode_fp8(bits):
  b = np.asarray(bits, np.uint8)
  exponent, mantissa = ((b >> 3) & 15).astype(np.int32), (b & 7).astype(np.float32)
  values = np.where(exponent == 0, mantissa*2.0**-9, (1+mantissa/8)*np.exp2(exponent-7))
  assert np.all((b & 127) != 127), 'Nonfinite FP8 fixture'
  return np.where(b & 128, -values, values).astype(np.float32)


def encode(values, dtype, base):
  from tinygrad import dtypes
  if dtype == dtypes.bfloat16: return base.bf16(np.asarray(values, np.float32))
  if dtype == dtypes.fp8e4m3: return base.fp8(np.asarray(values, np.float32))
  return np.asarray(values).astype(storage_dtype(dtype))


def decode(values, dtype, base):
  from tinygrad import dtypes
  if dtype == dtypes.bfloat16: return base.unbf16(values)
  if dtype == dtypes.fp8e4m3: return decode_fp8(values)
  return np.asarray(values, storage_dtype(dtype))


def fixtures(ast, seed, base):
  from tinygrad import dtypes
  values = []
  for p in params(ast):
    rng = np.random.default_rng(9016+seed*101+p.arg.slot)
    n = p.max_numel()
    if p.dtype in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.int32, dtypes.int64, dtypes.uint64):
      v = rng.integers(0, 8, n)
    elif p.dtype == dtypes.float32 and n <= 64:
      v = np.full(n, (.0625, .03125, .071234569)[seed-1], np.float32)
    else:
      v = rng.standard_normal(n, dtype=np.float32)*3
      if seed == 2: v[:max(1, n//4)] = 0
      v[:min(n, 8)] = np.array([0., -0., .5, -.5, 7., -7., 28., -28.], np.float32)[:min(n, 8)]
    values.append(encode(v, p.dtype, base))
  return values


def oracle(ast, raw, base, max_elements=2097152):
  """Independent NumPy evaluation of the captured pre-optimizer AST.

  It supports only pure elementwise/indexing ASTs. No reduction, WMMA, custom
  code, dynamic parameter, alias, or partial output can enter the tuning cache.
  """
  from tinygrad import dtypes
  from tinygrad.dtype import Invalid
  from tinygrad.uop.ops import Ops
  info = describe(ast, max_elements)
  assert info['eligible'], info.get('reason')
  ranges = [u for u in ast.toposort() if u.op is Ops.RANGE]
  memo = {r: np.arange(r.src[0].arg, dtype=np.int64).reshape(tuple(r.src[0].arg if i == j else 1 for j in range(len(ranges))))
          for i, r in enumerate(ranges)}
  decoded = [decode(v, p.dtype, base) for p, v in zip(params(ast), raw)]
  footprint = [sum(v.nbytes for v in decoded)]
  def cast(value, dt): return decode(encode(value, dt, base), dt, base)
  def evaluate(u):
    if u in memo: return memo[u]
    op = u.op.name
    if op == 'CONST':
      # Invalid addresses are the only permitted nonnumeric constants.
      assert isinstance(u.arg, (int, float, bool)) or u.arg is Invalid, 'Unknown constant'
      value = -1 if u.arg is Invalid else u.arg if u.dtype in (dtypes.weakint, dtypes.weakfloat) else cast(u.arg, u.dtype)
    elif op == 'INDEX':
      idx = np.asarray(evaluate(u.src[1]), np.int64)
      data = decoded[u.src[0].arg.slot]
      assert np.all((idx >= -1) & (idx < data.size)), 'Input address out of bounds'
      value = np.where(idx >= 0, data[np.maximum(idx, 0)], 0)
    elif op == 'LOAD': value = evaluate(u.src[0])
    elif op == 'CAST': value = cast(evaluate(u.src[0]), u.dtype)
    else:
      xs = [evaluate(s) for s in u.src]
      operations = {'ADD': lambda a,b:a+b, 'SUB':lambda a,b:a-b, 'MUL':lambda a,b:a*b, 'RECIP':lambda a:np.float32(1)/a,
                    'NEG':lambda a:-a, 'MAX':np.maximum, 'CMPLT':np.less, 'CMPEQ':np.equal, 'CMPNE':np.not_equal,
                    'WHERE':np.where, 'DIV':np.floor_divide, 'MOD':np.mod, 'TRUNCDIV':lambda a,b:np.trunc(a/b),
                    'TRUNCMOD':lambda a,b:a-np.trunc(a/b)*b, 'SHL':np.left_shift, 'SHR':np.right_shift,
                    'AND':np.bitwise_and, 'OR':np.bitwise_or, 'XOR':np.bitwise_xor}
      value = cast(operations[op](*xs), u.dtype)
    memo[u] = value
    footprint[0] += getattr(value, 'nbytes', 0)
    assert footprint[0] <= 256*1024*1024, 'Independent oracle memory bound'
    return value
  store = next(u for u in ast.toposort() if u.op is Ops.STORE)
  idx, result = np.broadcast_arrays(evaluate(store.src[0].src[1]), evaluate(store.src[1]))
  idx, result = idx.reshape(-1), result.reshape(-1)
  valid = idx >= 0
  idx, result = idx[valid].astype(np.int64), result[valid]
  slot = info['output_slot']
  assert np.array_equal(np.sort(idx), np.arange(raw[slot].size)), 'Output must cover every element exactly once'
  out = np.empty_like(raw[slot])
  out[idx] = encode(result, params(ast)[slot].dtype, base)
  return out


def runtime_provenance(base, wgp):
  import tinygrad.codegen as cg
  p = base.provenance(wgp)
  p['codegen_context'] = {v.key: v.value for v in cg.to_program_context if v.key not in ('DEV', 'BEAM')}
  return p


@contextmanager
def capture(base, directory, wgp, limit=16, max_elements=2097152, legacy_cache=None):
  import tinygrad.codegen as cg
  from tinygrad.helpers import Context
  assert 1 <= limit <= 32 and 1 <= max_elements <= 2097152
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=False)
  manifest = {'schema': 1, 'source_root': str(base.ROOT), 'helper_sha256': base.file_sha(__file__), 'records': [],
               'enabled': False, 'gpu_seen': 0, 'cpu_seen': 0, 'skips': {}, 'old_key_hits': 0, 'exact_ast_keys': [],
              'limits': {'kernels': limit, 'max_elements': max_elements}, 'parallel_forced_zero': True}
  old_entries = {} if legacy_cache is None else json.loads(Path(legacy_cache).read_text())['entries']
  original, seen = cg.apply_opts, set()
  def apply(ast, ren, beam=0):
    optimized = original(ast, ren, beam=0)
    if not manifest['enabled']: return optimized
    if repr(ren.target).split('+')[-1] != 'AMD:LLVM:gfx1200':
      manifest['cpu_seen'] += 1
      return optimized
    manifest['gpu_seen'] += 1
    assert (type(ren.compiler).__name__ == 'WGPCompiler') == wgp, 'Unexpected inference compiler'
    prov = runtime_provenance(base, wgp)
    key = base.cache_key(ast, prov)
    manifest['old_key_hits'] += key in old_entries
    if key in seen: return optimized
    seen.add(key)
    if len(manifest['exact_ast_keys']) < 512: manifest['exact_ast_keys'].append({'ast_key': ast.key.hex(), 'cache_key': key})
    info = describe(ast, max_elements)
    if not info['eligible'] or len(manifest['records']) >= limit:
      reason = info.get('reason', 'Capture count bound')
      manifest['skips'][reason] = manifest['skips'].get(reason, 0)+1
    else:
      assert not ast.arg.optimized and optimized.arg.optimized
      blob = pickle.dumps((ast, optimized), protocol=5)
      path = directory/('agent-tune-actual-'+key+'.pkl')
      path.write_bytes(blob)
      manifest['records'].append(dict(info, cache_key=key, artifact=path.name, artifact_sha256=base.sha(blob), provenance=prov))
    base.atomic_json(directory/'manifest.json', manifest)
    return optimized
  cg.apply_opts = apply
  try:
    with Context(PARALLEL=0): yield manifest
  finally:
    cg.apply_opts = original
    base.atomic_json(directory/'manifest.json', manifest)


@contextmanager
def replay(base, cache_path, wgp):
  import tinygrad.codegen as cg
  from tinygrad.codegen.opt.postrange import Scheduler
  from tinygrad.helpers import Context
  cache = json.loads(Path(cache_path).read_text())
  assert cache['schema'] == base.SCHEMA
  current = runtime_provenance(base, wgp)
  stats = {'hits': 0, 'gpu_misses': 0, 'cpu_skipped': 0, 'keys': {}, 'parallel_forced_zero': True, 'enabled': False,
           'cache_sha256': base.file_sha(cache_path)}
  original = cg.apply_opts
  def apply(ast, ren, beam=0):
    if not stats['enabled']: return original(ast, ren, beam=0)
    if repr(ren.target).split('+')[-1] != current['target']:
      stats['cpu_skipped'] += 1
      return original(ast, ren, beam=0)
    now = dict(current, codegen_context={v.key:v.value for v in cg.to_program_context if v.key not in ('DEV', 'BEAM')})
    key = base.cache_key(ast, now)
    row = cache['entries'].get(key)
    if row is None:
      stats['gpu_misses'] += 1
      return original(ast, ren, beam=0)
    assert row['validated'] and row['accepted'] and row['oracle'] == 'numpy-preopt-exact'
    assert row['ast_key'] == ast.key.hex() and base.cache_key(ast, row['provenance']) == key
    assert row['actual_helper_sha256'] == base.file_sha(__file__), 'Tuning helper changed after validation'
    assert (type(ren.compiler).__name__ == 'WGPCompiler') == wgp
    k = Scheduler(ast, ren)
    k.convert_loop_to_global()
    for opt in base.opts_from_json(row['opts']): k.apply_opt(opt)
    stats['hits'] += 1
    stats['keys'][key] = stats['keys'].get(key, 0)+1
    return k.get_optimized_ast(name_override=ast.arg.name)
  cg.apply_opts = apply
  try:
    with Context(PARALLEL=0): yield stats
  finally: cg.apply_opts = original


def artifacts(directory, base):
  directory = Path(directory)
  manifest = json.loads((directory/'manifest.json').read_text())
  assert manifest['schema'] == 1 and manifest['helper_sha256'] == base.file_sha(__file__)
  for row in manifest['records']:
    path = directory/row['artifact']
    assert path.parent == directory and path.name.startswith('agent-tune-actual-')
    blob = path.read_bytes()
    assert base.sha(blob) == row['artifact_sha256']
    # Only load locally generated, hash-bound AST artifacts from this capture.
    ast, optimized = pickle.loads(blob)
    assert ast.key.hex() == row['ast_key'] and base.cache_key(ast, row['provenance']) == row['cache_key']
    yield row, ast, optimized


def worker_probe():
  from tinygrad import Tensor, Device
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.engine.realize import lower_and_compile
  from tinygrad.engine.worker import terminate_worker_pool
  import tinygrad.codegen as cg
  calls = []
  for n in (17, 31):
    x = Tensor(np.arange(n, dtype=np.float32), device='CPU').realize()
    calls += list((x+1).contiguous().schedule_linear().src)
  assert len(calls) == 2
  original, results = cg.apply_opts, {}
  try:
    for parallel in (2, 0):
      observed = []
      def hook(ast, ren, beam=0):
        observed.append(ast.key.hex())
        return original(ast, ren, beam=0)
      cg.apply_opts = hook
      cg.to_program_cache.clear()
      with Context(PARALLEL=parallel): lower_and_compile(UOp(Ops.LINEAR, src=tuple(calls)))
      terminate_worker_pool()
      results[str(parallel)] = len(observed)
    assert results == {'2': 0, '0': 2}, results
    assert not any(d.startswith('AMD') for d in Device._opened_devices)
    return results
  finally:
    terminate_worker_pool()
    cg.apply_opts = original


def diagnose(base, args):
  saved = json.loads(args.invocation.read_text())
  for k, v in saved['environment'].items():
    if k == 'DEV': continue
    if v is None: os.environ.pop(k, None)
    else: os.environ[k] = v
  current = runtime_provenance(base, '--llvm-wgp' in saved['command'])
  cache = json.loads(args.cache.read_text())
  return {'binding_differences': [{'case': r['case'], 'ast_key': r['ast_key'],
           'differences': {k: {'cached': r['provenance'][k], 'pipeline': current[k]} for k in BINDING if r['provenance'][k] != current[k]}}
           for r in cache['entries'].values()], 'worker_hook_probe': worker_probe(), 'gpu_executed': False}


def buffers_for(ast, raw, device):
  from tinygrad.device import Buffer, BufferSpec
  from tinygrad.uop.ops import UOp
  buffers = [Buffer(device, v.nbytes, options=BufferSpec(cpu_access=True)).ensure_allocated() for v in raw]
  for b, v in zip(buffers, raw): b.host.view(size=b.nbytes, fmt='B')[:] = v.tobytes()
  return buffers, [UOp.from_buffer(b, p.dtype) for b, p in zip(buffers, params(ast))]


def self_test(base):
  from tinygrad import Device
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt.postrange import Scheduler
  from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
  from tinygrad.engine.realize import run_linear
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.helpers import Target
  checks = []
  for kind in ('residual', 'relu', 'quant-pack'):
    case = base.Case(kind, 17, 48, 'synthetic CPU oracle self-test only')
    _, _, linear, captured = base.prepare(case, Device['CPU'].renderer, 'CPU')
    for index, (ast, _) in enumerate(captured):
      from tinygrad import dtypes
      info = describe(ast)
      assert info['eligible'], info
      for seed in (1, 2, 3):
        raw = fixtures(ast, seed, base)
        if kind == 'quant-pack':
          # CPU software FP8 conversion flushes subnormals; use exactly
          # represented normal outputs for this independent packing control.
          for i, param in enumerate(params(ast)):
            v = (np.arange(raw[i].size)%15-7).astype(np.float32)/8
            raw[i] = encode(np.full(v.shape, .125, np.float32) if param.dtype == dtypes.float32 else v, param.dtype, base)
        expected = oracle(ast, raw, base)
        bs, arguments = buffers_for(ast, raw, 'CPU')
        base.poison_buffer(bs[info['output_slot']])
        program = linear.src[index].body
        run_linear(UOp(Ops.LINEAR, src=(program.call(*arguments),)), update_stats=False)
        actual = bytes(bs[info['output_slot']].as_memoryview())
        assert actual == expected.tobytes(), (kind, index, seed)
        checks.append({'kind': kind, 'index': index, 'seed': seed, 'numpy_preopt_exact': True})
      restored = pickle.loads(pickle.dumps(ast, protocol=5))
      assert restored.key == ast.key
      renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
      scheduler = Scheduler(ast, renderer)
      scheduler.convert_loop_to_global()
      seed = hand_coded_optimizations(scheduler.copy())
      candidates = base.choices([scheduler, seed], 3)
      built = [base.compile_candidate(k) for k in candidates]
      assert any(prg is not None for prg in built)
      checks.append({'kind': kind, 'index': index, 'pickle_key_exact': True, 'amd_candidates_compiled_offline': sum(prg is not None for prg in built)})
  # Reductions must be rejected rather than validated with an approximate oracle.
  case = base.Case('rms', 16, 64, 'unsupported reduction self-test')
  _, _, _, captured = base.prepare(case, Device['CPU'].renderer, 'CPU')
  assert any(not describe(ast)['eligible'] for ast, _ in captured)
  checks.append({'reduction_rejected': True})
  # Exercise capture -> persisted exact AST -> bounded AMD compilation without
  # opening AMD. The artifact is explicitly a synthetic CPU harness fixture.
  with tempfile.TemporaryDirectory(prefix='agent-tune-actual-', dir='/tmp/opencode') as temporary:
    directory = Path(temporary)/'capture'
    renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
    with capture(base, directory, False, limit=1) as captured:
      captured['enabled'] = True
      base.prepare(base.Case('residual', 17, 48, 'synthetic offline capture control'), renderer, 'CPU')
    assert captured['gpu_seen'] == 1 and len(captured['records']) == 1
    restored = list(artifacts(directory, base))
    assert len(restored) == 1 and restored[0][0]['ast_key'] == restored[0][1].key.hex()
    args = argparse.Namespace(mode='inspect', capture_dir=directory, kernels=1, budget=3, seconds=30,
                              wgp=False, output=Path(temporary)/'inspect.json', cache=None)
    inspected = process_capture(base, args)
    assert inspected['complete'] and any(a['compiled'] for a in inspected['rows'][0]['attempts'])
    checks.append({'offline_capture_artifact_inspect_exact': True, 'amd_candidates_compiled':
                   sum(a['compiled'] for a in inspected['rows'][0]['attempts'])})
    # An ephemeral synthetic cache tests the replay protocol, not performance.
    row, ast, optimized = restored[0]
    scheduler = Scheduler(ast, renderer)
    scheduler.convert_loop_to_global()
    scheduler = hand_coded_optimizations(scheduler)
    cache_path = Path(temporary)/'agent-tune-actual-synthetic-protocol-only.json'
    entry = {'validated': True, 'accepted': True, 'oracle': 'numpy-preopt-exact', 'actual_helper_sha256': base.file_sha(__file__),
             'ast_key': row['ast_key'], 'provenance': row['provenance'], 'opts': base.opts_json(scheduler.applied_opts)}
    base.atomic_json(cache_path, {'schema': base.SCHEMA, 'entries': {row['cache_key']: entry}})
    import tinygrad.codegen as cg
    original_hook = cg.apply_opts
    with replay(base, cache_path, False) as stats:
      stats['enabled'] = True
      actual = cg.apply_opts(ast, renderer)
      expected = scheduler.get_optimized_ast(name_override=ast.arg.name)
      assert actual.key == expected.key and stats['hits'] == 1
    assert cg.apply_opts is original_hook
    entry['actual_helper_sha256'] = 'changed'
    base.atomic_json(cache_path, {'schema': base.SCHEMA, 'entries': {row['cache_key']: entry}})
    with replay(base, cache_path, False) as stats:
      stats['enabled'] = True
      try: cg.apply_opts(ast, renderer)
      except AssertionError as exc: assert 'helper changed' in str(exc)
      else: raise AssertionError('Changed validation helper accepted')
    checks.append({'synthetic_cache_replay_exact': True, 'changed_helper_rejected': True, 'hooks_restored': True})
    # The inference boundary and paired route gate run without any real model.
    from contextlib import ExitStack
    from types import SimpleNamespace
    wrapper_path = Path(__file__).with_name('agent-tune-actual-run.py')
    spec = importlib.util.spec_from_file_location('agent_tune_actual_wrapper_check', wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    stats, status = {'enabled': False}, {'inference_boundary_armed': False}
    class Decoder:
      def __init__(self, model): assert model == 'fixture' and not stats['enabled']
    original_init = Decoder.__init__
    loader = lambda _: SimpleNamespace(DraftDecoder=Decoder)
    pipeline = SimpleNamespace(local_module=loader)
    with ExitStack() as stack:
      wrapper.arm_at_decoder(pipeline, stack, stats, status, base, Path(temporary)/'status.json')
      pipeline.local_module('draft-decode').DraftDecoder('fixture')
      assert stats['enabled'] and status['inference_boundary_armed']
    assert Decoder.__init__ is original_init and pipeline.local_module is loader
    sample = {'sample': 'fixture', 'input_latest_timestamp_us': 123, 'prediction':
              {'generated_tokens': ['x'], 'generated_token_ids': [1], 'reasoning': 'x', 'xyz': [[0., 1., -2.]], 'rot': [[0., 0., 0.]]}}
    paired = {'warmup': [], 'runs': [sample]}
    assert wrapper.compare(paired, paired)['predictions'] == 1
    changed = json.loads(json.dumps(paired))
    changed['runs'][0]['prediction']['xyz'][0][0] = -0.
    try: wrapper.compare(changed, paired)
    except AssertionError: pass
    else: raise AssertionError('Signed-zero route output change passed exact gate')
    checks.append({'decoder_boundary_armed_after_setup': True, 'paired_route_bit_change_rejected': True, 'hooks_restored': True})
  probe = worker_probe()
  assert not any(d.startswith('AMD') for d in Device._opened_devices)
  return {'checks': checks, 'worker_hook_probe': probe, 'gpu_executed': False, 'complete': True,
          'summary': {'cpu_checks': len(checks), 'worker_hook_probe': probe, 'gpu_executed': False}}


def process_capture(base, args):
  from tinygrad import Device, Tensor
  from tinygrad.codegen import to_program
  from tinygrad.codegen.opt.postrange import Scheduler
  from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
  from tinygrad.helpers import Target, Context
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.uop.ops import Ops, UOp
  timer = None
  report = {'mode': args.mode, 'gpu_executed': False, 'rows': [], 'limits': {'kernels': args.kernels, 'budget': args.budget, 'seconds': args.seconds},
            'timing': 'Full-size GPU timestamps; excludes USB reads and is not pipeline throughput', 'pipeline_validated': False}
  cache_path = args.cache or args.output.with_name('agent-tune-actual-cache.json')
  cache = json.loads(cache_path.read_text()) if cache_path.exists() else {'schema': base.SCHEMA, 'entries': {}}
  assert cache['schema'] == base.SCHEMA
  if args.mode == 'tune': base.atomic_json(cache_path, cache)
  if args.mode == 'tune':
    from tinygrad.device import Compiled
    Compiled.wait_timeout_ms = 10000
    base.local('prefill-kernels').fix_dispatch_packet()
    dev = Device['AMD']
    assert dev.is_usb and dev.arch == 'gfx1200' and isinstance(dev.renderer, AMDLLVMRenderer)
    if args.wgp: base.local('wgp_compiler').install(dev)
    renderer, timer = dev.renderer, base.Timer(dev)
  else:
    renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
    if args.wgp: renderer.compiler = base.local('wgp_compiler').WGPCompiler('gfx1200')
  try:
    for number, (record, ast, optimized) in enumerate(artifacts(args.capture_dir, base)):
      if number >= args.kernels: break
      started = time.monotonic()
      row = {'ast_key': record['ast_key'], 'cache_key': record['cache_key'], 'attempts': [], 'accepted': False}
      report['rows'].append(row)
      assert record['provenance']['wgp'] == args.wgp
      with Context(**record['provenance']['codegen_context']):
        current = runtime_provenance(base, args.wgp)
        if args.mode == 'tune': assert all(current[k] == record['provenance'][k] for k in BINDING), 'Capture/compiler binding changed'
        info = describe(ast)
        assert info['eligible'], info
        scheduler = Scheduler(ast, renderer)
        scheduler.convert_loop_to_global()
        seed = hand_coded_optimizations(scheduler.copy())
        pool = base.choices([scheduler, seed], args.budget)
        row['baseline_binary_sha256'] = base.sha(to_program(optimized, renderer).src[3].arg)
        if args.mode == 'inspect':
          for candidate in pool:
            program = base.compile_candidate(candidate)
            row['attempts'].append({'opts': base.opts_json(candidate.applied_opts), 'compiled': program is not None})
          row['status'] = 'cpu-compiled-only'
          continue
        data = [fixtures(ast, s, base) for s in (1, 2, 3)]
        try: expected = [oracle(ast, raw, base).tobytes() for raw in data]
        except (AssertionError, KeyError, ValueError, FloatingPointError) as exc:
          row.update(status='oracle-rejected', reason=str(exc))
          continue
        if len(set(expected)) < 2:
          row.update(status='oracle-rejected', reason='Mutations did not change reference output')
          continue
        bs, arguments = buffers_for(ast, data[0], 'AMD')
        output = Tensor(arguments[info['output_slot']])
        def graph(program): return base.linked(UOp(Ops.LINEAR, src=(program.call(*arguments),)))
        baseline = graph(to_program(optimized, renderer))
        def update(raw):
          dev.synchronize()
          for b, value in zip(bs, raw): b.host.view(size=b.nbytes, fmt='B')[:] = value.tobytes()
        def validate(compiled):
          checks = []
          for s, (raw, reference) in enumerate(zip(data, expected), 1):
            update(raw)
            base.poison_buffer(bs[info['output_slot']])
            timer.run(compiled)
            actual = base.raw_read(output, 'AMD')
            assert actual == reference, f'Exact NumPy oracle mismatch on mutation seed {s}'
            checks.append({'seed': s, 'exact': True, 'sha256': base.sha(actual)})
          return checks
        report['gpu_executed'] = True
        try: row['baseline_checks'] = validate(baseline)
        except AssertionError as exc:
          row.update(status='baseline-oracle-mismatch', reason=str(exc))
          continue
        best, seen = None, set()
        for candidate in pool:
          if time.monotonic()-started >= args.seconds: break
          trial = {'opts': base.opts_json(candidate.applied_opts)}
          row['attempts'].append(trial)
          program = base.compile_candidate(candidate)
          if program is None:
            trial['status'] = 'compile-rejected'
            continue
          binary = base.sha(program.src[3].arg)
          if binary in seen:
            trial['status'] = 'duplicate-binary'
            continue
          seen.add(binary)
          compiled = graph(program)
          try: trial['checks'] = validate(compiled)
          except AssertionError as exc:
            trial.update(status='incorrect', reason=str(exc))
            continue
          update(data[0])
          pairs, timestamps = [], []
          for i in range(args.samples):
            pair = {}
            for label, g in ([('baseline', baseline), ('candidate', compiled)] if i % 2 == 0 else [('candidate', compiled), ('baseline', baseline)]):
              pair[label], stamps = timer.run(g)
              timestamps.append({'pair': i, 'method': label, 'stamps': stamps})
            pairs.append(pair)
          b, c = (statistics.median(p[k] for p in pairs) for k in ('baseline', 'candidate'))
          accepted = c < b*.95 and sum(p['candidate'] < p['baseline'] for p in pairs) >= (args.samples+1)//2
          trial.update(status='exact', accepted=accepted, paired_gpu_us=pairs, timestamps=timestamps, baseline_us=b, candidate_us=c, binary_sha256=binary)
          if accepted and (best is None or c < best['candidate_us']): best = trial
          base.atomic_json(args.output, report)
        if best is not None:
          row.update(accepted=True, winner=best)
          cache['entries'][record['cache_key']] = {'validated': True, 'accepted': True, 'oracle': 'numpy-preopt-exact',
             'actual_helper_sha256': base.file_sha(__file__),
            'opts': best['opts'], 'provenance': record['provenance'], 'ast_key': record['ast_key'], 'binary_sha256': best['binary_sha256'],
            'checks': best['checks'], 'paired_gpu_us': best['paired_gpu_us'], 'capture': str(args.capture_dir), 'created_unix': time.time()}
          base.atomic_json(cache_path, cache)
        row['status'] = 'checked'
      base.atomic_json(args.output, report)
    report['complete'] = True
    report['summary'] = {'mode': args.mode, 'kernels': len(report['rows']), 'accepted': sum(r['accepted'] for r in report['rows']),
                         'gpu_executed': report['gpu_executed'], 'pipeline_validated': False}
    if args.mode == 'inspect': assert not any(d.startswith('AMD') for d in Device._opened_devices)
    return report
  except BaseException as exc:
    report.update(complete=False, failure=f'{type(exc).__name__}: {exc}')
    base.atomic_json(args.output, report)
    raise
  finally:
    if timer is not None: timer.close()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--mode', choices=('diagnose', 'self-test', 'inspect', 'tune'), default='inspect')
  p.add_argument('--source-root', type=Path, default=DEFAULT_ROOT)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--invocation', type=Path)
  p.add_argument('--cache', type=Path)
  p.add_argument('--capture-dir', type=Path)
  p.add_argument('--budget', type=int, default=8)
  p.add_argument('--kernels', type=int, default=4)
  p.add_argument('--samples', type=int, default=5)
  p.add_argument('--seconds', type=float, default=120)
  p.add_argument('--wgp', action='store_true')
  args = p.parse_args()
  assert 1 <= args.budget <= 16 and 1 <= args.kernels <= 8 and 3 <= args.samples <= 9 and 1 <= args.seconds <= 180
  os.environ.update(BEAM='0', PROFILE='0', DEBUG='0', PARALLEL='0')
  for key, value in {'BEAM_UOPS_MAX': '3000', 'BEAM_UPCAST_MAX': '64', 'BEAM_LOCAL_MAX': '256',
                     'BEAM_TIMEOUT_SEC': '10', 'MAX_BUFFER_SIZE': '268435456'}.items(): os.environ.setdefault(key, value)
  if args.mode != 'tune': os.environ['DEV'] = 'CPU'
  sys.path.insert(0, str(args.source_root))
  base = load_base(args.source_root)
  if args.mode == 'diagnose':
    report = diagnose(base, args)
  elif args.mode == 'self-test':
    report = self_test(base)
  else:
    report = process_capture(base, args)
  report['helper_sha256'] = base.file_sha(__file__)
  base.atomic_json(args.output, report)
  print(json.dumps(report.get('summary', {'mode': args.mode, 'gpu_executed': report.get('gpu_executed', False)})))


if __name__ == '__main__': main()

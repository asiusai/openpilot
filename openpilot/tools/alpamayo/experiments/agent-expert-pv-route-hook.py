"""Optional exact 2x2 PV dispatch; original expert attention body stays intact."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import CodeType, MethodType, ModuleType

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
LIVE = Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo')
REPORT = WORK/'agent-expert-pv-kernels/report.json'
REPORT_SHA = '6303e873c2c97896e43d20d7b1e1017ab977a19b540dddcc012ccc10fc24f2bd'
PINS = {'expert-kernels.py': '4676128ef21bd42f851311b3b5dba0563ab6cd1e54d367a3bca2e377f5a53b8b',
        'model.py': 'fce24a07fdd48bd24c05e67518c986bcb462d09dfb75b47daf4140dc4e67f5fb',
        'amd_attention.py': 'fa9d2f01c6557c97541f8a62e4e02ee7cf2608584787cad7d42b9922a00866f4',
        'agent-expert-pv-kernel.py': '1ad77b4271c722f2b64b3fb94a60c650385cf791de77b8232ca4a2fb51505e9a'}
SHAPE = (1, 16, 8, 64, 128, 2368)
QK_SHAPE = (1, 16, 8, 64, 2368, 128)
NAME = 'alpamayo_expert_pv_2x2'
BINARY = '7e6122334218c2f4723614a59620b32806f49ff527ac533725557c9a894deef2'
LLVM = '2b978f95330490c7cf0ed1de95a4ee5cdd3fd44062c1e0c130f159f353b93a1e'


def sha(data): return hashlib.sha256(data).hexdigest()


def checked(name, path):
  data = Path(path).read_bytes()
  assert sha(data) == PINS[name], f'Independently pinned source changed: {path}'
  return data


def verify_evidence(path=REPORT, source=HERE):
  data = Path(path).read_bytes()
  assert sha(data) == REPORT_SHA, 'Passed GPU evidence changed'
  result = json.loads(data)
  for flag in ('complete', 'gpu_executed', 'exactness_qualified', 'all_pre_timing_exact_gates_passed', 'final_post_timing_exact'):
    assert result[flag] is True, flag
  assert result['numerical_candidate'] is False and not result.get('failure')
  assert result['tinygrad_revision'] == '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
  assert len(result['cases']) == 3
  for case in result['cases']:
    assert case['complete'] and len(case['checks']) == 15 and all(c['exact_bf16_bits'] for c in case['checks'])
  actual = next(c for c in result['cases'] if c['case'] == 'expert')
  assert tuple(actual['shape_b_h_kvh_m_n_k']) == SHAPE
  for flag in ('masked_history_value_mutation_exact', 'changed_a_and_b_after_40_call_capture_exact', 'final_post_timing_fresh_a_b_replay_exact'):
    assert actual[flag]
  selected = actual['compiler']['2x2']
  assert selected['binary_sha256'] == BINARY and selected['llvm_sha256'] == LLVM
  assert actual['queue_proofs']['2x2']['captured_calls'] == 40
  assert actual['queue_proofs']['2x2']['ordered_execution_barriers'] == 39
  for original, digest in result['source_sha256'].items():
    old = Path(original)
    new = source.parent/old.relative_to(LIVE) if old.is_relative_to(LIVE) else old
    assert sha(new.read_bytes()) == digest, f'Micro-qualified dependency changed: {new}'
  for name in PINS:
    checked(name, (source.parent if name in ('model.py', 'amd_attention.py') else source)/name)
  return {'report_sha256': REPORT_SHA, 'shape': SHAPE, 'binary_sha256': BINARY, 'llvm_sha256': LLVM,
          'original_binary_sha256': actual['compiler']['original']['binary_sha256'], 'tile': [2, 2]}


def verify_function(function, path):
  function = getattr(function, '__func__', function)
  data = checked(Path(path).name, path)
  def codes(code):
    yield code
    for value in code.co_consts:
      if isinstance(value, CodeType): yield from codes(value)
  expected = next(c for c in codes(compile(data, function.__code__.co_filename, 'exec')) if c.co_qualname == function.__code__.co_qualname)
  assert function.__code__ == expected, f'Runtime function changed: {function.__qualname__}'


def load_kernel():
  path = HERE/'agent-expert-pv-kernel.py'
  value = ModuleType('agent_expert_pv_verified_numeric')
  value.__file__ = str(path)
  exec(compile(checked(path.name, path), str(path), 'exec'), value.__dict__)
  return value


def eligible(out, a, b, shape, causal, scaled, active):
  from tinygrad import dtypes
  return (active and tuple(shape) == SHAPE and causal is False and scaled is False
          and out.dtype == a.dtype == b.dtype == dtypes.bfloat16
          and out.shape == (1, 16, 64, 128) and a.shape == (1, 16, 64, 2368)
          and b.shape == (1, 8, 8, 148, 4, 16, 4))


class Installation:
  def close(self):
    self.active = False
    if self.module._matmul is self.dispatch: self.module._matmul = self.original_matmul
    for name, wrapper in self.wrappers.items():
      if getattr(self.model, name) is wrapper: setattr(self.model, name, self.originals[name])
    if getattr(self.model, 'agent_expert_pv', None) is self: del self.model.agent_expert_pv
    self.info['active'] = False

  @contextmanager
  def observe_programs(self):
    from tinygrad.runtime.support import hcq2
    from tinygrad.uop.ops import Ops
    original = hcq2._build_queues
    def build(ctx):
      for call, _, _ in ctx.batch:
        if call.body.op is not Ops.PROGRAM: continue
        program = call.body
        name = program.src[0].arg.function_name
        if name not in (NAME, 'alpamayo_attention_values', 'alpamayo_attention_scores', 'alpamayo_attention_softmax'): continue
        digest = sha(program.src[3].arg)
        if name == NAME:
          assert self.candidate and digest == BINARY, 'Actual captured candidate binary changed'
          assert sha(program.src[2].arg.encode()) == LLVM
        record = {'name': name, 'binary_sha256': digest}
        key = program.key.hex()
        assert key not in self.programs or self.programs[key] == record
        self.programs[key] = record
      return original(ctx)
    hcq2._build_queues = build
    try: yield
    finally: hcq2._build_queues = original

  def inspect_capture(self, jit):
    captured = jit.captured
    if captured is None or any(captured is old for old in self.seen): return
    records = []
    for call in captured._linear.src:
      aux = call.without_after.arg.aux
      assert hasattr(aux, 'kernels'), 'Expected pinned HCQ graph metadata'
      records.extend(aux.kernels)
    counts = Counter(str(k[1]) for k in records)
    pv = NAME if self.candidate else 'alpamayo_attention_values'
    assert counts[pv] == counts['alpamayo_attention_scores'] == counts['alpamayo_attention_softmax'] == 84, counts
    assert counts['alpamayo_attention_values' if self.candidate else NAME] == 0
    binaries = {}
    for record in records:
      name = str(record[1])
      if name not in (pv, 'alpamayo_attention_scores', 'alpamayo_attention_softmax'): continue
      key = record[4].hex()
      assert key in self.programs and self.programs[key]['name'] == name, 'Unobserved captured kernel'
      binaries.setdefault(name, set()).add(self.programs[key]['binary_sha256'])
    expected = BINARY if self.candidate else self.info['evidence']['original_binary_sha256']
    assert binaries[pv] == {expected}
    self.info['captures'].append({'pv_calls': 84, 'qk_calls': 84, 'softmax_calls': 84,
      'total_calls': len(records), 'binaries': {k: sorted(v) for k, v in binaries.items()}})
    self.seen.append(captured)  # Retain identity if the model later clears its cache.


def install(model, module, *, candidate=True, evidence=REPORT):
  from tinygrad import dtypes
  import amd_attention
  assert model.fp8_wmma and model.dtype == dtypes.bfloat16
  assert not hasattr(model, 'agent_expert_pv') and not getattr(model, 'trajectory_jits', {})
  assert not hasattr(model, 'agent_expert_tiles'), 'FP8 expert tile candidate is excluded'
  verified = verify_evidence(evidence)
  checked('amd_attention.py', amd_attention.__file__)
  checked('expert-kernels.py', module.__file__)
  assert module._matmul is amd_attention._matmul and module._softmax is amd_attention._softmax and module._pack is amd_attention._pack
  for function in (amd_attention._matmul, amd_attention._softmax, amd_attention._pack):
    verify_function(function, amd_attention.__file__)
  verify_function(module.attention, module.__file__)
  verify_function(model.attention, module.__file__)
  for name in ('trajectory', '_trajectory', 'expert_step', 'text_layer'):
    verify_function(getattr(model, name), HERE.parent/'model.py')
  assert model.config['expert_config']['llm_config']['num_hidden_layers'] == 28
  handle = Installation()
  handle.model, handle.module, handle.candidate, handle.active, handle.depth = model, module, candidate, True, 0
  handle.original_matmul, handle.numeric = module._matmul, load_kernel()
  handle.originals = {n: getattr(model, n) for n in ('trajectory', '_trajectory')}
  handle.counts, handle.programs, handle.seen, handle.wrappers = Counter(), {}, [], {}
  handle.info = {'active': True, 'candidate': candidate, 'evidence': verified, 'traces': [], 'captures': [],
                 'trajectory_calls': 0, 'unqualified_trajectory_calls': 0, 'outside_scope_selected': 0, 'runtime_programs': handle.programs}
  def dispatch(out, a, b, shape, causal=False, scaled=False):
    in_scope = handle.active and handle.depth > 0
    selected = eligible(out, a, b, shape, causal, scaled, in_scope)
    if in_scope:
      if selected: handle.counts['pv'] += 1
      elif tuple(shape) == QK_SHAPE and scaled is True and causal is False: handle.counts['qk'] += 1
      else: handle.counts['other'] += 1
    if selected and handle.candidate:
      return handle.numeric.kernel(out, a, b, shape=shape, tile_m=2, tile_n=2)
    return handle.original_matmul(out, a, b, shape=shape, causal=causal, scaled=scaled)
  handle.dispatch = dispatch
  module._matmul = dispatch
  def trace(self, *args, **kwargs):
    if not handle.active: return handle.originals['_trajectory'](*args, **kwargs)
    assert handle.depth == 0
    if args[4] != 3 or self.cache[0].shape[3] != 2304:
      handle.info['unqualified_trajectory_calls'] += 1
      return handle.originals['_trajectory'](*args, **kwargs)
    before = handle.counts.copy()
    handle.depth += 1
    try: result = handle.originals['_trajectory'](*args, **kwargs)
    finally: handle.depth -= 1
    delta = handle.counts-before
    assert delta == Counter(pv=84, qk=84), delta
    handle.info['traces'].append(dict(delta))
    return result
  def trajectory(self, *args, **kwargs):
    result = handle.originals['trajectory'](*args, **kwargs)
    if handle.active:
      handle.info['trajectory_calls'] += 1
      if args[4] == 3 and self.cache[0].shape[3] == 2304:
        for jit in self.trajectory_jits.values(): handle.inspect_capture(jit)
    return result
  for name, function in (('_trajectory', trace), ('trajectory', trajectory)):
    handle.wrappers[name] = MethodType(function, model)
    setattr(model, name, handle.wrappers[name])
  model.agent_expert_pv = handle
  return handle

"""Bind only eligible expert PV to the qualified WGP PROGRAM; observe current CU."""
from collections import Counter
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import MethodType

HERE = Path(__file__).resolve().parent


def common():
  spec = importlib.util.spec_from_file_location('qualified_hook_common', HERE/'agent-expert-pv-qualified-common.py')
  value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
  return value


class Installation:
  def close(self):
    if not self.active: return
    assert self.module._matmul is self.dispatch, 'PV dispatch replaced during run'
    self.module._matmul = self.original_matmul
    for name, wrapper in self.wrappers.items():
      assert getattr(self.model, name) is wrapper, 'Trajectory wrapper replaced during run'
      setattr(self.model, name, self.originals[name])
    assert self.model.agent_expert_pv_qualified is self
    del self.model.agent_expert_pv_qualified
    self.active = False
    self.info.update(active=False, cleanup_exact=True)

  @contextmanager
  def observe_programs(self):
    from tinygrad.runtime.support import hcq2
    from tinygrad.uop.ops import Ops
    previous = hcq2._build_queues
    def build(ctx):
      for call, _, _ in ctx.batch:
        p = call.body
        if p.op is not Ops.PROGRAM: continue
        name = p.src[0].arg.function_name
        if name not in ('alpamayo_attention_values', 'alpamayo_attention_scores', 'alpamayo_attention_softmax', 'alpamayo_expert_pv_2x2'): continue
        key = p.key.hex()
        if key in self.programs: continue
        ast = self.prepared['candidate' if self.candidate else 'current']['ast'] if key == self.expected['program_key'] else None
        record = self.C.usb().old().describe(p, ast)
        if name == 'alpamayo_expert_pv_2x2':
          assert self.candidate
          self.C.usb().validate_description(record, self.expected)
        self.programs[key] = record
      return previous(ctx)
    hcq2._build_queues = build
    try: yield
    finally:
      assert hcq2._build_queues is build
      hcq2._build_queues = previous
      self.info['observer_cleanup_exact'] = True

  def inspect_capture(self, jit):
    captured = jit.captured
    if captured is None or any(captured is prior for prior in self.seen): return
    records = [r for c in captured._linear.src for r in c.without_after.arg.aux.kernels]
    counts = Counter(str(r[1]) for r in records)
    pv = self.expected['name']
    assert counts[pv] == counts['alpamayo_attention_scores'] == counts['alpamayo_attention_softmax'] == 84, counts
    assert counts['alpamayo_attention_values' if self.candidate else 'alpamayo_expert_pv_2x2'] == 0
    binaries, program_keys = {}, set()
    for record in records:
      name, key = str(record[1]), record[4].hex()
      if name not in (pv, 'alpamayo_attention_scores', 'alpamayo_attention_softmax'): continue
      assert key in self.programs and self.programs[key]['name'] == name
      metadata = self.programs[key]
      assert metadata['target'] == self.info['evidence']['target']
      binaries.setdefault(name, set()).add(metadata['binary_sha256'])
      if name == pv:
        self.C.usb().validate_description(metadata, self.expected)
        assert list(record[6][0]) == [0] and list(record[6][1]) == [1, 2]
        program_keys.add(key)
    assert program_keys == {self.expected['program_key']}
    assert binaries[pv] == {self.expected['binary_sha256']}
    self.info['captures'].append({'pv_calls': 84, 'qk_calls': 84, 'softmax_calls': 84, 'total_calls': len(records),
      'program_keys': sorted(program_keys), 'target': self.expected['target'], 'pv_full_description': self.expected,
      'binaries': {name: sorted(values) for name, values in binaries.items()}})
    self.seen.append(captured)


def install(model, module, *, candidate=True, evidence=None):
  from tinygrad import Device, dtypes
  import amd_attention
  C = common()
  verified = evidence if isinstance(evidence, dict) else C.qualification()
  H = C.usb().load('agent-expert-pv-route-hook')
  assert model.fp8_wmma and model.dtype == dtypes.bfloat16 and not getattr(model, 'trajectory_jits', {})
  assert not any(hasattr(model, name) for name in ('agent_expert_pv', 'agent_expert_tiles', 'agent_expert_pv_qualified'))
  H.checked('amd_attention.py', amd_attention.__file__); H.checked('expert-kernels.py', module.__file__)
  assert module._matmul is amd_attention._matmul and module._softmax is amd_attention._softmax and module._pack is amd_attention._pack
  for fn in (amd_attention._matmul, amd_attention._softmax, amd_attention._pack): H.verify_function(fn, amd_attention.__file__)
  H.verify_function(module.attention, module.__file__); H.verify_function(model.attention, module.__file__)
  for name in ('trajectory', '_trajectory', 'expert_step', 'text_layer'): H.verify_function(getattr(model, name), C.SNAP/'model.py')
  assert model.config['expert_config']['llm_config']['num_hidden_layers'] == 28
  renderer = Device[Device.DEFAULT].renderer
  C.runtime(renderer, verified)
  prepared = C.prepare(verified)
  handle = Installation()
  handle.C, handle.model, handle.module, handle.candidate, handle.active, handle.depth = C, model, module, candidate, True, 0
  handle.expected, handle.prepared = verified['candidate' if candidate else 'current'], prepared
  handle.original_matmul = module._matmul
  handle.originals = {name: getattr(model, name) for name in ('trajectory', '_trajectory')}
  handle.counts, handle.programs, handle.seen, handle.wrappers = Counter(), {}, [], {}
  handle.info = {'candidate': candidate, 'active': True, 'cleanup_exact': False, 'observer_cleanup_exact': False,
    'evidence': verified, 'traces': [], 'captures': [], 'trajectory_calls': 0, 'unqualified_trajectory_calls': 0,
    'bound_program_calls': 0, 'outside_scope_selected': 0, 'runtime_programs': handle.programs}
  def dispatch(out, a, b, shape, causal=False, scaled=False):
    active = handle.active and handle.depth > 0
    selected = H.eligible(out, a, b, shape, causal, scaled, active)
    if active:
      if selected: handle.counts['pv'] += 1
      elif tuple(shape) == H.QK_SHAPE and scaled is True and causal is False: handle.counts['qk'] += 1
      else: handle.counts['other'] += 1
    if selected and handle.candidate:
      program = handle.prepared['candidate']['program']
      assert all(p.numel() == arg.numel() and p.dtype == arg.dtype for p, arg in zip(program.kernel_params, (out, a, b), strict=True))
      handle.info['bound_program_calls'] += 1
      return program
    result = handle.original_matmul(out, a, b, shape=shape, causal=causal, scaled=scaled)
    if selected: assert result.key.hex() == handle.expected['input_ast_key'], 'Current PV input AST changed'
    return result
  handle.dispatch = dispatch
  module._matmul = dispatch
  def trace(self, *args, **kwargs):
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
    handle.info['trajectory_calls'] += 1
    if args[4] == 3 and self.cache[0].shape[3] == 2304:
      for jit in self.trajectory_jits.values(): handle.inspect_capture(jit)
    return result
  for name, function in (('_trajectory', trace), ('trajectory', trajectory)):
    handle.wrappers[name] = MethodType(function, model); setattr(model, name, handle.wrappers[name])
  model.agent_expert_pv_qualified = handle
  return handle

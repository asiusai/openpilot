"""Parent-only, two-query mode inventory. --prepare and --check use no devices.

Original selected policy, followed optionally by one consistent LLVM-WGP arm.
Neither arm qualifies quality or speed. Complete HIP binaries remain unchanged.
"""
import argparse
import builtins
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-expert-pv-route-source'
SNAP = ROOT/'alpamayo'
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PY = '/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'
QUEUE = Path('/tmp/opencode/queue-alpamayo-expert-pv-route.py')
QUEUE_SHA = '8fbb9c5daa5eceb499fa852201a8537593b7e193f104a0f39c5f4ccd148775a9'
WRAPPER = Path('/tmp/opencode/run-alpamayo-speed-trial.py')
MODE = 'agent-compiler-mode-renderer'
PROOF = 'agent-compiler-mode-proof'


def sha(data): return hashlib.sha256(data).hexdigest()


def write(path, value):
  path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def frozen():
  assert sha(QUEUE.read_bytes()) == QUEUE_SHA
  queue = load(QUEUE, 'agent_compiler_mode_frozen_queue')
  manifest = queue.verify_sources()
  route = load(SNAP/'experiments/agent-expert-pv-route-run.py', 'agent_compiler_mode_route_plan')
  return queue, manifest, route


def metadata(binary):
  from tinygrad.runtime.support.elf import elf_loader
  _, sections, _ = elf_loader(binary)
  note = next(s.content for s in sections if s.name == '.note')
  result = {}
  for key in ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size',
              'group_segment_fixed_size', 'wavefront_size', 'workgroup_processor_mode'):
    field = ('.'+key).encode()
    pos = note.index(field)+len(field)
    tag = note[pos]
    if tag < 128: result[key] = tag
    elif tag in (204, 205, 206, 207):
      size = 1 << (tag-204)
      result[key] = int.from_bytes(note[pos+1:pos+1+size], 'big')
    else: raise ValueError((key, tag))
  # Cross-check the hardware descriptor, not just the YAML/MessagePack metadata.
  rodata = next(s.content for s in sections if s.name == '.rodata')
  descriptor_wgp = (int.from_bytes(rodata[48:52], 'little') >> 29) & 1
  assert descriptor_wgp == result['workgroup_processor_mode']
  result['descriptor_wgp_bit'] = descriptor_wgp
  return result


def source_pins():
  files = [HERE/(n+'.py') for n in (MODE, PROOF, 'agent-compiler-mode-inventory', 'agent-compiler-mode-check')]
  files += [HERE/'agent-compiler-mode-cpu.json', HERE/'agent-compiler-mode-reviewed-check.json', WRAPPER, QUEUE]
  return {str(p): sha(p.read_bytes()) for p in files}


def prepare(path):
  assert not path.exists()
  queue, manifest, route = frozen()
  cpu = json.loads((HERE/'agent-compiler-mode-cpu.json').read_text())
  check = json.loads((HERE/'agent-compiler-mode-reviewed-check.json').read_text())
  assert cpu['complete'] and cpu['cpu_only'] and not cpu['gpu_executed']
  assert check['complete'] and check['cpu_only'] and not check['gpu_executed']
  for report in (cpu, check):
    for p, digest in report['source_sha256'].items(): assert sha(Path(p).read_bytes()) == digest
  planned = route.plan(ROOT/'invocations/quality-0.json', WORK/'agent-compiler-mode-original/model.json')
  protocol = {'schema': 1, 'source_sha256': source_pins(), 'manifest_sha256': sha(queue.MANIFEST.read_bytes()),
    'maximum_model_queries': 2, 'timeout_seconds': 300, 'quality_qualified': False, 'speed_qualified': False,
    'policy': 'dynamic vision; fixed language; pad2048; target-verified NOMINAL; shuffle-FMA; full images; 3 Euler',
    'source_files': len(manifest['files']), 'external_pins': len(manifest['external_pins']),
    'input_pins': len(manifest['data_input_pins']), 'planned_identity': planned['pair_identity_sha256'],
    'samples': planned['scheduled_samples'][:2], 'parallel': 32, 'llvmopt': 1,
    'arms': ['original', 'consistent-wgp'],
    'diagnostic_changes': ['two original warmup queries; no measured queries', 'observation hooks',
      'controlled stop after second progress row', 'explicit original default PARALLEL=32 and LLVMOPT=1'],
    'candidate_change': 'one device renderer factory uses the importable fixed-WGP class before model compilation',
    'cache_policy': 'Original recorded compiler disk cache; distinct WGP/CU feature namespaces; fresh process per arm',
    'checkpoint_scope': 'Existing local checkpoint and FP8 cache; unchanged original loading; checkpoint tensors are not newly content-hashed',
    'qualification': 'If any model PROGRAM changes, fresh complete strict route quality is required before adoption.'}
  write(path, protocol)
  digest = sha(path.read_bytes())
  commands = {}
  for arm in protocol['arms']:
    cmd = [PY, str(WRAPPER), 'agent-compiler-mode-'+arm, '--program', str(Path(__file__).resolve()),
      '--timeout-seconds', '300', '--set', 'protocol', str(path), '--set', 'protocol-sha', digest,
      '--set', 'variant', arm, '--env', 'PYTHONPATH', ':'.join(map(str, (HERE, TINY, SNAP))),
      '--env', 'PYTHONDONTWRITEBYTECODE', '1', '--env', 'PARALLEL', '32', '--env', 'LLVMOPT', '1']
    if arm != 'original': cmd += ['--set', 'reference', str(WORK/'agent-compiler-mode-original/report.json')]
    commands[arm] = cmd
  commands_path = path.with_name(path.stem+'-commands.json')
  assert not commands_path.exists()
  write(commands_path, {'protocol_sha256': digest, 'commands': commands,
    'shell': {k: shlex.join(v) for k, v in commands.items()}, 'executed': False})
  print(json.dumps({'prepared': True, 'gpu_executed': False, 'protocol_sha256': digest, 'commands': str(commands_path)}))


class Inventory:
  def __init__(self, directory):
    self.directory = directory
    self.programs, self.compile_records, self.captures = {}, [], []
    self._captures_seen = set()
    self.batches = 0
    self.proof = importlib.import_module(PROOF)

  def record(self, program):
    from tinygrad.uop.ops import Ops
    if program.op is not Ops.PROGRAM or program.arg.target.device != 'AMD': return None
    key = program.key.hex()
    if key not in self.programs:
      row = self.proof.describe(program)
      row.update(name=program.src[0].arg.function_name, target=repr(program.arg.target), resources=metadata(program.src[-1].arg),
        source_kind='hip' if '__global__' in program.src[2].arg else 'llvm',
        params=[{'slot': p.arg.slot, 'dtype': str(p.dtype), 'shape': list(p.max_shape),
                 'addrspace': str(p.arg.addrspace)} for p in program.kernel_params], queue_occurrences=0)
      self.programs[key] = row
      if self.directory:
        self.directory.mkdir(exist_ok=True)
        stem = self.directory/('agent-compiler-mode-'+key)
        stem.with_suffix('.source.txt').write_text(program.src[2].arg)
        stem.with_suffix('.hsaco').write_bytes(program.src[3].arg)
        write(stem.with_suffix('.json'), row)
    return key

  def before_after_compile(self, original, linear, *args, **kwargs):
    from tinygrad.codegen import to_program_key, to_program_context, to_program_cache
    from tinygrad.engine.realize import _get_call_to_compile
    from tinygrad.uop.ops import Ops
    pending = []
    for call in linear.toposort():
      if call.op is not Ops.CALL: continue
      found = _get_call_to_compile(call)
      if found is None:
        if call.body.op is Ops.PROGRAM: self.record(call.body)
        continue
      ast, renderer = found
      if renderer.target.device != 'AMD': continue
      key = to_program_key(ast, renderer)
      precompiled = ast.op is Ops.PROGRAM and ast.src[-1].op is Ops.BINARY
      pending.append((key, {'input_ast_key': ast.key.hex(), 'input_op': ast.op.name,
        'origin': 'precompiled_binary' if precompiled else 'renderer_compilation',
        'input_binary_sha256': sha(ast.src[-1].arg) if precompiled else None,
        'parent_renderer': self.proof.identity(renderer), 'cache_hit_before': key in to_program_cache,
        'context': {v.key: v.value for v in to_program_context}, 'cache_key': self.proof.key_description(key)}))
    result = original(linear, *args, **kwargs)
    for key, row in pending:
      row['program_key'] = self.record(to_program_cache[key])
      if row['origin'] == 'precompiled_binary':
        assert self.programs[row['program_key']]['binary_sha256'] == row['input_binary_sha256']
      self.compile_records.append(row)
    return result

  def build(self, original, ctx):
    for call, _, _ in ctx.batch:
      if (key := self.record(call.body)) is not None: self.programs[key]['queue_occurrences'] += 1
    self.batches += 1
    return original(ctx)

  def captured(self, jit):
    from tinygrad.runtime.support.hcq2 import HCQInfo
    if jit.captured is None: return
    identity = (id(jit), jit.captured._linear.key.hex())
    if identity in self._captures_seen: return
    self._captures_seen.add(identity)
    kernels = []
    for call in jit.captured._linear.src:
      aux = call.without_after.arg.aux
      if not isinstance(aux, HCQInfo): continue
      for row in aux.kernels:
        if row[4] is None or str(row[0]).split(':')[0] != 'AMD': continue
        key = row[4].hex()
        assert key in self.programs, 'Captured AMD PROGRAM is missing from the binary inventory'
        kernels.append({'program_key': key, 'name': str(row[1]), 'global_input_slots': list(row[5]),
          'output_argument_indices': list(row[6][0]), 'input_argument_indices': list(row[6][1])})
    self.captures.append({'function': getattr(jit.fxn, '__qualname__', repr(jit.fxn)),
      'linear_key': jit.captured._linear.key.hex(), 'kernels': kernels})

  @contextmanager
  def observe(self):
    from tinygrad.engine import realize, jit
    from tinygrad.runtime.support import hcq2
    old_lower, old_alias = realize.lower_and_compile, hcq2.lower_and_compile
    old_build, old_call = hcq2._build_queues, jit._TinyJit.__call__
    assert old_lower is old_alias
    def lower(*args, **kwargs): return self.before_after_compile(old_lower, *args, **kwargs)
    def build(ctx): return self.build(old_build, ctx)
    def call(obj, *args, **kwargs):
      result = old_call(obj, *args, **kwargs)
      self.captured(obj)
      return result
    realize.lower_and_compile = hcq2.lower_and_compile = lower
    hcq2._build_queues, jit._TinyJit.__call__ = build, call
    try: yield
    finally:
      realize.lower_and_compile, hcq2.lower_and_compile = old_lower, old_alias
      hcq2._build_queues, jit._TinyJit.__call__ = old_build, old_call

  def export(self):
    return {'programs': self.programs, 'compile_records': self.compile_records, 'captures': self.captures,
      'queue_batches_observed': self.batches,
      'unique_binary_modes': dict(Counter(str(v['resources']['workgroup_processor_mode']) for v in self.programs.values()))}


class TwoQueriesComplete(Exception): pass


def stop_after_query(print_function, filename, save_model):
  """Stop at a source-pinned logging boundary after two completed original queries."""
  def emit(*values, **kwargs):
    print_function(*values, **kwargs)
    caller = sys._getframe(1)
    if caller.f_code.co_filename != str(filename) or caller.f_code.co_name != 'main': return
    if not values or not isinstance(values[0], str) or not values[0].startswith('Iteration 1: '): return
    report = caller.f_locals['report']
    assert len(report['warmup']) == 2 and report['runs'] == []
    assert caller.f_locals['iteration'] == 1
    if caller.f_locals.get('pool'): caller.f_locals['pool'].shutdown(wait=True)
    save_model(report)
    raise TwoQueriesComplete()
  return emit


def compare(before, after):
  # Compare actual PROGRAMs by observed input AST, preserving all argument/launch fields.
  def mapping(report):
    result = {}
    for r in report['inventory']['compile_records']:
      identity = (r['input_ast_key'], r['origin'], json.dumps(r['context'], sort_keys=True), r['parent_renderer']['target'])
      result.setdefault(identity, set()).add(r['program_key'])
    return result
  a, b = mapping(before), mapping(after)
  result = {'common_inputs': 0, 'changed_programs': [], 'only_original': [], 'only_candidate': []}
  for key in sorted(a.keys() | b.keys()):
    if key not in a: result['only_candidate'].append(list(key)); continue
    if key not in b: result['only_original'].append(list(key)); continue
    result['common_inputs'] += 1
    # Existing history-sensitive originals can contain multiple PROGRAMs per input.
    old = [before['inventory']['programs'][k] for k in sorted(a[key])]
    new = [after['inventory']['programs'][k] for k in sorted(b[key])]
    assert len(new) == 1, 'Candidate changed program identity within one process'
    for x in old:
      y = new[0]
      for field in ('lowered_ast', 'linear', 'source_sha256', 'program_info', 'params'):
        assert x[field] == y[field], f'Unexpected lowering change: {key} {field}'
      if key[1] == 'precompiled_binary': assert x['key'] == y['key'], 'Precompiled PROGRAM changed'
    if {x['key'] for x in old} != {new[0]['key']}:
      result['changed_programs'].append({'input_ast_key': key[0], 'name': new[0]['name'],
        'before': [{'key': x['key'], 'mode': x['resources']['workgroup_processor_mode']} for x in old],
        'after': {'key': new[0]['key'], 'mode': new[0]['resources']['workgroup_processor_mode']}})
  # Also bind already-complete PROGRAMs that never entered lower_and_compile's cache.
  explicit_before = {r['key'] for r in before['inventory']['programs'].values() if r['source_kind'] == 'hip'}
  explicit_after = {r['key'] for r in after['inventory']['programs'].values() if r['source_kind'] == 'hip'}
  assert explicit_before == explicit_after, 'Explicit non-LLVM program inventory differs'
  result['explicit_binary_set_unchanged'] = True
  return result


def run(args):
  assert args.protocol and args.protocol_sha and args.variant and args.output
  assert sha(args.protocol.read_bytes()) == args.protocol_sha
  protocol = json.loads(args.protocol.read_text())
  assert protocol['maximum_model_queries'] == 2 and protocol['timeout_seconds'] == 300
  for p, digest in protocol['source_sha256'].items(): assert sha(Path(p).read_bytes()) == digest, p
  _, _, route = frozen()
  assert not args.output.exists()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  model_output = args.output.with_name('agent-compiler-mode-model.json')
  assert not model_output.exists()
  planned = route.plan(ROOT/'invocations/quality-0.json', model_output)
  assert planned['pair_identity_sha256'] == protocol['planned_identity']
  command = planned['effective_command'].copy()
  command[command.index('--warmup')+1] = '2'
  command[command.index('--runs')+1] = '0'
  prior = json.loads(args.reference.read_text()) if args.reference else None
  if args.variant == 'consistent-wgp':
    assert prior and prior['complete'] and prior['variant'] == 'original'
    assert prior['protocol_sha256'] == args.protocol_sha
    assert prior['source_sha256'] == protocol['source_sha256']
    for path, digest in prior['model_artifact_sha256'].items(): assert sha(Path(path).read_bytes()) == digest
    for key, row in prior['inventory']['programs'].items():
      stem = Path(prior['artifacts'])/('agent-compiler-mode-'+key)
      assert sha(stem.with_suffix('.source.txt').read_bytes()) == row['source_sha256']
      assert sha(stem.with_suffix('.hsaco').read_bytes()) == row['binary_sha256']
  else: assert prior is None
  assert 'tinygrad' not in sys.modules, 'Set the recorded environment before importing tinygrad'
  env = json.loads((ROOT/'invocations/quality-0.json').read_text())['environment']
  for key, value in env.items():
    if value is None: os.environ.pop(key, None)
    else: os.environ[key] = value
  assert os.environ.get('PARALLEL', '32') == '32' and os.environ.get('LLVMOPT', '1') == '1'
  assert os.environ.get('BEAM', '0') == os.environ.get('JITBEAM', '0') == '0'
  os.environ.update(PYTHONDONTWRITEBYTECODE='1', LLVMOPT='1', PARALLEL='32', PROFILE='0')
  # Exact existing directories are declared in the command and checked in workers.
  # Spawn preserves this actual sys.path, even when the frozen wrapper resets PYTHONPATH.
  sys.path[:0] = [str(SNAP/'experiments'), str(SNAP)]
  fixed = importlib.import_module(MODE)
  assert Path(importlib.util.find_spec(MODE).origin).resolve() == HERE/(MODE+'.py')
  from tinygrad import Device
  from tinygrad.helpers import PARALLEL
  from tinygrad.codegen import to_program_context
  assert not Device._opened_devices and PARALLEL.value == 32
  shared = importlib.import_module('wgp_compiler')
  assert Path(shared.__file__).resolve() == SNAP/'experiments/wgp_compiler.py'
  inventory = Inventory(args.output.parent/'agent-compiler-mode-artifacts')
  report = {'complete': False, 'variant': args.variant, 'protocol_sha256': args.protocol_sha,
    'source_sha256': protocol['source_sha256'], 'model_report': str(model_output), 'artifacts': str(inventory.directory),
    'quality_qualified': False, 'speed_qualified': False, 'planned_identity': planned['pair_identity_sha256'],
    'actual_command': command, 'timings_valid_for_throughput': False, 'gpu_executed': False}
  if args.reference: report['reference_sha256'] = sha(args.reference.read_bytes())
  def save():
    report['inventory'] = inventory.export()
    write(args.output, report)
  old_install, old_print, old_argv = shared.install, builtins.print, sys.argv[:]
  def install(device):
    assert not report.get('installed') and not inventory.programs
    assert device.arch == 'gfx1200' and device.is_usb
    if args.variant == 'original': old_install(device)
    else: report['installation'] = fixed.install(device)
    report['installed'] = True
    report['runtime'] = {'renderer': inventory.proof.identity(device.renderer), 'parallel': PARALLEL.value,
      'context': {v.key: v.value for v in to_program_context},
      'environment': {k: os.environ.get(k) for k in sorted(set(env) | {'LLVMOPT', 'PARALLEL', 'BEAM', 'JITBEAM'})}}
    save()
  def save_model(model): write(model_output, model)
  started = time.monotonic()
  shared.install = install
  builtins.print = stop_after_query(old_print, SNAP/'experiments/benchmark-pipeline.py', save_model)
  save()
  try:
    sys.argv = command
    with inventory.observe():
      report['gpu_executed'] = True
      try: runpy.run_path(command[0], run_name='__main__')
      except TwoQueriesComplete: report['controlled_stop_after_two_queries'] = True
      else: raise AssertionError('Expected the exact two-query stop boundary')
    model = json.loads(model_output.read_text())
    spec = json.loads(model_output.with_suffix('.speculation.json').read_text())
    report['model_artifact_sha256'] = {str(path): sha(path.read_bytes()) for path in
      (model_output, model_output.with_suffix('.speculation.json'), model_output.with_suffix('.jsonl'))}
    short_plan = dict(planned, warmup=2, runs=0, scheduled_samples=planned['scheduled_samples'][:2])
    report['self_validation'] = route.compare_reports(model, model, short_plan, spec, spec)
    assert report['installed'] and inventory.programs and any(c['kernels'] for c in inventory.captures)
    report['inventory'] = inventory.export()
    if args.variant == 'consistent-wgp':
      assert prior['runtime']['context'] == report['runtime']['context']
      assert prior['runtime']['environment'] == report['runtime']['environment']
      assert prior['runtime']['parallel'] == report['runtime']['parallel'] == 32
      assert prior['runtime']['renderer']['target'] == report['runtime']['renderer']['target']
      for path, digest in prior['model_artifact_sha256'].items(): assert sha(Path(path).read_bytes()) == digest
      assert sha(args.reference.read_bytes()) == report['reference_sha256']
      rows = [r for r in inventory.compile_records if r['origin'] == 'renderer_compilation']
      assert rows and all(r['parent_renderer']['class'] == MODE+'.AMDLLVMRenderer' for r in rows)
      assert all(inventory.programs[r['program_key']]['resources']['workgroup_processor_mode'] == 1 for r in rows)
      before_model = json.loads(Path(prior['model_report']).read_text())
      before_spec = json.loads(Path(prior['model_report']).with_suffix('.speculation.json').read_text())
      report['strict_two_query_comparison'] = route.compare_reports(before_model, model, short_plan, before_spec, spec)
      report['program_comparison'] = compare(prior, report)
    report['complete'] = True
  except BaseException as exc:
    report['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
  finally:
    shared.install, builtins.print, sys.argv = old_install, old_print, old_argv
    report['elapsed_seconds'] = time.monotonic()-started
    save()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--prepare', type=Path)
  parser.add_argument('--protocol', type=Path)
  parser.add_argument('--protocol-sha')
  parser.add_argument('--variant', choices=('original', 'consistent-wgp'))
  parser.add_argument('--reference', type=Path)
  parser.add_argument('--output', type=Path)
  args = parser.parse_args()
  assert not sys.flags.optimize
  if args.prepare: prepare(args.prepare)
  else: run(args)


if __name__ == '__main__': main()

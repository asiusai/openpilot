"""Compiler-only proof using the pinned tinygrad spawn pool and task function."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

HERE = Path(__file__).resolve().parent
MODE = 'agent-compiler-mode-renderer'


def sha(data): return hashlib.sha256(data).hexdigest()


def identity(renderer):
  module = importlib.import_module(type(renderer).__module__)
  return {'class': type(renderer).__module__+'.'+type(renderer).__name__, 'target': repr(renderer.target),
          'origin': str(Path(module.__file__).resolve()), 'module_sha256': sha(Path(module.__file__).read_bytes()),
          'compiler_class': type(renderer.compiler).__module__+'.'+type(renderer.compiler).__name__,
          'compiler_cachekey': renderer.compiler.cachekey}


def key_description(key):
  return [key[0].hex(), key[1].__module__+'.'+key[1].__name__, repr(key[2]), *key[3:]]


def describe(p):
  return {'key': p.key.hex(), 'lowered_ast': p.src[0].key.hex(), 'linear': p.src[1].key.hex(),
          'source_sha256': sha(p.src[2].arg.encode()), 'binary_sha256': sha(p.src[3].arg),
          'program_info': repr(p.arg), 'params': [repr(x) for x in p.kernel_params]}


def worker_task(task):
  # This importable function calls the *actual*, unmodified tinygrad worker task.
  from tinygrad import Device
  from tinygrad.codegen import to_program_context, to_program_key
  from tinygrad.engine.realize import _compile_kernel
  from tinygrad.helpers import ALLOW_DEVICE_USAGE, Context
  index, (ast, renderer), context = task
  with Context(**context):
    observed = {v.key: v.value for v in to_program_context}
    key = key_description(to_program_key(ast, renderer))
  i, program = _compile_kernel(task)
  assert i == index and not Device._opened_devices and ALLOW_DEVICE_USAGE.value == 0
  return {'pid': os.getpid(), 'renderer': identity(renderer), 'context': observed,
          'ast': ast.key.hex(), 'cache_key': key, 'program': describe(program),
          'device_usage_allowed': ALLOW_DEVICE_USAGE.value, 'opened_devices': sorted(Device._opened_devices)}


def ast_for(count):
  # A fresh affine FP32 map, unrelated to the existing concat and expert-PV proofs.
  from tinygrad import Tensor, dtypes
  from tinygrad.uop.ops import UOp
  x = Tensor(UOp.new_buffer('AMD', count, dtypes.float32))
  linear = (x * 1.25 + 0.0625).schedule_linear()
  assert len(linear.src) == 1
  return linear.src[0].body


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  assert not args.output.exists() and not sys.flags.optimize
  from tinygrad import Device
  from tinygrad.codegen import do_to_program, to_program, to_program_cache, to_program_key, to_program_context
  from tinygrad.device import Compiled
  from tinygrad.engine import worker
  from tinygrad.engine.realize import _compile_kernel
  from tinygrad.helpers import ALLOW_DEVICE_USAGE, Context, Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer as Base, CPULLVMRenderer
  fixed = importlib.import_module(MODE)
  self_module = importlib.import_module(Path(__file__).stem)
  assert ALLOW_DEVICE_USAGE.value == 0 and not Device._opened_devices and not to_program_cache
  target = Target('AMD', 'LLVM', 'gfx1200', 'USB')
  # A plain Compiled object exercises real factory selection without a device open.
  fake = Compiled('AMD', None, [Base], None, arch='gfx1200')
  other = Compiled('AMD:1', None, [Base], None, arch='gfx1100')
  cpu = CPULLVMRenderer(Target('CPU', 'LLVM', 'x86_64,native'))
  cpu_before = (type(cpu), cpu.compiler, repr(cpu.target))
  with Context(DEV=target):
    other_factories = other.renderers[:]
    try: fixed.install(other)
    except ValueError: pass
    else: raise AssertionError('Another architecture was accepted')
    assert other.renderers == other_factories and not other.cached_renderer
    installed = fixed.install(fake)
    renderer = fake.renderer
    assert fake._renderer_name(type(renderer)) == 'LLVM'
    assert fake.renderer is renderer
  assert (type(cpu), cpu.compiler, repr(cpu.target)) == cpu_before
  assert importlib.import_module('tinygrad.renderer.llvmir').AMDLLVMRenderer is Base
  assert type(pickle.loads(pickle.dumps(renderer))) is fixed.AMDLLVMRenderer
  assert identity(pickle.loads(pickle.dumps(renderer))) == identity(renderer)
  base = Base(target)
  report = {'complete': False, 'cpu_only': True, 'gpu_executed': False, 'install': installed,
            'unrelated_cpu_other_arch_unchanged': True, 'cases': [], 'source_sha256': {}}
  for path in (Path(__file__), Path(fixed.__file__)):
    report['source_sha256'][str(path)] = sha(path.read_bytes())
  # Two context settings prove the task propagates context instead of inheriting it.
  with Context(PARALLEL=2):
    pool = worker.get_worker_pool()
    assert pool is not None
    try:
      for i, tc in enumerate((2, 1)):
        ast = ast_for(257+i)
        with Context(TC_OPT=tc):
          ctx = {v.key: v.value for v in to_program_context}
          cu = to_program(ast, base)  # Stale original-class CU cache entry, retained.
          kcu, kwgp = to_program_key(ast, base), to_program_key(ast, renderer)
          assert kcu != kwgp and kcu[0] == kwgp[0] and kcu[2:] == kwgp[2:]
          assert base.compiler.cachekey != renderer.compiler.cachekey
          parent = to_program(ast, renderer)
          # Independently bypass both caches for the intended parent WGP compiler.
          legacy = Base(target); legacy.compiler = fixed.WGPCompiler('gfx1200')
          independent = do_to_program(ast, legacy)
          assert describe(parent) == describe(independent)
          assert cu.src[3].arg != parent.src[3].arg
          assert to_program(ast, base) is cu and to_program_cache[kcu] is cu
          task = (i, (ast, renderer), ctx)
          # Exercise the actual task directly through the actual spawn pool too.
          _, direct = next(pool.imap_unordered(_compile_kernel, [task]))
          detail = next(pool.imap_unordered(self_module.worker_task, [task]))
          assert describe(direct) == describe(parent) == detail['program']
          assert detail['context'] == ctx and detail['ast'] == ast.key.hex()
          assert detail['cache_key'] == key_description(kwgp)
          assert detail['renderer'] == identity(renderer) and detail['pid'] != os.getpid()
          report['cases'].append({'context': ctx, 'parent': describe(parent), 'parent_renderer': identity(renderer),
            'cu': describe(cu), 'worker': detail, 'direct_spawned_task_matches': True,
            'stale_cu_entry_retained': True, 'class_is_only_process_key_difference': True,
            'compiler_disk_namespaces_distinct': True})
      # Installation after compilation must fail without replacing anything.
      late = Compiled('AMD', None, [Base], None, arch='gfx1200')
      with Context(DEV=target):
        try: fixed.install(late)
        except ValueError: pass
        else: raise AssertionError('Late installation was accepted')
      assert late.renderers == [Base]
    finally:
      worker.terminate_worker_pool()
  assert not Device._opened_devices
  report.update(complete=True, opened_devices=[], late_install_rejected=True,
    worker_import_resolution='Actual spawned worker imported the canonical module and matched its absolute path and SHA256')
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
  print(json.dumps({'complete': True, 'cases': len(report['cases']), 'gpu_executed': False}))


if __name__ == '__main__': main()

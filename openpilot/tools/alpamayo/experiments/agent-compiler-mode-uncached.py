"""Force LLVM compilation inside an actual spawned worker, bypassing disk caches."""
import importlib
import json
import os
from pathlib import Path
from types import FunctionType, SimpleNamespace


def worker_task(task):
  from tinygrad import Device
  from tinygrad.engine.realize import _compile_kernel
  from tinygrad.helpers import ALLOW_DEVICE_USAGE, Context
  proof = importlib.import_module('agent-compiler-mode-proof')
  with Context(**task[2]):
    _, program = _compile_kernel(task)
    # Deliberately bypass Compiler.compile_cached and both tinygrad caches.
    raw = task[1][1].compiler.compile(program.src[2].arg)
  assert raw == program.src[3].arg and ALLOW_DEVICE_USAGE.value == 0 and not Device._opened_devices
  return {'pid': os.getpid(), 'renderer': proof.identity(task[1][1]), 'program': proof.describe(program),
    'forced_uncached_binary_sha256': proof.sha(raw), 'raw_compile_matches_actual_task': True, 'opened_devices': []}


def main():
  from tinygrad import Device, dtypes
  from tinygrad.codegen import do_to_program, to_program_context
  from tinygrad.engine import worker
  from tinygrad.helpers import ALLOW_DEVICE_USAGE, Context, Target
  from tinygrad.uop.ops import UOp
  inv = importlib.import_module('agent-compiler-mode-inventory')
  proof = importlib.import_module('agent-compiler-mode-proof')
  fixed = importlib.import_module('agent-compiler-mode-renderer')
  module = importlib.import_module(Path(__file__).stem)
  output = Path(__file__).with_suffix('.json')
  assert not output.exists() and ALLOW_DEVICE_USAGE.value == 0 and not Device._opened_devices
  renderer = fixed.AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200', 'USB'))
  gemm_path = inv.SNAP/'amd_gemm.py'
  gemm = inv.load(gemm_path, 'agent_compiler_mode_uncached_gemm')
  class Devices:
    DEFAULT = 'AMD'
    def __getitem__(self, key): return SimpleNamespace(renderer=renderer)
  function = FunctionType(gemm._gemm.__code__, dict(gemm._gemm.__globals__, Device=Devices()), gemm._gemm.__name__)
  function.__kwdefaults__ = gemm._gemm.__kwdefaults__
  specs = (((2048, 2048), dtypes.bfloat16), ((2048, 9216), dtypes.fp8e4m3),
           ((2048, 9216), dtypes.fp8e4m3), ((1,), dtypes.float32), ((1,), dtypes.float32))
  ast = function(*(UOp.placeholder(shape, dtype, i) for i, (shape, dtype) in enumerate(specs)))
  parent = do_to_program(ast, renderer)
  parent_raw = renderer.compiler.compile(parent.src[2].arg)
  assert parent_raw == parent.src[3].arg
  with Context(PARALLEL=2):
    try:
      task = (0, (ast, renderer), {v.key: v.value for v in to_program_context})
      result = next(worker.get_worker_pool().imap_unordered(module.worker_task, [task]))
    finally: worker.terminate_worker_pool()
  assert result['pid'] != os.getpid() and result['program'] == proof.describe(parent)
  assert result['forced_uncached_binary_sha256'] == proof.sha(parent_raw)
  expected = json.loads((inv.HERE/'agent-compiler-mode-reviewed-check.json').read_text())['fc2_cpu_only']['programs']['wgp']
  assert expected['key'] == parent.key.hex() and expected['binary_sha256'] == proof.sha(parent_raw)
  assert not Device._opened_devices
  inv.write(output, {'complete': True, 'cpu_only': True, 'gpu_executed': False, 'opened_devices': [],
    'parent_uncached_binary_sha256': proof.sha(parent_raw), 'worker': result,
    'source_sha256': {str(p): proof.sha(p.read_bytes()) for p in
      (Path(__file__), Path(fixed.__file__), Path(proof.__file__), Path(inv.__file__), gemm_path)},
    'reviewed_check_sha256': proof.sha((inv.HERE/'agent-compiler-mode-reviewed-check.json').read_bytes())})
  print(json.dumps({'complete': True, 'raw_parent_worker_binary_equal': True, 'gpu_executed': False}))


if __name__ == '__main__': main()

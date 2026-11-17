"""Static gfx1200 compilation of indexed-copy kernels. Never opens a GPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=Path(__file__).with_name('agent-token-prune-compiler-report.json'))
  parser.add_argument('--artifacts', type=Path, default=Path(__file__).with_name('agent-token-prune-compiler-artifacts'))
  args = parser.parse_args()
  from tinygrad import Device, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.uop.ops import Ops, UOp
  path = Path(__file__).with_name('agent-token-prune-runtime.py')
  spec = importlib.util.spec_from_file_location('agent_token_prune_compile_runtime', path)
  runtime = importlib.util.module_from_spec(spec); sys.modules[spec.name] = runtime; spec.loader.exec_module(runtime)
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  args.artifacts.mkdir(exist_ok=True)
  assert all(p.name.startswith('agent-token-prune-') for p in args.artifacts.iterdir())
  records = []
  for name, source, output, fxn in (
      ('embedding-gather', (1, 2112, 2048), (1, 1152, 2048), runtime.gather_ast),
      ('rotary-gather', (1, 1, 2112, 128), (1, 1, 1152, 128), runtime.gather_ast),
      ('kv-scatter', (2, 1, 8, 1152, 128), (2, 1, 8, 2304, 128), runtime.scatter_ast)):
    destination = UOp.placeholder(output, dtypes.bfloat16, 0)
    value = UOp.placeholder(source, dtypes.bfloat16, 1)
    indices = UOp.placeholder((1152,), dtypes.int32, 2)
    program = to_program(fxn(destination, value, indices), renderer)
    text = next(u.arg for u in program.src if u.op is Ops.SOURCE)
    binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
    stem = 'agent-token-prune-'+name
    (args.artifacts/(stem+'.ll')).write_text(text)
    (args.artifacts/(stem+'.hsaco')).write_bytes(binary)
    local_size = list(program.arg.local_size)
    assert local_size != [1, 1, 1], 'Indexed copies must not dispatch one thread per workgroup'
    records.append({'name': name, 'source_shape': source, 'destination_shape': output,
                    'global_size': list(program.arg.global_size), 'local_size': local_size,
                    'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                    'binary_sha256': hashlib.sha256(binary).hexdigest(), 'binary_bytes': len(binary)})
  assert not Device._opened_devices, Device._opened_devices
  report = {'cpu_only': True, 'gpu_executed': False, 'gpu_correctness_or_speed_validated': False,
            'compiler': 'gfx1200 LLVM 20, -cumode; existing generic copy heuristics, no shape sweep',
            'runtime_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'records': records, 'opened_devices': []}
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__': main()

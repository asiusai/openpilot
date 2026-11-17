"""CPU-only gfx1200 compilation of inherited indexed copies at both declared sizes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=HERE/'agent-token-current-compiler-report.json')
  parser.add_argument('--artifacts', type=Path, default=HERE/'agent-token-current-compiler-artifacts')
  args = parser.parse_args()
  if os.environ.get('DEV') != 'CPU': parser.error('Set DEV=CPU explicitly')
  from tinygrad import Device, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.uop.ops import Ops, UOp
  spec = importlib.util.spec_from_file_location('agent_token_current_compile_base', HERE/'agent-token-prune-runtime.py')
  base = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = base
  spec.loader.exec_module(base)
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  args.artifacts.mkdir(exist_ok=False)
  records = []
  for policy, rows in (('latest-full-history-50',1344), ('latest-full-history-75',1728)):
    for name, source, output, fxn in (
      ('embedding-gather',(1,2112,2048),(1,rows,2048),base.gather_ast),
      ('rotary-gather',(1,1,2112,128),(1,1,rows,128),base.gather_ast),
      ('kv-scatter',(2,1,8,rows,128),(2,1,8,2304,128),base.scatter_ast)):
      destination = UOp.placeholder(output, dtypes.bfloat16, 0)
      value = UOp.placeholder(source, dtypes.bfloat16, 1)
      slots = UOp.placeholder((rows,), dtypes.int32, 2)
      program = to_program(fxn(destination,value,slots), renderer)
      text = next(value.arg for value in program.src if value.op == Ops.SOURCE)
      binary = next(value.arg for value in program.src if value.op == Ops.BINARY)
      stem = 'agent-token-current-'+policy+'-'+name
      (args.artifacts/(stem+'.ll')).write_text(text)
      (args.artifacts/(stem+'.hsaco')).write_bytes(binary)
      assert list(program.arg.local_size) != [1,1,1]
      records.append(dict(policy=policy, name=name, source_shape=source, destination_shape=output,
        global_size=list(program.arg.global_size), local_size=list(program.arg.local_size), binary_bytes=len(binary),
        source_sha256=hashlib.sha256(text.encode()).hexdigest(), binary_sha256=hashlib.sha256(binary).hexdigest()))
  assert not Device._opened_devices
  report = dict(cpu_only=True, opened_devices=[], gpu_executed=False, records=records,
    inherited_runtime_sha256=hashlib.sha256((HERE/'agent-token-prune-runtime.py').read_bytes()).hexdigest(),
    compiler='existing LLVM20 gfx1200 -cumode; inherited generic copy heuristics; no shape sweep')
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(dict(output=str(args.output), records=len(records), opened_devices=[]), indent=2))


if __name__ == '__main__': main()

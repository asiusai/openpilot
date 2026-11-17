"""CPU-only scheduling audit of the frozen v3 guarded-output baseline."""
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

EXP = Path(__file__).parent
ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-output')
ROOT.mkdir(exist_ok=True)
os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', CACHEDB=str(ROOT/'cpu-cache.db'), TMPDIR=str(ROOT))
sys.path.insert(0, '/tmp/opencode/alpamayo-asius-20261114/tinygrad')
from tinygrad import Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.support.hcq2 import unwrap_view


def source_baseline():
  source = (EXP/'agent-ck-attention-v3-run.py').read_text()
  assert hashlib.sha256(source.encode()).hexdigest() == '98ab25e20a9c250467f70919d6416fc25390ea0bdbf4e74c939a42559ce665d1'
  run = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
  functions = [n for n in run.body if isinstance(n, ast.FunctionDef) and n.name in ('baseline_program','baseline')]
  assert len(functions) == 2
  base = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-attention-v3/prepared-final')
  namespace = dict(Tensor=Tensor, dtypes=dtypes, UOp=UOp, Ops=Ops, KernelInfo=KernelInfo,
    baseline_blob=(base/'baseline.hsaco').read_bytes(), baseline_source=(base/'baseline.cpp').read_text())
  exec(compile(ast.Module(body=functions, type_ignores=[]), str(EXP/'agent-ck-attention-v3-run.py'), 'exec'), namespace)
  return namespace['baseline']


def tensor(shape, dtype):
  # Lazy AMD-labelled buffers permit scheduling; ALLOW_DEVICE_USAGE=0 forbids opening any GPU.
  return Tensor(UOp.from_buffer(Buffer('AMD', math.prod(shape)*dtype.itemsize), dtype)).reshape(shape)


def describe(u):
  base, offset = unwrap_view(u)
  return {'op':u.op.name, 'shape':list(u.shape), 'dtype':str(u.dtype),
    'base_op':base.op.name, 'base_slot':getattr(base.arg,'slot',None), 'byte_offset':offset,
    'buffer_bytes':getattr(getattr(base.arg,'buffer',None),'nbytes',None), 'uop_key':u.key.hex()}


def main():
  baseline = source_baseline()
  q, k, v = [tensor(shape,dtypes.bfloat16) for shape in [(1,16,2048,128),(1,8,2048,128),(1,8,2048,128)]]
  storage = tensor((4194304+512,), dtypes.bfloat16)
  inner = storage[256:256+4194304].reshape(1,16,2048,128)
  original_storage = storage.uop
  output = baseline(inner,q,k,v)
  before = {'storage':describe(storage.uop), 'inner':describe(inner.uop), 'returned':describe(output.uop)}
  linear, variables = output.linear_with_vars()
  assert not variables
  rows = []
  for call in linear.src:
    rows.append({'body_op':call.body.op.name, 'name':str(getattr(call.body.src[0].arg if call.body.op is Ops.PROGRAM else call.body.arg,'name','')),
      'arguments':[describe(x) for x in call.src[1:]]})
  result = {'gpu_executed':False, 'source':'unchanged v3 baseline AST and saved baseline.hsaco',
    'before':before, 'after':{'storage':describe(storage.uop),'returned':describe(output.uop)}, 'scheduled_calls':rows,
    'original_storage_unchanged_uop':storage.uop is original_storage}
  target = ROOT/'baseline-schedule.json'
  assert not target.exists()
  target.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))


if __name__ == '__main__': main()

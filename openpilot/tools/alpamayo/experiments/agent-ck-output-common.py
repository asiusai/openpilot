"""Shared graph construction for one guarded-output binding diagnostic."""
from functools import lru_cache
import ast
import hashlib
import importlib.util
import math
from pathlib import Path

EXP = Path(__file__).parent
ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-output')
QSHAPE, KVSHAPE, LSHAPE = (1,16,2048,128), (1,8,2048,128), (1,16,2048)
OC, LC, GUARD = math.prod(QSHAPE), math.prod(LSHAPE), 256


@lru_cache(None)
def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'), EXP/(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


@lru_cache(None)
def baseline():
  from tinygrad import Tensor,dtypes
  from tinygrad.uop.ops import UOp,Ops,KernelInfo
  source = (EXP/'agent-ck-attention-v3-run.py').read_text()
  assert hashlib.sha256(source.encode()).hexdigest() == '98ab25e20a9c250467f70919d6416fc25390ea0bdbf4e74c939a42559ce665d1'
  run = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'run')
  functions = [n for n in run.body if isinstance(n,ast.FunctionDef) and n.name in ('baseline_program','baseline')]
  assert len(functions) == 2
  folder = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-attention-v3/prepared-final')
  namespace = dict(Tensor=Tensor,dtypes=dtypes,UOp=UOp,Ops=Ops,KernelInfo=KernelInfo,
    baseline_blob=(folder/'baseline.hsaco').read_bytes(),baseline_source=(folder/'baseline.cpp').read_text())
  exec(compile(ast.Module(body=functions,type_ignores=[]),str(EXP/'agent-ck-attention-v3-run.py'),'exec'),namespace)
  return namespace['baseline']


def outputs(arm, normalize, q, k, v, out_storage, lse_storage):
  out = out_storage[GUARD:GUARD+OC].reshape(QSHAPE)
  lse = lse_storage[GUARD:GUARD+LC].reshape(LSHAPE)
  if normalize:
    # Register both contiguous offset views before custom_kernel callification.
    # CPU and parent checks require actual kernel pointers into the guarded allocations.
    out, lse = out.contiguous(), lse.contiguous()
  if arm == 'baseline': return (baseline()(out,q,k,v),)
  assert arm == 'candidate'
  return load('agent-ck-attention-v2-adapter').attention(q,k,v,out=out,lse=lse)


def execute(arm, normalize, *inputs):
  result = outputs(arm,normalize,*inputs)
  result[0].realize(*result[1:])
  return result


def storage_info(tensor, actual=False):
  b = tensor.uop.buffer
  row = {'tensor_id':id(tensor), 'buffer_id':id(b), 'base_buffer_id':id(b.base),
    'shape':list(tensor.shape), 'dtype':str(tensor.dtype), 'bytes':b.nbytes, 'offset_bytes':b.offset,
    'base_bytes':b.base.nbytes, 'is_allocated':b.is_allocated(), 'uop_key':tensor.uop.key.hex()}
  if actual:
    row.update(gpu_address=int(b.get_storage().buf), base_gpu_address=int(b.base.get_storage().buf))
    assert row['gpu_address'] == row['base_gpu_address']+b.offset
  return row


def view_info(u):
  from tinygrad.runtime.support.hcq2 import unwrap_view
  base, offset = unwrap_view(u)
  return {'op':u.op.name, 'base_op':base.op.name, 'base_slot':getattr(base.arg,'slot',None),
          'offset_bytes':offset, 'bytes':u.max_numel()*u.dtype.itemsize, 'base_key':base.key.hex()}

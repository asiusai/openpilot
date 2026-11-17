"""Exact copy-call templates and compiled host submit-program identities."""
from dataclasses import asdict
import hashlib
import json

def sha(data): return hashlib.sha256(data).hexdigest()
def key(value): return sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())


def copy_call(call):
  from tinygrad.uop.ops import Ops
  from tinygrad.runtime.support.hcq2 import unwrap_view
  assert call.op is Ops.CALL and call.body.op is Ops.STORE and len(call.src)==3
  args=[]
  for u in call.src[1:]:
    root,offset=unwrap_view(u)
    assert root.op is Ops.PARAM and root.arg.name=='lt_input'
    args.append({'slot':root.arg.slot,'base_bytes':root.nbytes(),'offset_bytes':offset,'bytes':u.nbytes(),
      'dtype':str(u.dtype),'device':u.device})
  return {'body_ast_key':call.body.key.hex(),'arguments':args,'outs':[0],'ins':[1],'dependencies':[],
    'device':'AMD','queue':'COPY:0','target_context':'USB+AMD:LLVM:gfx1200','byte_count':4}


def host_program(program):
  return {'source_sha256':sha(program.src[2].arg.encode()),'binary_sha256':sha(program.src[3].arg),
    'ast_key':program.src[0].key.hex(),'linear_key':program.src[1].key.hex(),'name':program.src[0].arg.name,
    'target':asdict(program.arg.target),'global':list(program.arg.global_size),'local':list(program.arg.local_size),
    'outs':list(program.arg.outs),'ins':list(program.arg.ins),
    'params':[{'slot':p.arg.slot,'name':p.arg.name,'dtype':str(p.dtype),'addrspace':str(p.addrspace),'shape':list(p.shape)}
      for p in program.kernel_params]}

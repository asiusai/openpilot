"""Exact program associations. Binary equality alone never authorizes a source."""
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace


def sha(data): return hashlib.sha256(data).hexdigest()
def key(value): return sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())


def identity(program):
  from tinygrad.runtime import ops_amd
  descriptor,_=ops_amd._amd_program_image(SimpleNamespace(arch='gfx1200',target=(12,0,0),
    iface=SimpleNamespace(props={'lds_size_in_kb':64})),program.src[3].arg)
  assert ((descriptor.rsrc1>>29)&1)==0 and not descriptor.enable_dispatch_ptr
  target=program.arg.target
  assert str(target)=='USB+AMD:LLVM:gfx1200', str(target)
  return {'binary_sha256':sha(program.src[3].arg),'source_sha256':sha(program.src[2].arg.encode()),
    'ast_key':program.src[0].key.hex(),'linear_key':program.src[1].key.hex(),
    'name':program.src[0].arg.name,'target':asdict(target),'descriptor':asdict(descriptor),'actual_mode':'CU',
    'global':list(program.arg.global_size),'local':list(program.arg.local_size),
    'outs':list(program.arg.outs),'ins':list(program.arg.ins),
    'params':[{'slot':p.arg.slot,'name':p.arg.name,'dtype':str(p.dtype),'addrspace':str(p.addrspace),
      'shape':list(p.shape)} for p in program.kernel_params]}


def authorize(program,plan):
  item=identity(program);association=key(item)
  assert association in plan['associations'], ('Unprepared exact program association',association,item)
  assert item==plan['associations'][association]
  assert association in plan['programs'][item['binary_sha256']]['associations']
  return association,item


def batch_rows(batch):
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import DepsTracker,unwrap_view
  tracker=DepsTracker([(d,q) for _,d,q in batch]);roots={};rows=[]
  for tag,(call,device,queue) in enumerate(batch):
    outs,ins=get_call_outs_ins(call);arguments=[]
    for u in call.src[1:]:
      assert not u.is_variable
      base,offset=unwrap_view(u)
      if base not in roots: roots[base]=len(roots)
      arguments.append({'root':roots[base],'offset_bytes':offset,'bytes':u.max_numel()*u.dtype.itemsize,'dtype':str(u.dtype)})
    rows.append({'tag':tag,'association':key(identity(call.body)),'device':device,'queue':queue,
      'outs':list(outs),'ins':list(ins),'dependencies':tracker.access_resources([(u,i in outs) for i,u in enumerate(call.src[1:])],tag),
      'arguments':arguments})
  return json.loads(json.dumps(rows))

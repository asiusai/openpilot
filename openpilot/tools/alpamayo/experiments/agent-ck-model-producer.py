"""Exact source associations for the existing Q/RoPE/cache-prefix producer schedule."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

def key(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def identity(p):
  from tinygrad.runtime import ops_amd
  descriptor,_=ops_amd._amd_program_image(SimpleNamespace(arch='gfx1200',target=(12,0,0),
    iface=SimpleNamespace(props={'lds_size_in_kb':64})),p.src[3].arg)
  assert str(p.arg.target)=='USB+AMD:LLVM:gfx1200' and not descriptor.enable_dispatch_ptr
  return {'binary_sha256':hashlib.sha256(p.src[3].arg).hexdigest(),'source_sha256':hashlib.sha256(p.src[2].arg.encode()).hexdigest(),
    'ast_key':p.src[0].key.hex(),'linear_key':p.src[1].key.hex(),'name':p.src[0].arg.name,'target':asdict(p.arg.target),
    'descriptor':asdict(descriptor),'actual_mode':'WGP' if (descriptor.rsrc1>>29)&1 else 'CU',
    'global':list(p.arg.global_size),'local':list(p.arg.local_size),'outs':list(p.arg.outs),'ins':list(p.arg.ins),
    'params':[{'slot':u.arg.slot,'name':u.arg.name,'shape':list(u.shape),'dtype':str(u.dtype),'addrspace':str(u.addrspace)} for u in p.kernel_params]}

def rows(batch):
  from tinygrad.runtime.support.hcq2 import DepsTracker,unwrap_view
  from tinygrad.engine.realize import get_call_outs_ins
  tracker=DepsTracker([(d,q) for _,d,q in batch]);roots={};result=[]
  for tag,(call,dev,queue) in enumerate(batch):
    outs,ins=get_call_outs_ins(call);args=[]
    for u in call.src[1:]:
      assert not u.is_variable
      base,offset=unwrap_view(u)
      if base not in roots: roots[base]=len(roots)
      args.append({'root':roots[base],'offset_bytes':offset,'bytes':u.max_numel()*u.dtype.itemsize,'dtype':str(u.dtype)})
    result.append({'tag':tag,'program_ast':call.body.src[0].key.hex(),'device':dev,'queue':queue,'arguments':args,
      'outs':list(outs),'ins':list(ins),'dependencies':tracker.access_resources([(u,i in outs) for i,u in enumerate(call.src[1:])],tag)})
  return json.loads(json.dumps(result))

@contextmanager
def observe(catalog,report,folder):
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  old_build,old_queues=ops_amd.amd_build_program,hcq2._build_queues
  def build(dev,p,devices):
    item=identity(p);aid=key(item)
    assert aid in catalog['associations'] and item==catalog['associations'][aid], ('Unknown live producer',aid,item)
    report.setdefault('producer_programs',{})[aid]=item
    for suffix,payload in [('source',p.src[2].arg.encode()),('hsaco',p.src[3].arg),('ast',str(p.src[0]).encode())]:
      path=folder/(aid+'.'+suffix)
      if not path.exists(): path.write_bytes(payload)
      else: assert path.read_bytes()==payload
    return old_build(dev,p,devices)
  def queues(ctx):
    actual=rows(ctx.batch)
    assert actual==catalog['batch'], ('Unknown live producer dependencies',actual)
    report.setdefault('producer_batches',[]).append(actual)
    return old_queues(ctx)
  ops_amd.amd_build_program,hcq2._build_queues=build,queues
  try: yield
  finally:
    assert ops_amd.amd_build_program is build and hcq2._build_queues is queues
    ops_amd.amd_build_program,hcq2._build_queues=old_build,old_queues

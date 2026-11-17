"""CPU compiler/catalog and host-memory fixture-bank simulation. No device runtime."""
import argparse
import ast
from contextlib import contextmanager,nullcontext
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-source-work')
OLD=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-qualified-work/prepared-final/protocol.json')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def extracted_run_nodes():
  tree=ast.parse((EXP/'agent-ck-qualified-run.py').read_text())
  run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  functions=[n for n in ast.walk(run) if isinstance(n,ast.FunctionDef) and n.name in ('call_helper','utility')]
  bank=next(n for n in ast.walk(run) if isinstance(n,ast.With) and isinstance(n.items[0].context_expr,ast.Call)
    and isinstance(n.items[0].context_expr.func,ast.Name) and n.items[0].context_expr.func.id=='phase'
    and n.items[0].context_expr.args[0].value=='resident_fixture_bank')
  return functions,bank


def main(target,catalog_path=None):
  target.mkdir(parents=True,exist_ok=False)
  os.environ.update(DEV='USB+AMD:LLVM',ALLOW_DEVICE_USAGE='0',PARALLEL='0',PROFILE='0',BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',
    LLVMOPT='1',GMMU='0',TC_OPT='2',DEBUG='0',TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'))
  sys.path.insert(0,str(TINY))
  import numpy as np
  from tinygrad import Tensor,TinyJit,dtypes
  from tinygrad.device import Buffer,Device,HostAllocator
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.engine import realize
  from tinygrad.runtime.support import hcq2
  tensor_module=importlib.import_module('tinygrad.tensor');jit_module=importlib.import_module('tinygrad.engine.jit')
  identity=load('agent-ck-source-identity');common=load('agent-ck-qualified-common')
  helper=common.load('agent-ck-output-common');adapter=helper.load('agent-ck-attention-v2-adapter')
  plan=json.loads(OLD.read_text())
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  class HostOnlyDevice:
    device='AMD';arch='gfx1200';host='CPU';prof_enabled=False;is_usb=True;target=(12,0,0)
    iface=SimpleNamespace(props={'lds_size_in_kb':64})
    def __init__(self): self.allocator=HostAllocator(self)
    def _select_renderer(self,*_): return renderer
    def synchronize(self,*_): pass
  dev=HostOnlyDevice()
  def host_getitem(self,name):
    assert name=='AMD', ('CPU preflight rejects device access',name)
    return dev
  assert not Device._opened_devices
  Device.__class__.__getitem__=host_getitem
  # Native hardware discovery is additionally forbidden, even if a fake-device hook regresses.
  def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      name=os.fsdecode(args[0]);assert not name.startswith(('/dev/bus/usb','/dev/dri','/dev/kfd')),name
  sys.addaudithook(audit)
  programs={};associations={};batches={};events=[];current={'name':'setup','emulate_copy':False}
  frozen=json.loads(catalog_path.read_text()) if catalog_path else None
  guard=load('agent-ck-source-guard')
  guard_report={'hcq_batches':[],'native_argument_records':[],'active_phase':'CPU_only'}
  (target/'guard-artifacts').mkdir()
  recording=guard.record_programs(guard_report,target/'guard-artifacts');recording.__enter__()
  def collect(linear):
    linear=realize.lower_and_compile(linear)
    if not linear.src: return linear
    batch=[]
    for call in linear.src:
      call=call.without_after;assert call.body.op is Ops.PROGRAM
      item=identity.identity(call.body);aid=identity.key(item);digest=item['binary_sha256']
      if aid not in associations:
        associations[aid]=item
        (target/(aid+'.source')).write_text(call.body.src[2].arg)
        (target/(aid+'.ast')).write_text(str(call.body.src[0])+'\n')
        (target/(aid+'.linear')).write_text(str(call.body.src[1])+'\n')
      if digest not in programs:
        (target/(digest+'.hsaco')).write_bytes(call.body.src[3].arg)
        programs[digest]={'associations':[]}
      if aid not in programs[digest]['associations']: programs[digest]['associations'].append(aid)
      batch.append((call,'AMD','COMPUTE:0'))
    rows=identity.batch_rows(batch);bid=identity.key(rows)
    batches[bid]=rows
    events.append({'case':current['name'],'batch':bid,'associations':[r['association'] for r in rows]})
    if frozen is not None:
      from tinygrad.runtime import ops_amd
      guard_report['active_phase']='CPU_only_'+current['name']
      old_build=ops_amd.amd_build_program
      def metadata_only(dev,prg,devices):
        return ops_amd._amd_program_image(dev,prg.src[3].arg)
      ops_amd.amd_build_program=metadata_only
      try:
        with (nullcontext() if adapter.ACTIVE else adapter.abi_scope()),guard.bindings(helper,frozen,guard_report):
          # Real HCQ dependency/barrier command construction; no command submission or device code allocation.
          commands=hcq2._build_queues(hcq2.BatchCtx(batch,False))
          assert set(commands)=={('AMD','COMPUTE:0')}
          queue=SimpleNamespace(dev=dev,devs=('AMD',))
          for call,_,_ in batch:
            descriptor,_=ops_amd.amd_build_program(dev,call.body,queue.devs)
            words=ops_amd.AMDComputeQueue.kernargs(queue,call,call.body,descriptor)
            assert words is not None
      finally: ops_amd.amd_build_program=old_build
    return linear
  def compile_only(linear,*args,**kwargs): return collect(linear)
  def link_only(linear,*args,**kwargs):
    replacements={u:UOp.new_buffer(u.device,u.max_numel(),u.dtype) for u in linear.toposort()
      if u.op is Ops.ALLOC and u.arg.name is None}
    assert not any(u.op is Ops.ALLOC and u.arg.name is not None for u in linear.toposort())
    return linear.substitute(replacements)
  def run_only(linear,*args,input_uops=(),**kwargs):
    linear=link_only(collect(linear))
    for call in linear.src:
      call=call.without_after
      buffers=[realize._resolve(u,tuple(input_uops)).buffer.ensure_allocated() for u in call.src[1:]]
      if current['emulate_copy']:
        item=identity.identity(call.body)
        assert item['outs']==[0] and item['ins']==[1] and len(buffers)==2
        # Bank scope contains only the reviewed plain uint16 copy and its contiguous return copy.
        assert len({b.nbytes for b in buffers})==1 and buffers[0].nbytes in (4194304,8388608)
        source=call.body.src[2].arg
        assert ('@llvm.exp' not in source and 'fmul' not in source and 'fadd' not in source)
        assert item['binary_sha256'] in {
          '1f33bf70ad896fc1cdc8b87ce2d894278c70a62d6eef85152ed9830c808aac34',
          'f20c5cb2dba4794debe9e4ebf61c272cae4d92685b2b028bafae8f7cbc0902d2',
          '2b3a9310397e930a052d5e9fd2ead57888c58cf984ec8e26586a422c49d5f0cc',
          '643a061c5519f47bc87b5d7c7092152cfd45d6384cb3ba1124aeac268c1c8841'}
        buffers[0].host.view(fmt='B')[:]=buffers[1].host.view(fmt='B')[:]
  tensor_module.run_linear=run_only
  jit_module.compile_linear=compile_only;jit_module.link_linear=link_only;jit_module.run_linear=run_only
  def tensor(shape,dt=dtypes.bfloat16):
    return Tensor(UOp.from_buffer(Buffer('AMD',math.prod(shape)*dt.itemsize).ensure_allocated(),dt)).reshape(shape)
  started=time.monotonic()
  # Extract and execute the original complete bank and original helper cache/TinyJit control flow.
  _,oracle,_,_=load('agent-ck-qualified-evidence').verify()
  cases={f['name']:{k:np.load(Path(plan['oracle'])/v['path'],allow_pickle=False) for k,v in f['files'].items()} for f in oracle['fixtures']}
  report={'io':{'host_write_bytes':0,'host_write_calls':0},'phases':[]}
  @contextmanager
  def phase(name):
    row={'name':name};report['phases'].append(row);yield row
  def write(t,data,offset=0):
    t.uop.buffer.host.view(offset=offset,size=len(data),fmt='B')[:]=data
    report['io']['host_write_bytes']+=len(data);report['io']['host_write_calls']+=1
  ns=dict(TinyJit=TinyJit,helpers={},helper_keys={},common=common,allocate=tensor,plan=plan,cases=cases,bank={},
    report=report,phase=phase,device=dev,write=write)
  functions,bank_node=extracted_run_nodes()
  code=ast.fix_missing_locations(ast.Module(body=functions+[bank_node],type_ignores=[]))
  current.update(name='actual_full_fixture_bank',emulate_copy=True)
  with adapter.abi_scope(): exec(compile(code,str(EXP/'agent-ck-qualified-run.py'),'exec'),ns)
  report['bank_hashes']={digest:hashlib.sha256(bytes(t.uop.buffer.host.view(fmt='B')[:])).hexdigest() for digest,t in ns['bank'].items()}
  assert all(a==b for a,b in report['bank_hashes'].items())
  report['helper_counts']=[{'signature':str(k),'count':v.cnt,'captured':v.captured is not None} for k,v in ns['helpers'].items()]
  current['emulate_copy']=False
  # Reuse the frozen preparation's exact shapes, functions, and case loop, then exercise normal TinyJit first/capture/replay.
  def compile_case(name,inputs,result):
    current['name']=name
    raw,variables=result[0].linear_with_vars(*result[1:]);assert not variables;collect(raw)
    if name.startswith('attention_'):
      _,boundary,arm=name.split('_');factory=lambda *xs:common.execute(boundary,arm,*xs)
    elif name.startswith('utility_'):
      mode=name.split('_')[1]
      if mode in ('q','kv'): mode+='_B'
      factory=lambda *xs:common.utility(*xs,mode=mode).realize()
    elif name.startswith('validator_'):
      boundary=name.split('_')[1];factory=lambda *xs:common.validator(*xs,boundary=boundary).realize()
    else: assert name=='tail_validator';factory=lambda *xs:common.tail_validator(*xs).realize()
    runner=TinyJit(factory)
    with adapter.abi_scope():
      for _ in range(3): runner(*inputs)
    assert runner.cnt==3 and runner.captured is not None and len(runner.captured._linears)==1
    assert all(not staging for _,staging in runner.captured._linears.values())
    print(json.dumps({'case':name,'associations':len(associations),'binaries':len(programs)}),flush=True)
  tree=ast.parse((EXP/'agent-ck-qualified-prepare.py').read_text())
  main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  nodes=[n for n in main_node.body if 92<=n.lineno<=113]
  ns2=dict(common=common,adapter=adapter,dtypes=dtypes,tensor=tensor,compile_case=compile_case,len=len)
  exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(EXP/'agent-ck-qualified-prepare.py'),'exec'),ns2)
  assert not Device._opened_devices
  assert set(programs)==set(plan['programs']), {'unexpected':list(set(programs)-set(plan['programs'])),'missing':list(set(plan['programs'])-set(programs))}
  for digest in programs: assert (target/(digest+'.hsaco')).read_bytes()==(OLD.parent/(digest+'.hsaco')).read_bytes()
  recording.__exit__(None,None,None)
  if frozen is not None:
    assert associations==frozen['associations'] and programs==frozen['programs'] and batches==frozen['batches']
    (target/'guard-preflight.json').write_text(json.dumps(guard_report,indent=2)+'\n')
  report.update(gpu_executed=False,opened_devices=[],classification='Host-memory copy simulation and real CPU compilation; no arithmetic/GPU evidence.',
    seconds=time.monotonic()-started,events=events,association_count=len(associations),binary_count=len(programs),
    actual_target=str(renderer.target),old_protocol_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),
    guard_catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest() if catalog_path else None,
    real_HCQ_command_and_kernarg_construction_checked=frozen is not None)
  (target/'catalog.json').write_text(json.dumps({'programs':programs,'associations':associations,'batches':batches},indent=2)+'\n')
  (target/'cpu-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'completed':True,'cpu_only':True,'target':str(target),'seconds':report['seconds'],'associations':len(associations)}))


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--catalog',type=Path);args=parser.parse_args()
  assert args.out.parent.resolve()==ROOT.resolve();main(args.out,args.catalog)

"""CPU-only prepare. All allocations are metadata or anonymous host memory."""
import argparse
from contextlib import contextmanager
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import runpy
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import patch

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_model_common',EXP/'agent-ck-model-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def main(target):
  target.mkdir(parents=True,exist_ok=False)
  (C.ROOT/'tmp').mkdir(exist_ok=True)
  os.environ.update(DEV='USB+AMD:LLVM',ALLOW_DEVICE_USAGE='0',PARALLEL='1',PROFILE='0',GMMU='0',BEAM='0',JITBEAM='0',
    TC_OPT='2',LLVMOPT='1',CACHEDB=str(C.ROOT/'cache.db'),TMPDIR=str(C.ROOT/'tmp'),PYTHONDONTWRITEBYTECODE='1',
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      assert not os.fsdecode(args[0]).startswith(('/dev/bus/usb','/dev/dri','/dev/kfd'))
  sys.addaudithook(audit)
  sys.path[:0]=[str(C.TINY),str(C.BASE),str(C.BASE/'experiments')]
  from tinygrad import Tensor,dtypes,Device
  from tinygrad.device import Buffer,HostAllocator
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.engine import realize
  from tinygrad.codegen import do_to_program
  from tinygrad.runtime.support import hcq2
  from tinygrad.runtime import ops_amd
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  class MetadataDevice:
    device='AMD';arch='gfx1200';target=(12,0,0);is_usb=True;prof_enabled=False;host='CPU'
    iface=SimpleNamespace(props={'lds_size_in_kb':64})
    def __init__(self): self.renderer=renderer;self.allocator=HostAllocator(self)
    @property
    def compiler(self): return self.renderer.compiler
    def _select_renderer(self,*args): return self.renderer
    def synchronize(self): pass
  device=MetadataDevice()
  python_device=SimpleNamespace(device='PYTHON',synchronize=lambda:None)
  python_device.allocator=HostAllocator(python_device)
  def getitem(self,name):
    assert name in ('AMD','PYTHON'),name
    return device if name=='AMD' else python_device
  assert not Device._opened_devices
  Device.__class__.__getitem__=getitem
  C.load('wgp_compiler',C.BASE/'experiments').install(device)
  worker_renderer=pickle.loads(pickle.dumps(renderer))
  assert type(renderer.compiler).__name__=='WGPCompiler' and type(worker_renderer.compiler).__name__=='AMDLLVMCompiler'
  proof={'gpu_executed':False,'opened_devices':[],'compiler_context':{
    'parent_requested_compiler':type(renderer.compiler).__name__,'worker_deserialized_compiler':type(worker_renderer.compiler).__name__,
    'target':str(renderer.target),'parallel':1,'compiler_fix_applied':False},'checks':[]}
  def tensor(shape,dt=dtypes.bfloat16): return Tensor(UOp.from_buffer(Buffer('AMD',math.prod(shape)*dt.itemsize),dt)).reshape(shape)
  identity=C.load('agent-ck-source-identity');helper=C.load('agent-ck-output-common')
  hook=C.load('agent-ck-model-hook');adapter=helper.load('agent-ck-attention-v2-adapter')
  selected=C.load('agent-attention-rowsoftmax',C.BASE/'experiments')
  catalog={'programs':{},'associations':{},'batches':{}}
  def compile_results(name,result):
    raw,variables=result[0].linear_with_vars(*result[1:]);assert not variables
    compiled=raw.replace(src=tuple(c.replace(src=(do_to_program(c.body,worker_renderer),*c.src[1:])) for c in raw.src))
    batch=[]
    for call in compiled.src:
      call=call.without_after;item=identity.identity(call.body);aid=identity.key(item);digest=item['binary_sha256']
      catalog['associations'][aid]=item
      if digest not in catalog['programs']: catalog['programs'][digest]={'associations':[]}
      if aid not in catalog['programs'][digest]['associations']: catalog['programs'][digest]['associations'].append(aid)
      for suffix,payload in [('source',call.body.src[2].arg.encode()),('hsaco',call.body.src[3].arg),
                             ('ast',str(call.body.src[0]).encode()),('linear',str(call.body.src[1]).encode())]:
        p=target/(aid+'.'+suffix)
        if not p.exists(): p.write_bytes(payload)
        else: assert p.read_bytes()==payload
      batch.append((call,'AMD','COMPUTE:0'))
    rows=identity.batch_rows(batch);catalog['batches'][identity.key(rows)]=rows
    proof['checks'].append({'case':name,'batch':identity.key(rows),'calls':rows})
    return compiled
  q,k,v=[tensor(shape) for shape in (helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE)]
  compile_results('actual_selected_original',(selected.attention(q,k,v,True,waves=8,transpose='shuffle-fma'),))
  for arm in ('baseline','candidate'):
    q,k,v=[tensor(shape) for shape in (helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE)]
    stores=[tensor((helper.OC+512,)),tensor((helper.LC+512,),dtypes.float32)]
    with adapter.abi_scope(): compile_results(arm,helper.outputs(arm,True,q,k,v,*stores))
  # Every introduced arithmetic binary stays equal to the completed micro protocol.
  previous=json.loads(C.OLD.read_text())
  for item in catalog['associations'].values():
    if item['name']=='alpamayo_ck_d128': assert item['binary_sha256']==adapter.ELF_SHA
    if item['name']=='agent_attention_rowsoftmax': assert item['binary_sha256']=='25eefe9f887548d55d03387d0dfba6b4f7ae7c4691d4527d4997975bcf4c4440'
    assert item['binary_sha256'] in previous['programs']
  # Real HCQ argument generation, barrier/dependency graph, and descriptor parsing; no link or submit.
  guard=C.load('agent-ck-model-guard');guard_report={'phase':'CPU_metadata'}
  old_build=ops_amd.amd_build_program
  ops_amd.amd_build_program=lambda dev,p,devs:ops_amd._amd_program_image(dev,p.src[3].arg)
  try:
    with adapter.abi_scope(),guard.guard(C,catalog,guard_report):
      q,k,v=[tensor(shape) for shape in (helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE)]
      stores=[tensor((helper.OC+512,)),tensor((helper.LC+512,),dtypes.float32)]
      outputs=helper.outputs('candidate',True,q,k,v,*stores)
      raw,_=outputs[0].linear_with_vars(*outputs[1:]);compiled=realize.lower_and_compile(raw)
      batch=[(x.without_after,'AMD','COMPUTE:0') for x in compiled.src]
      hcq2._build_queues(hcq2.BatchCtx(batch,False))
      queue=SimpleNamespace(dev=device,devs=('AMD',))
      for call,_,_ in batch:
        data,_=ops_amd.amd_build_program(device,call.body,queue.devs)
        assert ops_amd.AMDComputeQueue.kernargs(queue,call,call.body,data) is not None
  finally: ops_amd.amd_build_program=old_build
  proof['actual_HCQ_cpu_guard']=guard_report
  C.write_json(target/'catalog.json',catalog)
  # Direct production FP8 loader/cache admission, with the first conversion/upload blocked.
  import model
  saved_init=model.EdgeModel.__init__
  fast=runpy.run_path(str(C.BASE/'experiments/benchmark-fast.py'),run_name='ck_cpu_loader')
  class ConversionBoundary(Exception): pass
  def safe_load(path):
    index=json.loads((C.CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map']
    assert Path(path)==C.CHECKPOINT/index['vlm.model.language_model.layers.0.mlp.fc1.weight']
    raise ConversionBoundary()
  before_env=os.environ.get('ALPAMAYO_FP8_CACHE');os.environ['ALPAMAYO_FP8_CACHE']=str(C.CACHE)
  try:
    loader=fast['load'];loader.__globals__['safe_load']=safe_load
    loader.__globals__['upload']=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('CPU upload forbidden'))
    try: loader(object(),str(C.CHECKPOINT),precision='fp8')
    except ConversionBoundary: proof['actual_loader_cache_admission']=True
    else: raise AssertionError('Conversion boundary not reached')
  finally:
    model.EdgeModel.__init__=saved_init
    if before_env is None: os.environ.pop('ALPAMAYO_FP8_CACHE',None)
    else: os.environ['ALPAMAYO_FP8_CACHE']=before_env
  cachecheck=C.load('agent-expert-combined-cache-common')
  proof['fp8_cache']=cachecheck.cache_check()
  # Trace actual text_layer, packed quantization, FP8 GEMM, fused norm and RoPE.
  # Execution is replaced below; no tensor values or arithmetic outcomes are claimed.
  tm=importlib.import_module('tinygrad.tensor');old_run=tm.run_linear
  schedules=[]
  def no_execute(linear,*args,**kwargs):
    schedules.append({'key':linear.key.hex(),'calls':len(linear.src)})
  tm.run_linear=no_execute
  try:
    m=object.__new__(model.EdgeModel)
    m.config=json.loads((C.CHECKPOINT/'config.json').read_text());m.dtype=dtypes.bfloat16;m.fp8_wmma=True;m.cuda_kernels=False
    m.weights={};m.weight_scales={}
    from amd_gemm import unpack
    for rec in json.loads((C.CACHE/'index.json').read_text())['records']:
      if not rec['name'].startswith('vlm.model.language_model.layers.0.'): continue
      dt={'scale':dtypes.float32,'fp8':dtypes.fp8e4m3,'bf16':dtypes.bfloat16}[rec['kind']]
      value=tensor(rec['shape'],dt)
      if rec['packed']: value=unpack(value)
      (m.weight_scales if rec['kind']=='scale' else m.weights)[rec['name']]=value
    m.cache=[tensor((2,1,8,2304,128)) for _ in range(28)]
    C.load('fused-norm',C.BASE/'experiments').install(m)
    calibration=json.loads((C.WORK/'prefill-calibration/report.prefill-scales.json').read_text())['scales']
    C.load('quantize-packed',C.BASE/'experiments').install(m,fixed_scales={k:v for k,v in calibration.items() if k.startswith('vlm.model.language_model.layers.0/')})
    selected.install(m,'prefill','shuffle-fma',8)
    class StopTrace(Exception): pass
    captured={}
    def stop(handle,context,q,k,v,original):
      run=C.load('agent-ck-model-run')
      captured.update(layer=context['layer'],q=run.structural_graph(q),k=run.structural_graph(k),v=run.structural_graph(v))
      for n,t in zip(('q','k','v'),(q,k,v)): (target/(n+'-production-trace.txt')).write_text(str(t.uop)+'\n')
      assert context['cache'] is m.cache[0]
      assert any(u.op is Ops.STORE for u in k.uop.toposort(enter_calls=True))
      assert q.shape==helper.QSHAPE and k.shape==v.shape==helper.KVSHAPE
      producer=C.load('agent-ck-model-producer');inventory={'associations':{}}
      packed=[t.contiguous() for t in (q,k,v)]
      raw,variables=packed[0].linear_with_vars(*packed[1:]);assert not variables
      batches=[]
      for mode,ren in [('parent_WGP',renderer),('worker_CU',worker_renderer)]:
        batch=[]
        for call in raw.src:
          call=call.without_after;p=do_to_program(call.body,ren);item=producer.identity(p);aid=producer.key(item)
          inventory['associations'][aid]=item
          for suffix,payload in [('source',p.src[2].arg.encode()),('hsaco',p.src[3].arg),('ast',str(p.src[0]).encode())]:
            path=target/(aid+'.'+suffix)
            if not path.exists(): path.write_bytes(payload)
          batch.append((call.replace(src=(p,*call.src[1:])),'AMD','COMPUTE:0'))
        batches.append(producer.rows(batch))
      assert batches[0]==batches[1]
      inventory['batch']=batches[0]
      C.write_json(target/'producer-catalog.json',inventory)
      raise StopTrace()
    saved_attention,saved_layer=m.attention,m.text_layer
    h=hook.Hook(m,stop)
    try:
      try: m.text_layer(tensor((1,2048,2048)),tensor((1,1,2048,128)),tensor((1,1,2048,128)),
        'vlm.model.language_model.layers.0',m.cache[0],None,cache_pos=0)
      except StopTrace: pass
      else: raise AssertionError('Actual production text layer missed hook')
      assert h.calls==[0] and h.context is None
    finally: h.close()
    assert m.attention is saved_attention and m.text_layer is saved_layer
    proof['production_trace']=captured;proof['production_trace_schedules']=schedules
  finally: tm.run_linear=old_run
  proof['gpu_executed']=False;assert not Device._opened_devices
  C.write_json(target/'cpu-proof.json',proof)
  print(json.dumps({'cpu_prepared':str(target),'associations':len(catalog['associations']),'batches':len(catalog['batches']),
    'gpu_executed':False,'production_trace':bool(proof.get('production_trace'))}))

if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();main(args.output)

"""CPU-only execution of the full screen runner and real TinyJit lifecycle.

Model weights/vision/decoder work and native GPU arithmetic are mocked. Ordinary tensor operations run on CPU.
This tests interfaces, ownership, capture, controls, and cleanup; it cannot qualify CK numerical accuracy.
"""
import argparse
import ast
from contextlib import ExitStack
import copy
import ctypes
from dataclasses import replace
import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import MethodType,SimpleNamespace
from unittest.mock import patch

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_integrated_run',EXP/'agent-ck-integrated-run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);C=R.C

def main(output):
  root=output.parent/output.stem;root.mkdir(exist_ok=False)
  R.configure('candidate')
  os.environ.update(ALLOW_DEVICE_USAGE='0',CACHEDB=str(C.ROOT/'cpu-cache.db'),PARALLEL='0',DEV='USB+AMD:LLVM',PYTHONDONTWRITEBYTECODE='1')
  from tinygrad import Tensor,Device,dtypes,TinyJit
  from tinygrad.device import Buffer,BufferSpec,HostAllocator
  from tinygrad.uop.ops import UOp,Ops,ProgramInfo
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer,CPULLVMRenderer
  from tinygrad.engine import realize,jit
  from tinygrad.runtime.support import hcq2
  from tinygrad.codegen import do_to_program
  from tinygrad.runtime.ops_cpu import CPUDevice,CPUProgram
  from tinygrad.runtime import ops_amd
  import numpy as np
  def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      assert not os.fsdecode(args[0]).startswith(('/dev/bus/usb','/dev/dri','/dev/kfd'))
  sys.addaudithook(audit);assert not Device._opened_devices
  class PreOpen(Exception): pass
  def blocked(self,name): raise PreOpen(name)
  # Actual production runpy and FP8 loader imports, stopped at the first GPU lookup.
  old_argv=sys.argv;old_spec=importlib.util.spec_from_file_location;old_env=dict(os.environ)
  preopen=root/'preopen';preopen.mkdir()
  with patch.object(Device.__class__,'__getitem__',blocked):
    sys.argv=[str(C.BASE/'experiments/agent-speculation-run.py'),'--invocation',str(C.PREPARED/'candidate-invocation.json'),
      '--output',str(preopen/'unused.json'),'--mode','candidate','--prefill-attention','shuffle-fma','--pad-prompt-to','2048',
      '--warmup','0','--runs','8','--samples-json',str(C.PREPARED/'samples.json')]
    try: runpy.run_path(sys.argv[0],run_name='__main__')
    except PreOpen: pass
    else: raise AssertionError('Production pre-open boundary was not exercised')
    finally:
      sys.argv=old_argv;importlib.util.spec_from_file_location=old_spec;os.environ.clear();os.environ.update(old_env)
  cache=Path(old_env['ALPAMAYO_FP8_CACHE'])
  index=json.loads((cache/'index.json').read_text())
  assert C.sha(cache/'index.json')=='c4366c5b7e92b4cd5e2d826881b99b9a80361ec171ebd4a50d51b0f718dffa3e'
  # Import the actual loader without invoking model initialization or weight upload.
  C.load('benchmark-fast',C.BASE/'experiments')
  cpu=CPUDevice('CPU');cpu_renderer=CPULLVMRenderer(Target('CPU','LLVM',cpu.arch))
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  dev=SimpleNamespace(device='AMD',host='CPU',arch='gfx1200',target=(12,0,0),is_usb=True,prof_enabled=False,
    renderer=renderer,iface=SimpleNamespace(props={'lds_size_in_kb':64}),synchronize=lambda:None)
  class SimAllocator(HostAllocator):
    def _alloc(self,size,options):
      storage=super()._alloc(size,options)
      return storage if options.cpu_access else replace(storage,host=None)
  dev.allocator=SimAllocator(dev);dev._select_renderer=lambda *args:renderer;dev.compiler=renderer.compiler
  dev.runtime=lambda obj,links:CPUProgram(cpu,obj,links)
  def getitem(self,name):
    assert name in ('AMD','CPU'),name
    return dev if name=='AMD' else cpu
  Device.__class__.__getitem__=getitem
  C.load('wgp_compiler',C.BASE/'experiments').install(dev)
  H=C.load('agent-ck-integrated-hook');io=C.load('agent-ck-attention-v3-run')
  def raw(buffer,dt):
    assert buffer.device=='AMD' and isinstance(buffer.allocator,SimAllocator)
    return np.frombuffer((ctypes.c_ubyte*buffer.nbytes).from_address(buffer.get_storage().buf),dt)
  io.read=lambda t,d:raw(t.uop.buffer,np.uint8).tobytes()
  def bf(bits): return (bits.astype(np.uint32)<<16).view(np.float32)
  def bf_bits(values):
    words=np.asarray(values,np.float32).view(np.uint32)
    return ((words+0x7fff+((words>>16)&1))>>16).astype(np.uint16)
  def mock_lower(linear,verbose=True):
    replacements={}
    for call in linear.toposort():
      if call.op is not Ops.CALL: continue
      ast=call.body
      if ast.op is Ops.PROGRAM and isinstance(ast.arg,ProgramInfo): continue
      if ast.op not in (Ops.SINK,Ops.PROGRAM): continue
      native=ast.op is Ops.PROGRAM or ast.arg.name.startswith('ckq_')
      replacements[call]=call.replace(src=(do_to_program(ast,renderer if native else cpu_renderer),*call.src[1:]))
    return linear.substitute(replacements)
  executed={'ck':0,'baseline':0,'cpu':0,'poison':0};native_programs=[];native_linears=[];fail_next_native=[False]
  allocations={}
  def mock_link(linear):
    for u in linear.toposort():
      if u.op is Ops.ALLOC and u not in allocations: allocations[u]=UOp.new_buffer(u.device,u.max_numel(),u.dtype)
    return linear.substitute(allocations,walk=True)
  def execute(linear,var_vals=None,input_uops=(),jit=False,**kwargs):
    if realize.capturing:
      for capturer in realize.capturing: capturer.add_linear(linear)
      return
    compiled=realize.lower_and_compile(linear)
    ctx=realize.ExecContext(var_vals=var_vals or {},input_uops=tuple(input_uops))
    for item in compiled.src:
      call=item.without_after;p=call.body
      if p.op is Ops.STORE:
        dest,src=[u.buffer.ensure_allocated() for u in realize.resolve_params(call,ctx.input_uops)]
        raw(dest,np.uint8)[:]=raw(src,np.uint8);continue
      assert p.op is Ops.PROGRAM,p.op
      name=p.src[0].arg.name
      if name.startswith('ckq_'):
        assert name.startswith('ckq_poison_');b=realize.resolve_params(call,ctx.input_uops)[0].buffer.ensure_allocated()
        dt=np.uint16 if '_u16_' in name else np.uint32;data=raw(b,dt)
        data[:]=0x7fc1 if dt==np.uint16 else 0x7fc12345
        data[:256]=data[-256:]=0x5aa5 if dt==np.uint16 else 0x5aa55aa5
        executed['poison']+=1
      elif name in ('alpamayo_ck_d128','agent_attention_rowsoftmax'):
        buffers=[u.buffer.ensure_allocated() for u in realize.resolve_params(call,ctx.input_uops)]
        if name=='alpamayo_ck_d128':
          if not native_linears: native_linears.append(linear)
          descriptor,_=ops_amd._amd_program_image(dev,p.src[3].arg)
          ops_amd.AMDComputeQueue.kernargs(SimpleNamespace(dev=dev,devs=('AMD',)),call,p,descriptor)
          native_programs.append((call,p,descriptor))
          if fail_next_native[0]:
            fail_next_native[0]=False;raise RuntimeError('Deliberate CPU native execution failure')
          out,lse,q,k,v=buffers
          # Deliberate CPU stand-in: row-local, grouped V plus Q. Not CK or an FP32 attention oracle.
          values=bf(raw(v,np.uint16)).reshape(1,8,2048,128).repeat(2,axis=1)
          values=values*.5+bf(raw(q,np.uint16)).reshape(1,16,2048,128)*.001
          raw(out,np.uint16)[:]=bf_bits(values).reshape(-1);raw(lse,np.float32)[:]=0
          executed['ck']+=1
        else:
          out,q,*_=buffers;raw(out,np.uint16)[:]=bf_bits(bf(raw(q,np.uint16))*.25)
          executed['baseline']+=1
      else:
        realize.exec_kernel(ctx,call,p,devices=('AMD',));executed['cpu']+=1
  # Mock only GPU scheduling/execution. Keep actual Tensor scheduling, TinyJit capture/parametrization, and input guards.
  tm=importlib.import_module('tinygrad.tensor');real_path=runpy.run_path
  production_spec=importlib.util.spec_from_file_location
  def mocked_spec(name,location,*args,**kwargs):
    s=production_spec(name,location,*args,**kwargs)
    if Path(location).resolve()!=C.BASE/'experiments/lookup-decode.py': return s
    loader=s.loader
    class Loader:
      def create_module(self,value): return loader.create_module(value)
      def exec_module(self,module):
        loader.exec_module(module)
        def init(self,model): self.model=model;self.history=[];self.statistics=[];self.counter=0
        def fallback(self,*args):
          tokens=[7,135084];self.history=[tokens];self.statistics.append({'mock_fallback':True});return tokens
        def read_tokens(self,value):
          payload=np.frombuffer(io.read(value,dev),np.int32).copy().reshape(-1)
          assert payload.shape==(12,) and np.all(payload[6:]==1)
          payload[:6]=[124549,11135,3174,4286,1561,135084] if self.counter%2 else [0]*6
          return payload
        def model_run(self,model,x,cos,sin):
          self.counter+=1;logits=model.padded_prefill(x,cos,sin,64,2000)
          generated=self.generate(model,SimpleNamespace(max_tokens=64),SimpleNamespace(traj_ids={'future_start':135084}),None,logits,None,None,2000)
          return {'xyz':np.zeros((1,64,3)).tolist(),'rot':np.zeros((1,64,3,3)).tolist(),
            'generated_token_ids':generated,'reasoning':'CPU_MOCK_ONLY'}
        module.LookupDecoder.__init__=init;module.LookupDecoder.generate=fallback
        module.LookupDecoder.read_tokens=read_tokens;module.LookupDecoder.run=model_run
    s.loader=Loader();return s
  models=[]
  def mocked_pipeline(path,run_name=None):
    if Path(path)!=C.BASE/'experiments/benchmark-pipeline.py': return real_path(path,run_name=run_name)
    import model as core
    class Model:
      text_layer=core.EdgeModel.text_layer
      rope=staticmethod(core.EdgeModel.rope)
      def __init__(self):
        self.config={'vlm_config':{'text_config':{'num_hidden_layers':28}}};self.dtype=dtypes.bfloat16
        self.weight_scales={};self.cache=None;self.cuda_kernels=False;self.fp8_wmma=True;self.minimum_cache_capacity=0
        self.prefill_jit=self.decode_jit=self.greedy_decode_jit=SimpleNamespace(reset=lambda:None);self.trajectory_jits={}
        self.output_token_mapping=Tensor.arange(128,dtype=dtypes.int32).realize()
      def norm(self,x,*args,**kwargs): return x
      def linear(self,x,prefix,*args,heads=None,**kwargs):
        if heads is not None: return x[:,:,:heads*128]
        if prefix=='vlm.lm_head': return x[:,:,:128]
        return x*.01
      def attention(self,*args,**kwargs): raise AssertionError('Unexpected original attention')
    m=Model();models.append(m)
    C.load('agent-attention-rowsoftmax',C.BASE/'experiments').install(m,'prefill','shuffle-fma',8)
    node=next(n for n in ast.walk(ast.parse((C.BASE/'experiments/quantize-packed.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='text_layer')
    ns={'state':{'prefill':False,'prefix':None,'index':0},'old_layer':m.text_layer}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(C.BASE/'experiments/quantize-packed.py'),'exec'),ns)
    m.text_layer=MethodType(ns['text_layer'],m)
    # Use the exact production install interception order, including nominal-six target verification and fallback.
    def local(name):
      s=importlib.util.spec_from_file_location(name,C.BASE/'experiments'/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    local('padded-prefill').install(m);decoder=local('lookup-decode').LookupDecoder(m)
    paths=json.loads((C.PREPARED/'samples.json').read_text());rows=[]
    x=io.allocate('AMD',(1,2048,2048),dtypes.bfloat16);cs=io.allocate('AMD',(1,1,2048,128),dtypes.bfloat16);sn=io.allocate('AMD',(1,1,2048,128),dtypes.bfloat16)
    raw(cs.uop.buffer,np.uint16)[:]=0x3f80;raw(sn.uop.buffer,np.uint16)[:]=0
    for index,path in enumerate(paths):
      raw(x.uop.buffer,np.uint16)[:]=bf_bits(np.float32(.2+index*.01))
      pred=decoder.run(m,x,cs,sn);rows.append({'sample':path,'prediction':pred,'pipeline_ms':0})
    assert decoder.prefill_draft_records and any(r['accepted'] for r in decoder.prefill_draft_records)
    assert any(not r['accepted'] for r in decoder.prefill_draft_records)
    destination=Path(sys.argv[sys.argv.index('--output')+1]);destination.write_text(json.dumps({'warmup':[],'runs':rows,'summary':{}}))
  result={'gpu_executed':False,'completed_mock_runner':False,'native_GPU_arithmetic_executed':False,
    'mock_limit':'Vision/weights/decoder and native attention arithmetic are substituted. Normal tensor utilities execute on CPU. This is not accuracy or GPU capture evidence.',
    'actual_production_runpy_preopen':True,'FP8_loader_import_and_index':True,'arms':[]}
  # The full runner expects a protocol digest. The draft lives only in this CPU output directory.
  prepared=C.PREPARED;draft=root/'prepared';draft.mkdir()
  for name in ('candidate-invocation.json','baseline-invocation.json','samples.json','ck-identity.json'):
    content=json.loads((prepared/name).read_text())
    if name.endswith('invocation.json'): content['environment']['CACHEDB']=str(C.ROOT/'cpu-cache.db')
    (draft/name).write_text(json.dumps(content)+'\n')
  (draft/'protocol.json').write_text('{"CPU_MOCK_ONLY":true}\n');C.PREPARED=draft
  real_lower=realize.lower_and_compile
  try:
    with ExitStack() as stack:
      for obj,name,value in ((realize,'lower_and_compile',mock_lower),(tm,'run_linear',execute),(jit,'run_linear',execute),
        (jit,'compile_linear',lambda linear,**kw:realize.lower_and_compile(linear)),(jit,'link_linear',mock_link),
        (runpy,'run_path',mocked_pipeline),(importlib.util,'spec_from_file_location',mocked_spec)):
        stack.enter_context(patch.object(obj,name,value))
      for arm in ('baseline','candidate'):
        folder=root/arm;folder.mkdir()
        try: R.run({},arm,folder)
        finally:
          if (folder/'report.json').exists():
            report=json.loads((folder/'report.json').read_text())
            report.update(gpu_executed=False,CPU_MOCK_ONLY=True,numerical_acceptance=False)
            (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        report=json.loads((folder/'report.json').read_text())
        assert report['completed'] and report['hooks_restored'] and report['model_renderer_unchanged']
        report.update(gpu_executed=False,CPU_MOCK_ONLY=True,numerical_acceptance=False)
        (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        result['arms'].append({'arm':arm,'completed_mock':True,'prefills':len(report['prefills']),
          'controls_mock_passed':report.get('model_controls',{}).get('passed',False),'hooks_restored':report['hooks_restored']})
      failure=root/'candidate-deliberate-failure';failure.mkdir();fail_next_native[0]=True
      try: R.run({},'candidate',failure)
      except RuntimeError as error: assert str(error)=='Deliberate CPU native execution failure'
      else: raise AssertionError('Injected native failure did not propagate')
      failed=json.loads((failure/'report.json').read_text());assert failed['hooks_restored'] and failed['model_renderer_unchanged'] and not failed['completed']
      failed.update(gpu_executed=False,CPU_MOCK_ONLY=True,numerical_acceptance=False)
      (failure/'report.json').write_text(json.dumps(failed,indent=2)+'\n')
      result['exception_cleanup']=True
      candidate=json.loads((root/'candidate/report.json').read_text())
      valid=candidate['live_layouts'][0]['actual_arguments'];sizes=[r['bytes'] for r in valid]
      rejected=[]
      for mode in ('negative_offset','pointer','extent','alignment','alias','missing_offset'):
        bad=copy.deepcopy(valid)
        if mode=='negative_offset': bad[0]['offset_bytes']=-1
        elif mode=='pointer': bad[0]['gpu_address']+=16
        elif mode=='extent': bad[0]['base_bytes']=1
        elif mode=='alignment': bad[0]['gpu_address']+=1;bad[0]['base_gpu_address']+=1
        elif mode=='alias': bad[1].update(gpu_address=bad[0]['gpu_address'],base_gpu_address=bad[0]['gpu_address']-bad[1]['offset_bytes'])
        else: bad[0].pop('offset_bytes')
        try: C.storage_gate(bad,sizes)
        except (AssertionError,KeyError): rejected.append(mode)
        else: raise AssertionError(('Storage negative accepted',mode))
      result['storage_negatives']=rejected
      call,program,data=native_programs[0]
      malformed=[('source_same_ELF',program.replace(src=(*program.src[:2],UOp(Ops.SOURCE,arg=program.src[2].arg+'\n'),program.src[3]))),
        ('missing_LSE_dependency',program.replace(arg=replace(program.arg,outs=(0,)))),
        ('wrong_USB_target',program.replace(arg=replace(program.arg,target=Target('AMD','LLVM','gfx1200'))))]
      with H.native_scope(C,{}):
        for name,p in malformed:
          try: ops_amd.AMDComputeQueue.kernargs(SimpleNamespace(dev=dev,devs=('AMD',)),call,p,data)
          except AssertionError: pass
          else: raise AssertionError(('Native mutation accepted',name))
      result['native_association_negatives']=[name for name,_ in malformed]
      patches=candidate['model_controls']['negative_controls']
      assert all(r['patch']['host_mapped']==(r['kind']!='returned_last_word_bit') for r in patches)
      result['actual_patch_API_host_and_device_only_simulation']=True
      model=models[-1];q=Tensor(UOp.from_buffer(Buffer('AMD',H.OC*2),dtypes.bfloat16)).reshape(1,16,2048,128)
      kv=Tensor(UOp.from_buffer(Buffer('AMD',H.OC),dtypes.bfloat16)).reshape(1,8,2048,128)
      eligible=C.load('agent-ck-model-hook').eligible;context={'cache_pos':0,'cache':model.cache[0]}
      assert eligible(context,q,kv,kv,None,True)
      for args in ((None,q,kv,kv,None,True),(context,q[:,:,:1],kv,kv,None,True),
        (context,q,kv,kv,object(),True),(context,q,kv,kv,None,False)):
        assert not eligible(*args)
      result['wrong_scope_shape_mask_causal_fallback']=True
      try: C.load('agent-ck-output-common').load('agent-ck-attention-v2-adapter').attention(q,kv,kv)
      except AssertionError as e: assert 'retain abi_scope' in str(e)
      else: raise AssertionError('Inactive adapter accepted')
    assert all(not hasattr(m,'_agent_ck_model_hook') for m in models)
    assert not C.load('agent-ck-output-common').load('agent-ck-attention-v2-adapter').ACTIVE
    assert not Device._opened_devices
    # Compile the real first native boundary and audit graphs for USB gfx1200 using the existing model renderer.
    # Build dependency queues only. Do not link, upload, submit, or execute AMD code.
    compile_records=[]
    def compile_only(linear,*args,**kwargs):
      compiled=real_lower(linear)
      for call in compiled.src:
        p=call.body
        if p.op is not Ops.PROGRAM: continue
        data,_=ops_amd._amd_program_image(dev,p.src[3].arg)
        compile_records.append({'name':p.src[0].arg.name,'target':str(p.arg.target),
          'binary_sha256':__import__('hashlib').sha256(p.src[3].arg).hexdigest(),
          'source_sha256':__import__('hashlib').sha256(p.src[2].arg.encode()).hexdigest(),
          'actual_mode':'WGP' if (data.rsrc1>>29)&1 else 'CU','kernarg_bytes':data.kernargs_segment_size})
      return compiled
    with H.native_scope(C,{}):
      boundary=compile_only(native_linears[0])
      assert len(boundary.src)==3
      tracker=hcq2.DepsTracker([('AMD','COMPUTE:0')]*3);dependencies=[]
      for tag,call in enumerate(boundary.src):
        outs,_=realize.get_call_outs_ins(call)
        dependencies.append(tracker.access_resources([(u,i in outs) for i,u in enumerate(call.src[1:])],tag))
      assert dependencies==[[],[0],[0]],dependencies
      hcq2._build_queues(hcq2.BatchCtx([(call,'AMD','COMPUTE:0') for call in boundary.src],False))
    with patch.object(tm,'run_linear',compile_only):
      for store_shape,ret_shape,dt in (((H.OC+512,),(1,16,2048,128),dtypes.bfloat16),((H.LC+512,),(1,16,2048),dtypes.float32)):
        store=io.allocate('AMD',store_shape,dt);returned=io.allocate('AMD',ret_shape,dt);H.audit(store,returned)
        uint=dtypes.uint16 if dt==dtypes.bfloat16 else dtypes.uint32
        view=store.uop.buffer.view(uint.itemsize,256*uint.itemsize).ensure_allocated()
        Tensor(UOp.from_buffer(view,uint)).assign(0x7fc1 if uint.itemsize==2 else 0x7fc12345).realize()
    result.update(actual_AMD_compile_only=compile_records,O_and_LSE_HCQ_dependencies=dependencies,AMD_link_or_execution=False)
    result.update(completed_mock_runner=True,executed_CPU_mock_calls=executed)
  finally: C.PREPARED=prepared
  C.write(output,result);print(json.dumps({'completed_mock_runner':True,'gpu_executed':False,'report':str(output)}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)

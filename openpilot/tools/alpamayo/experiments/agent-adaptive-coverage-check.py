"""Actual production main/factory/18-query/finally execution with neural doubles."""
import argparse
from contextlib import ExitStack
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import socket
import sys
from types import MethodType,SimpleNamespace
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('coverage_check_common',HERE/'agent-adaptive-coverage-common.py')
K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)


class FactoryBoundary(Exception):pass


class Shape:
  def __init__(self,shape,dtype=None):self.shape=tuple(shape);self.dtype=dtype;self.ndim=len(self.shape);self.device='CPU'
  def clone(self):return Shape(self.shape,self.dtype)
  def realize(self):return self
  def __getitem__(self,key):
    if isinstance(key,Shape):return Shape((*key.shape,self.shape[-1]),self.dtype)
    key=key if isinstance(key,tuple) else (key,)
    shape=[]
    for i,n in enumerate(self.shape):
      item=key[i] if i<len(key) else slice(None)
      if isinstance(item,slice):shape.append(len(range(*item.indices(n))))
    return Shape(shape,self.dtype)
  def unsqueeze(self,axis):return Shape((*self.shape[:axis],1,*self.shape[axis:]),self.dtype)
  def cat(self,*others,dim=0):
    shape=list(self.shape);shape[dim]=sum(x.shape[dim] for x in (self,*others));return Shape(shape,self.dtype)


class Graph:
  def __init__(self,names=()):self.cnt=0;self.captured=None;self.names=list(names);self.resets=0
  def reset(self):self.cnt=0;self.captured=None;self.resets+=1
  def capture(self):
    call=SimpleNamespace(arg=SimpleNamespace(aux=SimpleNamespace(kernels=[(None,n) for n in self.names])))
    call.without_after=call;self.captured=SimpleNamespace(_linear=SimpleNamespace(src=[call]))


def main():
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inject-failure',action='store_true');args=p.parse_args()
  args0=args
  assert not args.output.exists()
  R,plan,N,C,evidence=K.frozen()
  from tinygrad import Device,dtypes
  assert not Device._opened_devices
  root=K.WORK/'agent-adaptive-coverage-cpu-artifacts'/(args.output.stem+'-'+K.sha(__file__)[:10]);root.mkdir(parents=True)
  H=K.load(HERE/'agent-adaptive-prefill-check.py','coverage_neural_arrays');Array=H.Array
  baseline=K.read(K.REFERENCE);saved_rows=baseline['warmup']+baseline['runs']
  prior=K.read(K.WORK/'agent-exact-combined-quality-candidate/report.json')['agent_exact_combined']['coverage']
  reference_pv=K.read(K.WORK/'agent-expert-combined-cache-quality-candidate/report.json')['agent_expert_combined']['pv']
  original_runpy=runpy.run_path;original_spec=importlib.util.spec_from_file_location
  events=[];observed=[];rejections=[];components=[];compositions=[]
  def blocked(*a,**kw):raise AssertionError('Device/network blocked in CPU production proof')
  def reject(name,fn):
    try:fn()
    except (AssertionError,ValueError,RuntimeError,KeyError):rejections.append(name)
    else:raise AssertionError('Invalid control accepted: '+name)
  M=K.load(HERE/'agent-adaptive-coverage-run.py','coverage_actual_runner')
  original_factory=K.factory

  with ExitStack() as stack:
    stack.enter_context(patch.dict(os.environ,dict(os.environ),clear=True))
    stack.enter_context(patch.object(type(Device),'__getitem__',blocked))
    stack.enter_context(patch.object(socket.socket,'connect',blocked))
    import torch
    torch.set_num_threads(4)
    def components_after_factory(ns,args,adapter,settings,sources,source_dir):
      comp=compositions[-1]
      events.append('actual_factory_index_read_then_neural_completion_double')
      model=SimpleNamespace(config=adapter.config.to_dict(),dtype=dtypes.bfloat16,cache=[Shape((2,1,8,2304,128),dtypes.bfloat16) for _ in range(28)],
        prefill_jit=Graph(['agent_packed_epilogue_cpu_neural_double']*28),decode_jit=Graph(),greedy_decode_jit=Graph(),
        trajectory_jits={},minimum_cache_capacity=0,prompt_length=None,output_token_mapping=Array(np.array([1],np.int32)),
        weights={'vlm.model.language_model.embed_tokens.weight':Shape((135168,2048),dtypes.bfloat16)},synchronize=lambda:None,
        encode=lambda *a:None)
      model.prefill_jit.capture()
      comp.model=model;comp.events.extend(['packed_quantizer','batched_vision','vision_tiles','language_fusion','factory_ready'])
      comp.arm_calls=1;comp.packed=(object(),{})
      comp.session=SimpleNamespace(stats=copy.deepcopy(prior['language_stats']),frame=None,state={'prefill':False})
      comp.session.stats['methods_restored']=False
      comp.vision_info=copy.deepcopy(prior['vision_hook'])
      comp.pv_handle=SimpleNamespace(info=copy.deepcopy(reference_pv));comp.pv_handle.info.update(active=True,cleanup_exact=False,observer_cleanup_exact=False,trajectory_calls=0)
      def cleanup():
        comp.session.stats['methods_restored']=True
        comp.pv_handle.info.update(active=False,cleanup_exact=True,observer_cleanup_exact=True)
        comp.restorations.append({'name':'CPU_NEURAL_installers_only','restored':True})
      comp.stack.callback(cleanup)
      pad=ns['local_module']('padded-prefill');pad.install(model)
      compact=ns['local_module']('compact-inputs').CompactInputs
      vision=ns['local_module']('vision-cache')
      class VisionJit(Graph):
        def __init__(self,fn):super().__init__();self.fn=fn
        def __call__(self,*values):
          self.cnt+=1
          if len(values)>=2 and isinstance(values[0],Shape) and values[0].shape==(1,720,768):
            batch=len(values)//2;self.names=[f'agent_vision_tiles_3x3_b{batch}']*25+[f'agent_vision_tiles_3x6_b{batch}']*25
          if self.cnt>=2:self.capture()
          return self.fn(*values)
      class TensorDouble:
        @staticmethod
        def realize(*values):return values
        cat=Shape.cat
      stack.enter_context(patch.object(vision,'Tensor',TensorDouble));stack.enter_context(patch.object(vision,'TinyJit',VisionJit))
      stack.enter_context(patch.object(vision.VisionCache,'image_view',staticmethod(lambda value,a,b:Shape((1,b-a,value.shape[-1]),value.dtype))))
      cache=vision.VisionCache(model,capacity=144,verify=False,grouped=True,views=True,
        batch_encoder=lambda *values:tuple(Shape((180,2048),dtypes.bfloat16) for _ in values[::2]),group_limit=2)
      lookup=ns['local_module']('lookup-decode')
      stack.enter_context(patch.object(lookup,'Tensor',Array))
      decoder=lookup.LookupDecoder(model)
      decoder.compact_io=True;decoder.carry_verified_token=True;decoder.fallback_block=6;decoder.direct_readback=True
      decoder.readback=lambda value:value.numpy();decoder.read_tokens=lambda value:value if isinstance(value,np.ndarray) else value.numpy()
      decoder.first_token=lambda value:int(value.data.reshape(-1)[0])
      decoder.select_logits=lambda values,index:values[:,index:index+1].clone()
      current={'index':-1,'tokens':None}
      anchor=adapter.config.traj_ids['future_start']
      def token_at(offset):return current['tokens'][offset] if offset<len(current['tokens']) else anchor
      def decode(logits,cos,sin,position,step,count):
        return Array(np.array([[[token_at(step+count)]]],np.float32)),Array(np.array([token_at(i) for i in range(step,step+count)],np.int32))
      decoder.decode=decode
      class Verifier:
        cnt=2
        def __call__(self,tokens,cos,sin,position,step):
          step=step.unbind()[1];count=tokens.shape[0]
          ids=np.array([token_at(i) for i in range(step+1,step+count+1)],np.int32)
          return Array(ids.astype(np.float32).reshape(1,count,1)),Array(ids)
      decoder.verifiers={(n,(1,1,64,128),2304):Verifier() for n in (16,32,48,64)}
      # No candidate code or policy changes: only the expensive neural prefill
      # result, GPU capture descriptors, and arithmetic buffers are doubled.
      adaptive=model.padded_prefill
      owner_state=next(s for s in test_overlay[0].states)
      def prefill(self,x,cos,sin,max_tokens,prompt):
        current['index']+=1;index=current['index'];tokens=saved_rows[index]['prediction']['generated_token_ids'];current['tokens']=tokens
        state=owner_state;draft=state.current['draft'];state.active=bool(draft)
        self.last_record=None;self.last_logits=self.last_payload=None
        if not draft:
          self.baseline.last_prompt=prompt
          self.baseline.last_targets=np.array([*saved_spec[index]['targets'],*([1]*6)],np.int32)
          self.baseline.prefix_audit=None
        else:
          targets=[token_at(i) for i in range(17)]
          self.last_payload=Array(np.array([*targets,*([1]*17)],np.int32))
          self.last_logits=Array(np.asarray(targets,np.float32).reshape(1,17,1))
          auditing=not state.audit_done
          self.last_record={'actual_prompt':prompt,'draft':list(draft),'width':16,'head_projections':17,
            'audit_prefix_once':auditing,'completed_before':state.completed,'previous_completed':list(state.previous),'numerical_candidate':True}
          if args0.inject_failure and index==3:raise AssertionError('CPU injected first adaptive prefix arithmetic failure')
          if auditing:self.last_record['prefix_audit']={'first_logits_exact':True,'all_layer_prefix_kv_exact':True,'layer_count':28,'rows':prompt};state.audit_done=True
        return Array(np.array([[[tokens[0]]]],np.float32))
      stack.enter_context(patch.object(type(adaptive),'__call__',prefill))
      def trajectory(actions,cos,sin,length,steps,offset):
        assert steps==3 and length==int(saved_rows[current['index']]['prediction']['prompt_tokens'])+offset
        assert offset==len(current['tokens'])
        comp.pv_handle.info['trajectory_calls']+=1
        return Array(np.zeros((1,64,2),np.float32))
      model.trajectory=trajectory
      class BufferArray:
        def __init__(self,value):self.data=np.array(value,copy=True);self.shape=self.data.shape;self.dtype=self.data.dtype
        def _buffer(self):return self
        def as_memoryview(self,**kw):return memoryview(self.data).cast('B')
      class Inputs(compact):
        def __init__(self,args,config,prepared,packed,arrays=None):
          self._signature=self.signature(args,config,prepared,packed);self.layout_key=('CPU_neural',)
          self.cpu=[BufferArray(a) for a in arrays];self.cache_rotary_inputs=False
          self.batches=tuple((i*720,(i+1)*720,1) for i in range(10))
          ids=packed[0];indices=np.flatnonzero(ids[0]==config.vlm_config.image_token_id)
          self.spans=tuple((int(x[0]),int(x[-1])+1) for x in np.split(indices,np.flatnonzero(np.diff(indices)!=1)+1))
          self.selective_cache=None;self.uploaded_pixel_bytes=0
        def upload(self,noise):
          assert self.selective_cache is cache
          self.uploaded_pixel_bytes=sum(720*768 for key in cache.keys if key not in cache.cache)
          return (Shape((7200,768),dtypes.bfloat16),Shape((7200,1152),dtypes.bfloat16),Shape((1,2048),dtypes.int32),
            Shape((1,1,2048,128)),Shape((1,1,2048,128)),Shape((1,1,64,128)),Shape((1,1,64,128)),
            Shape((1,1,128,128)),Shape((1,1,128,128)),Array(np.asarray(noise,np.float32)))
      settings.initial_noise=np.load(args.initial_noise,allow_pickle=False)
      report=copy.deepcopy(baseline);report['warmup']=[];report['runs']=[]
      report['source_sha256']=sources;report['source_snapshot']=str(source_dir)
      value=SimpleNamespace(adapter=adapter,model=model,settings=settings,input_class=Inputs,vision_cache=cache,
        decoder=decoder,runner=decoder.run,startup_active=[False],report=report,vision_provenance={},
        benchmark=sys.modules['benchmark'],Device=SimpleNamespace(DEFAULT='CPU'),
        GlobalCounters=SimpleNamespace(mem_used_per_device={'CPU_NEURAL_DOUBLE':0}),profiler=None)
      components.append(value);events.append('original_factory_completed_with_explicit_neural_doubles')
      return value

    # Observe original factory imports without changing any installed source.
    def observer(previous):
      def spec(name,location,*a,**kw):
        result=previous(name,location,*a,**kw)
        if location is None or Path(location).resolve()!=C.MLP/'experiments/benchmark-fast.py':return result
        loader=result.loader
        class Loader:
          def create_module(self,spec):return loader.create_module(spec)
          def exec_module(self,module):
            loader.exec_module(module)
            def stop(*a,**kw):events.append('actual_benchmark_fast_index_loaded');raise FactoryBoundary()
            stack.enter_context(patch.object(module,'safe_load',stop))
        result.loader=Loader();return result
      return spec
    def runpy_entry(path,*a,**kw):
      observed.append(str(path))
      if Path(path).resolve()!=C.MLP/'experiments/benchmark-pipeline.py':return original_runpy(path,*a,**kw)
      namespace=original_runpy(path,run_name='coverage_actual_pipeline')
      ns=namespace['main'].__globals__;initialize=ns['initialize']
      def initialize_boundary(*values,**options):
        import tinygrad
        from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler
        metadata=SimpleNamespace(arch='gfx1200',compiler=AMDLLVMCompiler('gfx1200'),renderer=SimpleNamespace(compiler=None))
        class DeviceMetadata:
          DEFAULT='USB+AMD:LLVM'
          def __class_getitem__(cls,key):assert key==cls.DEFAULT;return metadata
        events.append('actual_initialize_entered')
        try:
          with patch.object(tinygrad,'Device',DeviceMetadata):return initialize(*values,**options)
        except FactoryBoundary as error:
          trace=error.__traceback__;frame=None
          while trace:
            if trace.tb_frame.f_code is initialize.__code__:frame=trace.tb_frame
            trace=trace.tb_next
          assert frame is not None
          return components_after_factory(ns,values[0],frame.f_locals['adapter'],frame.f_locals['settings'],options['sources'],options['source_dir'])
      ns['initialize']=initialize_boundary
      namespace['main']()
      return namespace
    test_overlay=[]
    saved_spec=K.read(K.REFERENCE.with_suffix('.speculation.json'))['records']
    original_load=K.load
    def load(path,name):
      value=original_load(path,name)
      if Path(path).name=='agent-adaptive-prefill-hook.py':
        Original=value.Overlay
        class Tracked(Original):
          def __init__(self,*a,**kw):super().__init__(*a,**kw);test_overlay.append(self)
        value.Overlay=Tracked
      return value
    def factory(C):
      Base=original_factory(C)
      class Observed(Base):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);compositions.append(self)
      return Observed
    # The actual new runner, original speculation main, canonical pipeline main,
    # canonical initialize and adapter all execute. Production coverage is real.
    stack.enter_context(patch.object(M.K,'factory',factory));stack.enter_context(patch.object(M.K,'load',load))
    stack.enter_context(patch.object(importlib.util,'spec_from_file_location',observer(original_spec)))
    stack.enter_context(patch.object(runpy,'run_path',runpy_entry))
    pipeline=K.read(plan['pipeline']);pipeline['environment']['CACHEDB']=str(root/'cpu-cache.db');pipeline['environment']['TMPDIR']=str(root)
    cpu_pipeline=root/'pipeline.json';K.write(cpu_pipeline,pipeline,new=True)
    stack.enter_context(patch.object(M,'PIPELINE',cpu_pipeline))
    stack.enter_context(patch.object(M,'verify',lambda:{'CPU_TEST_ONLY':True}))
    receipt=K.reference_receipt(C)
    cpu_receipt=root/'reference-receipt.json';K.write(cpu_receipt,receipt,new=True)
    stack.enter_context(patch.object(M.K,'RECEIPT',cpu_receipt))
    cpu_manifest=root/'manifest.json';K.write(cpu_manifest,{'CPU_NEURAL_DOUBLES':True},new=True)
    stack.enter_context(patch.object(M,'MANIFEST',cpu_manifest))
    output=root/'production/report.json'
    failure=None
    try:M.run(output)
    except AssertionError as exc:
      if not args.inject_failure:raise
      failure=str(exc)
    status=K.read(output.parent/'adaptive-coverage-status.json')
    if args.inject_failure:
      assert failure=='CPU injected first adaptive prefix arithmetic failure'
      assert not status['complete'] and not status['gpu_executed']
      assert status['primary_failure']=={'type':'AssertionError','message':failure}
      assert status['import_hook_restored'] and status['adaptive_hook_restored']
      before=status['before_cleanup'];after=status['after_cleanup']
      assert before['adaptive']['prefills'][0]['last_record']['draft']
      assert before['adaptive']['states'][0]['completed']==3 and before['composition']['vision']
      assert before['composition']['pv']['trajectory_calls']==3
      assert after['composition']['pv']['cleanup_exact'] and after['composition']['pv']['observer_cleanup_exact']
      assert not after['composition']['vision'] and after['composition']['language_stats']['methods_restored']
      assert not status['finally_pv_validation']['passed']
      K.write(args.output,{'complete':True,'gpu_executed':False,'injected_failure_preserved':True,
        'actual_production_runpy_targets':observed,'current_partial_record_preserved':True,'finally_cleanup_exact':True,
        'production_status':str(output.parent/'adaptive-coverage-status.json'),
        'sources':{str(p):K.sha(p) for p in HERE.glob('agent-adaptive-coverage-*.py')}},new=True)
      print({'complete':True,'injected_failure':True,'gpu_executed':False});return
    assert status['complete'] and not status['gpu_executed'] and status['import_hook_restored'] and status['adaptive_hook_restored']
    assert status['checks']['shape_coverage']['vision_batches_exercised']==[1,2]
    assert status['checks']['adaptive']['genuine_partial_queries']>0
    assert status['finally_pv_validation']['passed']
    before=status['before_cleanup']['composition'];after=status['after_cleanup']['composition']
    assert before['vision'] and not after['vision'] and before['pv']['active'] and not after['pv']['active']
    assert after['pv']['cleanup_exact'] and after['pv']['observer_cleanup_exact'] and after['language_stats']['methods_restored']
    assert [r['misses'] for r in before['vision_calls']]==[r['vision_misses'] for r in saved_rows]
    # Rebuild only CPU diagnostic descriptors for mutation checks. The genuine
    # cleanup result remains preserved in the production status above.
    comp=compositions[0];comp.model.prefill_jit=Graph(['agent_packed_epilogue_cpu_neural_double']*28);comp.model.prefill_jit.capture()
    for row in before['vision']:
      graph=Graph(row['names']);graph.capture();comp.cache.encoders[tuple(map(tuple,row['shape']))]=graph
    assert comp.coverage()['validated']
    damaged=next(iter(comp.cache.encoders));value=comp.cache.encoders.pop(damaged)
    reject('missing observed graph',comp.coverage);comp.cache.encoders[damaged]=value
    value.names=['agent_vision_tiles_3x3_b2']*24+['agent_vision_tiles_3x6_b2']*25;value.capture()
    reject('missing projection or wrong batch kernel',comp.coverage)
    value.names=next(r['names'] for r in before['vision'] if tuple(map(tuple,r['shape']))==damaged);value.capture()
    comp.cache.encoders[((1,721,768),(1,721,1152))]=value
    reject('unqualified row shape',comp.coverage);del comp.cache.encoders[((1,721,768),(1,721,1152))]
    comp.cache.encoders[K.PAIR*3]=value;reject('unqualified batch3',comp.coverage);del comp.cache.encoders[K.PAIR*3]
    stats=comp.session.stats;key=next(iter(stats['layer_hits']));old=stats['layer_hits'].pop(key)
    reject('missing language layer',comp.coverage);stats['layer_hits'][key]=old
    comp.session.frame=object();reject('active quantizer frame',comp.coverage);comp.session.frame=None
    comp.coverage_vision_calls[-1]['completed']=False;reject('incomplete query ledger',comp.coverage);comp.coverage_vision_calls[-1]['completed']=True
    reject('unknown frozen source',lambda:comp.require_path(HERE/'agent-adaptive-coverage-common.py'))
    old=comp.vision_info['trace_calls'].pop(next(iter(comp.vision_info['trace_calls'])))
    reject('missing per-batch trace',comp.coverage)
    # Diagnostics remain serializable before and after a failed coverage check.
    failure_snapshot=M.safe_snapshot(comp,test_overlay[0]);assert failure_snapshot['composition']['pv']
    assert failure_snapshot['adaptive']['states'][0]['completed']==18
  assert not Device._opened_devices and importlib.util.spec_from_file_location is original_spec
  result={'complete':True,'gpu_executed':False,'actual_production_runpy_targets':observed,
    'events':events,'all_18_postvalidation':True,'factory_index_and_adapter_actual':True,
    'real_cache_grouping_matches_completed_gpu_report':True,'vision_calls':before['vision_calls'],
    'active_and_cleanup_lifecycle_checked':True,'negative_controls':rejections,
    'neural_doubles':['GPU factory completion and capture metadata','tensor allocation/upload','prefill and decode arithmetic','expert trajectory output'],
    'candidate_numerical_hook_file_unchanged':True,'reference_receipt':receipt,
    'production_status':str(output.parent/'adaptive-coverage-status.json'),
    'sources':{str(p):K.sha(p) for p in HERE.glob('agent-adaptive-coverage-*.py')}}
  K.write(args.output,result,new=True);print({'complete':True,'gpu_executed':False,'queries':18,'rejections':len(rejections)})


if __name__=='__main__':main()

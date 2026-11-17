"""Actual runpy launchers through factory, startup boundary, and Consumer construction.

Only GPU computation/allocation is doubled. Canonical mains, normalization, cache IO,
adapters, array preparation, Consumer construction, and its first binding call execute.
"""
import argparse
from contextlib import ExitStack,contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cpu_integrated_check_common',HERE/'agent-live-cpu-integrated-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

class FactoryBoundary(Exception):pass
class ComputeBoundary(Exception):pass
class ConstructedBoundary(Exception):pass
def blocked(*a,**kw):raise AssertionError('Device/network access forbidden in CPU preflight')


def admit():
  check=C.load(C.S.OLD/'agent-expert-live-check.py','cpu_integrated_cache_preflight');check.N=C.N
  result=check.admission()
  assert result['complete'] and result['factory']['actual_factory_load_executed']
  assert result['factory']['real_device_open_attempts']==[] and result['actual_storage_bindings']==C.N.storage()
  return result


class Boundary:
  def __init__(self,arm):
    self.arm=arm;self.events=[];self.hooks=None;self.consumer=None;self.components=None;self.factory_seen=False
    self.pad_checked=False;self.runpy_seen=[];self.owner_errors=[]

  def components_after_factory(self,pipeline,args,adapter,settings,sources,source_dir):
    # CPU inputs are real. Model execution and GPU allocations remain explicit doubles.
    import numpy as np
    settings.initial_noise=np.load(args.initial_noise,allow_pickle=False)
    assert settings.initial_noise.shape==(1,64,2) and settings.initial_noise.dtype==np.float32
    assert np.isfinite(settings.initial_noise).all()
    check=C.load(C.N.LIVE/'experiments/agent-live-consumer-check.py','cpu_integrated_cpu_storage')
    compact=pipeline.local_module('compact-inputs').CompactInputs
    vision_kind=pipeline.local_module('vision-cache').VisionCache
    pad=pipeline.local_module('padded-prefill').pad_inputs
    assert Path(pad.__code__.co_filename)==C.N.LIVE/'experiments/agent-speculation-run.py'
    self.pad_checked=True
    class Prefill:
      last_targets=None;audit=False
      def __call__(self,*a,**kw):blocked()
    class Model:
      def __init__(self):
        self.cache=[SimpleNamespace(uop=SimpleNamespace(base=object())) for _ in range(28)]
        self.minimum_cache_capacity=2304;self.trajectory_jits={};self.padded_prefill=Prefill();self.syncs=0
      def synchronize(self):self.syncs+=1
      def encode(self,*a,**kw):blocked()
    model=Model();boundary=self
    class Decoder:
      def __init__(self):
        self.model=model;self.history=[];self.prefill_draft_old_history=[];self.prefill_draft_records=[];self.statistics=[]
        self.jits={};self.verifiers={};self.select_jits={};self.warmup_graphs=False;self.before_diffusion=lambda:None
      def generate(self,*a,**kw):blocked()
      def run(self,*a,**kw):
        boundary.events.append('actual_consumer_runner_boundary')
        raise ComputeBoundary('Stop before first GPU inference')
    # Explicit state of the doubled GPU factory completion. The original cache
    # constructor and its ownership guard still execute; no GPU coverage is claimed.
    self.hooks.composition.model=model
    self.hooks.composition.events.append('batched_vision')
    self.events.append('CPU_double_GPU_factory_installation_metadata')
    decoder=Decoder();vision=vision_kind(model,capacity=144,verify=args.verify_vision_cache)
    value=SimpleNamespace(adapter=adapter,model=model,settings=settings,config=adapter.config,tokenizer=adapter.tokenizer,
      input_class=check.cpu_inputs_class(compact),vision_cache=vision,pad_inputs=pad,inputs=None,decoder=decoder,runner=decoder.run,
      startup=None,startup_active=(False,False),profiler=None,vision_provenance={'CPU_GPU_COMPUTE_DOUBLE':True},
      provenance={'CPU_GPU_COMPUTE_DOUBLE':True,'source_snapshot':str(source_dir)},
      report={'source_sha256':sources,'runtime_environment':{}},_startup_spec={'ready':False})
    self.components=value;self.hooks.composition.model=model;self.hooks.composition.cache=vision
    return value

  def startup_compute(self,components,effective,arrays,**kw):
    assert self.factory_seen and components is self.components
    import torch
    data={k:v if k=='nav_text' else torch.from_numpy(v) for k,v in arrays.items()}
    _,_,prepared=components.adapter.prepare(data);packed=components.adapter.native_inputs(prepared)
    prepared,packed=components.pad_inputs(prepared,packed,effective.pad_prompt_to)
    upload=components.input_class.arrays(components.settings,prepared,packed)
    components.inputs=components.input_class(components.settings,components.config,prepared,packed,arrays=upload)
    components._startup_spec={'ready':True,'CPU_GPU_COMPUTE_DOUBLE':True}
    self.events.append('cpu_inputs_at_original_startup_compute_boundary')
    return dict(components._startup_spec)

  @contextmanager
  def hooks_context(self,hooks):
    self.hooks=hooks
    with patch.object(hooks.warm,'initialize',self.startup_compute):yield

  def observe(self,previous,stack):
    boundary=self
    def spec(name,location,*a,**kw):
      result=previous(name,location,*a,**kw)
      if result is None or location is None:return result
      path=Path(location).resolve();loader=result.loader
      targets={'benchmark-pipeline.py','benchmark-fast.py','agent-live-runtime.py','agent-live-consumer.py'}
      ours=path==HERE/'agent-live-cpu-integrated-install.py'
      if not ours and (path.parent!=C.N.LIVE/'experiments' or path.name not in targets):return result
      class Loader:
        def create_module(self,spec):return loader.create_module(spec)
        def exec_module(self,module):
          loader.exec_module(module)
          if ours:
            factory=module.factory
            def cpu_factory(common):
              Base=factory(common)
              class CPUHooks(Base):
                def __enter__(self):
                  value=super().__enter__();self.stack.enter_context(boundary.hooks_context(self));return value
                def runtime(self,value):
                  super().runtime(value);original=value.bind_components
                  def bind(*a,**kw):
                    consumer=original(*a,**kw);boundary.consumer=consumer
                    assert boundary.factory_seen and boundary.pad_checked
                    assert self.cpu_handle.guard.shadow is None and self.cpu_info['calls']==[]
                    boundary.events.append('actual_Consumer_constructed_and_CPU_hook_installed')
                    # The actual binder rejects duplicate component ownership before another constructor.
                    try:original(*a,**kw)
                    except (ValueError,RuntimeError):boundary.owner_errors.append('duplicate owner rejected')
                    else:raise AssertionError('Duplicate Consumer owner accepted')
                    assert consumer._lock.acquire(blocking=False)
                    try:
                      try:consumer.predict({}, {'query_ns':1})
                      except RuntimeError as error:assert str(error).startswith('One live query is still active');boundary.owner_errors.append('active query rejected')
                      else:raise AssertionError('Nested query accepted')
                    finally:consumer._lock.release()
                    if boundary.arm=='paced':raise ConstructedBoundary('Stop before source clock or producer starts')
                    return consumer
                  self.patch(value,'bind_components',bind)
              return CPUHooks
            module.factory=cpu_factory
          elif path.name=='benchmark-fast.py':
            def conversion(*a,**kw):
              boundary.factory_seen=True;boundary.events.append('actual_factory_cache_read_and_conversion_boundary')
              raise FactoryBoundary()
            stack.enter_context(patch.object(module,'safe_load',conversion));stack.enter_context(patch.object(module,'upload',blocked))
          elif path.name=='benchmark-pipeline.py':
            original=module.initialize
            def initialize(*a,**kw):
              import tinygrad
              from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler
              fake=SimpleNamespace(arch='gfx1200',compiler=AMDLLVMCompiler('gfx1200'),renderer=SimpleNamespace(compiler=None))
              class MetadataDevice:
                DEFAULT='USB+AMD:LLVM'
                def __class_getitem__(cls,key):assert key==cls.DEFAULT;return fake
              boundary.events.append('actual_pipeline_initialize')
              try:
                with patch.object(tinygrad,'Device',MetadataDevice):return original(*a,**kw)
              except FactoryBoundary as error:
                trace=error.__traceback__;locals_=None
                while trace:
                  if trace.tb_frame.f_code is original.__code__:locals_=trace.tb_frame.f_locals
                  trace=trace.tb_next
                assert locals_ is not None and 'adapter' in locals_ and 'settings' in locals_
                return boundary.components_after_factory(module,a[0],locals_['adapter'],locals_['settings'],kw['sources'],kw['source_dir'])
            module.initialize=initialize
      result.loader=Loader();return result
    return spec


def production_case(root,arm,data):
  from tinygrad import Device
  output=root/arm/'report.json';protocol=root/(arm+'-protocol.json')
  C.write(protocol,dict(data,cpu_only=True),new=True)
  args=[str(HERE/'agent-live-cpu-integrated-run.py'),'--arm',arm,'--protocol',str(protocol),'--protocol-sha',C.sha(protocol),'--output',str(output)]
  if arm=='paced':args+=['--reviewed-parity-sha256',C.REFERENCE_SHA]
  boundary=Boundary(arm);original_spec=importlib.util.spec_from_file_location
  modules={name:sys.modules.get(name) for name in ('model','benchmark','inputs','native_warp','stream_inputs')}
  def observe_runpy(frame,event,arg):
    if event=='call' and frame.f_code.co_name in ('run_pipeline','launch'):
      if 'path' in frame.f_locals:boundary.runpy_seen.append(str(frame.f_locals['path']))
  error=None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ));stack.enter_context(patch.object(sys,'path',sys.path[:]))
      stack.enter_context(patch.object(sys,'argv',args))
      stack.enter_context(patch.object(type(Device),'__getitem__',blocked));stack.enter_context(patch.object(socket.socket,'connect',blocked))
      stack.enter_context(patch.object(importlib.util,'spec_from_file_location',boundary.observe(original_spec,stack)))
      old_profile=sys.getprofile();sys.setprofile(observe_runpy)
      try:runpy.run_path(args[0],run_name='__main__')
      except (ComputeBoundary,ConstructedBoundary) as e:error=e
      finally:sys.setprofile(old_profile)
  finally:
    for name,old in modules.items():
      if old is None:sys.modules.pop(name,None)
      else:sys.modules[name]=old
  assert type(error) is (ComputeBoundary if arm=='consumer' else ConstructedBoundary)
  assert importlib.util.spec_from_file_location is original_spec and not Device._opened_devices
  assert boundary.runpy_seen==[str(C.N.LIVE/'experiments/benchmark-pipeline.py')]
  assert boundary.owner_errors==['duplicate owner rejected','active query rejected']
  status=C.read(output.parent/'agent-live-cpu-integrated-status.json')
  assert not status['passed'] and not status['gpu_executed'] and status['hooks_restored'] and status['constructor_restored']
  info=status['cpu_hook'];assert info['cleanup_exact'] and info['shadow_after_cleanup']==0
  assert len(info['calls'])==int(arm=='consumer')
  if arm=='consumer':assert info['calls'][0]['completed'] and info['calls'][0]['shadow_before']==0 and info['calls'][0]['shadow_after']==33177600
  events=status['storage_events']
  assert [e['event'] for e in events].count('normalize')==1
  assert [e['path'] for e in events if e['event']=='cache_open']==[str(C.N.WEIGHTS/'index.json')]
  assert any(e['event']=='factory_enter' and e['cachedb_constant']==C.N.storage()['CACHEDB'] for e in events)
  assert boundary.consumer._closed and not boundary.consumer._lock.locked()
  return dict(complete=True,gpu_executed=False,events=boundary.events,actual_runpy_targets=boundary.runpy_seen,
    owner_errors=boundary.owner_errors,storage_events=events,cpu_hook=info,source_clock_started=False,
    mocked='GPU factory completion, GPU allocations and GPU startup computation only; real factory entry/index read and actual Consumer construction',
    cpu_protocol_uses_saved_parity_for_paced_setup_only=arm=='paced')


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--admit',action='store_true')
  args=p.parse_args();assert not args.output.exists();C.configure()
  if args.admit:result=admit()
  else:
    root=C.ARTIFACTS/args.output.stem;assert not root.exists();root.mkdir()
    data=dict(schema=1,files=C.files(),foundations=C.foundations(),arms=['consumer','paced'],automatic_paced=False)
    rows={arm:production_case(root,arm,data) for arm in ('consumer','paced')}
    result=dict(complete=True,gpu_executed=False,production=rows,source_sha256=C.files())
    # The actual strict comparator rejects canonical mutations; no tolerance or role weakening.
    baseline=C.read(C.REFERENCE);m,_=C.N.live();rejected=[]
    for label in ('token','input','history','speculation','kv','owner','unknown'):
      candidate=copy.deepcopy(baseline)
      if label=='token':candidate['rows'][0]['prediction']['generated_token_ids'][0]+=1
      elif label=='input':candidate['rows'][0]['prepared']['unexpected']=1
      elif label=='history':candidate['rows'][0]['history_before']['history'].append([1])
      elif label=='speculation':candidate['rows'][0]['speculation']['accepted']=False
      elif label=='kv':candidate['rows'][0]['defined_kv_prefix_bits'][0]['sha256']='0'*64
      elif label=='owner':candidate['rows'][0]['consumer_protocol']['input_reused']=False
      else:candidate['rows'][0]['unexpected']=1
      try:C.parity(baseline,candidate,C.REFERENCE)
      except (ValueError,AssertionError):rejected.append(label)
      else:raise AssertionError('Canonical mutation accepted: '+label)
    result['comparison_rejections']=rejected
  C.write(args.output,result,new=True);print(dict(complete=True,gpu_executed=False,admit_only=args.admit),flush=True)


if __name__=='__main__':main()

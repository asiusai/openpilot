"""CPU execution of both production wrappers and real paced setup through factory entry."""
import argparse
import copy
from contextlib import ExitStack
import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stem_check_common',HERE/'agent-expert-paced-stem-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C);N=C.N


def blocked(*a,**kw):raise AssertionError('Real device/network access forbidden')
class FactoryBoundary(Exception):pass


def admit():
  check=N.load(C.V.S.OLD/'agent-expert-live-check.py','stem_actual_cache_preflight');check.N=N
  proof=check.admission()
  assert proof['factory']['real_device_open_attempts']==[] and proof['actual_storage_bindings']==N.storage()
  return proof


def evidence_files():
  files=dict(C.read(C.QUALIFIED_PROTOCOL)['files'])
  files[str(C.QUALIFIED_PROTOCOL)]=C.QUALIFIED_SHA
  for p in [*C.CANDIDATE.parent.glob('*.json'),*C.FAILED.glob('*.json'),C.FAILED/'run.log',
            C.QUALIFIED/'candidate-parent.json',C.QUALIFIED/'paced-parent.json']:
    files[str(p)]=C.sha(p)
  folder=Path(C.read(C.CANDIDATE)['consumer_provenance']['initializer']['factory']['source_snapshot'])
  for p in folder.iterdir():files[str(p)]=C.sha(p)
  for p in HERE.glob('agent-expert-paced-stem-*.py'):files[str(p)]=C.sha(p)
  target=C.ARTIFACTS/'hooks/agent-exact-combined-install.py';files[str(target)]=C.sha(target)
  return files


def protocol(output,files,admission,*,cpu=True,repair=True):
  return dict(schema=1,qualified_protocol_sha256=C.QUALIFIED_SHA,candidate_sha256=C.CANDIDATE_SHA,
    files=files,output=str(output),metadata={k:str(v) for k,v in C.metadata(output).items()},cpu_only=cpu,repair=repair,
    candidate_admission=admission,failed_evidence=C.failed_evidence(),artifact_overlay=C.build(output,repair=repair)[2],
    overlay_helper_sha256={p.name:C.sha(p) for p in HERE.glob('agent-expert-paced-stem-*.py')},
    gpu_arms=['paced'],automatic_paced=False,timeout_seconds=900,pace=C.read(C.QUALIFIED_PROTOCOL)['pace'])


def production_case(directory,data,*,expected):
  from tinygrad import Device
  import tinygrad
  assert not Device._opened_devices
  proto=directory.parent/(directory.name+'-protocol.json');C.write(proto,data,new=True)
  run=C.load(HERE/'agent-expert-paced-stem-run.py','stem_cpu_production_entry')
  args=SimpleNamespace(protocol=proto,protocol_sha=C.sha(proto),output=Path(data['output']),reviewed_parity_sha256=C.CANDIDATE_SHA)
  original_spec=importlib.util.spec_from_file_location
  objects={};events=[];models=[];initial_modules={name:sys.modules.get(name) for name in ('model','benchmark','inputs','native_warp','stream_inputs')}
  def profile(frame,event,arg):
    name=frame.f_code.co_name
    if not ((event=='call' and name=='main') or (event=='return' and name=='__init__')):return
    file=frame.f_code.co_filename.rsplit('/',1)[-1]
    if event=='call' and name=='main' and file=='agent-paced-replay.py':
      events.append(dict(event='canonical_paced_main',reserved_before=list(sorted(p.name for p in args.output.parent.glob('report*')))))
    if event=='return' and name=='__init__' and 'self' in frame.f_locals:
      obj=frame.f_locals['self'];kind=type(obj).__name__
      if (file,kind) in (('agent-live-runtime.py','Output'),('agent-paced-replay.py','Trace'),
          ('agent-paced-replay.py','RGB'),('agent-paced-replay.py','Producer'),('agent-paced-replay.py','Clock'),
          ('agent-live-bridge.py','Preparer'),('agent-warp-fast-sampler.py','Sampler')):objects[kind]=obj
  from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler
  fake=SimpleNamespace(arch='gfx1200',compiler=AMDLLVMCompiler('gfx1200'),renderer=SimpleNamespace(compiler=None))
  class MetadataDevice:
    DEFAULT='USB+AMD:LLVM'
    def __class_getitem__(cls,key):assert key==cls.DEFAULT;return fake
  def observer(previous,doubles):
    def spec(name,location,*a,**kw):
      result=previous(name,location,*a,**kw)
      if result is None or location is None:return result
      path=Path(location).resolve()
      if path.parent!=N.LIVE/'experiments' or path.name not in ('benchmark-pipeline.py','benchmark-fast.py'):return result
      loader=result.loader
      class Loader:
        def create_module(self,spec):return loader.create_module(spec)
        def exec_module(self,module):
          loader.exec_module(module)
          if path.name=='benchmark-fast.py':
            def conversion(shard):
              events.append(dict(event='canonical_factory_conversion_boundary',shard=str(shard),cache=os.environ['ALPAMAYO_FP8_CACHE']))
              raise FactoryBoundary('Stop before real conversion/upload')
            doubles.enter_context(patch.object(module,'safe_load',conversion))
            doubles.enter_context(patch.object(module,'upload',blocked))
          else:
            original=module.initialize
            def initialize(*a,**kw):
              # Only factory computation is replaced; all setup leading here is production code.
              assert args.output.with_suffix('.jsonl').is_file()
              admission=C.read(C.metadata(args.output)['admission'])
              assert admission['complete'] and not admission['gpu_executed'] and admission['factory']['actual_factory_load_executed']
              assert admission['factory']['real_device_open_attempts']==[]
              assert {k:os.environ.get(k) for k in N.storage()}==N.storage()
              assert kw['sources']==C.read(C.CANDIDATE)['source_sha256'] and len(kw['sources'])==253
              source_dir=Path(kw['source_dir']);assert {p.name for p in source_dir.iterdir()}==set(kw['sources'])
              assert all(C.sha(source_dir/k)==v for k,v in kw['sources'].items())
              replay=sys.modules['agent_live_replay']
              assert replay.main.__globals__['normalize'] is replay.normalize
              pad=module.local_module('padded-prefill')
              launch=N.LIVE/'experiments/agent-speculation-run.py'
              assert Path(pad.pad_inputs.__code__.co_filename)==launch and Path(pad.install.__code__.co_filename)==launch
              events.append(dict(event='actual_pipeline_initialize',source_sha256=C.sha(path),snapshot_files=253,
                source_snapshot=str(source_dir),original_pad_hook=True,normalizer_before_copied_globals=True,
                actual_environment={k:os.environ.get(k) for k in N.storage()}))
              with patch.object(tinygrad,'Device',MetadataDevice):return original(*a,**kw)
            module.initialize=initialize
      result.loader=Loader();return result
    return spec
  exception=None
  try:
    with ExitStack() as doubles:
      doubles.enter_context(patch.dict(os.environ));doubles.enter_context(patch.object(sys,'argv',sys.argv[:]))
      doubles.enter_context(patch.object(sys,'path',sys.path[:]))
      doubles.enter_context(patch.object(type(Device),'__getitem__',blocked));doubles.enter_context(patch.object(socket.socket,'connect',blocked))
      doubles.enter_context(patch.object(importlib.util,'spec_from_file_location',observer(original_spec,doubles)))
      previous=sys.getprofile();sys.setprofile(profile)
      try:run.run(args)
      except (FactoryBoundary,FileExistsError,AssertionError) as error:exception=error
      finally:sys.setprofile(previous)
  finally:
    for name,old in initial_modules.items():
      if name=='model' and name in sys.modules:models.append(sys.modules[name])
      if old is None:sys.modules.pop(name,None)
      else:sys.modules[name]=old
  assert importlib.util.spec_from_file_location is original_spec and not Device._opened_devices
  assert exception is not None
  if expected=='collision':
    assert isinstance(exception,FileExistsError) and str(exception)=='Use a new output stem; retain all existing artifacts'
    assert events==[dict(event='canonical_paced_main',reserved_before=['report.agent-expert-live-storage.json'])]
    assert 'Output' not in objects and 'Producer' not in objects and not args.output.exists()
    return dict(original_collision_reproduced=True,gpu_executed=False,canonical_guard_unchanged=True,events=events)
  assert isinstance(exception,FactoryBoundary),repr(exception)
  assert events[0]==dict(event='canonical_paced_main',reserved_before=[])
  assert [e['event'] for e in events]==['canonical_paced_main','actual_pipeline_initialize','canonical_factory_conversion_boundary']
  assert set(objects)=={'Trace','RGB','Producer','Clock','Preparer','Sampler','Output'},set(objects)
  assert len(objects['Trace'].frames)==450 and objects['RGB'].inventory and objects['RGB'].reads==0
  assert objects['Producer'].thread is None and objects['Clock'].anchor is None
  assert objects['Output'].stream.closed and objects['Sampler'].pool is None
  rows=[json.loads(line) for line in args.output.with_suffix('.jsonl').read_text().splitlines()]
  assert len(rows)==1 and rows[0]['kind']=='protocol' and rows[0]['actual_consumer_parity_report_sha256']==C.CANDIDATE_SHA
  assert rows[0]['rgb']['sha256']==objects['RGB'].inventory['sha256']
  side=C.read(C.metadata(args.output)['storage'])
  assert side['constructor_restored'] and side['hooks_restored'] and not side['gpu_executed']
  assert len([e for e in side['storage_events'] if e['event']=='normalize'])==1
  assert [e['path'] for e in side['storage_events'] if e['event']=='cache_open']==[str(N.WEIGHTS/'index.json')]
  assert any(e['event']=='factory_enter' and e['cachedb_constant']==N.storage()['CACHEDB'] for e in side['storage_events'])
  assert objects['Preparer'].warp.__self__.sampler is objects['Sampler']
  assert not args.output.exists()
  return dict(complete=True,gpu_executed=False,production_outer_and_inner_executed=True,canonical_main_executed=True,
    original_collision_guard_passed=True,real_trace_frames=450,real_rgb_inventory=objects['RGB'].inventory,
    real_preparer_and_sampler=True,real_Output=True,Output_closed=True,source_clock_started=False,producer_thread_started=False,
    sampler_workers_closed=True,actual_factory_entry=True,actual_cache_index_open=True,constructor_restored=True,
    mocked_only_factory_compute=True,events=events,storage_events=side['storage_events'])


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--admit',action='store_true')
  args=parser.parse_args();assert not args.output.exists();C.configure()
  if args.admit:result=admit()
  else:
    root=C.ARTIFACTS/('checks-'+args.output.stem);assert not root.exists();root.mkdir()
    files=evidence_files();admission=C.candidate_admission()
    print('CPU original production collision',flush=True)
    collision=production_case(root/'collision',protocol(root/'collision/report.json',files,admission,repair=False),expected='collision')
    print('CPU repaired production factory entry',flush=True)
    actual=production_case(root/'actual',protocol(root/'actual/report.json',files,admission),expected='factory')
    print('CPU occupied output and negative controls',flush=True)
    rejected=[]
    def reject(name,fn):
      try:fn()
      except (AssertionError,ValueError,FileExistsError):rejected.append(name)
      else:raise AssertionError('Bad artifact control accepted: '+name)
    for name in ('report.json','report.foreign','report.launcher.json'):
      folder=root/('occupied-'+name);folder.mkdir();occupied=folder/name;occupied.write_text('{"preserve":true}\n')
      before=C.sha(occupied);output=folder/'report.json';data=protocol(output,files,admission)
      path=root/('occupied-'+name+'-protocol.json');C.write(path,data,new=True)
      runner=C.load(HERE/'agent-expert-paced-stem-run.py','stem_occupied_entry')
      options=SimpleNamespace(protocol=path,protocol_sha=C.sha(path),output=output,reviewed_parity_sha256=C.CANDIDATE_SHA)
      reject('occupied/'+name,lambda:runner.run(options));assert C.sha(occupied)==before
    for name in ('reserved_metadata','outside_directory','duplicate_metadata'):
      output=root/'invalid/report.json';paths=C.metadata(output)
      if name=='reserved_metadata':paths['storage']=output.with_suffix('.metadata.json')
      elif name=='outside_directory':paths['storage']=C.WORK/'foreign.json'
      else:paths['compare']=paths['storage']
      reject(name,lambda:C.validate_paths(output,paths))
    original_source=inspect.getsource(C.load(C.QUALIFIED/'agent-expert-live-compare-run.py','stem_source_negative').run)
    reject('unknown_source',lambda:C.rewrite(original_source+'\n',hashlib.sha256(original_source.encode()).hexdigest(),[]))
    changed=copy.deepcopy(C.read(C.CANDIDATE));changed['rows'][0]['prediction']['generated_token_ids'][0]+=1
    reject('changed_parity',lambda:C.candidate_admission(changed))
    result=dict(complete=True,gpu_executed=False,original_collision=collision,actual_production_path=actual,rejected=rejected,
      candidate_admission=admission,source_sha256={str(p):C.sha(p) for p in HERE.glob('agent-expert-paced-stem-*.py')})
  C.write(args.output,result,new=True);print(dict(complete=True,gpu_executed=False,admit_only=args.admit))


if __name__=='__main__':main()

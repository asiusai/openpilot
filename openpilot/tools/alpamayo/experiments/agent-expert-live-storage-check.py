"""CPU execution of actual transformed entrypoints and launcher environment boundaries."""
import argparse
import copy
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import runpy
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('storage_check_common',HERE/'agent-expert-live-storage-common.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S);N=S.N


def forbidden(*a,**kw):raise AssertionError('GPU/device/network/source-clock execution forbidden')
class Boundary(Exception):pass


def guard_controls():
  env=N.storage()
  events=[dict(event='normalize',actual_environment=env),
    dict(event='factory_enter',actual_environment=env,cachedb_constant=env['CACHEDB']),
    dict(event='cache_open',path=str(N.WEIGHTS/'index.json'),mode='r'),
    dict(event='cache_open',path=str(N.WEIGHTS/'weights.bin'),mode='rb'),
    dict(event='factory_temporary_directory',path=str(Path(env['TMPDIR'])/'cpu-metadata-fixture')),
    dict(event='cache_open',path=str(N.WEIGHTS/'weights.bin'),mode='rb'),
    dict(event='factory_complete',actual_environment=env)]
  S.check_storage(events)
  paced=events+[dict(event='paced_preparer',actual_environment=env,sampler_artifacts=str(S.ARTIFACTS/'hooks/agent-exact-live-artifacts'))]
  S.check_storage(paced,paced=True)
  rejected=[]
  for name in ('empty','normalize_missing','factory_missing','factory_incomplete','old_cache','old_cachedb','wrong_open',
               'missing_bulk_open','write_mode','root_temp','duplicate_normalize','wrong_completion_environment','missing_preparer'):
    value=copy.deepcopy(events)
    if name=='empty':value=[]
    elif name=='normalize_missing':value.pop(0)
    elif name=='factory_missing':value.pop(1)
    elif name=='factory_incomplete':value.pop()
    elif name=='old_cache':value[1]['actual_environment']['ALPAMAYO_FP8_CACHE']='/dev/shm/alpamayo-9060-fp8'
    elif name=='old_cachedb':value[1]['cachedb_constant']='/tmp/foreign-cache.db'
    elif name=='wrong_open':value[2]['path']='/dev/shm/alpamayo-9060-fp8/index.json'
    elif name=='missing_bulk_open':value.pop(5)
    elif name=='write_mode':value[3]['mode']='wb'
    elif name=='root_temp':value[4]['path']='/tmp/foreign-directory'
    elif name=='duplicate_normalize':value.append(value[0])
    elif name=='wrong_completion_environment':value[-1]['actual_environment']['TMPDIR']='/tmp'
    try:S.check_storage(value,paced=name=='missing_preparer')
    except AssertionError:rejected.append(name)
    else:raise AssertionError('Malformed storage audit accepted: '+name)
  return dict(synthetic_guard_fixtures=True,rejected=rejected,actual_io_proof='Separate transformed-entrypoint controls')


def boundary(directory,*,malformed=False,paced=False,candidate=True):
  """Run real main -> normalize -> launcher.main -> mocked initialize -> real cache load."""
  from tinygrad import Device
  assert not Device._opened_devices
  m,manifest=N.live()
  installer=N.load(S.OLD/'agent-expert-live-install.py' if malformed else HERE/'agent-expert-live-storage-install.py','storage_boundary_installer')
  Hooks=installer.factory(N) if malformed else installer.factory(S,pv_candidate=candidate)
  hooks=Hooks('paced' if paced else 'consumer');observed=[];opened=[]
  original_spec=importlib.util.spec_from_file_location
  directory.mkdir();out=directory/'report.json'
  initial_modules={k:sys.modules.get(k) for k in ('model','benchmark','inputs','native_warp','stream_inputs')}
  def initialize(args,**kw):
    observed.append(dict(event='pipeline_initialize',environment={k:os.environ.get(k) for k in N.storage()},
      samples=args.samples,warmup=args.warmup,runs=args.runs,pad=args.pad_prompt_to,steps=args.steps,vision_calibration=args.vision_calibration))
    fast=m.load(m.ROOT/'experiments/benchmark-fast.py','storage_boundary_real_fast')
    class MetadataDevice:
      DEFAULT='USB+AMD:LLVM'
      def __class_getitem__(cls,key):assert key==cls.DEFAULT;return SimpleNamespace(arch='gfx1200')
    original_path=fast.Path
    class WatchedPath(type(Path())):
      def open(path,*a,**kw):
        if path.name in ('index.json','weights.bin'):opened.append(str(path.resolve()))
        return super().open(*a,**kw)
    def conversion(path):
      observed.append(dict(event='actual_factory_conversion_boundary',shard=str(path),cache=os.environ['ALPAMAYO_FP8_CACHE']))
      raise Boundary('Canonical cache index admitted; stop before conversion/upload')
    import model
    previous=model.EdgeModel.__init__
    try:
      with patch.object(fast,'Device',MetadataDevice),patch.object(fast,'safe_load',conversion),patch.object(fast,'upload',forbidden):
        if malformed:
          with patch.object(fast,'Path',WatchedPath):model.EdgeModel.__init__(object(),args.checkpoint,precision='fp8')
        else:model.EdgeModel.__init__(object(),args.checkpoint,precision='fp8')
    finally:
      if malformed:model.EdgeModel.__init__=fast.original_init
      else:assert model.EdgeModel.__init__ is previous
      assert fast.Path is original_path
  def import_observer(previous):
    def spec(name,location,*a,**kw):
      result=previous(name,location,*a,**kw)
      if result is None or location is None or Path(location).resolve()!=m.ROOT/'experiments/benchmark-pipeline.py':return result
      loader=result.loader
      class Loader:
        def create_module(self,spec):return loader.create_module(spec)
        def exec_module(self,module):
          loader.exec_module(module);module.initialize=initialize;module.snapshot_sources=lambda *a:({},None)
      result.loader=Loader();return result
    return spec
  # These mocks replace computation and scheduling only, after real normalization.
  class Dummy:
    def __init__(self,*a,**kw):self.anchor=None;self.thread=None;self.inventory={};self.counts={};self.prime={}
    def close(self):pass
    def save(self,*a):pass
    def start(self):forbidden()
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ));stack.enter_context(patch.object(sys,'argv',sys.argv[:]))
      stack.enter_context(patch.object(sys,'path',sys.path[:]))
      stack.enter_context(patch.object(type(Device),'__getitem__',forbidden));stack.enter_context(patch.object(socket.socket,'connect',forbidden))
      stack.enter_context(hooks)
      stack.enter_context(patch.object(importlib.util,'spec_from_file_location',import_observer(importlib.util.spec_from_file_location)))
      from transformers import AutoTokenizer
      policy=m.load(m.ROOT/'experiments/agent-speculation-policy.py','storage_cpu_tokenizer_policy')
      stack.enter_context(patch.object(AutoTokenizer,'from_pretrained',lambda *a,**k:SimpleNamespace(encode=lambda *a,**k:list(policy.CHECKPOINT_DRAFT_IDS))))
      if not paced:
        module=m.load(m.ROOT/'experiments/agent-live-replay.py','storage_boundary_replay')
        detached=module.main.__globals__['normalize'] is not module.normalize
        opts=['--invocation',str(m.INVOCATION),'--mode','consumer','--decoder','speculative','--prefill-attention','shuffle-fma',
          '--execute-gpu','--output',str(out),'--compare-reference',str(N.REFERENCE)]
      else:
        module=m.load(m.ROOT/'experiments/agent-paced-replay.py','storage_boundary_paced')
        geometry=N.read(module.SOURCE/'manifest.json')
        trace=SimpleNamespace(frames=[],start=1,end=2,manifest=geometry,provenance={})
        stack.enter_context(patch.object(module,'Trace',lambda *a:trace))
        # model_run also owns a copied globals dictionary after startup transformation.
        stack.enter_context(patch.dict(module.model_run.__globals__,RGB=Dummy,Producer=Dummy,Clock=Dummy))
        stack.enter_context(patch.object(module.bridge,'Store',Dummy));stack.enter_context(patch.object(module.bridge,'Bridge',Dummy))
        actual_local=module.model_run.__globals__['local']
        def local(name):
          value=actual_local(name)
          if name=='agent-live-runtime':value.Output=Dummy
          return value
        stack.enter_context(patch.dict(module.model_run.__globals__,local=local))
        import native_warp
        stack.enter_context(patch.object(native_warp,'library',lambda:None))
        # Keep the original Preparer and ownership wrapper; no native library or worker starts.
        def sampler():hooks.sampler=SimpleNamespace(close=lambda:None,state=lambda:{})
        stack.enter_context(patch.object(hooks,'install_sampler',sampler))
        opts=['--mode','run','--invocation',str(m.INVOCATION),'--consumer-parity-report',str(N.CONSUMER),
          '--hz','10','--start-seconds','35','--seconds','8','--inference-budget-seconds','180',
          '--raw-frames',str(m.RGB),'--warp','native','--output',str(out)]
        detached=False
      try:module.main(opts)
      except Boundary:pass
      else:raise AssertionError('CPU boundary did not stop actual entrypoint')
      assert len(observed)==2 and observed[0]['event']=='pipeline_initialize'
      normalized=N.read(out.with_suffix('.normalized-invocation.json'))
      events=copy.deepcopy(hooks.storage_bindings if malformed else hooks.storage_events)
      if malformed:
        assert detached and events==[] and opened==['/dev/shm/alpamayo-9060-fp8/index.json']
        assert observed[0]['environment']['ALPAMAYO_FP8_CACHE']=='/dev/shm/alpamayo-9060-fp8'
        # Exercise the original final guard after stubbing unrelated coverage prerequisites.
        base=Hooks.__mro__[1];hooks.composition.pv_handle=SimpleNamespace()
        hooks.composition.session=SimpleNamespace(frame=None,state={'prefill':False})
        with patch.object(base,'coverage',lambda self:{}):
          try:hooks.coverage()
          except AssertionError:pass
          else:raise AssertionError('Original malformed empty audit was accepted')
        hooks.composition.session=None
      else:
        assert not detached and observed[0]['environment']==N.storage()
        assert normalized['environment']['ALPAMAYO_FP8_CACHE']==str(N.WEIGHTS)
        assert [x['path'] for x in events if x['event']=='cache_open']==[str(N.WEIGHTS/'index.json')]
        assert len([x for x in events if x['event']=='normalize'])==1
        assert len([x for x in events if x['event']=='paced_preparer'])==int(paced)
        try:S.check_storage(events,paced=paced)
        except AssertionError:pass  # Interrupted factory must not qualify.
        else:raise AssertionError('Partial cache load was incorrectly qualified')
  finally:
    for name,old in initial_modules.items():
      if old is None:sys.modules.pop(name,None)
      else:sys.modules[name]=old
  assert importlib.util.spec_from_file_location is original_spec and not Device._opened_devices
  if not malformed:assert hooks.composition.constructor_restored
  return dict(gpu_executed=False,actual_entrypoint=True,actual_launcher_environment=True,actual_cache_loader=True,
    transformed_main_detached_normalizer=detached,events=events,observed=observed,normalized_environment=normalized['environment'],
    source_clock_started=False,constructor_restored=not malformed and hooks.composition.constructor_restored,
    stopped_before_conversion_or_upload=True,mode='paced' if paced else 'consumer',pv_candidate=candidate)


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--admit',action='store_true')
  args=parser.parse_args();assert not args.output.exists();S.configure();N.paths()
  if args.admit:
    check=N.load(S.OLD/'agent-expert-live-check.py','storage_frozen_cache_check');check.N=N
    result=check.admission()
  else:
    directory=S.ARTIFACTS/('controls-'+args.output.stem);assert not directory.exists();directory.mkdir()
    diagnosis=boundary(directory/'malformed',malformed=True)
    current=boundary(directory/'current',candidate=False)
    candidate=boundary(directory/'candidate',candidate=True)
    paced=boundary(directory/'paced',paced=True)
    # Actual runpy import path and cache extent/hash admission remain independently exercised.
    check=N.load(S.OLD/'agent-expert-live-check.py','storage_independent_cache_check');check.N=N
    admission=check.admission()
    result=dict(complete=True,gpu_executed=False,diagnosis=diagnosis,current=current,candidate=candidate,paced=paced,guard_controls=guard_controls(),
      admission=admission,failed_evidence=S.failed_evidence(),fresh_gpu_equality_established=False)
  result['source_sha256']={str(p):N.sha(p) for p in HERE.glob('agent-expert-live-storage-*.py')}
  S.write(args.output,result,new=True);print({'complete':True,'gpu_executed':False,'admit_only':args.admit})


if __name__=='__main__':main()

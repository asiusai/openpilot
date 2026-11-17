"""Scoped adapters over the unchanged canonical speculative launcher and Consumer."""
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

class PreflightBoundary(Exception): pass

def kv_hashes(values,used):
  import numpy as np
  if len(values)!=28: raise ValueError('All28 KV buffers required')
  result=[]
  for value in values:
    shape=tuple(value.shape)
    if shape[:3]!=(2,1,8) or len(shape)!=5 or not 0<used<=shape[3] or value.dtype.itemsize!=2:
      raise ValueError('Unexpected defined KV prefix')
    bits=np.frombuffer(value._buffer().as_memoryview(allow_zero_copy=False),dtype=np.uint16)
    if bits.size!=np.prod(shape): raise ValueError('KV buffer extent differs')
    prefix=np.ascontiguousarray(bits.reshape(shape)[:,:,:,:used,:])
    result.append(dict(shape=list(prefix.shape),dtype='uint16_bits',sha256=hashlib.sha256(prefix.tobytes()).hexdigest()))
  return result

class Hooks:
  def __init__(self,m,manifest,role,environment,*,preflight=False):
    self.m,self.manifest,self.role,self.environment=m,manifest,role,environment
    self.preflight=preflight;self.stack=ExitStack();self.transforms={};self.startup={};self.details={};self.restored=[]
    self.binds=0;self.factory_calls=0;self.boundary={};self.factory_observation={};self.components=None
    self.warm=m.load(m.STARTUP/'agent-startup-spec-warm.py')
    self.transform=m.load(m.STARTUP/'agent-startup-spec-run.py').transform
    self.installer=m.load(m.EXP/'agent-startup-persist-install.py')

  def patch(self,obj,name,value,*,canonical_wrappers=False):
    before=getattr(obj,name);setattr(obj,name,value)
    def restore():
      current=getattr(obj,name)
      if current is not value:
        if not canonical_wrappers: raise RuntimeError('Hook ownership changed: '+name)
        # The pinned canonical launcher and replay both wrap initialize by assignment.
        # Accept only a verified closure chain which still contains this exact hook.
        allowed={str(p) for p in (self.m.ROOT/'experiments').glob('*.py')}
        allowed.add(str(self.m.STARTUP/'agent-startup-spec-run.py')+':execute_existing_launcher')
        seen=set()
        def contains(fn):
          if fn is value: return True
          if not inspect.isfunction(fn) or fn in seen or fn.__code__.co_filename not in allowed: return False
          seen.add(fn)
          return any(contains(cell.cell_contents) for cell in (fn.__closure__ or ()))
        if not contains(current): raise RuntimeError('Canonical initialize chain lost the scoped factory observer')
      setattr(obj,name,before);self.restored.append(name)
    self.stack.callback(restore)

  def __enter__(self):
    self.previous=importlib.util.spec_from_file_location
    self.stack.enter_context(patch.object(importlib.util,'spec_from_file_location',self.spec));return self
  def __exit__(self,*error): return self.stack.__exit__(*error)

  def spec(self,name,location,*args,**kwargs):
    result=self.previous(name,location,*args,**kwargs)
    if result is None or location is None: return result
    path=Path(location).resolve()
    if path.parent!=self.m.ROOT/'experiments' or path.name not in (
        'agent-live-replay.py','agent-live-runtime.py','benchmark-pipeline.py','benchmark-fast.py'): return result
    if self.m.sha(path)!=self.manifest['files'][str(path)]: raise ValueError('Unsealed factory/live source')
    owner,loader=self,result.loader
    class Loader:
      def create_module(self,spec): return loader.create_module(spec)
      def exec_module(self,module):
        loader.exec_module(module)
        if path.name=='agent-live-replay.py': owner.replay(module)
        elif path.name=='agent-live-runtime.py': owner.runtime(module)
        elif path.name=='benchmark-pipeline.py': owner.pipeline(module)
        else: owner.factory(module)
    result.loader=Loader();return result

  def pipeline(self,module):
    original=module.initialize;owner=self
    def initialize(*args,**kwargs):
      owner.factory_calls+=1
      if owner.preflight:
        effective=args[0]
        owner.boundary=dict(actual_runpy_factory_reached=True,environment={k:os.environ.get(k) for k in owner.environment},
          effective_samples=effective.samples,pad_prompt_to=effective.pad_prompt_to,vision_calibration=effective.vision_calibration,
          warm_current_shapes=effective.warm_current_shapes,checkpoint=str(effective.checkpoint),gpu_executed=False)
        raise PreflightBoundary('Stopped before original factory device access')
      return original(*args,**kwargs)
    self.patch(module,'initialize',initialize,canonical_wrappers=True)

  def factory(self,module):
    original=module.load;owner=self;realpath=module.Path
    class ObservedPath(type(realpath())):
      def open(self,*args,**kwargs):
        if self.name in ('index.json','weights.bin'):
          if self.parent!=owner.m.FP8: raise ValueError('Factory opened a different FP8 cache')
          owner.factory_observation.setdefault('opened',[]).append(str(self))
        return super().open(*args,**kwargs)
    self.patch(module,'Path',ObservedPath)
    def load(*args,**kwargs):
      from tinygrad.helpers import CACHEDB
      if Path(os.environ['ALPAMAYO_FP8_CACHE'])!=owner.m.FP8 or Path(CACHEDB)!=Path(owner.environment['CACHEDB']):
        raise ValueError('Actual factory cache binding differs')
      if any(Path(os.environ[k]).resolve()!=Path(owner.environment[k]).resolve() for k in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME')):
        raise ValueError('Actual factory writable paths differ')
      result=original(*args,**kwargs)
      if set(owner.factory_observation.get('opened',[]))!={str(owner.m.FP8/'index.json'),str(owner.m.FP8/'weights.bin')}:
        raise ValueError('Actual factory did not read both verified FP8 files')
      owner.factory_observation.update(completed=True,CACHEDB=str(CACHEDB),original_full_upload_verification_retained=True)
      return result
    # benchmark-fast installs the original load at import; scope the observer and restore the class on exit.
    self.stack.callback(setattr,module.EdgeModel,'__init__',module.original_init)
    self.patch(module.EdgeModel,'__init__',load)

  def replay(self,module):
    m,manifest=self.m,self.manifest
    m.load(m.ROOT/'experiments/agent-paced-dynamic-policy.py').adapt(module)
    original_dataset,original_normalize=module.Dataset,module.normalize
    class Dataset(original_dataset):
      def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        m.equal(self.provenance,manifest['full_dataset'],'full input qualification')
        self.__dict__.update(m.selected(self).__dict__)
    self.patch(module,'Dataset',Dataset)
    def normalize(invocation,dataset,pipeline,output):
      m.equal(dataset.provenance,manifest['selected_dataset'],'two-query dataset')
      full=SimpleNamespace(samples=manifest['full_samples'],provenance=manifest['full_dataset'])
      normalized,args,protocol=original_normalize(invocation,full,pipeline,output)
      m.equal(protocol,manifest['original_protocol'],'original dynamic recipe')
      normalized=copy.deepcopy(normalized);cmd=normalized['command']
      begin,end=cmd.index('--samples')+1,cmd.index('--initial-noise');cmd[begin:end]=dataset.samples
      cmd[cmd.index('--warmup')+1],cmd[cmd.index('--runs')+1]='0','2'
      normalized['environment'].update(self.environment)
      args=pipeline.parse_args(cmd[2:]);protocol=copy.deepcopy(protocol)
      protocol.update(dataset=dataset.provenance,replay_order=dataset.samples,
        flags=module.tree({k:v for k,v in vars(args).items() if k!='output'}),
        startup_persist_selection=dict(indices=list(m.INDICES),acceptance=[True,False],initial_history='empty'))
      return normalized,args,protocol
    self.patch(module,'normalize',normalize)
    original_complete=module.Audit.complete
    def complete(audit,*args,**kwargs):
      audit.components.model.synchronize()
      audit.pending['defined_kv_prefix_bits']=kv_hashes(audit.components.model.cache,audit.pending['speculation']['final_cache_length'])
      original_complete(audit,*args,**kwargs)
    self.patch(module.Audit,'complete',complete)
    def initialize(c,args,dataset,mode):
      if mode!='consumer' or self.components is not None: raise ValueError('One fresh current-input Consumer initialization required')
      self.components=c
      try:
        warm,details=self.installer.initialize(m,self.warm,c,args,dataset.load(dataset.rows[0]),self.role)
        self.startup.update(warm);self.details.update(details)
      finally: self.startup.update(getattr(c,'_startup_spec',{}))
      return warm
    self.transforms['execute']=self.transform(module,'execute_existing_launcher',[
      ('      audit.install(components)',"      result['startup_spec'] = _persist_initialize(components, values[0], dataset, mode)\n      audit.install(components)"),
      ('len(audit.rows) != 64','len(audit.rows) != 2'),
      ('Qualification requires 64 successfully synchronized predictions','Qualification requires two successfully synchronized predictions')],
      {'_persist_initialize':initialize})
    # Compile main only AFTER replacing Dataset/normalize/execute. Its namespace is copied by the frozen transformer.
    self.transforms['main']=self.transform(module,'main',[('queries=64','queries=2')],{})

  def runtime(self,module):
    original=module.bind_components
    def bind(c,*args,**kwargs):
      if self.binds or c is not self.components or not c._startup_spec['ready'] or c.vision_cache.cache:
        raise ValueError('Bind only the restored warmed current owner with cold logical vision cache')
      if any(getattr(c.decoder,k,[]) for k in ('history','statistics','prefill_draft_records')):
        raise ValueError('Startup consumed live history')
      owner=c.inputs;result=original(c,*args,**kwargs)
      if result.inputs is not owner: raise ValueError('Consumer replaced current input owner')
      self.binds+=1;self.stack.callback(result.close);return result
    self.patch(module,'bind_components',bind)

  def coverage(self):
    if self.binds!=1 or self.factory_calls!=1 or not self.factory_observation.get('completed'):
      raise ValueError('Require one actual factory load, startup and Consumer binding')
    if self.installer.signatures(self.m,self.components)!=self.details['ready_signatures']:
      raise AssertionError('Live predictions constructed or changed a warmed execution plan')
    return dict(factory_loads=self.factory_calls,consumer_bindings=self.binds,live_plan_signatures_unchanged=True,
      factory=self.factory_observation,transforms=self.transforms,**self.details)

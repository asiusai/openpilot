"""Adapt qualified hooks to byte-identical live foundations and original startup."""
from contextlib import ExitStack
import copy
import functools
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def composition_class(common):
  base = common.load(common.HERE/'agent-exact-combined-install.py','exact_live_qualified_installer')
  class LiveComposition(base.Composition):
    def require_path(self,path):
      path = Path(path).resolve()
      qualified = common.c.MLP/'experiments'/path.name
      if path.parent != common.ROOT/'experiments' or str(path) not in self.manifest['files']:
        raise ValueError('Unpinned live foundation import')
      if common.sha(path) != common.sha(qualified) or common.sha(path) != self.manifest['files'][str(path)]:
        raise ValueError('Live foundation differs from qualified composition')
  return LiveComposition


class Hooks:
  def __init__(self, common, manifest, mode, *, device_factory=None, mlp_dispatch=None):
    self.m, self.manifest, self.mode = common, manifest, mode
    self.stack = ExitStack()
    self.composition = composition_class(common)(common.c,manifest,'quality',mode != 'reference',
      device_factory=device_factory,mlp_dispatch=mlp_dispatch)
    self.startup = common.load(common.STARTUP/'agent-startup-spec-run.py','exact_live_startup_run')
    self.warm = common.load(common.STARTUP/'agent-startup-spec-warm.py','exact_live_startup_warm')
    self.policy = common.load(common.ROOT/'experiments/agent-paced-dynamic-policy.py','exact_live_dynamic_policy')
    self.startup_report, self.transforms, self.restorations = {}, {}, []
    self.sampler = None
    self.sampler_calls = self.preparers = self.binds = 0
    self.consumers = []

  def __enter__(self):
    self.stack.enter_context(self.composition)
    original_transform = self.startup.transform
    self.original_transform = original_transform
    def transform(module,name,changes,extra):
      additions = [('len(audit.rows) != 64','len(audit.rows) != 2'),
                   ('Qualification requires 64 successfully synchronized predictions','Qualification requires two successfully synchronized predictions')] if name == 'execute_existing_launcher' else [
                   ("gate['comparison']['queries'] != 64","gate['comparison']['queries'] != 2")]
      return original_transform(module,name,[*changes,*additions],extra)
    self.stack.enter_context(patch.object(self.startup,'transform',transform))
    self.stack.enter_context(self.startup.imports(self.m.ROOT,self.warm,self.startup_report,self.transforms))
    self.stack.enter_context(self.policy.imports())
    self.previous_spec = importlib.util.spec_from_file_location
    self.stack.enter_context(patch.object(importlib.util,'spec_from_file_location',self.spec))
    return self

  def __exit__(self,*error): return self.stack.__exit__(*error)

  def patch(self,obj,name,value):
    before = getattr(obj,name)
    setattr(obj,name,value)
    def restore():
      if getattr(obj,name) is not value: raise RuntimeError(f'Exact live hook conflict: {name}')
      setattr(obj,name,before)
      self.restorations.append(name)
    self.stack.callback(restore)

  def spec(self,name,location,*args,**kwargs):
    result = self.previous_spec(name,location,*args,**kwargs)
    if result is None or location is None: return result
    path = Path(location).resolve()
    targets = {'agent-live-replay.py','agent-live-bridge.py','agent-live-runtime.py'}
    if path.parent != self.m.ROOT/'experiments' or path.name not in targets: return result
    if self.m.sha(path) != self.manifest['files'][str(path)]: raise ValueError('Unpinned live machinery')
    owner, loader = self, result.loader
    class Loader:
      def create_module(self,spec): return loader.create_module(spec)
      def exec_module(self,module):
        loader.exec_module(module)
        if path.name == 'agent-live-replay.py': owner.replay(module)
        elif path.name == 'agent-live-bridge.py': owner.bridge(module)
        else: owner.runtime(module)
    result.loader = Loader()
    return result

  def replay(self,module):
    original_dataset, original_normalize = module.Dataset,module.normalize
    manifest, m = self.manifest,self.m
    class Dataset(original_dataset):
      def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.provenance != manifest['full_dataset']: raise ValueError('Original full dataset changed')
        subset = m.selected(self)
        self.__dict__.update(subset.__dict__)
    self.patch(module,'Dataset',Dataset)
    def normalize(invocation,dataset,pipeline,output):
      if dataset.samples != manifest['selected_samples'] or dataset.provenance != manifest['selected_dataset']:
        raise ValueError('Selected fresh query protocol differs')
      full = SimpleNamespace(samples=manifest['full_samples'],provenance=manifest['full_dataset'])
      normalized,args,protocol = original_normalize(invocation,full,pipeline,output)
      if protocol != manifest['original_protocol']: raise ValueError('Original normalized recipe changed')
      normalized = copy.deepcopy(normalized)
      cmd = normalized['command']
      begin,end = cmd.index('--samples')+1,cmd.index('--initial-noise')
      cmd[begin:end] = dataset.samples
      cmd[cmd.index('--warmup')+1],cmd[cmd.index('--runs')+1] = '0','2'
      args = pipeline.parse_args(cmd[2:])
      protocol = copy.deepcopy(protocol)
      protocol.update(dataset=dataset.provenance,replay_order=dataset.samples,
        flags=module.tree({k:v for k,v in vars(args).items() if k != 'output'}),
        exact_live_selection=dict(indices=list(m.INDICES),expected_acceptance=[True,False],initial_lookup_history='empty'))
      return normalized,args,protocol
    self.patch(module,'normalize',normalize)
    self.patch(module,'compare',m.compare)
    # Hash the valid model prefix after completion. Unused allocation tails have no defined model value.
    original_complete = module.Audit.complete
    def complete(audit,*args,**kwargs):
      audit.components.model.synchronize()
      row = audit.pending
      used = row['speculation']['final_cache_length']
      row['defined_kv_prefix_bits'] = kv_hashes(audit.components.model.cache,used)
      original_complete(audit,*args,**kwargs)
    self.patch(module.Audit,'complete',complete)
    self.transforms['parity_main'] = self.original_transform(module,'main',[('queries=64','queries=2')],{})

  def bridge(self,module):
    if self.mode != 'paced': return
    original = module.Preparer.__init__
    owner = self
    @functools.wraps(original)
    def initialize(preparer,*args,**kwargs):
      if owner.preparers: raise ValueError('Multiple paced preparers rejected')
      bound = inspect.signature(original).bind(preparer,*args,**kwargs)
      bound.apply_defaults()
      warp = bound.arguments['warp']
      source = Path(inspect.getsourcefile(warp)).resolve()
      if source != owner.m.ROOT/'experiments/native_warp.py' or owner.m.sha(source) != owner.m.c.sha(owner.m.c.MLP/'experiments/native_warp.py'):
        raise ValueError('Paced preparer must select the qualified native sampler foundation')
      owner.install_sampler()
      bound.arguments['warp'] = owner.sample
      original(*bound.args,**bound.kwargs)
      if preparer.warp.__self__ is not owner: raise ValueError('Preparer lost sampler ownership')
      owner.preparers += 1
    self.patch(module.Preparer,'__init__',initialize)

  def install_sampler(self):
    if self.sampler is not None: raise ValueError('Duplicate live sampler')
    m = self.m
    implementation = m.load(m.c.SAMPLER/'agent-warp-fast-sampler.py','exact_live_frozen_sampler')
    implementation.ARTIFACTS = m.HERE/'agent-exact-live-artifacts'
    config = m.read(m.c.SAMPLER/'agent-warp-fast-validated-portable-v6.json')['variants'][0]
    self.sampler = implementation.Sampler(config['kind'],config['workers'],Path(config['provenance']['command'][0]).name,
      config['provenance']['native'],config['max_maps'],config['max_bytes'])
    self.stack.callback(self.sampler.close)
    if self.sampler.lib.provenance['library_sha256'] != config['provenance']['library_sha256']:
      raise ValueError('Live sampler binary differs from qualification')

  def sample(self,*args,**kwargs):
    self.sampler_calls += 1
    return self.sampler.calibrated_image(*args,**kwargs)

  def runtime(self,module):
    original = module.bind_components
    owner = self
    @functools.wraps(original)
    def bind(components,*args,**kwargs):
      if (owner.binds or components.model is not owner.composition.model or components.vision_cache is not owner.composition.cache or
          components.decoder.model is not components.model): raise ValueError('Unexpected live model owner')
      state = components._startup_spec
      if not state['ready'] or components.inputs is None or components.vision_cache.cache:
        raise ValueError('Bind only the warmed owner with cold logical image cache')
      if any(getattr(components.decoder,k,[]) for k in ('history','statistics','prefill_draft_records')):
        raise ValueError('Startup changed initial live history')
      before = components.inputs
      result = original(components,*args,**kwargs)
      if result.inputs is not before: raise ValueError('Consumer replaced the warmed input owner')
      owner.binds += 1
      owner.consumers.append(result)
      owner.stack.callback(result.close)
      return result
    self.patch(module,'bind_components',bind)

  def coverage(self):
    if getattr(importlib.util.spec_from_file_location,'__self__',None) is not self:
      raise ValueError('Unexpected outer hook after the canonical launcher restored its imports')
    result = self.composition.coverage()
    if self.mode != 'reference' and self.binds != 1: raise ValueError('Require one actual Consumer owner')
    if self.mode == 'paced' and (self.preparers != 1 or self.sampler_calls <= 10 or self.sampler.misses <= 0):
      raise ValueError('Actual paced preparation did not use the qualified sampler')
    result.update(live_sampler_calls=self.sampler_calls,live_sampler=None if self.sampler is None else self.sampler.state(),
      consumer_bindings=self.binds,preparers=self.preparers)
    return result


def kv_hashes(values,used):
  import numpy as np
  result = []
  if len(values) != 28: raise ValueError('Expected all 28 language KV tensors')
  for value in values:
    shape = tuple(value.shape)
    if len(shape) != 5 or shape[:3] != (2,1,8) or not 0 < used <= shape[3] or value.dtype.itemsize != 2:
      raise ValueError('Unexpected valid KV layout')
    raw = value._buffer().as_memoryview(allow_zero_copy=False)
    bits = np.frombuffer(raw,dtype=np.uint16)
    if bits.size != np.prod(shape): raise ValueError('KV raw buffer includes a different allocation')
    prefix = np.ascontiguousarray(bits.reshape(shape)[:,:,:,:used,:])
    result.append(dict(shape=list(prefix.shape),dtype='uint16_bits',sha256=hashlib.sha256(prefix.tobytes()).hexdigest()))
  return result

"""Initialization-only installation into the actual BlockDecoder TinyJit dictionary."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

class LinkObserver:
  """Read-only Python profiling. Inclusive intervals overlap; never add them."""
  def __init__(self,m,arena_bytes): self.m,self.arena_bytes,self.events,self.active=m,arena_bytes,[],{}
  def profile(self,frame,event,arg):
    if frame.f_code.co_filename!=str(self.m.TG/'tinygrad/runtime/support/hcq2.py'): return
    name=frame.f_code.co_name
    if name not in ('hcq_link','_write_host','bufferize_buf'): return
    if event=='call': self.active[id(frame)]=time.perf_counter()
    elif event=='return' and id(frame) in self.active:
      row=dict(name=name,inclusive_ms=(time.perf_counter()-self.active.pop(id(frame)))*1000)
      if name=='_write_host': row['bytes']=sum(len(data) for _,data in frame.f_locals['writes'])
      if name=='bufferize_buf':
        b=frame.f_locals['b'];row.update(bytes=b.nbytes(),proven_arena_extent=b.nbytes()==self.arena_bytes)
      self.events.append(row)
  def __enter__(self):
    if sys.getprofile() is not None: raise ValueError('Cannot replace an existing profiler')
    sys.setprofile(self.profile);return self
  def __exit__(self,*error): sys.setprofile(None)

class SerialPlans:
  def __init__(self,m,components,args,*,builder=None):
    self.m,self.c,self.args=m,components,args
    self.plans,self.records={},[];self.builder=builder or self.build
    self.original=components.decoder.decode;self.old_instance=vars(components.decoder).get('decode')
    if components.decoder.jits: raise ValueError('Install before any serial graph exists')
    if self.old_instance is not None: raise ValueError('Unexpected existing instance decode hook')
    self.family=m.load(m.EXP/'agent-plan-family-reload-loader.py') if builder is None else None
    self.protocol=m.read(m.FAMILY) if builder is None else None
    self.owners=m.load(m.EXP/'agent-plan-family-reload-diagnostic.py').ClosedOwners(components.model) if builder is None else None

  def build(self,count,inputs):
    options=dict(bulk_index=Path(os.environ['ALPAMAYO_FP8_CACHE'])/'index.json',vocab=self.args.reduced_vocab,checkpoint=self.args.checkpoint)
    if count==1: return self.m.load(self.m.EXP/'agent-plan-reload-loader.py').build(self.c.model,inputs,**options)
    return self.family.build(count,self.c.model,inputs,self.protocol,**options)

  def decode(self,logits,cos,sin,position,step,count):
    if type(count) is not int or count not in range(1,7): raise ValueError('Only original serial counts1..6 are admitted')
    if type(position) is not int or not 0<=position<=2304-count or type(step) is not int or not 0<=step<=64-count:
      raise ValueError('Original symbolic decoder bounds exceeded')
    key=(cos.shape,count,self.c.model.cache[0].shape[3])
    if key[2]!=2304 or cos.shape!=sin.shape: raise ValueError('Unexpected current decoder layout')
    if key not in self.plans:
      if key in self.c.decoder.jits: raise ValueError('Original graph appeared before persistent admission')
      started=time.perf_counter();plan=self.builder(count,(logits,cos,sin))
      row=dict(count=count,key=[list(cos.shape),count,2304],declarative_load_rebind_proof_ms=(time.perf_counter()-started)*1000,
        original_calls=len(plan.document['calls']),scratch_bytes=plan.proof['scratch_bytes'],events=[],calls=0)
      started=time.perf_counter()
      with LinkObserver(self.m,plan.proof['scratch_bytes']) as observed: plan.compile_runtime(parent_gpu_authorized=True)
      row['HCQ_compile_ms']=(time.perf_counter()-started)*1000;row['events'].extend(observed.events)
      if plan.jit.cnt!=2 or plan.jit.fxn is not None: raise AssertionError('Require actual reconstructed CapturedJit at its normal replay state')
      self.plans[key]=plan;self.records.append(row);self.c.decoder.jits[key]=plan.jit
    plan=self.plans[key];plan.guard()
    if self.c.decoder.jits[key] is not plan.jit: raise AssertionError('Raw TinyJit owner changed')
    row=next(r for r in self.records if r['count']==count)
    before=plan.jit.cnt;started=time.perf_counter()
    # Observe actual normal runtime calls; never install runtime monkeypatches.
    with LinkObserver(self.m,plan.proof['scratch_bytes']) as observed:
      result=self.original(logits,cos,sin,position,step,count)
      self.c.model.synchronize()
    row['events'].extend(observed.events);row.setdefault('real_replay_ms',[]).append((time.perf_counter()-started)*1000)
    row['calls']+=1
    if plan.jit.cnt!=before+1: raise AssertionError('Real TinyJit replay did not advance its counter')
    return result

  @contextmanager
  def installing(self):
    decoder=self.c.decoder;decoder.decode=self.decode
    try: yield self
    finally:
      if getattr(decoder.decode,'__self__',None) is not self: raise RuntimeError('Startup decode hook conflict')
      del decoder.decode

  def seal(self,warm):
    decoder=self.c.decoder
    if type(decoder.jits) is not warm.ClosedGraphs or set(decoder.jits)!=set(self.plans) or len(self.plans)!=6:
      raise AssertionError('Original full warm must seal all six actual persistent members')
    for key,plan in self.plans.items():
      plan.guard();jit=decoder.jits[key]
      if jit is not plan.jit or jit.cnt<3 or len(jit.captured._linears)!=2:
        raise AssertionError('Both actual input/output-alias variants must link through original warm calls')
      if sorted(len(alias) for alias in jit.captured._linears)!=[0,1]: raise AssertionError('Unexpected alias contract')
    plans=self.plans;owners=self.owners
    class GuardedGraphs(warm.ClosedGraphs):
      def __getitem__(self,key):
        result=super().__getitem__(key)
        if owners is not None: owners.guard()
        plans[key].guard()
        if result is not plans[key].jit: raise AssertionError('Persistent raw TinyJit changed')
        return result
    decoder.jits=GuardedGraphs(decoder.jits)
    return dict(counts=list(range(1,7)),members=self.records,actual_raw_TinyJit_values=True,
      original_decode_restored='decode' not in vars(decoder),both_alias_variants=True,
      timing='Instrumented inclusive Python intervals; linking includes host writes and allocation; intervals overlap',
      scope='Serial construction only. Original replay, alias, verifier, selector and full real prefill restoration remain active')

def signatures(m,c):
  signature=m.load(m.EXP/'agent-startup-cache-capture.py').graph_signature
  result={}
  for owner,name in ((c.inputs,'selective_jits'),(c.vision_cache,'encoders'),(c.vision_cache,'stitchers'),
                     (c.decoder,'jits'),(c.decoder,'verifiers'),(c.decoder,'select_jits'),(c.model,'trajectory_jits')):
    result[name]=[{ 'key':repr(key),'signature':signature(jit)} for key,jit in getattr(owner,name).items()]
  result['choose_jit']=signature(c.decoder.choose_jit)
  result['prefill']=signature(c.model.padded_prefill.jit)
  return result

def initialize(m,warm,c,args,arrays,role):
  started=time.perf_counter();install=None;serial={}
  if role=='persisted':
    install=SerialPlans(m,c,args)
    with install.installing(): value=warm.initialize(c,args,arrays,source_root=m.ROOT,stage_limit_seconds=400)
    serial=install.seal(warm)
  elif role=='ordinary': value=warm.initialize(c,args,arrays,source_root=m.ROOT,stage_limit_seconds=400)
  else: raise ValueError('Unreviewed startup role')
  ready_ms=(time.perf_counter()-started)*1000
  plans=signatures(m,c)
  if install is not None: install.owners.guard()
  return value,dict(initializer_and_serial_install_ms=ready_ms,ready_with_plan_audit_ms=(time.perf_counter()-started)*1000,
    serial=serial,ready_signatures=plans)

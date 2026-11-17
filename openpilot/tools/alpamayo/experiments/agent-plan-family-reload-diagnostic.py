"""Original-versus-reloaded family proof; production mutation and poison paths."""
import gc
import json
import os
from pathlib import Path
import resource
import time

EXP=Path(__file__).resolve().parent

class ClosedOwners:
  def __init__(self,model):
    self.model=model;self.saved=self.current()

  def current(self):
    from tinygrad.uop.ops import UOp,Ops
    tensors={'weight/'+k:v for k,v in self.model.weights.items()}
    tensors.update({'scale/'+k:v for k,v in self.model.weight_scales.items()})
    tensors['mapping']=self.model.output_token_mapping
    tensors.update({'kv/'+str(i):v for i,v in enumerate(self.model.cache)})
    return {name:(t.uop.base,tuple(t.shape),t.dtype,t.uop.substitute({t.uop.base:UOp(Ops.NOOP)}).key) for name,t in tensors.items()}

  def guard(self):
    if self.current()!=self.saved: raise AssertionError('A model weight/scale/mapping/KV allocation or view changed')

class Mutations:
  def __init__(self,contract,model,decoder,loaded,original,inputs,first,snapshots,prompt,count):
    self.contract,self.model,self.decoder,self.loaded,self.original=contract,model,decoder,loaded,original
    self.logits,self.cos,self.sin=inputs;self.first,self.snapshots,self.prompt,self.count=first,snapshots,prompt,count
    if len(model.cache)!=28 or len(snapshots)!=28 or first.shape!=(1,1,2112): raise ValueError('Require all28 KV owners and actual logits shape')
    self.original_ret=original.captured.ret;self.arena_bytes=loaded.proof['scratch_bytes']
    self.allocation=self.identity();self.poison=contract.local(EXP/'agent-plan-poison-copy.py').poison_copy

  def identity(self): return tuple(id(t.uop.base.buffer.base) for t in self.model.cache)

  @staticmethod
  def exact(a,b):
    from tinygrad import dtypes
    if a.shape!=b.shape or a.dtype!=b.dtype: raise AssertionError('Tensor shape/dtype differs')
    dt={1:dtypes.uint8,2:dtypes.uint16,4:dtypes.uint32}[a.dtype.itemsize]
    if (a.bitcast(dt)!=b.bitcast(dt)).any().item(): raise AssertionError('Tensor bits differ')
    if not a.isfinite().all().item() or not b.isfinite().all().item(): raise AssertionError('Nonfinite tensor in exact comparison')

  def reset(self,mutate=False):
    for current,saved in zip(self.model.cache,self.snapshots,strict=True): current.assign(saved).realize()
    if mutate:
      prefix=self.model.cache[0][:,:,:,:self.prompt,:];prefix.assign(prefix+0.125).realize()
    self.model.synchronize()
    if self.identity()!=self.allocation: raise AssertionError('KV reset/mutation replaced an allocation')

  @staticmethod
  def clone(values): return tuple(t.clone().realize() for t in values)

  def same(self,actual,saved):
    for a,b in zip(actual,saved,strict=True): self.exact(a,b)

  def variables(self,position,step):
    from tinygrad.uop.ops import UOp
    limits=self.contract.family().bounds(self.count)
    return (UOp.variable('block_position',0,limits['block_position']['max']).bind(position),
            UOp.variable('block_step',0,limits['block_step']['max']).bind(step))

  def active_storage(self,jit,source,position,step):
    from tinygrad.engine.jit import _prepare_jit_inputs
    from tinygrad.uop.ops import Ops
    inputs,*_=_prepare_jit_inputs((source,self.cos,self.sin,*self.variables(position,step)),{})
    alias=tuple(u for u in inputs if u in jit.captured._jit_bufs)
    linked,staging=jit.captured._linears[alias]
    arenas={u.buffer.base for u in linked.toposort() if u.op is Ops.BUFFER and u.buffer.base.nbytes==self.arena_bytes
            and u.buffer.base.device==self.first.device}
    if len(arenas)!=1: raise AssertionError('Cannot identify exactly one active proven scratch arena')
    arena,=arenas
    owners=(*self.model.cache,self.logits,self.cos,self.sin,*jit.captured.ret)
    if arena in {t.uop.base.buffer.base for t in owners}: raise AssertionError('Arena overlaps an input/KV/output owner')
    if set(staging)!=set(alias): raise AssertionError('Alias staging owners differ')
    for source_uop,target in staging.items():
      if target.buffer.base is source_uop.buffer.base or target.nbytes()!=source_uop.nbytes() or target.dtype!=source_uop.dtype:
        raise AssertionError('Staging must be an independent complete copy')
    return arena,staging,alias

  def run(self,which,alias,state,position,step,mutate,pattern):
    from tinygrad import Tensor
    from tinygrad.engine.jit import _prepare_jit_inputs
    if (which,pattern) not in (('original',0xA5),('reloaded',0x5A)): raise ValueError('Preserve distinct poison patterns')
    jit,ret,invoke=(self.original,self.original_ret,
      lambda x:self.decoder.decode(x,self.cos,self.sin,position,step,self.count)) if which=='original' else \
      (self.loaded.jit,self.loaded.ret,lambda x:self.loaded(x,self.cos,self.sin,position,step))
    self.reset(mutate);source=ret[0] if alias else state
    if alias: ret[0].assign(state).realize()
    inputs,*_=_prepare_jit_inputs((source,self.cos,self.sin,*self.variables(position,step)),{})
    alias_key=tuple(u for u in inputs if u in jit.captured._jit_bufs)
    if alias_key not in jit.captured._linears:
      Tensor.realize(*invoke(source));self.model.synchronize();self.reset(mutate)
    arena,staging,actual_alias=self.active_storage(jit,source,position,step)
    if bool(actual_alias)!=alias or len(staging)!=int(alias): raise AssertionError('Original alias contract differs')
    self.model.synchronize()
    buffers=[arena,*[t.uop.base.buffer for t in ret],*[u.buffer for u in staging.values()]]
    for buffer in buffers: self.poison(buffer,pattern)
    if alias: ret[0].assign(state).realize()
    Tensor.realize(*invoke(source));self.model.synchronize()
    if self.identity()!=self.allocation: raise AssertionError('Poison replay replaced KV storage')
    return ret,dict(scratch_bytes=arena.nbytes,output_bytes=sum(t.numel()*t.dtype.itemsize for t in ret),
      staging_bytes=sum(u.buffer.nbytes for u in staging.values()),pattern=pattern,alias=alias,
      aliased_logits_input_restored_after_poison=alias)

  def exercise(self,result,event):
    from tinygrad import Tensor,dtypes
    first=self.first
    mask=(Tensor.arange(2112).to(first.device)==2111).reshape(1,1,2112)
    changed=mask.where(first.abs().max()+256,first).cast(dtypes.bfloat16).realize()
    if not (changed.bitcast(dtypes.uint16)!=first.bitcast(dtypes.uint16)).any().item(): raise AssertionError('Logit mutation ineffective')
    limits=self.contract.family().bounds(self.count)
    states=[('current',first,self.prompt,0,False),('changed_logits_KV_position_step',changed,self.prompt+3,3,True),
            ('last_legal_position_step',changed,limits['block_position']['max'],limits['block_step']['max'],True)]
    for name,state,position,step,mutate in states:
      for alias in (False,True):
        reference,original_poison=self.run('original',alias,state,position,step,mutate,0xA5)
        expected=self.clone(reference);expected_kv=self.clone(self.model.cache)
        candidate,reload_poison=self.run('reloaded',alias,state,position,step,mutate,0x5A)
        self.same(candidate,expected);self.same(self.model.cache,expected_kv)
        del expected,expected_kv
        result['cases'].append(dict(name=name,position=position,step=step,mutated_KV=mutate,alias=alias,
          output_bits_equal=True,all28_KV_bits_equal=True,finite=True,original_poison=original_poison,reload_poison=reload_poison))
      event('passed_'+name)
    if len(self.original.captured._linears)!=2 or len(self.loaded.jit.captured._linears)!=2:
      raise AssertionError('Both alias variants must be exercised')

def compare_original(contract,model,inputs,original,record,loaded,options):
  """Compare fresh original calls, complete owner registry, views, and arena."""
  from tinygrad.uop.ops import Ops
  audit=contract.local(EXP/'agent-plan-bind-audit.py');family=contract.family()
  registry=audit.model_registry(model,inputs,original.captured.ret,**options)
  rows=[]
  for index,outer in enumerate(record['linear'].src):
    call=outer.without_after;p=call.body
    if p.op is not Ops.PROGRAM: raise AssertionError('Fresh original has an unreviewed operation')
    rows.append(dict(index=index,program_key=p.key.hex(),arguments=[registry.expression(u) for u in call.src[1:]],
      outputs=list(p.arg.outs),inputs=list(p.arg.ins),name=p.src[0].arg.name,
      ordered_dependencies=[u.op.name for u in outer.src[1:]] if outer.op is Ops.AFTER else []))
  if rows!=loaded.document['calls'] or registry.describe()!=loaded.document['registry']:
    raise AssertionError('Fresh original PROGRAM/argument/owner/arena sequence differs')
  if [c.body.key for c in loaded.linear.src]!=[c.body.key for c in record['linear'].src]:
    raise AssertionError('Reload changed the numerical PROGRAM sequence')
  expected_info=contract.local(EXP/'agent-plan-reload-loader.py').contract_json(original.captured.expected_input_info)
  if expected_info!=loaded.document['input_contract'] or original.captured.expected_names!=[0,1,2]:
    raise AssertionError('Original input contract differs')
  actual_views={n:family.view_recipe(t) for n,t in zip(('return/logits','return/token'),original.captured.ret,strict=True)}
  loaded_views={n:family.view_recipe(t) for n,t in zip(('return/logits','return/token'),loaded.ret,strict=True)}
  if actual_views!=loaded_views: raise AssertionError('Original return view differs')

def session(context,contract,protocol,started):
  from tinygrad import Tensor
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.helpers import GlobalCounters
  model,decoder,logits,dc,ds=(context[k] for k in ('model','decoder','logits','dc','ds'))
  report,args,output=(context[k] for k in ('report','args','output'))
  prompt,first,reset=(context[k] for k in ('prompt','first','reset_kv'))
  loader=contract.local(EXP/'agent-plan-family-reload-loader.py');audit=contract.local(EXP/'agent-plan-bind-audit.py')
  options=dict(bulk_index=Path(os.environ['ALPAMAYO_FP8_CACHE'])/'index.json',vocab=args.reduced_vocab,checkpoint=args.checkpoint)
  result=dict(schema=1,qualified=False,ready_for_live_use=False,requested_counts=protocol['counts'],completed_counts=[],
    current_inputs=1,future_inputs=0,count1_reexecuted=False,token_policy_changed=False,members={},stages=[],
    constructor_derivation=loader.Prepared.derivation,snapshot_bytes=sum(t.numel()*t.dtype.itemsize for t in context['snapshots']))
  report['plan_family']=result;report['capture']=dict(qualified_numerical_identities_equal=False)
  def event(name,count):
    model.synchronize();row=dict(name=name,count=count,elapsed_ms=(time.perf_counter()-started)*1000,
      live_tensor_bytes=GlobalCounters.mem_used,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['stages'].append(row);print(json.dumps({'agent_plan_family_reload_stage':row}),flush=True)
  for count in protocol['counts']:
    if time.perf_counter()-started>=protocol['limits']['start_next_count_before_seconds']:
      result['budget_stopped_before_count']=count;break
    if GlobalCounters.mem_used>=protocol['limits']['live_tensor_bytes']: raise MemoryError('Live tensor memory budget exceeded')
    key=(dc.shape,count,2304)
    if key in decoder.jits: raise ValueError('Original member exists before reload')
    row=dict(count=count,qualified=False,cases=[],loaded_before_original_same_count=False);result['members'][str(count)]=row
    reset();loaded=loader.build(count,model,(logits,dc,ds),protocol,**options)
    loaded.compile_runtime(parent_gpu_authorized=True)
    reload_first=loaded(logits,dc,ds,prompt,0);Tensor.realize(*reload_first);model.synchronize()
    saved=Mutations.clone(reload_first);saved_kv=Mutations.clone(model.cache)
    if key in decoder.jits: raise AssertionError('Original JIT appeared before first persisted execution')
    row['loaded_before_original_same_count']=True;event('fresh_member_reloaded_before_original',count)
    reset();eager=decoder.decode(logits,dc,ds,prompt,0,count);Tensor.realize(*eager)
    for a,b in zip((*eager,*model.cache),(*saved,*saved_kv),strict=True): Mutations.exact(a,b)
    reset();limits=contract.family().bounds(count)
    from tinygrad.uop.ops import UOp
    pos=UOp.variable('block_position',0,limits['block_position']['max']).bind(prompt)
    step=UOp.variable('block_step',0,limits['block_step']['max']).bind(0)
    inputs,*_=_prepare_jit_inputs((logits,dc,ds,pos,step),{})
    with audit.capture_boundary(inputs) as boundary: captured=decoder.decode(logits,dc,ds,prompt,0,count)
    Tensor.realize(*captured)
    for a,b in zip((*captured,*model.cache),(*saved,*saved_kv),strict=True): Mutations.exact(a,b)
    del saved,saved_kv
    original=decoder.jits[key];compare_original(contract,model,(logits,dc,ds),original,boundary,loaded,options)
    event('fresh_original_eager_capture_and_fullKV_exact',count)
    mutations=Mutations(contract,model,decoder,loaded,original,(logits,dc,ds),first,context['snapshots'],prompt,count)
    mutations.exercise(row,lambda name:event(name,count))
    loaded.guard()
    row.update(qualified=True,original_alias_variants=2,reloaded_alias_variants=2,numerical_PROGRAMs_exact=len(loaded.document['calls']),
      proof_sha256=protocol['members'][str(count)]['proof_sha256'],scratch_bytes=loaded.proof['scratch_bytes'],
      original_and_reload_owner_contract_exact=True,closed_owner_identity_retained=True)
    result['completed_counts'].append(count);event('member_complete',count)
    del decoder.jits[key]
    del loaded,original,mutations,reload_first,eager,captured,boundary
    gc.collect()
  complete=result['completed_counts']==protocol['counts']
  result.update(all_requested_members_passed=complete,whole_family_2_through_6_passed=complete and protocol['counts']==[2,3,4,5,6])
  # The frozen wrapper's legacy gate describes identity, not whole-family qualification.
  report['capture']=dict(qualified_numerical_identities_equal=all(r.get('qualified',False) for r in result['members'].values()))

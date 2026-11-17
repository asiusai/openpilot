"""Parent-only count1 persistent-reload diagnostic using the frozen actual factory."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
PROTOCOL=WORK/'agent-plan-reload-protocol2.json'

def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def capture_one(c,args,arrays,report,output,expected,reference,replay):
  import numpy as np
  import torch
  from tinygrad import Tensor,dtypes
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.uop.ops import Ops,UOp
  loader=local(EXP/'agent-plan-reload-loader.py');audit=local(EXP/'agent-plan-bind-audit.py')
  warm=local(WORK/'startup-cache-parent-source/agent-startup-spec-warm.py')
  if sha(warm.__file__)!='3b7f2d7d5e1888243271fa34b7bb8f91d7b7f7c5dd3789503bec1071a3e2900c':
    raise ValueError('Qualified restoration helper changed')
  model,decoder=c.model,c.decoder
  if c.inputs is not None or decoder.jits or c.startup is not None or any(c.startup_active): raise ValueError('Require fresh factory owners')
  if args.vision_calibration is not None or args.pad_prompt_to!=2048 or args.max_tokens!=64: raise ValueError('Require qualified dynamic vision recipe')
  history=copy.deepcopy({name:getattr(decoder,name) for name in ('history','statistics','prefill_draft_records')})
  data={k:list(v) if k=='nav_text' else torch.from_numpy(v.copy()) for k,v in arrays.items()}
  config,tokenizer,prepared=c.adapter.prepare(data)
  if config is not c.config or tokenizer is not c.tokenizer: raise ValueError('Initialized input owners changed')
  packed=c.adapter.native_inputs(prepared);prepared,packed=c.pad_inputs(prepared,packed,args.pad_prompt_to)
  if not prepared.get('prefill_draft_eligible') or not np.all(prepared['tokenized_data']['image_grid_thw'].numpy()==(1,20,36)):
    raise ValueError('Unreviewed first-current-input layout')
  if replay.tree(prepared)!=reference['prepared'] or replay.tree(packed)!=reference['packed']:
    raise AssertionError('First actual model input differs from qualified evidence')
  c.vision_cache.prepare(prepared)
  owner=c.input_class(c.settings,config,prepared,packed);owner.selective_cache=c.vision_cache;c.inputs=owner
  if len(owner.batches)!=10: raise ValueError('Require ten complete current images')
  pixels,positions,ids,cos,sin,dc,ds,*_=owner.upload(c.settings.initial_noise)
  embedding=model.encode(pixels,positions,ids,owner.batches,owner.spans);prompt=int(prepared['actual_prompt_length'])
  prefill=lambda:model.padded_prefill(embedding,cos,sin,c.settings.max_tokens,prompt)
  logits=prefill();Tensor.realize(logits,dc,ds);model.synchronize()
  snapshots=[t.clone().realize() for t in model.cache]
  first=logits.clone().realize();payload=model.padded_prefill.last_targets.clone().realize()
  identity=lambda:tuple(id(t.uop.base.buffer.base) for t in model.cache)
  original_identity=identity()
  def exact(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype: raise AssertionError('Tensor shape/dtype differs')
    dt={1:dtypes.uint8,2:dtypes.uint16,4:dtypes.uint32}[a.dtype.itemsize]
    if (a.bitcast(dt)!=b.bitcast(dt)).any().item(): raise AssertionError('Actual tensor bits differ')
  def reset(mutate=False):
    for current,saved in zip(model.cache,snapshots,strict=True): current.assign(saved).realize()
    if mutate:
      prefix=model.cache[0][:,:,:,:prompt,:]
      prefix.assign(prefix+0.125).realize()
    model.synchronize()
    if identity()!=original_identity: raise AssertionError('KV mutation replaced storage')
  def clone(values): return tuple(t.clone().realize() for t in values)
  def same(values,saved):
    for a,b in zip(values,saved,strict=True): exact(a,b)
  def kv_equal(saved):
    for current,value in zip(model.cache,saved,strict=True): exact(current,value)
  def variables(position,step):
    return UOp.variable('block_position',0,2303).bind(position),UOp.variable('block_step',0,63).bind(step)
  start=time.perf_counter()
  report['plan_reload']=dict(qualified=False,current_inputs=1,future_inputs=0,predictions_published=0,ready_for_live_use=False,
    frozen_capture_sha256='7fe7e733e3c53f5d7d67c7e5959d2f01896110c41263f75714d85f179cbaa275',cases=[],stages=[])
  result=report['plan_reload']
  def stage(name):
    model.synchronize();row=dict(name=name,elapsed_ms=(time.perf_counter()-start)*1000)
    result['stages'].append(row);print(json.dumps({'agent_plan_reload_stage':row}),flush=True)
  try:
    # This loads yesterday's declarative plan before any current count1 Python
    # eager/capture execution exists. All owners and device addresses are fresh.
    loaded=loader.build(model,(logits,dc,ds),bulk_index=Path(os.environ['ALPAMAYO_FP8_CACHE'])/'index.json',
      vocab=args.reduced_vocab,checkpoint=args.checkpoint)
    result['proof']=dict(calls=loaded.proof['calls'],programs=loaded.proof['programs'],scratch_bytes=loaded.proof['scratch_bytes'],
      scratch_read_checks=len(loaded.proof['scratch_read_checks']),scratch_instruction_bounds=len(loaded.proof['scratch_instruction_bounds']))
    if decoder.jits: raise AssertionError('Original count1 unexpectedly exists before reload')
    loaded.compile_runtime(parent_gpu_authorized=True)
    reload_first=loaded(logits,dc,ds,prompt,0);Tensor.realize(*reload_first);model.synchronize()
    saved_first=clone(reload_first);saved_kv=clone(model.cache);stage('fresh_plan_reloaded_before_original_count1')
    reset();eager=decoder.decode(logits,dc,ds,prompt,0,1);Tensor.realize(*eager)
    same(eager,saved_first);kv_equal(saved_kv);reset()
    input_uops,*_=_prepare_jit_inputs((logits,dc,ds,*variables(prompt,0)),{})
    with audit.capture_boundary(input_uops) as boundary: captured=decoder.decode(logits,dc,ds,prompt,0,1)
    Tensor.realize(*captured);same(captured,saved_first);kv_equal(saved_kv)
    del saved_kv,saved_first
    key=(dc.shape,1,2304);original=decoder.jits[key]
    original_ret=original.captured.ret
    reset();original_ret[0].assign(first).realize()
    Tensor.realize(*decoder.decode(original_ret[0],dc,ds,prompt,0,1));model.synchronize()
    registry=audit.model_registry(model,(logits,dc,ds),original_ret,
      bulk_index=Path(os.environ['ALPAMAYO_FP8_CACHE'])/'index.json',vocab=args.reduced_vocab,checkpoint=args.checkpoint)
    report['capture']=audit.export_capture(boundary,original,registry,output.parent/'agent-plan-reload-original',expected)
    if not report['capture']['qualified_numerical_identities_equal']: raise AssertionError('Fresh original PROGRAM identities changed')
    if [call.body.key for call in loaded.linear.src]!=[call.body.key for call in boundary['linear'].src]:
      raise AssertionError('Reloaded numerical PROGRAM sequence differs from fresh original')
    stage('fresh_original_eager_capture_and_KV_equal')

    def active_storage(jit,source,position,step):
      inputs,*_=_prepare_jit_inputs((source,dc,ds,*variables(position,step)),{})
      alias=tuple(u for u in inputs if u in jit.captured._jit_bufs)
      linked,staging=jit.captured._linears[alias]
      candidates={u.buffer.base for u in linked.toposort() if u.op is Ops.BUFFER and u.buffer.base.nbytes==10499328
                  and u.buffer.base.device=='AMD'}
      if len(candidates)!=1: raise AssertionError('Cannot identify the active linked scratch arena exactly')
      arena,=candidates
      forbidden={t.uop.base.buffer.base for t in (*model.cache,logits,dc,ds,*jit.captured.ret)}
      if arena in forbidden: raise AssertionError('Scratch overlaps current owners')
      return arena,staging,alias

    def poisoned_run(which,alias,state,position,step,mutate,pattern):
      jit,ret,invoke=(original,original_ret,lambda x:decoder.decode(x,dc,ds,position,step,1)) if which=='original' else \
        (loaded.jit,loaded.ret,lambda x:loaded(x,dc,ds,position,step))
      reset(mutate)
      source=ret[0] if alias else state
      if alias: ret[0].assign(state).realize()
      # Link the normal alias path before poisoning. Its first execution is discarded.
      inputs,*_=_prepare_jit_inputs((source,dc,ds,*variables(position,step)),{})
      alias_key=tuple(u for u in inputs if u in jit.captured._jit_bufs)
      if alias_key not in jit.captured._linears:
        Tensor.realize(*invoke(source));model.synchronize();reset(mutate)
      arena,staging,actual_alias=active_storage(jit,source,position,step)
      if bool(actual_alias)!=alias or len(staging)!=int(alias): raise AssertionError('Output alias staging contract differs')
      model.synchronize()
      buffers=[arena,*[t.uop.base.buffer for t in ret],*[u.buffer for u in staging.values()]]
      for buffer in buffers:
        buffer.ensure_allocated().copyin(memoryview(bytes([pattern])*buffer.nbytes))
      # An aliased output is also the required current input. Restore that input
      # after poisoning; token output, arena, and staging remain poisoned.
      if alias: ret[0].assign(state).realize()
      Tensor.realize(*invoke(source));model.synchronize()
      return ret,dict(scratch_bytes=arena.nbytes,output_bytes=sum(t.numel()*t.dtype.itemsize for t in ret),
        staging_bytes=sum(u.buffer.nbytes for u in staging.values()),pattern=pattern,alias=alias,
        aliased_logits_input_restored_after_poison=alias)

    mask=(Tensor.arange(2112,device='AMD')==2111).reshape(1,1,2112)
    changed=mask.where(first.abs().max()+256,first).cast(dtypes.bfloat16).realize()
    if not (changed.bitcast(dtypes.uint16)!=first.bitcast(dtypes.uint16)).any().item(): raise AssertionError('Logit mutation was ineffective')
    states=[('current',first,prompt,0,False),('changed_logits_KV_position_step',changed,prompt+3,3,True),
            ('last_position_last_step',changed,2303,63,True)]
    for index,(name,state,position,step,mutate) in enumerate(states):
      for alias in (False,True):
        reference_values,original_poison=poisoned_run('original',alias,state,position,step,mutate,0xA5)
        expected_values=clone(reference_values);expected_kv=clone(model.cache)
        candidate_values,reload_poison=poisoned_run('reloaded',alias,state,position,step,mutate,0x5A)
        same(candidate_values,expected_values);kv_equal(expected_kv)
        del expected_values,expected_kv
        result['cases'].append(dict(name=name,position=position,step=step,mutated_KV=mutate,alias=alias,
          prediction_bits_equal=True,all28_KV_bits_equal=True,original_poison=original_poison,reload_poison=reload_poison))
      stage('passed_'+name)
    if len(original.captured._linears)!=2 or len(loaded.jit.captured._linears)!=2: raise AssertionError('Both alias variants must be exercised')
    result.update(fresh_process_original_vs_reloaded_exact=True,loaded_before_original_count1=True,
      original_alias_variants=2,reloaded_alias_variants=2,numerical_PROGRAMs_exact=765,token_policy_changed=False)
  finally:
    restored=warm.restore_prefill(model.cache,snapshots,prefill,identity=identity,equal=exact,
      copy_into=lambda target,saved:target.assign(saved).realize())
    exact(restored,first);exact(model.padded_prefill.last_targets,payload);model.synchronize()
    if identity()!=original_identity or c.inputs is not owner: raise AssertionError('Actual current input/KV owner changed')
    if history!={name:getattr(decoder,name) for name in history}: raise AssertionError('Generation history changed')
    result.update(full_KV_restored=True,first_logits_restored=True,target_payload_restored=True,current_owner_retained=True,
      allocation_identity_retained=True,generation_history_unchanged=True)
  result['qualified']=True  # This count1 diagnostic only; no whole-model or live adoption.
  stage('count1_diagnostic_complete_real_prefill_restored')

class PreflightComplete(Exception): pass

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache-db',type=Path)
  parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
  if not args.output.resolve().is_relative_to(WORK) or args.output.exists(): raise ValueError('Use a new WORK output')
  if args.preflight and args.execute_gpu: raise ValueError('Preflight cannot execute hardware')
  # Set qualified numeric ContextVar defaults before importing tinygrad. Storage
  # paths stay on this run's WORK directory. The original launcher repeats them.
  environment=json.loads((WORK/'agent-plan-bind-capture1/report.json').read_text())['protocol']['environment']
  for key,value in environment.items():
    if key in ('CACHEDB','TMPDIR','PYTHONDONTWRITEBYTECODE'): continue
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=value
  sys.dont_write_bytecode=True
  loader=local(EXP/'agent-plan-reload-loader.py')
  if not args.preflight:
    if not args.execute_gpu or args.cache_db is None: raise ValueError('Parent GPU flag and WORK compiler cache required')
    protocol=json.loads(args.protocol.read_text());loader.source_gate(protocol)
  frozen=local(EXP/'agent-plan-bind-run.py');frozen.capture_one=capture_one
  report_path=args.output if not args.preflight else args.output.with_suffix('.factory-preflight.json')
  command=['--output',str(report_path),'--cache-db',str(args.cache_db or Path(os.environ['CACHEDB'])),'--execute-gpu']
  if args.preflight:
    original_local=frozen.local;observed={}
    def imports(path):
      module=original_local(path)
      if Path(path).name=='benchmark-pipeline.py':
        def initialize(selected,**kwargs):
          if selected.vision_calibration is not None or selected.pad_prompt_to!=2048: raise AssertionError('Wrong actual launch recipe')
          observed.update(actual_factory=str(path),source_dir=str(kwargs['source_dir']),recipe_parsed=True)
          raise PreflightComplete('CPU import/actual runpy launcher preflight stopped before factory allocation')
        module.initialize=initialize
      return module
    from tinygrad import Device
    def forbidden(*args,**kwargs): raise AssertionError('Preflight attempted device access')
    with patch.object(type(Device),'__getitem__',forbidden),patch.object(frozen,'local',imports):
      try: frozen.main(command)
      except PreflightComplete: pass
      else: raise AssertionError('Preflight failed to reach the actual factory boundary')
    if not observed or Device._opened_devices: raise AssertionError('Invalid CPU preflight')
    args.output.write_text(json.dumps(dict(preflight=True,gpu_executed=False,devices=[],**observed),indent=2)+'\n')
  else:
    try: frozen.main(command)
    finally:
      if args.output.exists():
        report=json.loads(args.output.read_text());report['reload_protocol_sha256']=sha(args.protocol)
        report['reload_sources']=protocol['reload_sources'];report['diagnostic_only']=True
        args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()

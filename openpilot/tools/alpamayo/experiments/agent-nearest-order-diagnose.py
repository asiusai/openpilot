"""CPU-only canonical lookup-history reconstruction; no model-generated new output."""
import copy,importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace,MethodType
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_order_diagnose_common',HERE/'agent-nearest-order-common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.setup_cpu()
import numpy as np

def simulate(module,history,target,expected_stats,prompt=2026):
  """Execute the unchanged generate control flow with saved-token CPU oracles.

  This identifies query grouping. It cannot reproduce floating-point GPU KV.
  Saved current tokens are diagnostic oracle answers, never a parent trial input.
  """
  trace=[];anchor=135084
  def token(index):return target[index] if index<len(target) else anchor
  class Value:
    def __init__(self,value,**kw):self.value=np.asarray(value)
    def to(self,*a):return self
    def realize(self):return self
    def numpy(self):return self.value
  class Var:
    def bind(self,value):return value
  decoder=object.__new__(module.LookupDecoder)
  decoder.history=copy.deepcopy(history);decoder.statistics=[];decoder.verifiers={};decoder.warmup_graphs=False
  decoder.fallback_block=6;decoder.carry_verified_token=True;decoder.compact_io=True
  decoder.model=SimpleNamespace(cache=[SimpleNamespace(shape=(2,1,8,2304,128))])
  decoder.first_token=lambda logits:token(logits.offset)
  decoder.select_logits=lambda logits,index:SimpleNamespace(offset=logits.offset+index+1,device='CPU')
  decoder.read_tokens=lambda value:value.numpy()
  decoder.warm_verifiers=lambda *a:None
  def decode(logits,dc,ds,position,step,count):
    assert position==prompt+step
    trace.append(dict(kind='serial',offset=step,width=count,tokens=[token(i) for i in range(step,step+count)]))
    return SimpleNamespace(offset=step+count,device='CPU'),Value([token(i) for i in range(step,step+count)])
  def verify(tokens,dc,ds,position,step):
    values=tokens.numpy().tolist();trace.append(dict(kind='verify',offset=step,width=len(values),proposed=values))
    return SimpleNamespace(offset=step,device='CPU'),Value([token(i) for i in range(step+1,step+1+len(values))])
  decoder.decode=decode;decoder.verify=verify
  with patch.object(module,'Tensor',Value),patch.object(module,'TinyJit',lambda fn:fn),patch.object(module,'UOp',SimpleNamespace(variable=lambda *a:Var())):
    result=module.LookupDecoder.generate(decoder,decoder.model,SimpleNamespace(max_tokens=64),
      SimpleNamespace(traj_ids={'future_start':anchor}),None,SimpleNamespace(offset=0,device='CPU'),
      SimpleNamespace(shape=(1,1,64,128)),None,prompt)
  assert result==target
  actual=decoder.statistics[-1];assert actual==expected_stats,(actual,expected_stats)
  assert decoder.history==c.remember(history,target)
  return dict(statistics=actual,rounds=trace,generated_exact=True,history_update_exact=True,
    limit='Canonical control flow with saved-token oracle; no inference, numerical KV proof or quality result')

def diagnose():
  plan,policy,startup,dynamic=c.verify_assets();saved=c.validate_saved(plan,startup)
  baseline=c.read(plan['saved_nearest_report']);rows=baseline['warmup']+baseline['runs'];assert len(rows)==64
  module=c.local('lookup-decode');prior={};history=[]
  for index,row in enumerate(rows):
    prior[Path(row['sample']).name]=(index,copy.deepcopy(history),row)
    history=c.remember(history,row['prediction']['generated_token_ids'])
  result=[]
  for case,actual in zip(plan['cases'],saved['rows'],strict=True):
    index,before,old=prior[Path(case['sample']).name];target=actual['prediction']['generated_token_ids']
    assert target==old['prediction']['generated_token_ids']
    a,b=(np.asarray(p['xyz']) for p in (actual['prediction'],old['prediction']))
    ar,br=(np.asarray(p['rot']) for p in (actual['prediction'],old['prediction']))
    stats=lambda row:{k:v for k,v in row.items() if k!='prefill_draft'}
    item=dict(sample=case['sample'],display_time_s=case['display_time_s'],reference_index=index,
      old_reference_prior_history=before,consumer_prior_history=actual['history_before']['history'],
      old_reference_history_entries=len(before),consumer_history_entries=len(actual['history_before']['history']),
      old_statistics=stats(old['draft']),consumer_statistics=stats(actual['decoder_statistics']),
      target_verification_record_exact=old['draft']['prefill_draft']==actual['speculation'],
      generated_tokens_exact=True,reasoning_exact=actual['prediction']['reasoning']==old['prediction']['reasoning'],
      max_xyz_coord_difference_m=float(np.abs(a-b).max()),max_rotation_coord_difference=float(np.abs(ar-br).max()),
      all_nontiming_prediction_fields_exact=c.digest({k:v for k,v in old['prediction'].items() if k!='timings_ms'})==
        c.digest({k:v for k,v in actual['prediction'].items() if k!='timings_ms'}))
    if not actual['speculation']['accepted']:
      item['canonical_64_history_schedule']=simulate(module,before,target,item['old_statistics'],actual['speculation']['actual_prompt'])
      item['canonical_four_history_schedule']=simulate(module,actual['history_before']['history'],target,item['consumer_statistics'],actual['speculation']['actual_prompt'])
    result.append(item)
  from tinygrad import Device
  assert not Device._opened_devices
  return dict(cpu_only=True,gpu_executed=False,source_sha256=c.own_sources(),pins=c.PINS,queries=result,
    conclusion='Reference query-order/history mismatch is demonstrated. Different recorded decode grouping is reconstructed by unchanged control flow.',
    unresolved='Whether grouping explains every XYZ/rotation bit requires a same-order actual reference. No Consumer bug is excluded by CPU evidence.',
    old_failure_retained=True,saved_consumer_reusable_for_source_bound_same_order_comparison=True,
    conditioned_parent_outputs=False,automatic_pace=False,automatic_adoption=False,opened_devices=[])

if __name__=='__main__':
  result=diagnose();c.save(HERE/'agent-nearest-order-diagnosis.json',result)
  print(json.dumps({k:v for k,v in result.items() if k not in ('queries','source_sha256')},indent=2))

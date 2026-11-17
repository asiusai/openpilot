"""CPU checks for evidence reuse, poison/guard rejection, and the exact V patch."""
import copy
import importlib.util
import json
from pathlib import Path

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-output-finish-work')


def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  import numpy as np
  evidence=load('agent-ck-output-finish-evidence');runner=load('agent-ck-output-finish-run')
  previous,report,reuse=evidence.verify();plan,arrays=runner.verify()
  assert reuse==plan['reuse']
  checked=[]
  def reject(label,callback):
    try: callback()
    except (AssertionError,KeyError): checked.append(label);return
    raise AssertionError('Negative gate failed: '+label)
  def change(path,value):
    result=copy.deepcopy(report);target=result
    for item in path[:-1]: target=target[item]
    target[path[-1]]=value
    return result
  for label,path,value in [
    ('prior_timeout_phase',['failure','phase'],'candidate_normalized_first_A'),
    ('baseline_restore_hash',['phases',4,'O','guarded','sha256'],'0'*64),
    ('baseline_not_captured',['phases',2,'capture','expected_names'],[]),
    ('candidate_poison',['phases',5,'O','guarded','poison_count'],1),
    ('candidate_redzone',['phases',5,'LSE','guard_mismatch_count'],1),
    ('candidate_alias_mismatch',['phases',5,'O','guarded_returned_mismatch_count'],1),
    ('candidate_error_increase',['phases',5,'fp32_error','rmse'],1),
    ('candidate_LSE_error',['phases',5,'lse_error','rmse'],1),
    ('candidate_wrong_CU_WGP',['native_argument_records',-1,'CU_mode_bit'],1),
    ('candidate_wrong_O_pointer',['native_argument_records',-1,'arguments',0,'offset_bytes'],0)]:
    reject(label,lambda p=path,v=value:evidence.claims(change(p,v),previous))
  bf=np.array([0x3f80,0x7fc1,0x7f80,0xff80],np.uint16)
  stats=runner.statistics(bf,'bf16',0x7fc1)
  assert stats['elements']==4 and stats['finite_count']==1 and stats['nonfinite_count']==3 and stats['poison_count']==1
  fp=np.array([0x3f800000,0x7fc12345,0x7f800000,0xff800000],np.uint32)
  stats=runner.statistics(fp,'f32',0x7fc12345)
  assert stats['finite_count']==1 and stats['nonfinite_count']==3 and stats['poison_count']==1
  # The same range and slicing operation used by parent MMIO must not write adjacent bytes.
  offset,length=plan['input_patch']['byte_offset'],plan['input_patch']['byte_length']
  host=bytearray(arrays['a']['v'].tobytes())
  memoryview(host)[offset:offset+length]=arrays['mutate_v']['v'].tobytes()[offset:offset+length]
  assert bytes(host)==arrays['mutate_v']['v'].tobytes()
  memoryview(host)[offset:offset+length]=arrays['a']['v'].tobytes()[offset:offset+length]
  assert bytes(host)==arrays['a']['v'].tobytes()
  # Independent reference mutation respects precisely the new causal/GQA support gate.
  changed=arrays['a']['output_fp32']!=arrays['mutate_v']['output_fp32']
  allowed=np.zeros(changed.shape,bool);allowed[:,4:6,128:,:]=True
  assert np.count_nonzero(changed & ~allowed)==0 and changed.any()
  result={'cpu_checks':'passed','gpu_executed':False,'negative_checks_rejected':checked,
    'full_fixture_mutation_and_restoration':True,'statistics_poison_and_nonfinite_detection':True,
    'independent_reference_causal_GQA_mutation_support':True,'source_bound_reuse_valid':True,
    'protocol_sha256':evidence.sha(runner.PROTOCOL)}
  with (ROOT/'cpu-check.json').open('x') as output: output.write(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))


if __name__=='__main__': main()

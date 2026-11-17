"""CPU controls for the exact same-arm timing comparison and history rewind."""
import argparse
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('quality64_controls_common',HERE/'agent-adaptive-quality64-common.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)


def main():
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  assert not args.output.exists();K,_,_,_,C,_=Q.base()
  from tinygrad import Device
  assert not Device._opened_devices
  baseline=Q.read(Q.WORK/'agent-exact-combined-quality-candidate/report.json')
  actual={'warmup':copy.deepcopy(Q.rows(baseline)),'runs':copy.deepcopy(Q.rows(baseline))}
  Q.same_arm(K,baseline,actual);rejections=[]
  def reject(label,fn):
    try:fn()
    except (AssertionError,ValueError,KeyError):rejections.append(label)
    else:raise AssertionError('Accepted corrupt control: '+label)
  def mutation(label,mutate):
    changed=copy.deepcopy(actual);mutate(changed);reject(label,lambda:Q.same_arm(K,baseline,changed))
  mutation('warm token change',lambda r:r['warmup'][23]['prediction']['generated_token_ids'].__setitem__(0,1))
  mutation('measured trajectory change',lambda r:r['runs'][40]['prediction']['xyz'][0].__setitem__(0,123.))
  mutation('unknown prediction field',lambda r:r['runs'][3]['prediction'].update(unknown_new_field=True))
  mutation('nonfinite trajectory',lambda r:r['runs'][1]['prediction']['xyz'][0].__setitem__(0,float('nan')))
  mutation('query order',lambda r:r['runs'].reverse())
  mutation('cache/missing image change',lambda r:r['runs'][20].update(vision_misses=0))
  mutation('target-verification change',lambda r:r['runs'][20]['draft'].update(unexpected_target=True))
  zero=copy.deepcopy(baseline);Q.rows(zero)[0]['prediction']['xyz'][0][0]=0.
  negative={'warmup':copy.deepcopy(Q.rows(zero)),'runs':copy.deepcopy(Q.rows(zero))}
  negative['runs'][0]['prediction']['xyz'][0][0]=-0.
  reject('signed zero',lambda:Q.same_arm(K,zero,negative))
  B=Q.load(HERE/'agent-adaptive-quality64-boundary.py','quality64_control_boundary')
  state=SimpleNamespace(previous=None,completed=0,running=False,current=None,active=None,audit_prefix_once=False,audit_done=False)
  encoder=object();cache=SimpleNamespace(cache={},keys=(),output_grids=None,hits=0,misses=0,encoders={'compiled':encoder})
  decoder=SimpleNamespace(history=[],statistics=[],prefill_draft_old_history=[])
  components=SimpleNamespace(decoder=decoder,vision_cache=cache)
  b=B.Boundary(C.MLP/'experiments/benchmark-pipeline.py',SimpleNamespace(states=[state]),True);b.attach(components)
  state.previous=(1,2);state.completed=64;decoder.history=[[1,2]];cache.cache['old_future_image']=object();cache.keys=('old',)
  b.begin(64,components)
  assert state.previous is None and state.completed==0 and decoder.history==[] and not cache.cache and cache.keys==()
  assert cache.encoders['compiled'] is encoder
  b.end(64,components);assert not b.records[0]['exclude_from_paired_timing']
  b.begin(65,components);b.cold=1;b.end(65,components);assert b.records[1]['exclude_from_paired_timing']
  state.completed=64;state.running=True
  reject('active adaptive owner at rewind',lambda:b.begin(64,components));state.running=False
  state.audit_prefix_once=True
  reject('audit in timing pass',lambda:b.begin(64,components));state.audit_prefix_once=False
  # Malformed shape must still fail the inherited shape contract.
  assert callable(Q.counted_coverage(K,64)) and callable(Q.counted_coverage(K,128))
  assert not Device._opened_devices
  Q.write(args.output,{'complete':True,'gpu_executed':False,'same_arm_all128_checked':True,
    'negative_controls':rejections,'rewind_keeps_graphs_clears_history_and_features':True,
    'cold_jit_query_marked_for_pairwise_exclusion':True,'sources':{str(p):Q.sha(p) for p in HERE.glob('agent-adaptive-quality64-*.py')}})
  print({'complete':True,'negative_controls':len(rejections),'gpu_executed':False})

if __name__=='__main__':main()

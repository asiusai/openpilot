"""Actual turn entrypoint and inherited arrival ownership coverage; neural work only is doubled."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('turn_entry_run',HERE/'agent-turn-paced-run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)

def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  assert not args.output.exists();A=R.context();A.configure();C=A.C
  assert os.environ['DEV']=='CPU' and os.environ['ALLOW_DEVICE_USAGE']=='0'
  data=R.read(args.protocol);A.verify(args.protocol,R.sha(args.protocol),cpu_only=True)
  B=A.original_load(R.EXP/'agent-arrival-warp-ownership-check.py','turn_original_full_ownership_check');B.A=A;B.C=C
  original_run=runpy.run_path;entries=[]
  def selected_entry(path,run_name):
    assert Path(path)==R.EXP/'agent-arrival-warp-ownership-run.py' and run_name=='__main__'
    target=HERE/'agent-turn-paced-run.py';entries.append(str(target));sys.argv[0]=str(target)
    return original_run(str(target),run_name=run_name)
  B.runpy=SimpleNamespace(run_path=selected_entry)
  root=R.ARTIFACTS/args.output.stem;assert not root.exists();root.mkdir()
  from tinygrad import Device
  from tinygrad.helpers import Context
  with Context(ALLOW_DEVICE_USAGE=0):production=B.production(root,data)
  status=R.read(root/'entry/agent-turn-paced-status.json')
  assert not status['complete'] and not status['gpu_executed'] and status['cpu_only']
  assert len(status['invocation_adapter'])==1 and status['invocation_adapter'][0]['only_interval_and_rgb_path_changed']
  options=status['invocation_adapter'][0]['actual_arguments']
  assert options[options.index('--start-seconds')+1]=='111' and options[options.index('--raw-frames')+1]==str(R.CACHE/'rgb')
  assert status['preparer_restored']==[True] and status['history_count']==len(status['history_capture'])==2
  import numpy as np
  startup,history=status['history_capture']
  assert {k:v for k,v in startup.items() if k!='audit_ms'}=={k:v for k,v in history.items() if k!='audit_ms'}
  assert history['full_images']==10 and history['patches_per_image']==720 and history['camera_indices']==[1]*5+[6]*5
  expected=history['query_ns']+np.arange(-19,1,dtype=np.int64)*100_000_000
  assert history['history_sample_ns']==expected.tolist()
  for key,shape in [('ego_history_xyz',(1,20,3)),('ego_history_rot',(1,20,3,3))]:
    record=history[key];array=np.asarray(record['values'],dtype=np.dtype(record['dtype']))
    assert array.shape==shape and array.tobytes().hex()==record['float_bits_hex'] and np.isfinite(array).all()
  for index,offset in enumerate([-2_000_000_000,-1_000_000_000,-500_000_000,-100_000_000,0]*2):
    assert history['absolute_timestamps_us'][index]*1000<=history['query_ns']+offset
  assert not Device._opened_devices and production['all_actual_super_coverage_executed'] and production['actual_factory_storage']['complete']
  groups=R.breakdown([dict(nominal_eligible=True,nominal_accepted=True,generated_tokens=6,consumer_ms=1.,road_sof_to_result_ms=2.),
    dict(nominal_eligible=True,nominal_accepted=False,generated_tokens=28,consumer_ms=3.,road_sof_to_result_ms=4.),
    dict(nominal_eligible=False,nominal_accepted=False,generated_tokens=26,consumer_ms=5.,road_sof_to_result_ms=6.)])
  assert groups['nominal_accepted']['count']==groups['nominal_rejected']['count']==groups['nominal_ineligible']['count']==1
  assert groups['longer_than_six_tokens']['count']==2
  result=dict(complete=True,gpu_executed=False,devices=[],production=production,actual_entrypoints=entries,
    actual_turn_history_captured=True,model_ready_history=history,invocation_adapter=status['invocation_adapter'],
    initial_state='Actual past-only raw/state prime for 111s; original full startup input boundary with neural computation doubled.',
    startup_history_equals_cpu_consumer_history=True,breakdown_controls_passed=True,
    no_actual_prediction_claim=True,ssse3_packing_installed=False,source_files=R.files())
  R.write(args.output,result);print(json.dumps(dict(complete=True,gpu_executed=False,history_query_ns=history['query_ns'],all_actual_super_coverage_executed=True)),flush=True)

if __name__=='__main__':main()

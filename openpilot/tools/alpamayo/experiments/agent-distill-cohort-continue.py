"""One bounded CPU-only missing-scene attempt plus independent immutable finalization."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_cohort_common_continue',HERE/'agent-distill-cohort-common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)

def bounded(command,log,seconds):
  started=time.monotonic();timed_out=False
  with log.open('x') as stream:
    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
      timed_out=True;os.killpg(process.pid,signal.SIGTERM)
      try:code=process.wait(timeout=5)
      except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
  return dict(returncode=code,timed_out=timed_out,seconds=time.monotonic()-started,deadline_seconds=seconds)

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scene',required=True)
  parser.add_argument('--run-name',required=True);parser.add_argument('--manifest-output',type=Path,required=True)
  parser.add_argument('--seconds',type=int,default=120);parser.add_argument('--execute-cpu',action='store_true',default=False)
  args=parser.parse_args();assert 1<=args.seconds<=120
  assert args.run_name and all(char.isalnum() or char=='-' for char in args.run_name)
  assert args.scene in {row['id'] for row in C.coverage()['anchors']}
  output=args.manifest_output.resolve();assert output.is_relative_to(C.ROOT) and not output.exists()
  run=C.ROOT/('continuation-'+args.run_name);prepared=C.ROOT/('recovery-'+args.run_name)
  assert not run.exists() and not prepared.exists();C.storage(C.ROOT);sys.addaudithook(C.cpu_guard)
  python=str(C.BASE/'upstream/.venv/bin/python');helper=HERE/'agent-distill-cohort-recover.py'
  command=[python,'-B',str(helper),'--decode-scene',args.scene,'--finish-existing','--output-dir',str(prepared),
    '--manifest-output',str(run/'agent-distill-cohort-attempt.json')]
  finalizer=[python,'-B',str(helper),'--manifest-output',str(output)]
  plan=dict(schema=1,scene=args.scene,command=command,child_deadline_seconds=args.seconds,termination_grace_seconds=5,
    finalizer=finalizer,finalizer_deadline_seconds=60,max_total_child_seconds=args.seconds+70,
    preserve_all_original_and_partial_files=True,skip_all_verified_existing_frames_and_inputs=True,
    GPU_execution=False,training=False,execute_cpu=args.execute_cpu,
    source_sha256={str(path):C.sha(path) for path in [Path(__file__),helper,HERE/'agent-distill-cohort-prepare.py',HERE/'agent-distill-cohort-common.py',C.COVERAGE]})
  if not args.execute_cpu:print(json.dumps(plan,indent=2));return
  run.mkdir();C.write(run/'agent-distill-cohort-invocation.json',plan)
  attempt=bounded(command,run/'attempt.log',args.seconds)
  # Always re-audit completed receipts after the child is gone. No inferred or partial input is admitted.
  final=bounded(finalizer,run/'finalize.log',60)
  result=dict(attempt=attempt,finalizer=final,manifest=str(output),manifest_available=output.is_file(),gpu_executed=False,training_executed=False)
  if output.is_file():
    value=C.read(output);result.update(manifest_sha256=C.sha(output),completed_scene_ids=value['completed_scene_ids'],
      missing_scene_ids=value['missing_scene_ids'],full_cohort_complete=value['full_cohort_complete'])
  C.write(run/'agent-distill-cohort-result.json',result);print(json.dumps(result,indent=2))
  assert final['returncode']==0 and result['manifest_available'],'Finalization failed; preserve logs and previous immutable partial manifest'

if __name__=='__main__':main()

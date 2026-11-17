"""Fresh matched standalone quality experiment. Parent alone executes GPU arms."""
import argparse
from contextlib import ExitStack,nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ARTIFACTS=WORK/'agent-adaptive-prefill-artifacts'
BASE=WORK/'agent-expert-combined-cache-source'
PROTOCOL=WORK/'agent-expert-combined-cache-protocol.json'
PROTOCOL_SHA='723432734e33005b7a16c6756cb5ee64d2b88b692520ed9b63be6395537172fd'
REFERENCE64=WORK/'agent-expert-pv-qualified-quality-reference/report.json'
REFERENCE64_SHA='069e11d02b7424f1504b80dc80d887775c3ce5c9895d447402d1d5892751e842'
PY='/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'
WRAPPER=Path('/tmp/opencode/run-alpamayo-speed-trial.py')
MANIFEST=WORK/'agent-adaptive-prefill-plan.json'
CPU_REPORT=WORK/'agent-adaptive-prefill-cpu-v2.json'
INDICES=list(range(14,32))


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value,new=True):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('x' if new else 'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
  sys.modules[name]=module;spec.loader.exec_module(module);return module


def option(command,flag,value):command[command.index('--'+flag)+1]=str(value)


def reserve():
  disk=shutil.disk_usage(WORK)
  assert WORK.stat().st_dev==Path('/mnt/nvme-500').stat().st_dev!=Path('/').stat().st_dev
  assert disk.free>=max(5*1024**3,disk.total*.10), 'Preserve 10% / 5 GiB storage reserve'
  for p in (ARTIFACTS,ARTIFACTS/'tmp',ARTIFACTS/'cache'):p.mkdir(parents=True,exist_ok=True)


def base():
  N=load(BASE/'agent-expert-combined-cache-common.py','adaptive_existing_cache_common')
  C,data,evidence=N.verify(PROTOCOL,PROTOCOL_SHA)
  C.paths()
  return N,C,data,evidence


def prepare():
  reserve();N,C,data,evidence=base()
  assert sha(REFERENCE64)==REFERENCE64_SHA
  rows=read(REFERENCE64);rows=rows['warmup']+rows['runs']
  samples=[rows[i]['sample'] for i in INDICES]
  assert len(samples)==18 and all(Path(p).is_dir() for p in samples)
  # No sample pixels/learned history/nav/noise are modified or re-encoded here.
  sample_files={str(p):sha(p) for directory in samples for p in sorted(Path(directory).glob('*.npy'))}
  cpu=read(CPU_REPORT);assert cpu['complete'] and not cpu['gpu_executed']
  for name,digest in cpu['sources'].items():assert sha(name)==digest, 'CPU-tested helper changed: '+name
  terminal=read(data['stages']['quality']['pipeline'])
  cmd=terminal['command'];begin=cmd.index('--samples')+1;end=cmd.index('--initial-noise')
  cmd[begin:end]=samples;option(cmd,'warmup',2);option(cmd,'runs',16)
  option(cmd,'output',WORK/'agent-adaptive-prefill-unused-output.json')
  terminal['environment'].update(CACHEDB=str(ARTIFACTS/'gpu-cache.db'),TMPDIR=str(ARTIFACTS/'tmp'),
    XDG_CACHE_HOME=str(ARTIFACTS/'cache'),HF_HOME=str(ARTIFACTS/'cache/huggingface'),TORCH_HOME=str(ARTIFACTS/'cache/torch'))
  pipeline=WORK/'agent-adaptive-prefill-pipeline.json';write(pipeline,terminal)
  source_names=('policy','hook','check','run','queue')
  sources={str(HERE/f'agent-adaptive-prefill-{name}.py'):sha(HERE/f'agent-adaptive-prefill-{name}.py') for name in source_names}
  plan={'numerical_candidate':True,'width':16,'quality_only':True,'no_automatic_paced':True,
    'original_protocol':str(PROTOCOL),'original_protocol_sha256':PROTOCOL_SHA,
    'baseline_reference64':{'path':str(REFERENCE64),'sha256':REFERENCE64_SHA,'use':'input order only; outputs are freshly paired'},
    'indices':INDICES,'samples':samples,'warmup':2,'runs':16,'pipeline':str(pipeline),'pipeline_sha256':sha(pipeline),
    'input_files':sample_files,'sources':sources,'cpu_report':str(CPU_REPORT),
    'cpu_report_sha256':sha(CPU_REPORT),'current_kernel_recipe':'combined MLP2048 + vision tiles + qualified expert PV; dynamic vision, fixed language, shuffle-FMA',
    'cache':N.cache_check(),'gpu_execution_during_preparation':False,
    'scope':'Prepared standalone pipeline. No Consumer/arrival/startup-wrapper installation or continuous timing claim.',
    'candidate_audit':'Exactly one extra nominal prefill at first adaptive query, first logits + all 28 prompt-KV bit checks; quality timing not throughput.'}
  write(MANIFEST,plan)
  print(json.dumps({'ready':True,'plan':str(MANIFEST),'sha256':sha(MANIFEST),'queries_per_arm':18}))


def verify_plan():
  plan=read(MANIFEST)
  assert plan['width']==16 and plan['indices']==INDICES and plan['warmup']==2 and plan['runs']==16
  for group in ('sources','input_files'):
    for name,digest in plan[group].items():assert sha(name)==digest,name
  assert sha(plan['pipeline'])==plan['pipeline_sha256'] and sha(plan['cpu_report'])==plan['cpu_report_sha256']
  assert sha(PROTOCOL)==plan['original_protocol_sha256']==PROTOCOL_SHA
  return plan


def compare(reference,candidate):
  before,after=read(reference),read(candidate)
  x,y=before['warmup']+before['runs'],after['warmup']+after['runs']
  assert len(x)==len(y)==18
  comparisons=[]
  for old,new in zip(x,y,strict=True):
    assert old['sample']==new['sample'] and old['input_latest_timestamp_us']==new['input_latest_timestamp_us']
    a,b=old['prediction'],new['prediction']
    assert a['prompt_tokens']==b['prompt_tokens'] and a['prefill_tokens']==b['prefill_tokens']==2048
    row={'sample':old['sample'],'tokens_equal':a['generated_token_ids']==b['generated_token_ids'],
      'reasoning_equal':a['reasoning']==b['reasoning']}
    for key in ('xyz','rot'):
      aa,bb=np.asarray(a[key],dtype=np.float64),np.asarray(b[key],dtype=np.float64)
      assert aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
      row[key+'_all_float_bits_equal']=bool(np.array_equal(aa.view(np.uint64),bb.view(np.uint64)))
      row[key+'_max_absolute_delta']=float(np.abs(aa-bb).max())
      row[key+'_rms_delta']=float(np.sqrt(np.mean(np.square(aa-bb))))
    row['timings']={arm:{k:pred['timings_ms'][k] for k in ('prefill','reasoning','total')} for arm,pred in (('reference',a),('candidate',b))}
    comparisons.append(row)
  return {'numerical_candidate':True,'matched_queries':18,'rows':comparisons,
    'tokens_equal_count':sum(x['tokens_equal'] for x in comparisons),
    'all_predictions_bitexact':all(x['tokens_equal'] and x['reasoning_equal'] and x['xyz_all_float_bits_equal'] and x['rot_all_float_bits_equal'] for x in comparisons),
    'accuracy_gate':'No tolerance accepted here. Parent must assess changed tokens and trajectory errors against recorded turn motion.',
    'timings_valid_for_throughput':False,'reason':'Candidate one-time prefix audit plus cold graph transitions; includes every extra head/transfer.'}


def run(arm,output,reference=None):
  reserve();plan=verify_plan();N,C,data,evidence=base()
  assert not output.exists() and not list(output.parent.glob(output.stem+'.*')), 'Use an exclusive new output stem'
  output.parent.mkdir(parents=True,exist_ok=True)
  if arm=='candidate':
    assert reference is not None and reference.exists()
    status=read(reference.parent/'adaptive-prefill-status.json')
    assert status['complete'] and status['arm']=='reference' and status['plan_sha256']==sha(MANIFEST)
  else:assert reference is None
  terminal=read(plan['pipeline']);environment=terminal['environment']
  original_spec=importlib.util.spec_from_file_location;overlay=None;composition=None
  status={'complete':False,'arm':arm,'numerical_candidate':arm=='candidate','plan_sha256':sha(MANIFEST),
    'gpu_executed':False,'scope':plan['scope'],'timings_valid_for_throughput':False}
  status_path=output.parent/'adaptive-prefill-status.json';write(status_path,status)
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ,dict(os.environ),clear=True))
      for key,value in environment.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=str(value)
      from tinygrad import Device
      assert not Device._opened_devices
      status['cache']=N.cache_check()
      installer=N.load(N.OLD/'agent-expert-combined-install.py','adaptive_frozen_install')
      composition=installer.factory(C)('quality',True,evidence)
      with composition:
        if arm=='candidate':overlay=load(HERE/'agent-adaptive-prefill-hook.py','adaptive_runtime_hook').Overlay(C.MLP/'experiments',True)
        with overlay if overlay else nullcontext():
          inner_base=importlib.util.spec_from_file_location
          command=[str(C.MLP/'experiments/agent-speculation-run.py'),'--output',str(output),
            '--invocation',str(plan['pipeline']),'--mode','candidate','--prefill-attention','shuffle-fma',
            '--warmup','2','--runs','16','--pad-prompt-to','2048']
          stack.enter_context(patch.object(sys,'argv',command))
          try:
            status['gpu_executed']=True
            runpy.run_path(command[0],run_name='__main__')
            status['coverage']=composition.coverage()
            if overlay:
              assert len(overlay.states)==1
              state=overlay.states[0]
              assert state.completed==18 and state.audit_done and not state.running
              assert state.records and all(r['all_targets_finite'] for r in state.records)
              status['adaptive']={'records':state.records,'completed':state.completed,'ast':overlay.ast_proof,
                'prefix_audit_count':sum(r['audit_prefix_once'] for r in state.records),'heads':17,'width':16}
              assert status['adaptive']['prefix_audit_count']==1
          finally:importlib.util.spec_from_file_location=inner_base
      status['pv']=composition.pv_handle.info
      original=C.load(N.OLD/'agent-expert-combined-run.py','adaptive_existing_validation')
      original.validate_pv(status['pv'],True,evidence,18)
      status['restorations']=composition.restorations
      assert all(x['restored'] for x in composition.restorations) and composition.session.stats['methods_restored']
      assert overlay is None or overlay.restored
      if arm=='candidate':
        status['comparison']=compare(reference,output)
        status['reference_sha256']=sha(reference)
      status['complete']=True;status['report_sha256']=sha(output)
  except BaseException as exc:
    status['failure']={'type':type(exc).__name__,'message':str(exc)};raise
  finally:
    status['import_hook_restored']=importlib.util.spec_from_file_location is original_spec
    status['adaptive_hook_restored']=overlay is None or overlay.restored
    write(status_path,status,new=False)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output',type=Path);p.add_argument('--arm',choices=('reference','candidate'))
  p.add_argument('--reference',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--plan',action='store_true')
  args=p.parse_args()
  if args.prepare:return prepare()
  if args.plan:
    plan=verify_plan();N,C,_,_=base()
    namespace=runpy.run_path(str(C.MLP/'experiments/benchmark-pipeline.py'),run_name='adaptive_actual_import_probe')
    assert callable(namespace['initialize']) and callable(namespace['main'])
    from tinygrad import Device
    assert not Device._opened_devices
    print(json.dumps({'ready':True,'plan':str(MANIFEST),'sha256':sha(MANIFEST),'opened_devices':[],
      'sample_count':len(plan['samples']),'actual_runpy_pipeline_import':True}));return
  if args.output is None or args.arm is None:p.error('Require --output and --arm, or --prepare/--plan')
  run(args.arm,args.output,args.reference)


if __name__=='__main__':main()

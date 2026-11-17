"""Candidate-only numerical check against admitted observed reference predictions."""
import argparse
from contextlib import ExitStack,nullcontext
import copy
import importlib.util
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('adaptive_coverage_common',HERE/'agent-adaptive-coverage-common.py')
K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
ARTIFACTS=K.WORK/'agent-adaptive-coverage-artifacts'
MANIFEST=K.WORK/'agent-adaptive-coverage-plan.json'
PIPELINE=K.WORK/'agent-adaptive-coverage-pipeline.json'


def safe_snapshot(comp,overlay):
  value={}
  if comp is not None:
    try:value['composition']=K.inventory(comp)
    except BaseException as exc:value['composition_snapshot_error']={'type':type(exc).__name__,'message':str(exc)}
  if overlay is not None:
    value['adaptive']={'states':[{'completed':s.completed,'previous':s.previous,'running':s.running,
      'audit_done':s.audit_done,'records':copy.deepcopy(s.records)} for s in overlay.states],
      'prefills':[{'last_record':copy.deepcopy(p.last_record),
        'last_logits_shape':None if p.last_logits is None else list(p.last_logits.shape),
        'last_payload_shape':None if p.last_payload is None else list(p.last_payload.shape)} for _,_,p in overlay.prefills],
      'ast':overlay.ast_proof,'restored':overlay.restored}
  return value


def adaptive_checks(overlay):
  assert overlay is not None and len(overlay.states)==1
  state=overlay.states[0]
  assert state.completed==18 and state.audit_done and not state.running and state.records
  assert all(r['head_projections']==17 and r['width']==16 and r['all_targets_finite'] and r['finite_flags']==[1]*17 for r in state.records)
  audits=[r for r in state.records if r['audit_prefix_once']]
  assert len(audits)==1
  assert audits[0]['prefix_audit']=={'first_logits_exact':True,'all_layer_prefix_kv_exact':True,'layer_count':28,'rows':audits[0]['actual_prompt']}
  partial=[r for r in state.records if 0<r['accepted_prefix']<len(r['draft'])]
  assert partial, 'No genuine partial acceptance in the declared turn fixture'
  return {'complete':True,'genuine_partial_queries':len(partial),'records':copy.deepcopy(state.records),
    'completed':state.completed,'prefix_audit_count':1,'head_projections':17,'numerical_candidate':True}


def report_checks(report,plan):
  rows=report['warmup']+report['runs']
  assert len(report['warmup'])==2 and len(report['runs'])==16 and [r['sample'] for r in rows]==plan['samples']
  baseline=K.read(K.REFERENCE)
  for key in ('source_sha256','settings','runtime_environment','initial_noise_sha256','prefill_calibration_sha256',
    'precision_restoration','reduced_vocab','pixel_scale','history_pixel_scale','history_feature_scale',
    'vision_calibration_sha256','pad_prompt_to','trajectory_solver','vision_group_limit','batched_vision'):
    K.exact(report[key],baseline[key],key)
  K.snapshot_files(report)
  for i,(old,new) in enumerate(zip(baseline['warmup']+baseline['runs'],rows,strict=True)):
    for key in ('sample','input_latest_timestamp_us','vision_hits','vision_misses','uploaded_pixel_bytes'):
      K.exact(old[key],new[key],str(i)+'/'+key)
  return {'all_18_input_order_and_cache_miss_fields_equal':True,'source_and_recipe_exact':True}


def prepare():
  R,plan,N,C,evidence=K.frozen();R.reserve()
  for name in ('tmp','cache'):(ARTIFACTS/name).mkdir(parents=True,exist_ok=True)
  receipt=K.reference_receipt(C,write_result=True)
  pipeline=K.read(plan['pipeline'])
  pipeline['environment'].update(CACHEDB=str(ARTIFACTS/'gpu-cache.db'),TMPDIR=str(ARTIFACTS/'tmp'),
    XDG_CACHE_HOME=str(ARTIFACTS/'cache'),HF_HOME=str(ARTIFACTS/'cache/huggingface'),TORCH_HOME=str(ARTIFACTS/'cache/torch'))
  K.write(PIPELINE,pipeline,new=True)
  K.write(MANIFEST,{'original_plan_sha256':K.PLAN_SHA,'reference_receipt':str(K.RECEIPT),'reference_receipt_sha256':K.sha(K.RECEIPT),
    'pipeline':str(PIPELINE),'pipeline_sha256':K.sha(PIPELINE),
    'sources':{str(p):K.sha(p) for p in HERE.glob('agent-adaptive-coverage-*.py')},
    'candidate_hook_sha256':plan['sources'][str(HERE/'agent-adaptive-prefill-hook.py')],
    'numerical_hook_changed':False,'numerical_reference_only':True,'automatic_paced':False,
    'cpu_report':str(K.WORK/'agent-adaptive-coverage-cpu.json'),
    'cpu_report_sha256':K.sha(K.WORK/'agent-adaptive-coverage-cpu.json')},new=True)
  print({'ready':True,'manifest':str(MANIFEST),'sha256':K.sha(MANIFEST),'reference_reused_for_predictions':receipt['reference_predictions_admitted']})


def verify():
  data=K.read(MANIFEST)
  assert data['original_plan_sha256']==K.PLAN_SHA and data['numerical_hook_changed'] is False and not data['automatic_paced']
  for path,digest in data['sources'].items():assert K.sha(path)==digest,path
  for pathkey in ('pipeline','reference_receipt','cpu_report'):assert K.sha(data[pathkey])==data[pathkey+'_sha256']
  assert K.read(data['cpu_report'])['complete']
  return data


def run(output):
  manifest=verify();R,plan,N,C,evidence=K.frozen();R.reserve()
  K.exact(K.reference_receipt(C),K.read(K.RECEIPT),'reference_receipt')
  assert not output.exists() and not list(output.parent.glob(output.stem+'.*')), 'Use a new output stem'
  output.parent.mkdir(parents=True,exist_ok=True)
  status_path=output.parent/'adaptive-coverage-status.json'
  status={'complete':False,'numerical_candidate':True,'gpu_executed':False,'reference_receipt_sha256':K.sha(K.RECEIPT),
    'new_plan_sha256':K.sha(MANIFEST),'original_plan_sha256':K.PLAN_SHA,'timings_valid_for_throughput':False,
    'numerical_hook_changed':False,'checks':{},'primary_failure':None}
  K.write(status_path,status,new=True)
  comp=overlay=None;original_spec=importlib.util.spec_from_file_location;original_error=None
  from tinygrad import Device
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ,dict(os.environ),clear=True))
      for key,value in K.read(PIPELINE)['environment'].items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=str(value)
      assert not Device._opened_devices
      status['cache']=N.cache_check()
      comp=K.factory(C)('quality',True,evidence)
      with comp:
        overlay=K.load(HERE/'agent-adaptive-prefill-hook.py','coverage_unchanged_adaptive').Overlay(C.MLP/'experiments',True)
        with overlay:
          inner=importlib.util.spec_from_file_location
          command=[str(C.MLP/'experiments/agent-speculation-run.py'),'--output',str(output),'--invocation',str(PIPELINE),
            '--mode','candidate','--prefill-attention','shuffle-fma','--warmup','2','--runs','16','--pad-prompt-to','2048']
          stack.enter_context(patch.object(sys,'argv',command))
          try:
            runpy.run_path(command[0],run_name='__main__')
            status['checks']['report']=report_checks(K.read(output),plan)
            status['checks']['shape_coverage']=comp.coverage()
            status['checks']['adaptive']=adaptive_checks(overlay)
          finally:
            status['before_cleanup']=safe_snapshot(comp,overlay)
            K.write(status_path,status)
            importlib.util.spec_from_file_location=inner
      original=K.load(N.OLD/'agent-expert-combined-run.py','coverage_original_pv_validation')
      original.validate_pv(comp.pv_handle.info,True,evidence,18)
      status['checks']['pv_complete']=True
      assert all(r['restored'] for r in comp.restorations) and comp.session.stats['methods_restored']
      assert overlay.restored
      status['comparison']=R.compare(K.REFERENCE,output)
      status['complete']=True
  except BaseException as exc:
    original_error=exc;status['primary_failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
  finally:
    status['gpu_executed']=bool(Device._opened_devices)
    status['after_cleanup']=safe_snapshot(comp,overlay)
    status['import_hook_restored']=importlib.util.spec_from_file_location is original_spec
    status['adaptive_hook_restored']=overlay is None or overlay.restored
    # Validate available PV diagnostics independently, even when arithmetic or
    # shape coverage failed earlier. Never overwrite the primary exception.
    if comp is not None and comp.pv_handle is not None:
      try:
        validator=K.load(N.OLD/'agent-expert-combined-run.py','coverage_finally_pv_validation')
        validator.validate_pv(comp.pv_handle.info,True,evidence,18)
        status['finally_pv_validation']={'passed':True}
      except BaseException as exc:status['finally_pv_validation']={'passed':False,'type':type(exc).__name__,'message':str(exc)}
    status['available_reports']={p.name:K.sha(p) for p in (output,output.with_suffix('.jsonl'),output.with_suffix('.speculation.json')) if p.exists()}
    K.write(status_path,status)


def main():
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--plan',action='store_true')
  args=p.parse_args()
  if args.prepare:return prepare()
  if args.plan:
    verify();_,plan,_,C,_=K.frozen();K.reference_receipt(C)
    print({'ready':True,'new_plan_sha256':K.sha(MANIFEST),'queries':len(plan['samples']),'candidate_only':True,'gpu_executed':False});return
  if args.output is None:p.error('--output is required')
  run(args.output)


if __name__=='__main__':main()

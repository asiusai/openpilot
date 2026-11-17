"""Parent-only full64 quality pair or separately requested matched timing arms."""
import argparse
from contextlib import ExitStack,nullcontext
import importlib.util
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('adaptive_quality64_common',HERE/'agent-adaptive-quality64-common.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)


def adaptive_checks(overlay,timing):
  assert len(overlay.states)==1;s=overlay.states[0]
  assert s.completed==64 and not s.running and s.records
  for r in s.records:
    assert r['width']==16 and r['head_projections']==17 and r['all_targets_finite'] and r['finite_flags']==[1]*17
    assert r['target_rows']==list(range(r['actual_prompt']-1,r['actual_prompt']+16))
    assert r['retained_draft_rows']==[r['actual_prompt'],r['actual_prompt']+r['accepted_prefix']]
    assert r['next_cache_write']==r['actual_prompt']+r['accepted_prefix'] and r['next_decode_rope_step']==r['accepted_prefix']
    assert r['final_cache_length']==r['actual_prompt']+len(r['generated']) and r['action_rotary_offset']==len(r['generated'])
  audits=[r for r in s.records if r['audit_prefix_once']]
  assert len(audits)==(0 if timing else 1) and s.audit_done==bool(audits)
  for r in audits:assert r['prefix_audit']=={'first_logits_exact':True,'all_layer_prefix_kv_exact':True,'layer_count':28,'rows':r['actual_prompt']}
  partial=[r for r in s.records if 0<r['accepted_prefix']<len(r['draft'])];assert partial
  return {'records':s.records,'attempts':len(s.records),'partial':len(partial),
    'zero':sum(r['accepted_prefix']==0 for r in s.records),'full':sum(r['accepted_prefix']==len(r['draft']) for r in s.records),
    'prefix_audit_count':len(audits),'completed_per_pass':64,'numerical_candidate':True}


def require_quality():
  path=Q.WORK/'agent-adaptive-quality64-review.json';review=Q.read(path)
  assert review['complete'] and review['five_percent_screen_passed']
  for arm in ('reference','candidate'):
    output=Q.output('quality',arm);status=Q.read(Q.status_path(output))
    assert status['complete'] and status['gpu_executed'] and status['plan_sha256']==Q.sha(Q.PLAN)
    assert review[arm+'_sha256']==Q.sha(output)==status['report_sha256']
  return review


def run(output,arm='candidate',stage='quality'):
  plan=Q.verify();K,R,old,N,C,evidence=Q.base();R.reserve()
  timing=stage=='timing';count=128 if timing else 64
  if timing:require_quality()
  if arm=='candidate' and not timing:
    reference=Q.output('quality','reference');prior=Q.read(Q.status_path(reference))
    assert prior['complete'] and prior['gpu_executed'] and prior['arm']=='reference' and prior['plan_sha256']==Q.sha(Q.PLAN)
    assert prior['report_sha256']==Q.sha(reference)
  assert not output.exists() and not list(output.parent.glob(output.stem+'.*')), 'Use a new output stem'
  output.parent.mkdir(parents=True,exist_ok=True)
  status={'complete':False,'gpu_executed':False,'arm':arm,'stage':stage,'plan_sha256':Q.sha(Q.PLAN),
    'numerical_candidate':arm=='candidate','timings_valid_for_throughput':False,'automatic_continuous':False,'checks':{}}
  status_path=Q.status_path(output);Q.write(status_path,status)
  original_spec=importlib.util.spec_from_file_location;comp=overlay=boundary=None
  previous=Q.load(HERE/'agent-adaptive-coverage-run.py','quality64_saved_diagnostics')
  from tinygrad import Device
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ,dict(os.environ),clear=True))
      for key,value in Q.read(plan['pipeline'])['environment'].items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=str(value)
      assert not Device._opened_devices;status['cache']=N.cache_check()
      comp=Q.composition(K,C,count)('quality',True,evidence)
      with comp:
        if arm=='candidate':overlay=Q.load(HERE/'agent-adaptive-prefill-hook.py','quality64_unchanged_numerics').Overlay(C.MLP/'experiments',not timing)
        with overlay if overlay else nullcontext():
          inner=importlib.util.spec_from_file_location
          boundary=Q.load(HERE/'agent-adaptive-quality64-boundary.py','quality64_boundary').Boundary(
            C.MLP/'experiments/benchmark-pipeline.py',overlay,timing)
          command=[str(C.MLP/'experiments/agent-speculation-run.py'),'--output',str(output),'--invocation',plan['pipeline'],
            '--mode','candidate','--prefill-attention','shuffle-fma','--warmup',str(64 if timing else 2),
            '--runs',str(64 if timing else 62),'--pad-prompt-to','2048']
          stack.enter_context(patch.object(sys,'argv',command))
          try:
            with boundary:runpy.run_path(command[0],run_name='__main__')
            report=Q.read(output);status['checks']['report']=Q.check_report(K,report,plan,count)
            assert len(boundary.records)==count and boundary.active is None
            assert boundary.records[0]['before']=={'history':[],'adaptive_previous':None,'adaptive_completed':0 if overlay else None}
            status['checks']['coverage']=comp.coverage()
            if overlay:status['checks']['adaptive']=adaptive_checks(overlay,timing)
            if timing:
              quality=Q.read(Q.output('quality',arm));status['checks']['same_arm']=Q.same_arm(K,quality,report)
              old_status=Q.read(Q.status_path(Q.output('quality',arm)))
              for i,record in enumerate(boundary.records):
                for field in ('before','after'):K.exact(record[field],old_status['boundary_records'][i%64][field],f'{i}/{field}')
              status['checks']['own_history_exact']=True
          finally:
            status['before_cleanup']=previous.safe_snapshot(comp,overlay)
            status['boundary_records']=boundary.records;status['active_query']=boundary.active;status['boundary_ast']=boundary.proof
            Q.write(status_path,status,new=False);importlib.util.spec_from_file_location=inner
      validator=Q.load(N.OLD/'agent-expert-combined-run.py','quality64_pv_validation')
      validator.validate_pv(comp.pv_handle.info,True,evidence,count)
      assert all(r['restored'] for r in comp.restorations) and comp.session.stats['methods_restored']
      assert overlay is None or overlay.restored
      assert boundary.restored
      status['checks']['pv_complete']=True;status['complete']=True;status['report_sha256']=Q.sha(output)
    if stage=='quality' and arm=='candidate':
      review=Q.review(Q.output('quality','reference'),output);Q.write(Q.WORK/'agent-adaptive-quality64-review.json',review)
      status['review_sha256']=Q.sha(Q.WORK/'agent-adaptive-quality64-review.json')
      assert review['five_percent_screen_passed'], 'Full64 or one of four 16-query windows exceeds 5% ADE/FDE'
  except BaseException as exc:
    status['complete']=False;status['failure']={'type':type(exc).__name__,'message':str(exc)};raise
  finally:
    status['gpu_executed']=bool(Device._opened_devices)
    status['after_cleanup']=previous.safe_snapshot(comp,overlay)
    status['import_hook_restored']=importlib.util.spec_from_file_location is original_spec
    status['adaptive_hook_restored']=overlay is None or overlay.restored
    status['boundary_restored']=boundary is None or boundary.restored
    if comp is not None and comp.pv_handle is not None:
      try:
        validator=Q.load(N.OLD/'agent-expert-combined-run.py','quality64_final_pv_validation');validator.validate_pv(comp.pv_handle.info,True,evidence,count)
        status['finally_pv_validation']={'passed':True}
      except BaseException as exc:status['finally_pv_validation']={'passed':False,'type':type(exc).__name__,'message':str(exc)}
    status['available_reports']={p.name:Q.sha(p) for p in (output,output.with_suffix('.jsonl'),output.with_suffix('.speculation.json')) if p.exists()}
    Q.write(status_path,status,new=False)


def main():
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--arm',choices=('reference','candidate'),default='candidate')
  p.add_argument('--stage',choices=('quality','timing'),default='quality');p.add_argument('--prepare',action='store_true');p.add_argument('--plan',action='store_true')
  args=p.parse_args()
  if args.prepare:Q.prepare();print({'prepared':True,'plan_sha256':Q.sha(Q.PLAN)});return
  if args.plan:
    Q.verify();Q.base();print({'ready':True,'fresh_reference_required':True,'quality_queries_per_arm':64,
      'separate_timing_passes_per_arm':[64,64],'plan_sha256':Q.sha(Q.PLAN),'gpu_executed':False});return
  assert args.output is not None;run(args.output,args.arm,args.stage)

if __name__=='__main__':main()

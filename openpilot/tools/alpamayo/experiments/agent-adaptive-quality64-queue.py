"""One fresh quality pair; timing is a separate explicit command after review."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('quality64_parent_run',HERE/'agent-adaptive-quality64-run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);Q=R.Q


def timing_summary():
  reports={a:Q.read(Q.output('timing',a)) for a in ('reference','candidate')}
  statuses={a:Q.read(Q.status_path(Q.output('timing',a))) for a in reports}
  for arm,s in statuses.items():
    assert s['complete'] and s['gpu_executed'] and s['stage']=='timing' and s['checks']['own_history_exact']
    assert s['report_sha256']==Q.sha(Q.output('timing',arm)) and len(s['boundary_records'])==128
  excluded=sorted({r['original64_index'] for s in statuses.values() for r in s['boundary_records'][64:] if r['exclude_from_paired_timing']})
  retained=[i for i in range(64) if i not in excluded];assert retained, 'No comparable already-captured queries remain'
  summary={}
  for arm,report in reports.items():
    chosen=[report['runs'][i] for i in retained]
    fields={k:[r[k] for r in chosen] for k in ('pipeline_ms','service_ms','cpu_buffers_ms')}
    fields.update({k:[r['prediction']['timings_ms'][k] for r in chosen] for k in ('prefill','reasoning','total')})
    summary[arm]={k:{'median_ms':float(np.median(v)),'mean_ms':float(np.mean(v)),'p95_ms':float(np.percentile(v,95))} for k,v in fields.items()}
  return {'complete':True,'paired_retained_indices':retained,'paired_excluded_indices':excluded,'summary':summary,
    'every_warm_and_measured_output_exact_to_own_quality':True,'empty_history_each_pass':True,'audit_in_measured_clock':False,
    'all_head_stitch_transfer_materialization_continuation_costs_in_original_clock':True,
    'report_sha256':{arm:Q.sha(Q.output('timing',arm)) for arm in reports},
    'scope':'Prepared-sample original pipeline; subset paired latency, not continuous-camera Hz. Excluded queries remain in original reports.',
    'adoption_allowed':False,'automatic_continuous':False}


def main():
  p=argparse.ArgumentParser();p.add_argument('--stage',choices=('quality','timing'),default='quality')
  p.add_argument('--review-sha256');p.add_argument('--plan',action='store_true');args=p.parse_args()
  Q.verify();_,old,_,_,_,_=Q.base()
  if args.stage=='timing':
    if not args.plan:
      R.require_quality();assert args.review_sha256==Q.sha(Q.WORK/'agent-adaptive-quality64-review.json'), 'Supply the reviewed full64 report SHA'
  else:assert args.review_sha256 is None
  commands=[]
  for arm in ('reference','candidate'):
    output=Q.output(args.stage,arm)
    if output.parent.exists():raise FileExistsError(output.parent)
    commands.append([old.PY,str(old.WRAPPER),output.parent.name,'--program',str(HERE/'agent-adaptive-quality64-run.py'),
      '--timeout-seconds','900','--env','PYTHONDONTWRITEBYTECODE','1','--env','PYTHONPATH','/tmp/opencode/alpamayo-asius-20261114/tinygrad',
      '--env','CACHEDB',str(Q.ART/'gpu-cache.db'),'--env','TMPDIR',str(Q.ART/'tmp'),
      '--set','arm',arm,'--set','stage',args.stage])
  if args.plan:
    print(json.dumps({'commands':commands,'stage':args.stage,'plan_sha256':Q.sha(Q.PLAN),
      'timing_requires_reviewed_full64':True,'automatic_continuous':False},indent=2));return
  queue=Q.WORK/f'agent-adaptive-quality64-{args.stage}-queue.json';assert not queue.exists()
  receipts=[]
  for command in commands:
    child=subprocess.run(command,timeout=960);receipts.append({'command':command,'returncode':child.returncode})
    Q.write(queue,{'stage':args.stage,'children':receipts,'review_sha256':args.review_sha256},new=len(receipts)==1)
    if child.returncode:raise SystemExit(child.returncode)
  if args.stage=='timing':Q.write(Q.WORK/'agent-adaptive-quality64-timing-summary.json',timing_summary())

if __name__=='__main__':main()

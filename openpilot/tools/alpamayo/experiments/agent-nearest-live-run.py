"""Parent-only four-query parity or gated8s recorded replay. Default: CPU preflight."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
DEFAULT=WORK/'agent-nearest-live-data/agent-nearest-live-plan.json'
sys.dont_write_bytecode=True


def load(path,name):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def read(path):return json.loads(Path(path).read_text())


def verify(plan,policy):
  for path,digest in plan['frozen_sha256'].items():
    if policy.sha(path)!=digest:raise ValueError('Frozen nearest-live artifact changed: '+path)
  repository=plan['tinygrad']['path']
  if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()!=plan['tinygrad']['revision'] or
      subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repository,text=True).strip()):
    raise ValueError('Pinned tinygrad source changed')
  if plan['policy']!=policy.POLICY or [case['display_time_s'] for case in plan['cases']]!=[35.,116.5,167.,228.]:
    raise ValueError('Only the four predeclared queries are supported')
  root=Path(plan['base_source']);sys.path[:0]=[str(root),str(root/'experiments')]
  startup=load(plan['startup_runner'],'agent_nearest_live_qualified_startup')
  dynamic=load(root/'experiments/agent-paced-dynamic-policy.py','agent_nearest_live_qualified_dynamic')
  dynamic.verify_snapshot()
  reference,candidate=read(plan['startup_reference']),read(plan['startup_candidate'])
  qualification=startup.qualify(reference,candidate,dynamic)
  if qualification!=plan['startup_qualification']:raise ValueError('Startup qualification changed')
  if not all(case['exact_saved_nearest_input'] for case in plan['cases']):raise ValueError('These available-only inputs need a fresh paired reference, not saved-nearest parity')
  return root,startup,dynamic


def parity(plan,output,root,startup,dynamic,policy):
  exp=root/'experiments'
  def local(name):return load(exp/(name+'.py'),'agent_nearest_live_gpu_'+name.replace('-','_'))
  replay=local('agent-live-replay');dynamic.adapt(replay)
  qualified=read(plan['startup_candidate'])
  recipe=SimpleNamespace(samples=qualified['protocol']['replay_order'],provenance=qualified['protocol']['dataset'])
  normalized,args,protocol=replay.normalize(Path(plan['invocation']),recipe,local('benchmark-pipeline'),output.with_suffix('.launcher.json'))
  if protocol!=qualified['protocol']:raise ValueError('Four-query factory recipe differs from qualified startup')
  cases=plan['cases']
  rows=[dict(sample=case['sample'],query_ns=case['query_ns']) for case in cases]
  dataset=SimpleNamespace(rows=rows,samples=[row['sample'] for row in rows],inputs=[replay.tree(replay.Dataset.load(row)) for row in rows])
  audit=replay.Audit(dataset);result={};initialized=[]
  warm=load(Path(plan['startup_runner']).with_name('agent-startup-spec-warm.py'),'agent_nearest_live_unchanged_warm')
  invocation=output.with_suffix('.normalized-invocation.json');replay.save(invocation,normalized)
  launcher=local('agent-speculation-run');runtime=local('agent-live-runtime')
  def launch(path,run_name):
    if Path(path).resolve()!=exp/'benchmark-pipeline.py' or run_name!='__main__':raise ValueError('Unexpected canonical factory target')
    pipeline=local('benchmark-pipeline');effective=pipeline.parse_args()
    sources,source_dir=pipeline.snapshot_sources(output.with_suffix('.factory.json'))
    components=pipeline.initialize(effective,sources=sources,source_dir=source_dir);initialized.append(components)
    # The SAME qualified helper, current-input-only, before Audit and Consumer binding.
    result['startup_spec']=warm.initialize(components,effective,replay.Dataset.load(rows[0]),source_root=root)
    audit.install(components)
    clock=replay.ReplayClock();spec=local('agent-speculation-policy')
    live=local('agent-live-consumer')
    consumer=runtime.bind_components(components,replay.binding_args(effective),consumer_module=live,
      speculation_policy=spec,speculation_draft=spec.CHECKPOINT_DRAFT_IDS,clock=clock)
    result['consumer_provenance']=consumer.provenance
    for case,row in zip(cases,rows,strict=True):
      arrays=replay.Dataset.load(row);metadata=copy.deepcopy(case['metadata']);clock.now=metadata['prepared_ns']
      value=consumer.predict(arrays,metadata);consumer.synchronize();audit.complete(consumer_record=value['live_consumer'])
      for key,expected in case['expected_cpu_hashes'].items():
        if audit.rows[-1][key]!=expected:raise ValueError('Actual Consumer CPU packing differs: '+key)
    consumer.close()
    result['initialized']=dict(source_sha256=components.report['source_sha256'],settings=replay.tree(vars(components.settings)),
      runtime_environment=components.report['runtime_environment'])
    return {}
  argv=[str(exp/'agent-speculation-run.py'),'--invocation',str(invocation),'--output',str(args.output),
    '--mode','candidate','--warmup',str(args.warmup),'--runs',str(args.runs),'--pad-prompt-to','2048','--prefill-attention','shuffle-fma']
  with patch.dict(os.environ),patch.object(sys,'argv',argv),patch.object(sys,'path',list(sys.path)),patch.object(runpy,'run_path',launch),policy.imports(root):
    sys.path[:0]=[part for part in normalized['environment']['PYTHONPATH'].split(os.pathsep) if part]
    launcher.main()
  if len(initialized)!=1 or audit.pending is not None or len(audit.rows)!=4:raise ValueError('Expected one model and exactly four completed queries')
  expected=read(plan['saved_nearest_report']);by_sample={Path(row['sample']).name:row for row in expected['warmup']+expected['runs']}
  comparisons=[]
  for case,row in zip(cases,audit.rows,strict=True):
    saved=by_sample[Path(case['saved_nearest_sample']).name]['prediction']
    a,b=({k:v for k,v in prediction.items() if k!='timings_ms'} for prediction in (saved,row['prediction']))
    comparisons.append(dict(query_ns=case['query_ns'],all_nontiming_prediction_fields_exact=replay.digest(a)==replay.digest(b),
      actual_consumer_cpu_hashes_exact=all(row[key]==value for key,value in case['expected_cpu_hashes'].items()),
      expected_speculative_accept=case['expected_speculative_accept'],actual_speculative_accept=row['speculation']['accepted']))
  exact=all(row['all_nontiming_prediction_fields_exact'] and row['actual_consumer_cpu_hashes_exact'] and
    row['expected_speculative_accept']==row['actual_speculative_accept'] for row in comparisons)
  result.update(mode='nearest-live-parity4',policy=policy.POLICY,gpu_executed=True,predictions_qualified=exact,
    queries=4,rows=audit.rows,comparison=dict(exact=exact,queries=4,rows=comparisons),
    qualified_factory_protocol=protocol,
    startup_sources=startup.own_sources(),plan_sha256=policy.sha(DEFAULT),adapter_sha256=policy.sha(HERE/'agent-nearest-live-policy.py'),
    timing='Instrumented4query parity; synthetic protocol clock; no throughput or input-latency claim',automatic_adoption=False)
  replay.save(output,result)
  if not exact:raise RuntimeError('Four-query Consumer/saved-nearest parity failed; paced replay remains blocked')


def verify_parity4(gate,plan,policy,startup):
  if not (gate.get('mode')=='nearest-live-parity4' and gate.get('policy')==policy.POLICY and
          gate.get('gpu_executed') is True and gate.get('predictions_qualified') is True and gate.get('queries')==4 and
          gate.get('plan_sha256')==policy.sha(DEFAULT) and gate.get('adapter_sha256')==policy.sha(HERE/'agent-nearest-live-policy.py') and
          gate.get('startup_sources')==startup.own_sources()):raise ValueError('Missing source-bound real-GPU four-query gate')
  qualified=read(plan['startup_candidate'])
  if gate.get('qualified_factory_protocol')!=qualified['protocol'] or gate['initialized']['source_sha256']!=qualified['initialized']['source_sha256']:
    raise ValueError('Four-query factory/source protocol changed')
  if not all(gate.get('startup_spec',{}).get(key) is True for key in
             ('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained')):
    raise ValueError('Four-query owner was not initialized/restored by the qualified warm helper')
  rows=gate.get('rows',[])
  if len(rows)!=4 or not rows[0].get('consumer_protocol',{}).get('input_reused'):raise ValueError('Missing warmed-owner reuse or four completions')
  saved=read(plan['saved_nearest_report']);original={Path(row['sample']).name:row['prediction'] for row in saved['warmup']+saved['runs']}
  expected=[]
  for case,row in zip(plan['cases'],rows,strict=True):
    if not row.get('completed') or row['query_ns']!=case['query_ns'] or row['sample']!=case['sample']:
      raise ValueError('Unsynchronized or unrelated parity query')
    prediction=original[Path(case['saved_nearest_sample']).name]
    a,b=({k:v for k,v in value.items() if k!='timings_ms'} for value in (prediction,row['prediction']))
    item=dict(query_ns=case['query_ns'],all_nontiming_prediction_fields_exact=json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False),
      actual_consumer_cpu_hashes_exact=all(row[key]==value for key,value in case['expected_cpu_hashes'].items()),
      expected_speculative_accept=case['expected_speculative_accept'],actual_speculative_accept=row['speculation']['accepted'])
    if not (item['all_nontiming_prediction_fields_exact'] and item['actual_consumer_cpu_hashes_exact'] and
            item['expected_speculative_accept']==item['actual_speculative_accept']):raise ValueError('Independent four-query parity failed')
    expected.append(item)
  if gate.get('comparison')!=dict(exact=True,queries=4,rows=expected):raise ValueError('Stored comparison differs from independent recomputation')
  return dict(exact=True,queries=4,accepted=2,fallback=2)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--mode',choices=('parity','paced'),required=True);p.add_argument('--plan',type=Path,default=DEFAULT)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--parity-report',type=Path)
  p.add_argument('--execute-gpu',action='store_true',help='PARENT ONLY, explicit separate gate/run')
  cli=p.parse_args();policy=load(HERE/'agent-nearest-live-policy.py','agent_nearest_live_run_policy')
  if cli.plan.resolve()!=DEFAULT:raise ValueError('Use the single frozen nearest-live plan')
  plan=read(cli.plan);root,startup,dynamic=verify(plan,policy)
  if cli.mode=='paced':
    if cli.parity_report is None:raise ValueError('Completed real-GPU nearest4 gate required before paced replay')
    gate=read(cli.parity_report)
    verify_parity4(gate,plan,policy,startup)
  if not cli.execute_gpu:
    print(json.dumps(dict(cpu_only=True,gpu_executed=False,mode=cli.mode,policy=policy.POLICY,queries=4 if cli.mode=='parity' else None,
      startup_actual_GPU_qualified=True,nearest_parity_pending=cli.mode=='parity',source_seconds=8,requested_ceiling_hz=10,native_camera_hz=20,
      inference_budget_seconds=180,maximum_active=1,automatic_adoption=False),indent=2));return
  if cli.output.exists():raise FileExistsError('Never overwrite evidence')
  cli.output.parent.mkdir(parents=True,exist_ok=True)
  if cli.mode=='parity':parity(plan,cli.output,root,startup,dynamic,policy)
  else:
    # Unchanged qualified paced lifecycle: prime -> warm first current input -> bind -> start clock/producer -> observe.
    with policy.imports(root):
      startup.main(['--task','paced','--base-source',str(root),'--reference-report',plan['startup_reference'],
        '--consumer-parity-report',plan['startup_candidate'],'--invocation',plan['invocation'],'--mode','run','--output',str(cli.output),
        '--hz','10','--start-seconds','35','--seconds','8','--inference-budget-seconds','180','--raw-frames',plan['raw_frames'],'--warp','native'])
    result=read(cli.output);result['nearest_live_policy']=dict(policy=policy.POLICY,plan_sha256=policy.sha(cli.plan),
      adapter_sha256=policy.sha(HERE/'agent-nearest-live-policy.py'),parity4_sha256=policy.sha(cli.parity_report),
      numerical_input_change=True,availability='Actually admitted/ready by each dispatch; no all-capture lookup',adopted=False)
    cli.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

"""Optional parent-only canonical four-query reference against immutable Consumer evidence.

No Consumer execution, history injection, tolerance, pacing or adoption path.
Default invocation verifies CPU/source gates and prints the bounded protocol.
"""
import argparse,copy,fcntl,importlib.util,json,os,runpy,sys
from pathlib import Path
from types import SimpleNamespace,MethodType
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_order_run_common',HERE/'agent-nearest-order-common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sys.dont_write_bytecode=True

def recipe(plan,dynamic,output):
  replay=c.local('agent-live-replay');dynamic.adapt(replay);pipeline=c.local('benchmark-pipeline')
  qualified=c.read(plan['startup_candidate'])
  data=SimpleNamespace(samples=qualified['protocol']['replay_order'],provenance=qualified['protocol']['dataset'])
  normalized,args,protocol=replay.normalize(Path(plan['invocation']),data,pipeline,output)
  if protocol!=qualified['protocol']:raise ValueError('Original qualified64 factory recipe differs')
  bounded=copy.deepcopy(normalized);command=bounded['command'];begin=command.index('--samples')+1
  end=next((i for i in range(begin,len(command)) if command[i].startswith('--')),len(command))
  command[begin:end]=[case['sample'] for case in plan['cases']]
  for flag,value in (('--warmup','0'),('--runs','4')):command[command.index(flag)+1]=value
  # Cache-file placement only. Never write the parent/shared compiler database.
  bounded['environment']['CACHEDB']=str(HERE/'agent-nearest-order-gpu-cache.db')
  effective=pipeline.parse_args(command[2:]);dynamic.selected_args(effective)
  changes={key:dict(before=replay.tree(getattr(args,key)),after=replay.tree(getattr(effective,key))) for key in vars(args)
    if getattr(args,key)!=getattr(effective,key)}
  if set(changes)!={'samples','warmup','runs'}:raise ValueError('Unexpected bounded recipe change: '+str(changes))
  if effective.validation_reference is not None or effective.pipeline_inputs or effective.warmup_graphs or effective.warm_current_shapes:
    raise ValueError('No extra predictions, overlap or active warming in the four-query loop')
  return replay,bounded,effective,protocol,dict(loop_only=changes,environment_only={'CACHEDB':dict(before=normalized['environment']['CACHEDB'],
    after=bounded['environment']['CACHEDB'])})

def ownership(components,inputs,case,live):
  import numpy as np
  if inputs is not components.inputs or inputs.selective_cache is not components.vision_cache:
    raise ValueError('Canonical reference replaced the warmed input/cache owner')
  signature=live.digest((inputs._signature,[(value.shape,str(np.dtype(value.dtype.fmt))) for value in inputs.cpu]))
  entries={}
  for label,owner,name in [('VisionCache.cache',components.vision_cache,'cache'),('VisionCache.encoders',components.vision_cache,'encoders'),
      ('VisionCache.stitchers',components.vision_cache,'stitchers'),('Decoder.jits',components.decoder,'jits'),
      ('Decoder.verifiers',components.decoder,'verifiers'),('Decoder.select_jits',components.decoder,'select_jits'),
      ('EdgeModel.trajectory_jits',components.model,'trajectory_jits'),('Decoder.history',components.decoder,'history'),
      ('Decoder.prefill_draft_old_history',components.decoder,'prefill_draft_old_history'),
      ('CurrentInputs.selective_jits',inputs,'selective_jits'),('CurrentInputs.positions',inputs,'positions')]:entries[label]=len(getattr(owner,name))
  return dict(input_reused=True,input_signature_sha256=signature,upload_array_bytes=sum(v.numel()*v.dtype.itemsize for v in inputs.cpu),
    source_capture_timestamps_ns=[f['capture_ns'] for f in case['metadata']['frames']],actual_prompt_tokens=components.model.padded_prefill.last_prompt,
    padded_prompt_tokens=2048,cache_entries=entries,owned_input_objects=1)

def execute(plan,policy,startup,dynamic,saved,output,protocol_pin):
  root=c.BASE;exp=root/'experiments';raw=output.with_suffix('.canonical.json')
  replay,normalized,args,factory_protocol,changes=recipe(plan,dynamic,raw)
  rows=[dict(sample=case['sample'],query_ns=case['query_ns']) for case in plan['cases']]
  dataset=SimpleNamespace(rows=rows,samples=[r['sample'] for r in rows],inputs=[replay.tree(replay.Dataset.load(r)) for r in rows])
  audit=replay.Audit(dataset);result=dict(schema=1,mode='nearest-order-canonical-reference4',gpu_executed=True,queries=4,
    source_sha256=c.own_sources(),protocol_sha256=c.sha(c.PROTOCOL),saved_consumer_sha256=c.sha(c.SAVED),
    old_comparison_still_failed=True,automatic_pace=False,automatic_adoption=False,conditioned_on_saved_outputs=False,
    numerical_difference_vs64='Retained in the unchanged failed report and CPU diagnosis; not relabeled or tolerated',
    loop_and_cache_location_changes=changes,qualified_factory_protocol=factory_protocol,reference_ownership=[],
    timing='Instrumented canonical4 diagnostic only; no pacing, throughput or quality claim')
  warm=c.load(Path(plan['startup_runner']).with_name('agent-startup-spec-warm.py'))
  launcher=c.local('agent-speculation-run');initialized=[]
  invocation=output.with_suffix('.invocation.json');c.save(invocation,normalized)
  def launch(path,run_name):
    if Path(path).resolve()!=exp/'benchmark-pipeline.py' or run_name!='__main__':raise ValueError('Unexpected factory target')
    pipeline=c.local('benchmark-pipeline');initialize=pipeline.initialize;live=c.local('agent-live-consumer')
    def begin(effective,**kwargs):
      if vars(effective)!=vars(args):raise ValueError('Launcher changed the bounded inference recipe')
      components=initialize(effective,**kwargs);initialized.append(components)
      init=dict(source_sha256=components.report['source_sha256'],settings=replay.tree(vars(components.settings)),
        runtime_environment=components.report['runtime_environment'])
      if init!=saved['initialized']:raise ValueError('Actual reference factory/settings/source differ from saved Consumer')
      result['initialized']=init
      try:result['startup_spec']=warm.initialize(components,effective,replay.Dataset.load(rows[0]),source_root=root)
      finally:result['startup_spec']=dict(getattr(components,'_startup_spec',{}))
      state=result['startup_spec']
      if not all(state.get(key) is True for key in c.RESTORE):raise ValueError('Reference startup failed an existing bit/owner gate')
      for key in ('source_sha256','graph_counts','decoder_plan','snapshot_bytes','actual_prompt','accepted_suffix_rows'):
        if state[key]!=saved['startup_spec'][key]:raise ValueError('Initial warmed state contract differs: '+key)
      if components.decoder.history or components.decoder.statistics or components.decoder.prefill_draft_records or components.decoder.prefill_draft_old_history:
        raise ValueError('Reference startup retained prediction history')
      if components.vision_cache.cache or components.vision_cache.hits or components.vision_cache.misses:
        raise ValueError('Reference startup retained logical image features')
      owner=components.inputs;cache_id=tuple(id(v.uop.base) for v in components.model.cache)
      components.decoder.before_diffusion=live.live_before_diffusion
      audit.install(components);audited=components.runner
      def completed(bound,*values,**kw):
        if bound is not components.decoder or kw['inputs'] is not owner or any(components.startup_active):
          raise ValueError('Reference owner or initialization phase changed')
        if tuple(id(v.uop.base) for v in components.model.cache)!=cache_id:raise ValueError('KV owner changed before prediction')
        prediction=audited(*values,**kw);components.model.synchronize()
        if tuple(id(v.uop.base) for v in components.model.cache)!=cache_id:raise ValueError('KV owner changed during prediction')
        audit.complete();index=len(audit.rows)-1
        result['reference_ownership'].append(ownership(components,owner,plan['cases'][index],live))
        return prediction
      components.runner=MethodType(completed,components.decoder)
      return components
    pipeline.initialize=begin
    result['canonical_main_transform']=c.transform_main(pipeline)
    if result['canonical_main_transform']!=protocol_pin['canonical_main_transform']:raise ValueError('Unreviewed canonical wrapper')
    pipeline.main()
  argv=[str(exp/'agent-speculation-run.py'),'--invocation',str(invocation),'--output',str(raw),'--mode','candidate',
    '--warmup','0','--runs','4','--pad-prompt-to','2048','--prefill-attention','shuffle-fma']
  try:
    with patch.dict(os.environ),patch.object(sys,'argv',argv),patch.object(sys,'path',list(sys.path)),patch.object(runpy,'run_path',launch),policy.imports(root):
      sys.path[:0]=[part for part in normalized['environment']['PYTHONPATH'].split(os.pathsep) if part]
      launcher.main()
    if len(initialized)!=1 or audit.pending is not None or len(audit.rows)!=4:raise ValueError('One model, four completed predictions required')
    result['rows']=audit.rows
    result['comparison']=c.compare_rows(audit.rows,saved['rows'])
    result['ownership_exact']=result['reference_ownership']==[r['consumer_protocol'] for r in saved['rows']]
    result['same_order_exact']=result['comparison']['exact'] and result['ownership_exact']
    if not result['same_order_exact']:raise RuntimeError('Same-order exact reference failed; inspect fields and retain both failures')
    result['interpretation']='Canonical original four-query order matches saved Consumer exactly; old64 numerical order sensitivity remains a separate finding'
  except BaseException as error:
    result['error']=repr(error);result['same_order_exact']=False;result['rows']=audit.rows
    raise
  finally:c.save(output,result)

def verify_protocol():
  protocol=c.read(c.PROTOCOL)
  if protocol['source_sha256']!=c.own_sources():raise ValueError('Owned source changed since CPU protocol freeze')
  for path,digest in protocol['artifact_sha256'].items():
    if c.sha(path)!=digest:raise ValueError('Protocol artifact changed: '+path)
  if protocol['automatic_pace'] or protocol['automatic_adoption'] or protocol['queries']!=4:raise ValueError('Unreviewed scope')
  checks=c.read(HERE/'agent-nearest-order-checks.json')
  if not checks['passed'] or checks['gpu_executed'] or checks['source_sha256']!=c.own_sources():raise ValueError('Current CPU checks required')
  return protocol

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute-gpu',action='store_true')
  parser.add_argument('--output',type=Path,default=HERE/'agent-nearest-order-parent-reference/report.json')
  args=parser.parse_args();c.setup_cpu()
  plan,policy,startup,dynamic=c.verify_assets();saved=c.validate_saved(plan,startup);protocol=verify_protocol()
  if not args.execute_gpu:
    print(json.dumps(dict(cpu_only=True,gpu_executed=False,queries=4,display_times_s=[x['display_time_s'] for x in plan['cases']],
      saved_consumer_reused=True,source_sha256=c.own_sources(),canonical_main_transform=protocol['canonical_main_transform'],
      automatic_pace=False,automatic_adoption=False,output=str(args.output)),indent=2));return
  if args.output.parent.parent.resolve()!=HERE or not args.output.parent.name.startswith('agent-nearest-order-'):
    raise ValueError('All new parent output must stay in a new owned agent-nearest-order-* directory')
  if args.output.parent.exists():raise FileExistsError('Use a new output directory; retain every earlier result')
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:pass
    else:raise RuntimeError('Parent must hold the shared GPU flock')
  if os.environ.get('AGENT_NEAREST_ORDER_PARENT_TIMEOUT')!='1':raise RuntimeError('Use the reviewed parent timeout command')
  args.output.parent.mkdir()
  execute(plan,policy,startup,dynamic,saved,args.output,protocol)

if __name__=='__main__':main()

"""CPU checks for strict outputs, actual-mode classification, and timing isolation."""
import argparse
import builtins
import copy
import importlib
import json
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace

C=importlib.import_module('agent-compiler-speed-common')


def reject(f):
  try:f()
  except (AssertionError,ValueError,FileNotFoundError):return
  raise AssertionError('Negative control was accepted')


def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args();assert not args.output.exists()
  from tinygrad import Device
  from tinygrad.helpers import ALLOW_DEVICE_USAGE,Context,Target
  from tinygrad.codegen import to_program,to_program_context
  from tinygrad.engine import worker
  from tinygrad.engine.realize import _compile_kernel
  assert ALLOW_DEVICE_USAGE.value==0 and not Device._opened_devices
  run=importlib.import_module('agent-compiler-speed-run')
  queue=importlib.import_module('agent-compiler-speed-queue')
  fixed=importlib.import_module('agent-compiler-mode-renderer')
  P=importlib.import_module('agent-compiler-mode-proof')
  assert Path(fixed.__file__).resolve()==C.PRIVATE/'agent-compiler-mode-renderer.py'
  renderer=fixed.AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  ast=P.ast_for(263);program=to_program(ast,renderer)
  with Context(PARALLEL=2):
    try:
      task=(0,(ast,renderer),{v.key:v.value for v in to_program_context})
      _,spawned=next(worker.get_worker_pool().imap_unordered(_compile_kernel,[task]))
      detail=next(worker.get_worker_pool().imap_unordered(P.worker_task,[task]))
    finally:worker.terminate_worker_pool()
  assert P.describe(spawned)==P.describe(program)==detail['program']
  assert detail['renderer']['origin']==str(C.PRIVATE/'agent-compiler-mode-renderer.py')
  # Use saved actual GPU artifacts as data; run no GPU work.
  anchors=[C.read(C.ROOT/'evidence'/f'mode-{arm}.json')['inventory'] for arm in C.ARMS]
  mode_check=C.compare_warm(*anchors)
  assert mode_check['explicit_programs']==26 and mode_check['changed']==131
  kinds=C.origins(anchors[1])
  assert sum('renderer_compilation' in x for x in kinds.values())==163
  assert all(anchors[1]['programs'][k]['resources']['workgroup_processor_mode']==1 for k,v in kinds.items() if 'renderer_compilation' in v)
  explicit=next(k for k,v in kinds.items() if 'precompiled_binary' in v)
  bad=copy.deepcopy(anchors[1]);bad['programs'][explicit]['binary_sha256']='00'*32
  reject(lambda:C.compare_warm(anchors[0],bad))
  bad=copy.deepcopy(anchors[1]);bad['captures'][0]['kernels'][0]['global_input_slots'].append(999999)
  reject(lambda:C.compare_warm(anchors[0],bad))
  # An existing complete 64-query report supplies realistic nested schemas.
  old=C.WORK/'agent-expert-tiles-quality-reference/report.json'
  base=C.read(old);spec=C.read(old.with_suffix('.speculation.json'))
  samples=[x['sample'] for x in base['warmup']+base['runs']]
  C.exact_reports(base,base,spec,spec,'quality',samples)
  mutations=[]
  def bad_model(label,edit):
    value=copy.deepcopy(base);edit(value)
    reject(lambda:C.exact_reports(base,value,spec,spec,'quality',samples));mutations.append(label)
  bad_model('last-row-token',lambda x:x['runs'][-1]['prediction']['generated_token_ids'].__setitem__(0,-1))
  bad_model('last-row-history',lambda x:x['runs'][-1]['draft']['prefill_draft'].__setitem__('final_cache_length',-1))
  bad_model('unknown-top',lambda x:x.__setitem__('unknown_new_policy',True))
  bad_model('unknown-summary',lambda x:x['summary'].__setitem__('unknown_new_policy',True))
  bad_model('unknown-row',lambda x:x['runs'][-1].__setitem__('unknown_input_history',17))
  bad_model('unknown-prediction',lambda x:x['runs'][-1]['prediction'].__setitem__('unknown',17))
  bad_model('unknown-timing-field',lambda x:x['runs'][-1]['prediction']['timings_ms'].__setitem__('unknown',17))
  bad_model('freeze-policy',lambda x:x['warmup_gc_freeze'].__setitem__('limit','changed'))
  bad_model('nonfinite-output',lambda x:x['runs'][-1]['prediction'].__setitem__('xyz',[float('nan')]))
  bad_model('sample-order',lambda x:x['runs'].reverse())
  zero=copy.deepcopy(base);zero['runs'][-1]['prediction']['xyz']=[0.0]
  neg=copy.deepcopy(zero);neg['runs'][-1]['prediction']['xyz']=[-0.0]
  reject(lambda:C.exact_reports(zero,neg,spec,spec,'quality',samples));mutations.append('signed-zero')
  bad_spec=copy.deepcopy(spec);bad_spec['records'][-1]['unknown_target_record']=True
  reject(lambda:C.exact_reports(base,base,spec,bad_spec,'quality',samples));mutations.append('complete-speculation')
  measured=copy.deepcopy(base);measured['runs'][-1]['pipeline_ms']*=2
  C.exact_reports(base,measured,spec,spec,'quality',samples)
  native=C.read(C.WORK/'dynamic-vision-native-5hz/report.json')
  metric_check=C.native_summary(native,native)
  assert metric_check['throughput_change_percent']==metric_check['latency_median_change_percent']==0
  bad_native=copy.deepcopy(native);bad_native['summary']['throughput_fps']*=2
  reject(lambda:C.native_summary(native,bad_native))
  # The actual observer context must close before the synthetic measured clock.
  audit=run.audit_class()(C.WORK/'agent-compiler-speed-cpu-artifacts')
  original_hooks=run.hooks();observer=audit.observe();observer.__enter__()
  real_print=builtins.print;baseline_print=lambda *a,**kw:None
  detached=[]
  def finish():
    observer.__exit__(None,None,None);builtins.print=baseline_print
    assert run.hooks()==original_hooks
    detached.append(audit.callbacks)
  filename=C.SNAP/'experiments/benchmark-pipeline.py'
  code="""def main():
  report={'warmup':[],'runs':[]};pending=None;measured_begin=None
  for iteration in range(12):
    report['warmup'].append({})
    print(f'Iteration {iteration}: synthetic')
  assert len(detached)==1 and print is baseline_print
  measured_begin=1.0
  for _ in range(24):
    assert hooks()==original_hooks
    print('measured synthetic')
"""
  namespace={'detached':detached,'baseline_print':baseline_print,'hooks':run.hooks,'original_hooks':original_hooks}
  builtins.print=run.boundary(baseline_print,filename,12,finish)
  try:
    exec(compile(code,str(filename),'exec'),namespace);namespace['main']()
  finally:
    builtins.print=real_print;observer.__exit__(None,None,None)
  assert audit.callbacks==detached[0] and run.hooks()==original_hooks
  # Quality rejection must prevent the first native subprocess call.
  calls=[];fake_commands={s+'/'+a:['placeholder','--timeout-seconds','900'] for s in C.COUNTS for a in C.ARMS}
  fake_protocol={'run_order':[[s,a] for s in C.COUNTS for a in C.ARMS]}
  fake=SimpleNamespace(ROOT=C.ROOT,WORK=C.WORK,verify=lambda *a,**kw:fake_protocol,
    read=lambda *a:{'protocol_sha256':'test','commands':fake_commands},commands=lambda *a:fake_commands,
    out=lambda s,a:C.WORK/f'agent-compiler-speed-cpu-nonexistent-{s}-{a}',
    metadata_path=lambda *a:'unused',checked_result=lambda *a:None,
    validate_pair=lambda *a:{'exact':True,'native_throughput':None},write=lambda *a:None,
    quality_gate=lambda *a:(_ for _ in ()).throw(AssertionError('injected quality rejection')))
  old_c,old_run,old_argv=queue.C,queue.subprocess.run,sys.argv[:]
  try:
    queue.C=fake;queue.subprocess.run=lambda cmd,**kw:calls.append(cmd)
    sys.argv=['queue','--protocol-sha','test','--execute']
    reject(queue.main)
  finally:queue.C,queue.subprocess.run,sys.argv=old_c,old_run,old_argv
  assert len(calls)==2,'Native work launched after rejected quality'
  assert not Device._opened_devices
  files=[*C.PRIVATE.glob('agent-compiler-speed-*.py'),*(C.PRIVATE/n for n in
    ('agent-compiler-mode-renderer.py','agent-compiler-mode-proof.py','agent-compiler-mode-inventory.py'))]
  result={'complete':True,'cpu_only':True,'gpu_executed':False,'opened_devices':[],
    'source_sha256':{str(p):C.file_sha(p) for p in files},'actual_mode_origin_check':mode_check,
    'renderer_copy_spawned_import':detail,'negative_controls':mutations,
    'explicit_binary_mutation_rejected':True,'captured_argument_mutation_rejected':True,
    'native_quality_gate_blocks_subprocess':True,'timing_summary_corruption_rejected':True,
    'all_observers_restored_before_clock':True,'measured_observer_callbacks':audit.callbacks-detached[0],
    'measurement_changes_do_not_change_quality':True,'native_metric_self_check':metric_check,
    'reference_choice':'Fresh original64; inspected existing reports cannot prove the exact current recipe and compiler history'}
  C.write(args.output,result)
  print(json.dumps({'complete':True,'gpu_executed':False,'negative_controls':len(mutations)}))


if __name__=='__main__':main()

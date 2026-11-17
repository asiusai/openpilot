#!/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python
"""Exclusive-output cache retry; quality4 exact gate still precedes native36."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('expert_combined_cache_queue_common',HERE/'agent-expert-combined-cache-common.py')
N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)


def out(stage,mode): return N.WORK/f'agent-expert-combined-cache-{stage}-{mode}'


def prepare():
  assert not N.ROOT.exists() and not N.PROTOCOL.exists()
  C = N.original(); old,evidence = C.protocol(N.OLD_PROTOCOL,N.OLD_SHA)
  assert all(not out(s,m).exists() for s in C.COUNTS for m in ('reference','candidate'))
  wall = N.read(N.FAILURE/'wall.json')
  assert wall['returncode'] == 1 and not wall['timed_out'] and wall['timeout_seconds'] == 900
  failure = N.read(N.FAILURE/'report.agent-expert-combined.json')
  assert not failure['passed'] and failure['failure']['type'] == 'FileNotFoundError'
  assert str(N.WORK/'agent-expert-combined-artifacts/fp8-cache/index.json') in failure['failure']['message']
  copy_path = N.WORK/'agent-expert-combined-cache-copy.json'
  if copy_path.exists():
    copied = N.read(copy_path); assert copied['copied']['weights.bin']['destination_readback_sha256'] == N.WEIGHTS_SHA
    N.cache_check()
  else: N.write(copy_path,N.materialize(),new=True)
  (N.ARTIFACTS/'tmp').mkdir(exist_ok=True)
  R = N.load(HERE/'agent-expert-combined-cache-run.py','expert_combined_cache_prepare_run')
  cpu = N.WORK/'agent-expert-combined-cache-cpu.json'
  assert not cpu.exists()
  terminal = N.read(old['stages']['quality']['pipeline'])['command']
  command = [N.PY,str(HERE/'agent-expert-combined-cache-preflight.py'),'--controls','--output',str(cpu),
    '--checkpoint',terminal[terminal.index('--checkpoint')+1]]
  checked = subprocess.run(command,env=R.cpu_env(),capture_output=True,text=True,timeout=90)
  cpu.with_suffix('.log').write_text(checked.stdout+checked.stderr); checked.check_returncode()
  proof = N.read(cpu)
  assert proof['complete'] and not proof['gpu_executed'] and len(proof['rejected']) == 8
  N.ROOT.mkdir()
  data = copy.deepcopy(old)
  files = data['files']
  for path in sorted(HERE.glob('agent-expert-combined-cache-*.py')):
    dest = N.ROOT/path.name; dest.write_bytes(path.read_bytes()); dest.chmod(0o444)
    files[str(path)] = N.sha(path); files[str(dest)] = N.sha(dest)
  for stage in C.COUNTS:
    invocations = {suffix:N.read(old['stages'][stage][suffix]) for suffix in ('invocation','pipeline')}
    for suffix,invocation in invocations.items():
      path = N.ROOT/f'{stage}-{suffix}.json'
      cmd = invocation['command']; cmd[cmd.index('--output')+1] = str(N.ROOT/f'{stage}-unused-output.json')
      if '--invocation' in cmd: cmd[cmd.index('--invocation')+1] = str(N.ROOT/f'{stage}-pipeline.json')
      invocation['environment'].update(ALPAMAYO_FP8_CACHE=str(N.CACHE),CACHEDB=str(N.ARTIFACTS/'gpu-cache.db'),TMPDIR=str(N.ARTIFACTS/'tmp'))
      N.write(path,invocation,new=True); path.chmod(0o444)
      files[str(path)] = N.sha(path); data['stages'][stage][suffix] = str(path)
  old_failure_files = [p for p in N.FAILURE.rglob('*') if p.is_file()]
  for path in [N.OLD_PROTOCOL,N.MANIFEST,copy_path,cpu,N.CACHE/'index.json',N.CACHE/'weights.bin',N.SAMPLER,
               N.OLD/'queue-results.json',*old_failure_files]: files[str(path)] = N.sha(path)
  data['cache_retry'] = {'original_protocol':str(N.OLD_PROTOCOL),'original_protocol_sha256':N.OLD_SHA,
    'previous_failure':str(N.FAILURE),'previous_failure_sha256':N.sha(N.FAILURE/'report.agent-expert-combined.json'),
    'cache':str(N.CACHE),'independent_cache_qualification':N.qualification(),
    'copy_report_sha256':N.sha(copy_path),'actual_factory_preflight_sha256':N.sha(cpu),
    'only_changes':'Exact cache materialization, storage paths, pre-GPU CPU cache admission, exclusive retry output names',
    'original_numerical_sources_and_gates_unchanged':True,'repeated_original_test_suites':False}
  N.write(N.PROTOCOL,data,new=True); N.PROTOCOL.chmod(0o444)
  print({'prepared':True,'gpu_executed':False,'protocol_sha256':N.sha(N.PROTOCOL)})


def command(stage,mode,digest):
  cmd = [N.PY,str(N.WRAPPER),out(stage,mode).name,'--program',str(N.ROOT/'agent-expert-combined-cache-run.py'),
    '--timeout-seconds','900','--set','stage',stage,'--set','variant',mode,'--set','protocol',str(N.PROTOCOL),
    '--set','protocol-sha',digest,'--env','PYTHONDONTWRITEBYTECODE','1','--env','TMPDIR',str(N.ARTIFACTS/'tmp'),
    '--env','CACHEDB',str(N.ARTIFACTS/'gpu-cache.db'),'--env','ALPAMAYO_FP8_CACHE',str(N.CACHE),
    '--env','PYTHONPATH',':'.join(map(str,(N.TINY,N.LOADER.parent,N.LOADER.parent.parent)))]
  if mode == 'candidate': cmd += ['--set','reference',str(out(stage,'reference')/'report.json')]
  return cmd


def check_plans(C,data,digest):
  R = N.load(N.ROOT/'agent-expert-combined-cache-run.py','expert_combined_cache_plan_run')
  result = {}
  # Both distinct recipes exercise the real canonical cache loader. Arm-only behavior
  # remains covered by the unchanged original four plans and composed CPU controls.
  for stage,mode in (('quality','reference'),('native','candidate')):
    target = N.ROOT/'preflight'/f'{stage}-{mode}.json'; assert not target.exists()
    args = [N.PY,str(N.ROOT/'agent-expert-combined-cache-run.py'),'--plan','--stage',stage,'--variant',mode,
      '--protocol',str(N.PROTOCOL),'--protocol-sha',digest,'--output',str(target)]
    done = subprocess.run(args,env=R.cpu_env(),capture_output=True,text=True,timeout=120)
    target.parent.mkdir(exist_ok=True); target.with_suffix('.log').write_text(done.stdout+done.stderr); done.check_returncode()
    planned = N.read(target); admission = N.read(target.with_suffix('.agent-expert-combined-cache.json'))
    assert planned['complete'] and planned['plan_only'] and not planned['gpu_executed']
    assert admission['cache_admitted'] and admission['original_runner_completed']
    assert planned['plan']['scheduled_samples'] == data['stages'][stage]['scheduled_samples']
    result[stage+'/'+mode] = {'plan':planned,'cache_admission':admission}
  return result


def main():
  p = argparse.ArgumentParser(description=__doc__)
  mode = p.add_mutually_exclusive_group(); mode.add_argument('--prepare',action='store_true'); mode.add_argument('--check',action='store_true')
  p.add_argument('--protocol-sha'); args = p.parse_args()
  if args.prepare: prepare(); return
  assert args.protocol_sha
  C,data,evidence = N.verify(N.PROTOCOL,args.protocol_sha)
  Q = N.load(N.OLD/'agent-expert-combined-queue.py','expert_combined_cache_original_queue')
  R = N.load(N.OLD/'agent-expert-combined-run.py','expert_combined_cache_original_run')
  # Only output resolution changes in this private queue instance. Its validator is unchanged.
  Q.out = out
  check = N.ROOT/'cpu-preflight.json'
  if not check.exists():
    N.write(check,{'complete':True,'gpu_executed':False,'protocol_sha256':args.protocol_sha,
      'plans':check_plans(C,data,args.protocol_sha),
      'original_cpu_controls_retained':True,'repeated_numeric_suites':False,
      'commands':[command(s,m,args.protocol_sha) for s in C.COUNTS for m in ('reference','candidate')]},new=True)
  proof = N.read(check); assert proof['complete'] and not proof['gpu_executed'] and proof['protocol_sha256'] == args.protocol_sha
  if args.check: print({'cpu_cache_preflight_complete':True,'gpu_executed':False}); return
  assert all(not out(s,m).exists() for s in C.COUNTS for m in ('reference','candidate')), 'Never resume or overwrite a retry result'
  target = N.ROOT/'queue-results.json'; assert not target.exists()
  state = {'complete':False,'protocol_sha256':args.protocol_sha,'jobs':[],'automatic_paced':False}
  def execute(stage,mode):
    N.verify(N.PROTOCOL,args.protocol_sha)
    job = {'stage':stage,'mode':mode,'command':command(stage,mode,args.protocol_sha),'complete':False}
    state['jobs'].append(job); N.write(target,state)
    child = subprocess.run(job['command'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    job['returncode'] = child.returncode; N.write(target,state); child.check_returncode()
    N.verify(N.PROTOCOL,args.protocol_sha)
    admission = N.read((out(stage,mode)/'report.json').with_suffix('.agent-expert-combined-cache.json'))
    assert admission['cache_admitted'] and admission['original_runner_completed'] and 'failure' not in admission
    job['gate'] = Q.validate_child(R,data,evidence,stage,mode,args.protocol_sha)
    job.update(complete=True,report_sha256=N.sha(out(stage,mode)/'report.json')); N.write(target,state)
  try: Q.sequence(execute); state['complete'] = True
  except BaseException as exc:
    state['failure'] = {'type':type(exc).__name__,'message':str(exc)}; raise
  finally: N.write(target,state)


if __name__ == '__main__': main()

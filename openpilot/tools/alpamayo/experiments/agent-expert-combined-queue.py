#!/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python
"""CPU preparation or parent-only gated composition quality4 and native36 queue."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('expert_combined_queue_common',HERE/'agent-expert-combined-common.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
WRAPPER = '/tmp/opencode/run-alpamayo-speed-trial.py'


def out(stage,mode): return C.WORK/f'agent-expert-combined-{stage}-{mode}'


def cpu_env():
  return {**os.environ,'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','TC_OPT':'2','CCACHE':'0','PYTHONDONTWRITEBYTECODE':'1',
    'TMPDIR':str(C.WORK/'agent-expert-combined-artifacts/tmp'),
    'CACHEDB':str(C.WORK/'agent-expert-combined-artifacts/cpu-cache.db'),
    'PYTHONPATH':':'.join(map(str,(C.TINY,C.MLP/'experiments',C.MLP)))}


def prepare():
  assert not C.ROOT.exists() and not C.PROTOCOL.exists()
  assert all(not out(s,m).exists() for s in C.COUNTS for m in ('reference','candidate'))
  c,base = C.base(); _,evidence = C.pv()
  (c.HERE/'tmp').mkdir(exist_ok=True)
  cpu = C.WORK/'agent-expert-combined-cpu.json'
  assert not cpu.exists()
  done = subprocess.run([C.PY,str(HERE/'agent-expert-combined-check.py'),'--output',str(cpu)],
    env=cpu_env(),capture_output=True,text=True,timeout=300)
  (C.WORK/'agent-expert-combined-cpu.log').write_text(done.stdout+done.stderr)
  done.check_returncode()
  reviewed = C.read(cpu); assert reviewed['complete'] and not reviewed['gpu_executed']
  for p,h in reviewed['sources'].items(): assert C.sha(p) == h
  C.ROOT.mkdir()
  files = {str(cpu):C.sha(cpu),str(WRAPPER):C.sha(WRAPPER)}
  for path in sorted(HERE.glob('agent-expert-combined-*.py')):
    dest = C.ROOT/path.name
    dest.write_bytes(path.read_bytes()); dest.chmod(0o444)
    files[str(path)] = C.sha(path); files[str(dest)] = C.sha(dest)
  stages = {}
  for stage,counts in C.COUNTS.items():
    outer = C.read(C.BASE.parent/f'{stage}-invocation.json')
    pipeline = C.read(C.BASE.parent/f'{stage}-pipeline.json')
    cmd = pipeline['command']
    start,end = cmd.index('--samples')+1,cmd.index('--initial-noise')
    original_samples = cmd[start:end]
    selected = [original_samples[i] for i in C.INDICES] if stage == 'quality' else original_samples
    cmd[start:end] = selected
    destination = C.ROOT/f'{stage}-pipeline.json'
    outer['command'][outer['command'].index('--invocation')+1] = str(destination)
    for invocation in (outer,pipeline):
      command = invocation['command']
      for flag,value in (('--output',str(C.ROOT/f'{stage}-unused-output.json')),('--warmup',str(counts[0])),('--runs',str(counts[1]))):
        command[command.index(flag)+1] = value
      invocation['environment'].update(CACHEDB=str(c.HERE/'gpu-cache.db'),TMPDIR=str(c.HERE/'tmp'),
        ALPAMAYO_FP8_CACHE=str(c.HERE/'fp8-cache'),PYTHONDONTWRITEBYTECODE='1')
    C.write(destination,pipeline,new=True)
    outer_path = C.ROOT/f'{stage}-invocation.json'; C.write(outer_path,outer,new=True)
    for path in (destination,outer_path): files[str(path)] = C.sha(path); path.chmod(0o444)
    stages[stage] = {'invocation':str(outer_path),'pipeline':str(destination),'scheduled_samples':selected[:sum(counts)]}
  # Pin all previous evidence separately; do not rewrite its source hashes.
  evidence_paths = [C.BASE,C.PV_PROTOCOL]
  for name in C.PRIOR:
    path = C.WORK/name/'report.json'
    evidence_paths += [path,path.with_suffix('.speculation.json'),path.with_suffix(
      '.agent-exact-combined.json' if name.startswith('agent-exact-combined') else '.agent-expert-pv-qualified.json')]
  for path in evidence_paths: files[str(path)] = C.sha(path)
  data = {'schema':1,'base_sha256':C.BASE_SHA,'pv_protocol_sha256':C.PV_SHA,'files':files,'prior':C.prior(),
    'pv_evidence':evidence,'counts':C.COUNTS,'quality_indices':C.INDICES,'stages':stages,
    'parent_child_timeout_seconds':900,'automatic_paced':False,'gpu_executed_during_preparation':False,
    'policy':'Existing combined MLP2048 + dynamic vision tiles + CPU sampler in both arms; only qualified PV differs',
    'scope':'Fresh four-query composition check, not replacement of either independent quality64 qualification',
    'empty_initial_lookup_and_vision_cache':True,'vision_batches_in_fresh_pair':[2],
    'relocated_byte_identical_core':C.CORE,'source_relocation':'Only private PV common.SNAP/paths and hook.common adapter; numeric source unchanged'}
  C.write(C.PROTOCOL,data,new=True); C.PROTOCOL.chmod(0o444)
  print({'prepared':True,'gpu_executed':False,'protocol_sha256':C.sha(C.PROTOCOL)})


def command(stage,mode,digest):
  args = [C.PY,WRAPPER,out(stage,mode).name,'--program',str(C.ROOT/'agent-expert-combined-run.py'),
    '--timeout-seconds','900','--set','stage',stage,'--set','variant',mode,
    '--set','protocol',str(C.PROTOCOL),'--set','protocol-sha',digest,
    '--env','PYTHONDONTWRITEBYTECODE','1','--env','TMPDIR',str(C.WORK/'agent-expert-combined-artifacts/tmp'),
    '--env','CACHEDB',str(C.WORK/'agent-expert-combined-artifacts/gpu-cache.db'),
    '--env','PYTHONPATH',':'.join(map(str,(C.TINY,C.MLP/'experiments',C.MLP)))]
  if mode == 'candidate': args += ['--set','reference',str(out(stage,'reference')/'report.json')]
  return args


def sequence(execute):
  for stage in C.COUNTS:
    for mode in ('reference','candidate'): execute(stage,mode)


def sequence_checks():
  for index in range(4):
    seen = []
    def execute(stage,mode):
      seen.append((stage,mode))
      if len(seen) == index+1: raise RuntimeError('injected')
    try: sequence(execute)
    except RuntimeError: pass
    else: raise AssertionError('Queue swallowed a failed child')
    assert len(seen) == index+1
  return {'four_stops_checked':True,'quality_failure_blocks_native':True}


def plans(digest):
  result = {}
  for stage in C.COUNTS:
    for mode in ('reference','candidate'):
      target = C.ROOT/'preflight'/f'{stage}-{mode}.json'
      assert not target.exists()
      done = subprocess.run([C.PY,str(C.ROOT/'agent-expert-combined-run.py'),'--plan','--stage',stage,'--variant',mode,
        '--protocol',str(C.PROTOCOL),'--protocol-sha',digest,'--output',str(target)],
        env=cpu_env(),capture_output=True,text=True,timeout=90)
      target.parent.mkdir(exist_ok=True); target.with_suffix('.log').write_text(done.stdout+done.stderr)
      done.check_returncode()
      value = C.read(target)
      assert value['complete'] and value['plan_only'] and not value['gpu_executed'] and value['import_path']['actual_runpy_wgp_import_and_pickle']
      result[stage+'/'+mode] = value
  return result


def validate_child(R,data,evidence,stage,mode,digest):
  directory = out(stage,mode); path = directory/'report.json'
  wall = C.read(directory/'wall.json')
  assert wall['returncode'] == 0 and not wall['timed_out'] and wall['timeout_seconds'] == 900
  report,spec = C.read(path),C.read(path.with_suffix('.speculation.json'))
  assert report['agent_expert_combined'] == C.read(path.with_suffix('.agent-expert-combined.json'))
  assert Path(report['source_snapshot']) == directory/'report-sources'
  gate = R.validate(C,report,spec,C.plan(data,stage,path),stage,mode == 'candidate',evidence,digest)
  if mode == 'candidate':
    ref = out(stage,'reference')/'report.json'
    gate = R.compare(C,C.read(ref),report,C.plan(data,stage,path),C.read(ref.with_suffix('.speculation.json')),spec)
    assert gate == report['agent_expert_combined']['paired_exact'] and gate['samples'] == sum(C.COUNTS[stage])
  return gate


def main():
  p = argparse.ArgumentParser(description=__doc__)
  modes = p.add_mutually_exclusive_group()
  modes.add_argument('--prepare',action='store_true'); modes.add_argument('--check',action='store_true')
  p.add_argument('--protocol-sha')
  args = p.parse_args(); assert sys.flags.optimize == 0
  if args.prepare: prepare(); return
  assert args.protocol_sha
  data,evidence = C.protocol(C.PROTOCOL,args.protocol_sha)
  R = C.load(C.ROOT/'agent-expert-combined-run.py','expert_combined_queue_run')
  check = C.ROOT/'cpu-preflight.json'
  if not check.exists():
    C.write(check,{'complete':True,'gpu_executed':False,'protocol_sha256':args.protocol_sha,
      'plans':plans(args.protocol_sha),'queue_checks':sequence_checks(),
      'commands':[command(s,m,args.protocol_sha) for s in C.COUNTS for m in ('reference','candidate')]},new=True)
  checked = C.read(check)
  assert checked['complete'] and not checked['gpu_executed'] and checked['protocol_sha256'] == args.protocol_sha
  if args.check: print({'cpu_preflight_complete':True,'gpu_executed':False}); return
  assert all(not out(s,m).exists() for s in C.COUNTS for m in ('reference','candidate')), 'Never resume or overwrite GPU evidence'
  target = C.ROOT/'queue-results.json'; assert not target.exists()
  state = {'complete':False,'protocol_sha256':args.protocol_sha,'jobs':[],'automatic_paced':False}
  def execute(stage,mode):
    C.protocol(C.PROTOCOL,args.protocol_sha)
    job = {'stage':stage,'mode':mode,'command':command(stage,mode,args.protocol_sha),'complete':False}
    state['jobs'].append(job); C.write(target,state)
    child = subprocess.run(job['command'],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    job['returncode'] = child.returncode; C.write(target,state); child.check_returncode()
    C.protocol(C.PROTOCOL,args.protocol_sha)
    job['gate'] = validate_child(R,data,evidence,stage,mode,args.protocol_sha)
    job.update(complete=True,report_sha256=C.sha(out(stage,mode)/'report.json')); C.write(target,state)
  try: sequence(execute); state['complete'] = True
  except BaseException as exc:
    state['failure'] = {'type':type(exc).__name__,'message':str(exc)}; raise
  finally: C.write(target,state)


if __name__ == '__main__': main()

"""Real alarm lifecycle and wrapper timeout tests. No GPU or USB operations."""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

EXP=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('ck_deadline',EXP/'agent-ck-paced-deadline-entry.py')
D=importlib.util.module_from_spec(s);s.loader.exec_module(D)

def write(path,value):
  assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def unit(output):
  assert not output.exists();D.ROOT.mkdir(exist_ok=True)
  tree,proof=D.transformed()
  canonical=runpy.run_path(str(D.CANONICAL),run_name='deadline_cpu_canonical')
  budget=canonical['hard_budget'];assert signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
  old=signal.getsignal(signal.SIGALRM);handler=lambda *args:None
  signal.signal(signal.SIGALRM,handler);rejected=[]
  try:
    with budget(180):
      assert signal.getsignal(signal.SIGALRM)==signal.SIG_DFL
      assert 179<signal.getitimer(signal.ITIMER_REAL)[0]<=180
    assert signal.getsignal(signal.SIGALRM) is handler and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
    try:
      with budget(180):raise LookupError('deliberate inner failure')
    except LookupError:pass
    else:raise AssertionError('Inner exception was suppressed')
    assert signal.getsignal(signal.SIGALRM) is handler and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
    for seconds in (0,181):
      try:
        with budget(seconds):raise AssertionError('Invalid budget accepted')
      except ValueError:rejected.append('invalid_budget_'+str(seconds))
    signal.setitimer(signal.ITIMER_REAL,30,2)
    try:
      with budget(180):raise AssertionError('Foreign alarm accepted')
    except ValueError:rejected.append('occupied_foreign_alarm')
    assert signal.getsignal(signal.SIGALRM) is handler
    left,interval=signal.getitimer(signal.ITIMER_REAL);assert 29<left<=30 and interval==2
  finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
  # Execute the wrapper's exact Popen/wait/interrupt/kill statements with a short fixture budget.
  wrapper=D.verify_sources();main=D.function(wrapper,'main')
  log_with=next(n for n in ast.walk(main) if isinstance(n,ast.With) and any(isinstance(x,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='child' for t in x.targets) for x in n.body))
  assert len(log_with.body)==3 and isinstance(log_with.body[1],ast.Try)
  fixture=ast.parse('def wait_fixture(cmd,env,log,args):\n  timed_out=False\n  return timed_out,code\n')
  fixture.body[0].body[1:1]=copy.deepcopy(log_with.body[:2])
  ns={'subprocess':subprocess,'signal':signal};exec(compile(ast.fix_missing_locations(fixture),'<exact-wrapper-wait-fixture>','exec'),ns)
  trials=[]
  for ignore in (False,True):
    code='import signal\n'+('signal.signal(signal.SIGINT,signal.SIG_IGN)\n' if ignore else '')+'signal.pause()\n'
    start=time.monotonic()
    with (D.ROOT/f'child-ignore-{ignore}.log').open('x') as log:
      timed_out,ret=ns['wait_fixture']([sys.executable,'-B','-c',code],os.environ.copy(),log,SimpleNamespace(timeout_seconds=.3))
    elapsed=time.monotonic()-start
    assert timed_out and ret==(-signal.SIGKILL if ignore else -signal.SIGINT) and elapsed<8
    trials.append(dict(ignore_sigint=ignore,returncode=ret,elapsed_seconds=elapsed,original_grace_seconds=5))
  # Actual canonical SIG_DFL expiry, protected by an independent subprocess timeout.
  fn=D.function(ast.parse(D.CANONICAL.read_text()),'hard_budget')
  child='import signal\nfrom contextlib import contextmanager\n'+ast.unparse(fn)+'\nwith hard_budget(.05):signal.pause()\n'
  assert subprocess.run([sys.executable,'-B','-c',child],timeout=3).returncode==-signal.SIGALRM
  name='agent-ck-selected-recovery-parent-paced-v3';out=D.WORK/name/'report.json'
  words=[sys.executable,str(D.WRAPPER),name,'--program',str(EXP/'agent-ck-paced-deadline-entry.py'),'--timeout-seconds','600']
  invocation={'timeout_seconds':600,'command':[sys.executable,words[4],'--output',str(out),'--parent-run']}
  D.parent_contract(words,invocation,out,[str(D.LOCK)])
  for label,changed,record,fds in (
      ('wrong_parent',[words[0],'/foreign.py',*words[2:]],invocation,[str(D.LOCK)]),
      ('wrong_parent_budget',[*words[:-1],'180'],invocation,[str(D.LOCK)]),
      ('wrong_invocation_budget',words,dict(invocation,timeout_seconds=900),[str(D.LOCK)]),
      ('missing_parent_lock',words,invocation,[])):
    try:D.parent_contract(changed,record,out,fds)
    except AssertionError:rejected.append(label)
    else:raise AssertionError('Invalid wrapper contract accepted: '+label)
  write(output,dict(complete=True,gpu_executed=False,source_proof=proof,real_hard_budget_normal_and_exception_restoration=True,
    foreign_alarm_unchanged_after_rejection=True,canonical_default_signal_expiry=True,wrapper_exact_wait_fixture=trials,
    rejections=rejected,source_files={str(p):D.sha(p) for p in D.PINS}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();unit(a.output)

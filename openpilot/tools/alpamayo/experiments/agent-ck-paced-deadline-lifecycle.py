"""Actual new runpy entry through mocked GPU startup, then real canonical hard_budget."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from types import ModuleType

EXP=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('ck_deadline',EXP/'agent-ck-paced-deadline-entry.py')
D=importlib.util.module_from_spec(s);s.loader.exec_module(D)
events=[]

def adjust(source):
  def change(old,new):
    nonlocal source
    assert source.count(old)==1,old;source=source.replace(old,new,1)
  change("argv = [str(HERE/'agent-ck-selected-paced.py'),'--output',str(output),'--cpu-preflight','--quality-report',str(_S.QUALITY)]",
    "argv = [str(HERE/'agent-ck-paced-deadline-entry.py'),'--output',str(output),'--parent-run',"
    "'--quality-report',str(_real_recovery.FAILED/'report.json'),'--quality-report-sha',_real_recovery.FAILED_SHA,"
    "'--recovery-receipt',str(_real_recovery.ROOT/'receipt-v1.json'),'--recovery-receipt-sha',_real_recovery.S.sha(_real_recovery.ROOT/'receipt-v1.json')]")
  change("original(*a,**kw)\n                      raise K.ConstructedBoundary('After complete selected bind')",
    "consumer=original(*a,**kw)\n                      raise K.ConstructedBoundary('After complete selected bind')")
  change('boundary.neural_complete = True\n                      raise', 'boundary.neural_complete = True\n                      return consumer')
  marker="path = Path(location).resolve()\n        if path == HERE/'agent-ck-selected-common.py':"
  inserted="""path = Path(location).resolve()
        if path == HERE/'agent-ck-selected-recovery-common.py':
          original_loader=result.loader
          class RecoveryLoader:
            def create_module(self,spec):return original_loader.create_module(spec)
            def exec_module(self,module):
              original_loader.exec_module(module)
              module.gate=_real_recovery.gate
              module.verify_protocol=_real_recovery.verify_protocol
              module.configure=_S.configure
          result.loader=RecoveryLoader();return result
        if path == C.N.LIVE/'experiments/agent-paced-replay.py':
          original_loader=result.loader
          class BudgetLoader:
            def create_module(self,spec):return original_loader.create_module(spec)
            def exec_module(self,module):
              original_loader.exec_module(module)
              assert Path(module.hard_budget.__wrapped__.__code__.co_filename)==path
              def start(clock):
                assert boundary.factory_seen and boundary.neural_complete
                assert boundary.hooks.ck_report['startup_ready']
                assert boundary.hooks.ck_report['consumer_owns_warmed_model']
                assert clock.anchor is None
                remaining,interval=_signal.getitimer(_signal.ITIMER_REAL)
                assert 179<remaining<=180 and interval==0
                assert _signal.getsignal(_signal.SIGALRM)==_signal.SIG_DFL
                _deadline_events.append('actual_startup_bind_then_canonical_180s_budget_entered')
                raise K.ConstructedBoundary('CPU stop after original hard_budget entry, before clock/thread start')
              stack.enter_context(patch.object(module.Clock,'start',start))
          result.loader=BudgetLoader();return result
        if path == HERE/'agent-ck-selected-common.py':"""
  change(marker,inserted)
  return source

def main(output):
  assert not output.exists();os.environ.update(ALLOW_DEVICE_USAGE='0',DEV='CPU')
  # Parent-seeded firmware/cache remains untouched. CPU storage uses only this new subtree.
  root=output.parent/(output.stem+'-state');root.mkdir()
  old=D.WORK/'agent-ck-selected-recovery-work/artifacts/hooks';new=root/'artifacts/hooks'
  for rel in ('agent-exact-combined-install.py','agent-exact-live-artifacts/12460b8ec1eae4609835e65cab03ec855bff660cc872123cf1494f652cee8dd7.so'):
    (new/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/rel,new/rel)
  for rel in ('tmp','cache/huggingface','cache/torch'):(root/'artifacts'/rel).mkdir(parents=True,exist_ok=True)
  real_s=importlib.util.spec_from_file_location('deadline_actual_recovery',EXP/'agent-ck-selected-recovery-common.py')
  recovery=importlib.util.module_from_spec(real_s);real_s.loader.exec_module(recovery)
  text=(EXP/'agent-ck-selected-check.py').read_text()
  marker="  m=ModuleType('selected_actual_factory_check');m.__file__=str(EXP/'agent-arrival-warp-ownership-check.py')"
  assert text.count(marker)==1;text=text.replace(marker,'  source=_adjust(source)\n'+marker,1)
  marker='  m.A,m.C,m._K,m._A,m._C,m._S,m._H,m._data=A,C,K,A,C,S,H,data'
  assert text.count(marker)==1;text=text.replace(marker,marker+'\n  m._real_recovery,m._deadline_events,m._signal=_real_recovery,_deadline_events,_signal',1)
  m=ModuleType('deadline_actual_selected_check');m.__file__=str(EXP/'agent-ck-selected-check.py')
  exec(compile(text,m.__file__,'exec'),m.__dict__)
  m.S.ROOT=root;m._adjust=adjust;m._real_recovery=recovery;m._deadline_events=events;m._signal=signal
  # The normal parent process/lock is a declared CPU fixture. Its real strict contract is tested unchanged.
  def parent_fixture(a):
    assert signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
    name='agent-ck-selected-recovery-parent-paced-v3';out=D.WORK/name/'report.json'
    words=[sys.executable,str(D.WRAPPER),name,'--program',str(EXP/'agent-ck-paced-deadline-entry.py'),'--timeout-seconds','600']
    D.parent_contract(words,{'timeout_seconds':600,'command':[sys.executable,words[4],'--output',str(out),'--parent-run']},out,[str(D.LOCK)])
    events.append('wrapper_600s_contract_fixture_no_process_alarm')
  old_profile=sys.getprofile();old_handler=signal.getsignal(signal.SIGALRM)
  def profile(frame,event,arg):
    if event=='return' and frame.f_code.co_name=='build' and Path(frame.f_code.co_filename)==EXP/'agent-ck-paced-deadline-entry.py':
      arg[0]._require_wrapper=parent_fixture
  sys.setprofile(profile)
  try:m.paced(output)
  finally:sys.setprofile(old_profile)
  assert signal.getsignal(signal.SIGALRM)==old_handler and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
  assert events==['wrapper_600s_contract_fixture_no_process_alarm','actual_startup_bind_then_canonical_180s_budget_entered']
  result=json.loads(output.read_text());assert result['complete'] and not result['gpu_executed']
  result['deadline_lifecycle']=dict(events=events,original_hard_budget_unmodified=True,handler_and_timer_restored=True,
    source_clock_started=False,parent_identity_mocked=True,actual_receipt_gate_unmodified=True)
  output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)

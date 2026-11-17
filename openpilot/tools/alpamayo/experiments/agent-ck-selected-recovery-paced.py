"""Same selected paced computation; admit only a hash-bound independent recovery receipt."""
import argparse
import importlib.util
import os
from pathlib import Path
import signal
from types import ModuleType,SimpleNamespace

s=importlib.util.spec_from_file_location('ck_selected_recovery',Path(__file__).with_name('agent-ck-selected-recovery-common.py'))
R=importlib.util.module_from_spec(s);s.loader.exec_module(R)

def execute(a):
  R.verify_protocol();R.configure()
  receipt=R.gate(a.quality_report,a.quality_report_sha,a.recovery_receipt,a.recovery_receipt_sha)
  # Private namespace adapter only. Installer, native scope, startup, Consumer, source and final validators are unchanged.
  selected=SimpleNamespace(**vars(R.S));selected.configure=R.configure
  selected.quality_gate=lambda path:R.gate(path,a.quality_report_sha,a.recovery_receipt,a.recovery_receipt_sha)
  m=ModuleType('ck_selected_recovery_paced_original');m.__file__=str(R.EXP/'agent-ck-selected-paced.py')
  exec(compile(Path(m.__file__).read_text(),m.__file__,'exec'),m.__dict__);m.S=selected
  try:
    m.execute(a.output,a.quality_report)
  finally:
    side=a.output.parent/'ck-selected-status.json'
    if side.exists():
      state=R.S.read(side);state['quality_recovery']={'original_failed_report_sha256':a.quality_report_sha,
        'receipt_sha256':a.recovery_receipt_sha,'original_completed':False,'exact64':receipt['exact']['exact'],
        'recovery_protocol_sha256':R.S.sha(R.ROOT/'protocol.json')};R.S.write(side,state)
  report=R.S.read(a.output);report['agent_ck_selected']['quality_recovery']=R.S.read(side)['quality_recovery']
  R.verify_protocol();R.S.write(a.output,report)

def main():
  p=argparse.ArgumentParser();p.add_argument('--parent-run',action='store_true');p.add_argument('--output',type=Path,required=True)
  for n in ('quality-report','quality-report-sha','recovery-receipt','recovery-receipt-sha'):p.add_argument('--'+n,required=True)
  a=p.parse_args();a.quality_report=Path(a.quality_report);a.recovery_receipt=Path(a.recovery_receipt)
  # Apply private cache and numerical environment before receipt validation imports tinygrad.
  R.verify_protocol();R.configure();R.gate(a.quality_report,a.quality_report_sha,a.recovery_receipt,a.recovery_receipt_sha)
  assert not a.output.exists() and a.output.resolve().is_relative_to(R.S.WORK)
  if not a.parent_run:print({'verified':True,'gpu_executed':False,'recovery_exact64':True});return
  os.environ['ALLOW_DEVICE_USAGE']='1'
  def expired(*args):raise TimeoutError('Selected recovery paced child reached 580 seconds')
  old=signal.signal(signal.SIGALRM,expired);signal.alarm(580)
  try:execute(a)
  finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)

if __name__=='__main__':main()

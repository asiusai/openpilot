"""Separate eventual paced launch. A completed selected quality result is mandatory."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import signal
from types import ModuleType,SimpleNamespace

s=importlib.util.spec_from_file_location('ck_selected_common',Path(__file__).with_name('agent-ck-selected-common.py'))
S=importlib.util.module_from_spec(s);s.loader.exec_module(S)

def execute(output,quality,*,cpu_preflight=False):
  plan=S.verify();S.quality_gate(quality);A,C,N=S.configure()
  source=(S.EXP/'agent-arrival-warp-integrated-run.py').read_text()
  start=source.index("   admission =") if "   admission =" in source else source.index("  admission =")
  end=source.index('  m, manifest =',start)
  # The source-pinned CPU report is checked by this entry. Reuse completed cache admission, do not launch an old probe into old artifacts.
  source=source[:start]+"  assert data['cpu_proof']['complete'] and not data['cpu_proof']['gpu_executed']\n"+source[end:]
  oldload=A.load
  created=[]
  def load(path,name):
    if Path(path).resolve()==S.EXP/'agent-arrival-warp-integrated-install.py':
      module=S.module(S.EXP/'agent-ck-selected-install.py',name)
      real=module.factory
      def factory(common):
        Base=real(S)
        class Owned(Base):
          def __init__(self,*a,**kw):super().__init__(*a,**kw);created.append(self)
        return Owned
      module.factory=factory;return module
    return oldload(path,name)
  adapter=SimpleNamespace(**vars(A));adapter.load=load
  adapter.configure=lambda:S.configure()
  def verify(path,digest,**kwargs):
    assert Path(path)==S.PREPARED/'protocol.json' and S.sha(path)==digest
    return S.verify()
  adapter.verify=verify
  # Old pins remain evidence labels. They never claim the new CK numerics equal the prior original-attention paced predictions.
  m=ModuleType('ck_selected_paced_original');m.__file__=str(S.EXP/'agent-arrival-warp-integrated-run.py')
  exec(compile(source,m.__file__,'exec'),m.__dict__);m.A=adapter;m.C=C
  args=SimpleNamespace(output=output,protocol=S.PREPARED/'protocol.json',protocol_sha=S.sha(S.PREPARED/'protocol.json'),cpu_preflight=cpu_preflight)
  side=output.parent/'ck-selected-status.json'
  state=dict(completed=False,gpu_executed=False,quality_report_sha256=S.sha(quality),protocol_sha256=args.protocol_sha,
    current_input_startup=True,auto_chained=False,default_adoption=False)
  try:
    m.run(args)
    assert len(created)==1;hooks=created[0]
    assert hooks.ck_report['completed'] and hooks.ck_report['cleanup_exact']
    assert hooks.ownership['finally_restored']
    report=S.read(output);report['agent_ck_selected']=copy.deepcopy(hooks.ck_report)
    report['agent_ck_selected']['quality_report_sha256']=S.sha(quality)
    S.write(output,report);state.update(completed=True,gpu_executed=True)
  except BaseException as e:
    state['failure']={'type':type(e).__name__,'message':str(e)};raise
  finally:
    state['owners']=[{'ck':x.ck_report,'arrival_ownership':x.ownership} for x in created]
    if output.parent.exists():S.write(side,state)

def main():
  p=argparse.ArgumentParser();p.add_argument('--parent-run',action='store_true');p.add_argument('--quality-report',type=Path)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-preflight',action='store_true');a=p.parse_args();S.verify()
  if a.cpu_preflight:
    assert os.environ.get('ALLOW_DEVICE_USAGE')=='0' and os.environ.get('DEV')=='CPU'
    execute(a.output,a.quality_report,cpu_preflight=True);return
  if not a.parent_run:print({'verified':True,'gpu_executed':False,'quality_review_before_launch':True});return
  assert a.quality_report is not None;S.quality_gate(a.quality_report)
  os.environ['ALLOW_DEVICE_USAGE']='1'
  def expired(*args):raise TimeoutError('Selected paced child reached 580 seconds')
  signal.signal(signal.SIGALRM,expired);signal.alarm(580)
  try:execute(a.output,a.quality_report)
  finally:signal.alarm(0)

if __name__=='__main__':main()

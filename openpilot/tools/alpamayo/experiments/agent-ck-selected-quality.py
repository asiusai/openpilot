"""One fresh selected+CK chronological64 run. No automatic Consumer/paced launch."""
import argparse
from contextlib import ExitStack
import copy
import importlib.util
import os
from pathlib import Path
import sys

s=importlib.util.spec_from_file_location('ck_selected_common',Path(__file__).with_name('agent-ck-selected-common.py'))
S=importlib.util.module_from_spec(s);s.loader.exec_module(S)

def execute(plan,folder):
  A,C,N=S.configure()
  installer=C.load(C.OLD/'agent-expert-live-storage-install.py','ck_selected_original_numerical')
  hooks=installer.factory(C.S,pv_candidate=True)('consumer')
  composition=hooks.composition
  R=S.runner(S.C);R.configure('candidate')
  # Storage agrees with the genuine selected FP8 constructor observer.
  os.environ.update(N.storage());report=None;execution_ok=False
  def coverage(report):
    report['selected_coverage']=composition.coverage()
    assert composition.session.frame is None and not composition.session.state['prefill']
    _,evidence=N.pv();N.validate_pv(composition.pv_handle.info,True,64,evidence)
    report['selected_pv']=copy.deepcopy(composition.pv_handle.info)
    report['selected_storage']=copy.deepcopy(composition.storage_events)
  S.C.selected_coverage=coverage
  try:
    with composition:
      report=R.run(plan,'candidate',folder)
      S.T.validate_arm(report)
      report['selected_exact']=S.exact(S.read(S.QUALITY),report)
      report['source_after_exact']=S.verify()['files']==plan['files']
      assert report['source_after_exact']
    execution_ok=True
  finally:
    if (folder/'report.json').exists():
      report=S.read(folder/'report.json') if report is None else report
      report['selected_protocol_sha256']=S.sha(S.PREPARED/'protocol.json')
      report['selected_restorations']=composition.restorations
      report['selected_constructor_restored']=composition.constructor_restored
      report['selected_pv_cleanup']=composition.pv_handle.info if composition.pv_handle else None
      cleanup=bool(composition.constructor_restored and composition.session is not None and composition.session.stats['methods_restored']
        and composition.pv_handle is not None and composition.pv_handle.info['cleanup_exact'] and composition.pv_handle.info['observer_cleanup_exact']
        and all(r['restored'] for r in composition.restorations))
      report['completed']=bool(execution_ok and report['completed'] and report.get('selected_exact',{}).get('exact') and cleanup)
      report['hooks_restored']=bool(report['hooks_restored'] and (cleanup or not report['gpu_executed']))
      S.write(folder/'report.json',report)
  return report

def main():
  p=argparse.ArgumentParser();p.add_argument('--parent-run',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  plan=S.verify();assert not a.output.exists() and a.output.resolve().is_relative_to(S.WORK)
  if not a.parent_run:print({'verified':True,'gpu_executed':False});return
  folder=a.output.parent
  assert not any((folder/name).exists() for name in ('report.json','pipeline.json','pipeline-sources','pipeline.speculation.json'))
  folder.mkdir(parents=True,exist_ok=True)
  os.environ['ALLOW_DEVICE_USAGE']='1'
  result=execute(plan,folder);assert result['completed'];S.write(a.output,result)

if __name__=='__main__':main()

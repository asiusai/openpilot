"""Parent-only lean timing arm, admitted by an exact completed quality64 review."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

s=importlib.util.spec_from_file_location('ck_timing_common',Path(__file__).with_name('agent-ck-quality64-timing-common.py'))
T=importlib.util.module_from_spec(s);s.loader.exec_module(T)

def main():
  assert not sys.flags.optimize
  p=argparse.ArgumentParser();p.add_argument('--arm',choices=('baseline','candidate'),required=True)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path);p.add_argument('--parent-run',action='store_true');a=p.parse_args()
  plan=T.verify();folder=T.ROOT/a.arm;original=folder/'report.json'
  assert not folder.exists() and a.output.parent.is_dir() and not a.output.exists() and a.output.resolve()!=original.resolve()
  if not a.parent_run:
    print(json.dumps({'verified':True,'gpu_executed':False,'launch_still_requires_full64_review':True}));return
  assert a.review is not None,'Timing requires the completed full64 review'
  review,quality=T.review_gate(a.review)
  folder.mkdir();R=T.runner(T.C);R.configure(a.arm);os.environ['ALLOW_DEVICE_USAGE']='1';passed=False;failure=None
  try:
    r=R.run(plan,a.arm,folder);T.validate_arm(r);passed=T.exact_quality_parity(quality[a.arm],r)
  except BaseException as e: failure={'type':type(e).__name__,'message':str(e)};raise
  finally:
    if original.exists():
      r=json.loads(original.read_text());r['inference_completed']=r['completed'];r['completed']=r['completed'] and passed
      r['timing_identity_passed']=passed;r['timing_review_sha256']=T.Q.sha(a.review);r['quality_comparison_sha256']=review['comparison_sha256']
      r['quality_report_sha256']=T.Q.sha(Path(review[a.arm+'_report']))
      if failure is not None: r['timing_failure']=failure
      r['parent_bridge']={'original_report':str(original),'original_sha256':T.Q.sha(original),'entry_sha256':T.Q.sha(__file__)}
      T.Q.write(a.output,r)

if __name__=='__main__': main()

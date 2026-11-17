"""Complete lean CPU lifecycle and parity checks. No timing admission or GPU evidence."""
import argparse
import importlib.util
import json
from pathlib import Path

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_timing_common',EXP/'agent-ck-quality64-timing-common.py')
T=importlib.util.module_from_spec(s);s.loader.exec_module(T)

def main(output):
  checker=T.Q.module('agent-ck-quality64-check');checker.run_check(output,T.C)
  root=output.parent/output.stem
  for arm in ('baseline','candidate'):
    q=json.loads((T.Q.ROOT/'cpu-final'/arm/'report.json').read_text());r=json.loads((root/arm/'report.json').read_text())
    assert not q['gpu_executed'] and not r['gpu_executed'];T.exact_quality_parity(q,r)
  r=json.loads(output.read_text())
  # Only first-call seeding plus the deliberately injected first-call failure may execute poison in this CPU test.
  assert r['executed_CPU_mock_calls']['poison']==58,r['executed_CPU_mock_calls']
  for arm,n in (('baseline',336),('candidate',5376)):
    report=json.loads((root/arm/'report.json').read_text());assert report['readback_bytes']==n
  r['lean_checks']={'no_captured_poison':True,'audit_reads_only_queries_0_1_63':True,'same_arm_CPU_mock_predictions_and_history_exact':True,
    'native_arithmetic_not_proven_by_mock':True,'captured_audit_construction_forbidden':True}
  output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
  print(json.dumps({'lean_full64_CPU_lifecycle_passed':True,'gpu_executed':False}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)

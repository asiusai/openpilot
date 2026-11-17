"""Opt-in connected 25-update feasibility pilot using the qualified KFD1 training launcher."""
import importlib.util
import inspect
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('connected_joint_pilot_core',HERE/'agent-distill-objective-pilot-core.py')
P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)


def load():
  for name,digest in [('agent-distill-objective-qualify-run.py','da33d23791d1f66c4e2b85cffe6c5904d228ba52eac467946bbfe9c1b7e6ed47'),
      ('agent-distill-objective-qualify-core.py','3ddab9f2ee0ad218e28599677416a8cc0d7ff6fe295c84e3782ce90325f381eb')]:
    if P.sha(HERE/name)!=digest:raise ValueError('Changed qualified training implementation')
  original=P.module('connected_joint_pilot_launcher',HERE/'agent-distill-objective-qualify-run.py')
  original.Q=P
  source=inspect.getsource(original.main)
  addition="parser.add_argument('--validation-report',type=Path);parser.add_argument('--validation-report-sha')"
  before="parser.add_argument('--aligned-report',type=Path);parser.add_argument('--aligned-report-sha')"
  assert source.count(before)==1;source=source.replace(before,before+'\n  '+addition)
  before='row,context=Q.control(args.aligned_report,args.aligned_report_sha)'
  assert source.count(before)==1
  source=source.replace(before,'row,context=Q.control(args.aligned_report,args.aligned_report_sha,args.validation_report,args.validation_report_sha)')
  source=source.replace('--aligned-report','--train-report').replace('args.aligned_report','args.train_report')
  source=source.replace('--execute-qualification','--execute-pilot').replace('args.execute_qualification','args.execute_pilot')
  exec(compile(source,str(Path(__file__))+':qualified-launcher-overlay','exec'),original.__dict__)
  old_admission=original.admission
  def admission(args,context):
    receipt=old_admission(args,context)
    if receipt.get('aligned_target_sha256')!=context['aligned_target_sha256']:raise ValueError('Parent admission must bind both completed targets')
    if receipt.get('fixed_updates')!=P.UPDATES or receipt.get('qualified_report_sha256')!=P.QUALIFIED_SHA:raise ValueError('Changed bounded pilot')
    checked=P.read(receipt['cpu_check']['path'])
    for name in ('agent-distill-objective-pilot-core.py','agent-distill-objective-pilot-run.py','agent-distill-objective-pilot-check.py'):
      if checked['source_sha256'].get(str(HERE/name))!=P.sha(HERE/name):raise ValueError('Changed pilot CPU qualification')
    assert checked['full_shape']['qualified_graph_identical'] and checked['schedule']['validation_steps']==[0,25]
    assert checked['new_bounds']==P.BOUNDS and P.estimate()['projected_seconds']<900
    return receipt
  original.admission=admission
  old_child=original.child
  def child(args,row,context,receipt):
    try:return old_child(args,row,context,receipt)
    finally:
      path=args.output_dir/'report.json'
      if path.exists():
        report=P.read(path);report.update(qualified_resource_report_sha256=P.QUALIFIED_SHA,
          initial_weights='Immutable language100/expert25 masters with fresh joint moments; qualification step1 excluded',
          full_model_comparison_required=True,automatic_model_adoption=False)
        report['source_sha256'].update({str(p):P.sha(p) for p in HERE.glob('agent-distill-objective-pilot-*.py')})
        P.S.atomic_json(path,report)
  original.child=child;original.__file__=str(Path(__file__).resolve())
  return original


R=load();Q=P;TARGET=R.TARGET;LEDGER_SHA=R.LEDGER_SHA;main=R.main
if __name__=='__main__':main()

"""Narrow 320MiB single-allocation overlay; unchanged objective, optimizer, state context and KFD ledger."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-objective-scratch-artifacts'
FAILED=WORK/'agent-distill-objective-qualify-artifacts/live-v1'
FAILED_SHA='f351bf84b354a1d425fec7c0783144cc8617c10d918f7123c5f1cb0c7fb79ee6'
STATE_SHA='e0931a98bdf39a3b05ca2a35027f5a057e0a341b0e008761badb3b24f38064d3'
EVIDENCE=ROOT/'compiler-v1/report.json'
EVIDENCE_SHA='1111881fefc3f6b30a6b185b554f2d6e632df0e7a424232bbd6f5c55b436ca30'
EXCEPTION=dict(old=256*1024**2,new=320*1024**2,failed_report_sha256=FAILED_SHA)
PINS={'agent-distill-objective-qualify-run.py':'da33d23791d1f66c4e2b85cffe6c5904d228ba52eac467946bbfe9c1b7e6ed47',
      'agent-distill-objective-qualify-core.py':'3ddab9f2ee0ad218e28599677416a8cc0d7ff6fe295c84e3782ce90325f381eb'}


def load():
  for name,digest in PINS.items():
    if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:raise ValueError('Reviewed qualification source changed')
  spec=importlib.util.spec_from_file_location('scratch_original_qualification',HERE/'agent-distill-objective-qualify-run.py')
  original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
  Q=original.Q;old_bounds=Q.BOUNDS.copy()
  assert old_bounds['max_single_buffer']==EXCEPTION['old']
  Q.BOUNDS=dict(old_bounds,max_single_buffer=EXCEPTION['new']);Q.ROOT=ROOT

  def transition(context):
    if Q.sha(FAILED/'report.json')!=FAILED_SHA or Q.sha(EVIDENCE)!=EVIDENCE_SHA:raise ValueError('Changed completed scratch evidence')
    prior=Q.read(FAILED/'report.json');pointer=Q.read(FAILED/'latest.json')
    if prior['context']!=context or pointer['context']!=context:raise ValueError('Recovery context changed; no restamp permitted')
    if prior['bounds']!=old_bounds or prior['complete'] or prior['failure']['message']!='buffer of size 279.97MB is too large':
      raise ValueError('Unexpected original qualification result')
    if pointer['metadata']['step']!=0 or Q.read(FAILED/'journal.json')!=dict(committed=0,intended=None):raise ValueError('Original update state is not clean zero')
    if pointer['file']!='checkpoint-0000.npz' or pointer['sha256']!=STATE_SHA or Q.sha(FAILED/pointer['file'])!=STATE_SHA:
      raise ValueError('Original complete step-zero checkpoint changed')
    evidence=Q.read(EVIDENCE)
    assert evidence['requested_scratch_bytes']==279969792 and evidence['maximum_kernel']['recompiled_binary_exact']
    assert {k:v for k,v in Q.BOUNDS.items() if k!='max_single_buffer'}=={k:v for k,v in old_bounds.items() if k!='max_single_buffer'}
    return dict(single_buffer_cap=EXCEPTION,other_bounds_unchanged=True,graph_or_optimizer_change=False,
      original_report=str(FAILED/'report.json'),original_report_sha256=FAILED_SHA,
      original_step_zero=str(FAILED/'checkpoint-0000.npz'),original_step_zero_sha256=STATE_SHA,
      compiler_evidence=str(EVIDENCE),compiler_evidence_sha256=EVIDENCE_SHA,
      optimizer_context_unchanged=True,original_artifacts_modified=False)
  original.scratch_transition=transition

  def checked_resume(path,context):
    if path is None:return
    if path.resolve()==FAILED:original.scratch_transition(context);return
    if not path.resolve().is_relative_to(ROOT):raise ValueError('Resume only the clean original zero state or this isolated experiment')
  original.scratch_checked_resume=checked_resume
  source=inspect.getsource(original.main)
  before="if args.resume_from and not args.resume_from.resolve().is_relative_to(Q.ROOT):raise ValueError('Resume only an isolated qualification checkpoint')"
  if source.count(before)!=1:raise ValueError('Qualification resume boundary changed')
  exec(compile(source.replace(before,'scratch_checked_resume(args.resume_from,context)'),str(Path(__file__))+':resume-overlay','exec'),original.__dict__)

  base_admission=original.admission
  def admission(args,context):
    receipt=base_admission(args,context)
    if receipt.get('single_buffer_exception')!=EXCEPTION:raise ValueError('Parent review must identify the exact single-buffer change')
    checked=Q.read(receipt['cpu_check']['path'])
    for name in ('agent-distill-objective-scratch-run.py','agent-distill-objective-scratch-check.py'):
      if checked['source_sha256'].get(str(HERE/name))!=Q.sha(HERE/name):raise ValueError('Changed scratch CPU qualification')
    if not checked['scratch_link']['all_passed'] or checked['new_bounds']!=Q.BOUNDS:raise ValueError('New cap behavior lacks CPU evidence')
    original.scratch_transition(context)
    return receipt
  original.admission=admission
  base_child=original.child
  def child(args,row,context,receipt):
    change=original.scratch_transition(context)
    change.update(resume_from=str(args.resume_from.resolve()) if args.resume_from else None,checkpoint_context_rewritten=False)
    try:return base_child(args,row,context,receipt)
    finally:
      path=args.output_dir/'report.json'
      if path.exists():
        report=Q.read(path);report['resource_transition']=change
        report['source_sha256'].update({str(p):Q.sha(p) for p in HERE.glob('agent-distill-objective-scratch-*.py')})
        Q.S.atomic_json(path,report)
  original.child=child
  # The original bounded dispatcher must launch this overlay in its child.
  original.__file__=str(Path(__file__).resolve())
  return original


R=load();Q=R.Q;TARGET=R.TARGET;LEDGER_SHA=R.LEDGER_SHA;main=R.main
if __name__=='__main__':main()

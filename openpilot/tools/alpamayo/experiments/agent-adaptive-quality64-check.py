"""Execute the existing full production CPU harness on all64 and the new runner.

Only the CPU harness fixture/count/entrypoint doubles are adapted. The actual
pipeline main, factory entry, coverage, numeric policy, and cleanup execute.
The deliberately zero-action neural double must fail the real motion screen.
"""
import argparse
import ast
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('quality64_check_common',HERE/'agent-adaptive-quality64-common.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)


def main():
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
  p.add_argument('--timing',action='store_true');p.add_argument('--quality-cpu-output',type=Path)
  args=p.parse_args();assert not args.output.exists()
  plan=Q.read(Q.PLAN);K,R,old,N,C,evidence=Q.base()
  assert plan['samples']==[item['sample'] for item in Q.verify_inputs()]
  baseline=Q.WORK/'agent-exact-combined-quality-candidate/report.json'
  assert Q.sha(baseline)==C.PRIOR['agent-exact-combined-quality-candidate']
  source=HERE/'agent-adaptive-coverage-check.py';assert Q.sha(source)==Q.PINS[source]
  text=source.read_text();edits=[]
  def change(old,new,n=1):
    nonlocal text
    assert text.count(old)==n,(old,text.count(old),n)
    text=text.replace(old,new);edits.append({'before':old,'after':new,'occurrences':n})
  change("R,plan,N,C,evidence=K.frozen()", "R,plan,N,C,evidence=CPU_R,CPU_PLAN,CPU_N,CPU_C,CPU_EVIDENCE\n  K.REFERENCE=CPU_BASELINE")
  change("'agent-adaptive-coverage-cpu-artifacts'", "'agent-adaptive-quality64-cpu-artifacts'")
  change("M=K.load(HERE/'agent-adaptive-coverage-run.py','coverage_actual_runner')", "M=K.load(HERE/'agent-adaptive-quality64-run.py','quality64_actual_runner')\n  Q=M.Q")
  change('original_factory=K.factory','original_factory=Q.composition')
  change('def factory(C):\n      Base=original_factory(C)','def factory(K,C,count):\n      Base=original_factory(K,C,count)')
  change("patch.object(M.K,'factory',factory)","patch.object(M.Q,'composition',factory)")
  change("patch.object(M.K,'load',load)","patch.object(M.Q,'load',load)")
  change("current['index']+=1;index=current['index'];tokens=saved_rows[index]['prediction']['generated_token_ids'];current['tokens']=tokens",
    "current['index']+=1;index=current['index']%64;tokens=saved_rows[index]['prediction']['generated_token_ids'];current['tokens']=tokens")
  change("auditing=not state.audit_done", "auditing=state.audit_prefix_once and not state.audit_done")
  change("saved_rows[current['index']]", "saved_rows[current['index']%64]")
  change("if args0.inject_failure and index==3:", "if args0.inject_failure and index==17:")
  change("namespace['main']()\n       return namespace".replace('       ','      '),
    "if kw.get('run_name')!='quality64_original_pipeline':namespace['main']()\n      return namespace")
  change("stack.enter_context(patch.object(M,'PIPELINE',cpu_pipeline))", "plan=dict(plan,pipeline=str(cpu_pipeline))")
  change("stack.enter_context(patch.object(M,'verify',lambda:{'CPU_TEST_ONLY':True}))", "stack.enter_context(patch.object(M.Q,'verify',lambda:plan))")
  change("receipt=K.reference_receipt(C)", "receipt={'CPU_NEURAL_DOUBLES':True,'reference':str(CPU_BASELINE)}")
  change("stack.enter_context(patch.object(M.K,'RECEIPT',cpu_receipt))", "pass # CPU receipt is explicit, never admitted GPU evidence")
  change("stack.enter_context(patch.object(M,'MANIFEST',cpu_manifest))", "stack.enter_context(patch.object(M.Q,'PLAN',cpu_manifest))")
  change("try:M.run(output)", """original_output=M.Q.output
    fake_reference=root/'reference/report.json';K.write(fake_reference,baseline,new=True)
    K.write(M.Q.status_path(fake_reference),{'complete':True,'gpu_executed':True,'arm':'reference',
      'plan_sha256':K.sha(cpu_manifest),'report_sha256':K.sha(fake_reference),'CPU_NEURAL_ROLE_ONLY':True},new=True)
    def paths(stage,arm):
      if stage=='quality':return CPU_QUALITY if CPU_TIMING and arm=='candidate' else fake_reference
      return original_output(stage,arm)
    stack.enter_context(patch.object(M.Q,'output',paths))
    if CPU_TIMING:stack.enter_context(patch.object(M,'require_quality',lambda:{'CPU_ONLY_QUALITY_BOUNDARY':True}))
    # Motion computation remains real. Route the new review artifact into this CPU directory.
    production_write=M.Q.write
    def write(path,value,new=True):
      if Path(path)==M.Q.WORK/'agent-adaptive-quality64-review.json':path=root/'cpu-motion-review.json'
      return production_write(path,value,new=new)
    production_sha=M.Q.sha
    def sha(path):
      if Path(path)==M.Q.WORK/'agent-adaptive-quality64-review.json':path=root/'cpu-motion-review.json'
      return production_sha(path)
    stack.enter_context(patch.object(M.Q,'write',write));stack.enter_context(patch.object(M.Q,'sha',sha))
    try:M.run(output,stage='timing' if CPU_TIMING else 'quality')""")
  change("if not args.inject_failure:raise\n      failure=str(exc)",
    "if not args.inject_failure and (CPU_TIMING or str(exc)!='Full64 or one of four 16-query windows exceeds 5% ADE/FDE'):raise\n      failure=str(exc)")
  change("status=K.read(output.parent/'adaptive-coverage-status.json')", "status=K.read(M.Q.status_path(output))")
  change("assert status['complete'] and not status['gpu_executed'] and status['import_hook_restored'] and status['adaptive_hook_restored']",
    "assert status['complete']==CPU_TIMING and not status['gpu_executed'] and status['import_hook_restored'] and status['adaptive_hook_restored']\n    assert CPU_TIMING or failure=='Full64 or one of four 16-query windows exceeds 5% ADE/FDE'")
  change("status['checks']['shape_coverage']", "status['checks']['coverage']")
  change("status['checks']['adaptive']['genuine_partial_queries']", "status['checks']['adaptive']['partial']")
  change("[r['vision_misses'] for r in saved_rows]", "[r['vision_misses'] for r in saved_rows]*(2 if CPU_TIMING else 1)")
  change("failure_snapshot=M.safe_snapshot(comp,test_overlay[0])", "failure_snapshot=K.load(HERE/'agent-adaptive-coverage-run.py','cpu_saved_snapshot').safe_snapshot(comp,test_overlay[0])")
  change("assert failure_snapshot['adaptive']['states'][0]['completed']==18", "assert failure_snapshot['adaptive']['states'][0]['completed']==64")
  change("'all_18_postvalidation':True", "'all_64_postvalidation':True,'timing_rewind_and_same_arm_exact':CPU_TIMING,'motion_gate_rejection_exercised':not CPU_TIMING")
  change("'production_status':str(output.parent/'adaptive-coverage-status.json')", "'production_status':str(M.Q.status_path(output))",n=2)
  change("HERE.glob('agent-adaptive-coverage-*.py')", "HERE.glob('agent-adaptive-quality64-*.py')",n=2)
  change("'queries':18", "'queries':128 if CPU_TIMING else 64")
  # Coverage body keeps every assertion; only this obsolete text label differs.
  tree=ast.parse(text);compile(tree,str(source)+':full64-cpu-fixture','exec')
  namespace={'__name__':'quality64_reused_cpu_harness','__file__':str(Path(__file__)),
    'CPU_R':R,'CPU_PLAN':plan,'CPU_N':N,'CPU_C':C,'CPU_EVIDENCE':evidence,'CPU_BASELINE':baseline,
    'CPU_TIMING':args.timing,'CPU_QUALITY':args.quality_cpu_output}
  sys.argv=[str(__file__),'--output',str(args.output)]
  exec(compile(tree,str(source)+':full64-cpu-fixture','exec'),namespace);namespace['main']()
  proof=Q.read(args.output);proof['cpu_harness_fixture_edits']=edits
  Q.write(args.output,proof,new=False)

if __name__=='__main__':main()

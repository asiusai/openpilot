"""Run the same real full64 CPU entrypoint without the adaptive overlay."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=HERE/'agent-adaptive-quality64-check.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='515fb1e538494b08e1d6b05d1b5326d531d9223f2bd9861967a6d70420ca727a'
text=source.read_text()
anchor='  # Coverage body keeps every assertion; only this obsolete text label differs.\n'
assert text.count(anchor)==1
insert='''  # Reference arm uses the actual original nominal decoder, with neural outputs doubled.
  change("owner_state=next(s for s in test_overlay[0].states)", "owner_state=None")
  change("state=owner_state;draft=state.current['draft'];state.active=bool(draft)", "state=owner_state;draft=()")
  change("self.baseline.last_prompt=prompt", "self.last_prompt=prompt")
  change("self.baseline.last_targets=", "self.last_targets=")
  change("self.baseline.prefix_audit=None", "self.prefix_audit=None")
  change("M.run(output,stage='timing' if CPU_TIMING else 'quality')", "M.run(output,arm='reference',stage='timing' if CPU_TIMING else 'quality')")
  change("CPU_TIMING and arm=='candidate'", "CPU_TIMING")
  change("assert status['complete']==CPU_TIMING", "assert status['complete']")
  change("assert CPU_TIMING or failure=='Full64 or one of four 16-query windows exceeds 5% ADE/FDE'", "assert failure is None")
  change("assert status['checks']['adaptive']['partial']>0", "assert 'adaptive' not in status['checks']")
  change("safe_snapshot(comp,test_overlay[0])", "safe_snapshot(comp,None)")
  change("assert failure_snapshot['adaptive']['states'][0]['completed']==64", "assert status['checks']['report']['count']==(128 if CPU_TIMING else 64)")
  change("'motion_gate_rejection_exercised':not CPU_TIMING", "'motion_gate_rejection_exercised':False,'reference_arm_exercised':True")
'''
text=text.replace(anchor,insert+anchor)
exec(compile(text,str(source)+':reference-cpu-fixture','exec'),globals())

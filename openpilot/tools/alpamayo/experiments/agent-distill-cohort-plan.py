"""Pin two small cohort adapters, reusing the existing teacher/runtime/cache without copying them."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_cohort_common_plan',HERE/'agent-distill-cohort-common.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)

def main():
  root=c.ROOT/'capture-v1';assert not root.exists();c.storage(c.ROOT)
  inputs=c.validate_rows(c.read(c.INPUTS));previous=c.read(c.OLD_PLAN)
  for filename,digest in previous['source_sha256'].items():assert c.sha(filename)==digest,filename
  root.mkdir();frozen=root/'frozen';frozen.mkdir()
  for name in ('agent-distill-cohort-common.py','agent-distill-cohort-capture.py'):(frozen/name).write_bytes((HERE/name).read_bytes())
  adapter=c.module('agent_distill_cohort_generated_adapter_check',frozen/'agent-distill-cohort-capture.py')
  source=adapter.capture_source();ast.parse(source);(root/'agent-distill-cohort-expanded-execute.py').write_text(source)
  by_id={row['id']:row for row in inputs['samples']};commands=[]
  for job in (0,1):
    ids=c.coverage()['capture_plan']['job_'+str(job)+'_ids'];rows=[by_id[value] for value in ids]
    invocation=c.read(previous['invocation_path']);command=invocation['command']
    a,b=command.index('--samples')+1,command.index('--initial-noise');command[a:b]=[row['sample'] for row in rows]
    command[command.index('--runs')+1]='12';command[command.index('--output')+1]=str(root/('job-'+str(job)+'-parent-placeholder.json'))
    invocation_path=root/('agent-distill-cohort-invocation-'+str(job)+'.json');c.write(invocation_path,invocation)
    plan=copy.deepcopy(previous);plan.update(protocol=c.protocol()['protocol'],job=job,coverage_sha256=c.COVERAGE_SHA,scene_ids=ids,
      expected_examples=12,max_model_prefills=12,inputs_manifest=dict(path=str(c.INPUTS),sha256=c.sha(c.INPUTS)),
      invocation_path=str(invocation_path),original_capture_plan=dict(path=str(c.OLD_PLAN),sha256=c.sha(c.OLD_PLAN)),
      scheduling='Twelve different scenes, one experimental Follow the road. condition each. No duplicated warmups or prior outputs.',
      sealed_policy='Only existing train/validation drives. Sealed drive and protected route excluded entirely.',
      source_sha256={str(path):c.sha(path) for path in [*HERE.glob('agent-distill-cohort-*.py'),*frozen.glob('*.py'),
        root/'agent-distill-cohort-expanded-execute.py',invocation_path,c.INPUTS,c.OLD_PLAN,Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments/agent-distill-command-capture.py')]} )
    path=root/('agent-distill-cohort-job-'+str(job)+'.json');c.write(path,plan)
    prefix=['/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python',str(frozen/'agent-distill-cohort-capture.py'),'--plan',str(path)]
    commands.append(dict(job=job,plan=str(path),plan_sha256=c.sha(path),
      CPU_preflight=prefix+['--output',str(c.ROOT/('agent-distill-cohort-preflight-job-'+str(job)+'.json')),'--check-imports'],
      parent_GPU=['/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python','/tmp/opencode/run-alpamayo-speed-trial.py',
        'agent-distill-cohort-gpu-job-'+str(job),'--program',str(frozen/'agent-distill-cohort-capture.py'),'--set','plan',str(path),'--timeout-seconds','300','--flag','execute-gpu']))
  c.write(root/'agent-distill-cohort-commands.json',dict(executed_GPU=False,inputs_manifest=str(c.INPUTS),inputs_manifest_sha256=c.sha(c.INPUTS),jobs=commands))
  print(json.dumps(dict(plans=[item['plan'] for item in commands],inputs_manifest_sha256=c.sha(c.INPUTS),gpu_executed=False),indent=2))

if __name__=='__main__':main()

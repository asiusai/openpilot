"""CPU-only freeze after ACTUAL startup qualification. Never executes a GPU job."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
DATA=WORK/'agent-nearest-live-data'
BASE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
OUT=WORK/'agent-nearest-live-source'
PLAN=DATA/'agent-nearest-live-plan.json'
PY='/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'
sys.dont_write_bytecode=True


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


def main():
  if OUT.exists() or PLAN.exists():raise FileExistsError('Frozen nearest-live sources must remain immutable')
  startup_path=WORK/'startup-spec-frozen-source/agent-startup-spec-run.py'
  startup=load(startup_path,'agent_nearest_live_prepare_startup')
  dynamic=load(BASE/'experiments/agent-paced-dynamic-policy.py','agent_nearest_live_prepare_dynamic');dynamic.verify_snapshot()
  a,b=(read(WORK/('agent-startup-spec-'+name)/'report.json') for name in ('reference','candidate'))
  qualified=startup.qualify(a,b,dynamic)  # No fabricated warm-success status; runs the actual frozen qualification logic.
  cpu=read(DATA/'agent-nearest-live-full64.json')
  controls=read(DATA/'agent-nearest-live-controls.json')
  if not (controls['source_exact_startup_import_composition'] and controls['original_module_aliases_restored'] and
          all(controls['poisoned_metadata_rejections'].values()) and controls['adapter_sha256']==sha(HERE/'agent-nearest-live-policy.py')):
    raise ValueError('Missing unchanged startup composition and rejection controls')
  for path,digest in cpu['source_sha256'].items():
    if sha(path)!=digest:raise ValueError('CPU qualification source changed: '+path)
  if not (cpu['cpu_only'] and cpu['gpu_executed'] is False and cpu['queries']==64 and cpu['consumer_cpu_hash_cases']==64 and
          cpu['full_array_exact_count']==63 and cpu['original_methods_unchanged']):raise ValueError('Missing honest received-only CPU audit')
  cases=copy.deepcopy(cpu['parent_gpu_cases'])
  if [case['display_time_s'] for case in cases]!=[35.,116.5,167.,228.] or not all(case['exact_saved_nearest_input'] for case in cases):
    raise ValueError('Do not substitute a nonmatching query or weaken saved-nearest parity')
  saved=read(WORK/'agent-history-nearest-quality-nearest/report.json')
  spec=read(WORK/'agent-history-nearest-quality-nearest/report.speculation.json')['records']
  by_sample={Path(row['sample']).name:(row,record) for row,record in zip(saved['warmup']+saved['runs'],spec,strict=True)}
  for case in cases:case['expected_speculative_accept']=by_sample[Path(case['saved_nearest_sample']).name][1]['accepted']
  if [case['expected_speculative_accept'] for case in cases]!=[True,False,True,False]:raise ValueError('Predeclared acceptance/fallback coverage changed')
  OUT.mkdir();hashes={}
  for path in sorted(HERE.glob('agent-nearest-live-*.py')):
    shutil.copyfile(path,OUT/path.name)
    if sha(path)!=sha(OUT/path.name):raise ValueError('Owned helper copy changed bytes')
    hashes[str(OUT/path.name)]=sha(path)
  paths=[DATA/'agent-nearest-live-full64.json',DATA/'agent-nearest-live-metadata64.json',DATA/'agent-nearest-live-controls.json',BASE.parent/'manifest.json',
    BASE.parent/'pipeline-invocation.json',startup_path,startup_path.with_name('agent-startup-spec-warm.py'),
    WORK/'agent-startup-spec-reference/report.json',WORK/'agent-startup-spec-candidate/report.json',
    WORK/'agent-startup-spec-paced/report.json',WORK/'agent-history-nearest-quality-nearest/report.json',
    WORK/'agent-history-nearest-quality-nearest/report.speculation.json',WORK/'agent-history-nearest-quality-nearest/comparison-floor.json',
    WORK/'agent-history-nearest-quality-nearest/longitudinal-review.json',
    WORK/'paced-cache-audit.json',Path('/tmp/opencode/alpamayo-paced-cache-audit.py'),
    WORK/'agent-paced-replay-cache-35-43/paced-cache.json',Path('/tmp/opencode/run-alpamayo-speed-trial.py')]
  for path in paths:hashes[str(path)]=sha(path)
  hashes.update(cpu['input_sha256']);hashes.update(cpu['raw_sha256'])
  invocation=read(BASE.parent/'pipeline-invocation.json');pythonpath=invocation['environment']['PYTHONPATH']
  jobs=[]
  for mode in ('parity','paced'):
    name='agent-nearest-live-'+('parity4' if mode=='parity' else 'paced35-43')
    command=[PY,'/tmp/opencode/run-alpamayo-speed-trial.py',name,'--program',str(OUT/'agent-nearest-live-run.py'),
      '--timeout-seconds','1200','--set','mode',mode,'--set','plan',str(PLAN),'--flag','execute-gpu',
      '--env','PYTHONPATH',pythonpath,'--env','PYTHONDONTWRITEBYTECODE','1']
    if mode=='paced':command.extend(['--set','parity-report',str(WORK/'agent-nearest-live-parity4/report.json')])
    jobs.append(dict(mode=mode,command=command,output=str(WORK/name/'report.json'),parent_launch_only=True))
  tinygrad=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
  revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tinygrad,text=True).strip()
  if revision!='89846dcfb383a98ee01cfec3bba440ae7dcf9b1e' or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=tinygrad,text=True).strip():
    raise ValueError('Require the clean tracked pinned compiler/runtime source')
  plan=dict(schema=1,policy='nearest-received-history-past-tie-v1',classification='NUMERICAL INPUT POLICY; optional; no adoption/full-route qualification',
    tinygrad=dict(path=str(tinygrad),revision=revision),
    base_source=str(BASE),invocation=str(BASE.parent/'pipeline-invocation.json'),startup_runner=str(startup_path),
    startup_qualification=qualified,startup_reference=str(WORK/'agent-startup-spec-reference/report.json'),
    startup_candidate=str(WORK/'agent-startup-spec-candidate/report.json'),
    saved_nearest_report=str(WORK/'agent-history-nearest-quality-nearest/report.json'),
    raw_frames=str(WORK/'agent-paced-replay-cache-35-43/rgb'),cases=cases,frozen_sha256=hashes,jobs=jobs,
    CPU_audit=dict(queries=64,exact_saved_nearest_full_inputs=63,changed_query_seconds=36.5,
      changed_fields=['image_frames','absolute_timestamps'],all_learned_history_and_navigation_exact=True,
      actual_subscriber_receipt_times_known=False,available_only_reference=str(DATA/'agent-nearest-live-available-only')),
    recipe='Unchanged qualified dynamic vision/fixed language/pad2048/target-verified NOMINAL/shuffle-FMA/3Euler/full outputs',
    completed_schedule_cache_audit=dict(report=str(WORK/'paced-cache-audit.json'),floor_mean_misses=8.4,
      nearest_same_query_times_mean_misses=8.6,capacity_enlargement_helped=False,
      limit='Metadata at fixed completed FLOOR query times; excludes receipt, pixel content, calibration changes, and changed completion times'),
    parity=dict(queries=4,times_seconds=[35.,116.5,167.,228.],required='Every actual Consumer prepared/packed/CPU-buffer/vision-key hash and every saved-nearest non-timing prediction field exact; actual acceptance/fallback flags exact'),
    paced=dict(source_start_seconds=35,source_seconds=8,requested_ceiling_hz=10,native_camera_hz=20,maximum_active=1,
      inference_budget_seconds=180,child_timeout_including_initialization_seconds=1200,no_future_model_warming=True,
      lifecycle='Same qualified startup.main/imports and warm.initialize bytes; only Store/Preparer/Consumer aliases are scoped subclasses',
      cache_miss_claim='None. Report received-only misses at actual elapsed cadence; no ideal two-new-image assumption'),
    automatic_parity_to_paced=False,automatic_adoption_or_fullroute=False,gpu_executed_by_agent=False,
    known_perception_limit='Parent review:116.5s green/flashing-amber unresolved;167s changed reasoning. Numerical screen is not perception or route qualification.')
  save(PLAN,plan)
  print(json.dumps(dict(prepared=True,plan=str(PLAN),startup_qualification=qualified,GPU_executed=False,
    cases=[(c['display_time_s'],c['expected_speculative_accept']) for c in cases],automatic_paced=False),indent=2))


if __name__=='__main__':main()

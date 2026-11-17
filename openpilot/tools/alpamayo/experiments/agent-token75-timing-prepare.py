"""Freeze one ordinary-lookup native36 timing pair. CPU file operations only."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_token75_timing_prepare_common',HERE/'agent-token75-timing-common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def copy_exact(source,target,hashes):
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():c.require(c.sha(target)==c.sha(source),'Snapshot collision: '+str(target))
  else:shutil.copy2(source,target)
  c.require(c.sha(target)==c.sha(source),'Source bytes changed')
  hashes[str(target)]=c.sha(target)


def main():
  c.require(not c.ROOT.exists(),'Existing frozen timing protocol must remain immutable')
  old_plan=c.read(c.OLD_ROOT/'agent-token-current-queue-plan.json')
  for path,record in old_plan['snapshot_files'].items():c.require(c.sha(path)==record['sha256'],'Original queued source changed')
  quality={ratio:c.read(c.WORK/('agent-token-current-quality-history'+ratio)/'comparison-vs-full-padded-quality.json') for ratio in ('50','75')}
  c.require(quality['50']['windows']['112-120']['within_five_percent_trajectory_error'] is False,'Expected rejected50% evidence')
  c.require(quality['75']['matched']['within_five_percent_trajectory_error'] and
            all(row['within_five_percent_trajectory_error'] for row in quality['75']['windows'].values()),'75% short-window screens did not pass')
  existing_native=c.read(c.WORK/'full-native-5hz-padded/report.json')
  samples=[row['sample'] for row in c.entries(existing_native)]
  c.require(len(samples)==36 and len(set(samples))==36,'Existing native fixture is not36 distinct inputs')
  native_inv=c.read(c.WORK/'full-native-5hz-padded/invocation.json')
  nc=native_inv['command']
  source_manifest=Path(nc[nc.index('--native-manifest')+1]);source_calibration=Path(nc[nc.index('--native-calibrations')+1])
  raw_cache=Path(nc[nc.index('--native-frame-cache')+1]);manifest=c.read(source_manifest)
  rows={Path(row['sample']).name:row for row in manifest['schedule']}
  queries=[rows[Path(sample).name]['query_ns'] for sample in samples]
  c.require(all(195_000_000<=b-a<=205_000_000 for a,b in zip(queries,queries[1:])),'Fixture does not advance at approximately5Hz')
  c.ROOT.mkdir();hashes={}
  for path,digest in ((Path(path),record['sha256']) for path,record in old_plan['snapshot_files'].items()):
    if path.is_relative_to(c.OLD_SNAP):copy_exact(path,c.SNAP/path.relative_to(c.OLD_SNAP),hashes)
  copy_exact(c.OLD_ROOT/'agent-token-current-prefill-scales.json',c.ROOT/'agent-token-current-prefill-scales.json',hashes)
  for path in HERE.glob('agent-token75-timing-*.py'):copy_exact(path,c.FEXP/path.name,hashes)
  manifest_path=c.ROOT/'agent-token75-timing-native-manifest.json';calibration_path=c.ROOT/'agent-token75-timing-native-calibrations.json'
  copy_exact(source_manifest,manifest_path,hashes);copy_exact(source_calibration,calibration_path,hashes)
  # Relocate path-bearing qualification JSON only. Executable helper bytes remain identical to passed75% GPU sources.
  original_cpu=c.read(c.OLD_ROOT/'agent-token-current-queue-cpu-report.json')
  cpu=c.relocate(original_cpu,c.OLD_ROOT,c.ROOT)
  c.require(cpu['source_sha256']==original_cpu['source_sha256'],'A source digest changed during relocation')
  cpu_path=c.ROOT/'agent-token75-timing-relocated-cpu-report.json';c.write_new(cpu_path,cpu);hashes[str(cpu_path)]=c.sha(cpu_path)
  invocation=c.relocate(c.read(c.OLD_ROOT/'agent-token-current-quality-invocation.json'),c.OLD_ROOT,c.ROOT)
  command=invocation['command'];begin,end=command.index('--samples')+1,command.index('--initial-noise')
  command[begin:end]=samples
  for name,value in (('warmup',12),('runs',24),('native-manifest',manifest_path),('native-calibrations',calibration_path),('native-frame-cache',raw_cache)):
    c.set_option(command,name,value)
  # Preserve the GPU-qualified ordinary lookup recipe; avoid all current2048/NOMINAL compatibility changes.
  for flag in ('--coarse-timing','--cache-rotary-inputs','--late-inputs','--freeze-warmup-gc','--phrase-drafts','--vision-calibration'):
    c.require(flag not in command,'Unexpected recipe option: '+flag)
  c.require(command[command.index('--pad-prompt-to')+1]=='2112','Expected GPU-qualified2112 padding')
  invocations={};jobs=[]
  for arm in ('reference','candidate'):
    saved=copy.deepcopy(invocation);name='agent-token75-timing-native-'+arm
    c.set_option(saved['command'],'output',c.WORK/name/'report.json')
    path=c.ROOT/f'agent-token75-timing-{arm}-invocation.json';c.write_new(path,saved);hashes[str(path)]=c.sha(path);invocations[arm]=str(path)
    jobs.append(dict(arm=arm,name=name,report=str(c.WORK/name/'report.json'),command=[str(c.PY),'/tmp/opencode/run-alpamayo-speed-trial.py',name,
      '--program',str(c.FEXP/'agent-token75-timing-run.py'),'--timeout-seconds','900','--set','arm',arm,'--set','protocol',str(c.PROTOCOL),
      '--flag','execute-gpu','--env','PYTHONPATH',saved['environment']['PYTHONPATH'],'--env','PYTHONDONTWRITEBYTECODE','1']))
  # Pin each actual native input and raw capture. No images are regenerated or selected differently.
  for sample in samples:
    for path in Path(sample).glob('*.npy'):hashes[str(path)]=c.sha(path)
    row=rows[Path(sample).name]
    c.require(tuple(frame['view'] for frame in row['frames'])==(1,)*5+(6,)*5,'Incomplete front histories')
    for frame in row['frames']:
      c.require(frame['capture_ns']<=row['query_ns'],'Unavailable camera frame')
      path=raw_cache/str(frame['view'])/f'{frame["video_index"]}.npy';hashes[str(path)]=c.sha(path)
  for ratio in ('50','75'):
    path=c.WORK/('agent-token-current-quality-history'+ratio)/'comparison-vs-full-padded-quality.json'
    copy_exact(path,c.ROOT/f'agent-token75-timing-quality-history{ratio}.json',hashes)
  for name in ('prefill-calibration','initial-noise'):
    path=Path(command[command.index('--'+name)+1]);hashes[str(path)]=c.sha(path)
  checkpoint=Path(command[command.index('--checkpoint')+1])
  for name in ('config.json','model.safetensors.index.json'):hashes[str(checkpoint/name)]=c.sha(checkpoint/name)
  hashes['/tmp/opencode/run-alpamayo-speed-trial.py']=c.sha('/tmp/opencode/run-alpamayo-speed-trial.py')
  runner_template=Path('/mnt/nvme-500/tmp/alpamayo-calibrated-two-six-20261116/matched-warm/2cam-invocation.json')
  hashes[str(runner_template)]=c.sha(runner_template)
  plan=dict(schema=1,policy=c.POLICY,warmup=12,measured=24,samples=samples,queries_ns=queries,
    invocations=invocations,jobs=jobs,relocated_cpu_report=str(cpu_path),frozen_sha256=hashes,
    native_manifest=str(manifest_path),native_manifest_sha256=c.sha(manifest_path),native_calibrations=str(calibration_path),
    native_calibrations_sha256=c.sha(calibration_path),raw_cache=str(raw_cache),
    fixed_language_scale_sha256=c.sha(command[command.index('--prefill-calibration')+1]),
    recipe=dict(pad_prompt_to=2112,decoder='ordinary lookup with genuine target verification',nominal_prefill=False,
      prefill_attention='original online attention; no shuffle-FMA addition',vision='dynamic activation scales',language='fixed scales',
      expert='three Euler evaluations',max_tokens=64,full_images_per_camera=5,vision_group_limit=1,
      synchronized_stages=True,freeze_gc=False,late_inputs=False,rotary_cache=False),
    source_input_policy='Unchanged existing native5Hz ring fixture; no nearest-history policy or frame substitutions',
    current2048_compatibility_blocker='NOMINAL SpeculativePrefill replaces model.padded_prefill, the same method owned by the pruner; draft targets and selected-prefix cache writes need new joint integration. Not attempted.',
    compatibility_gpu_gate_required=False,compatibility_reason='No new arithmetic/kernel/decoder hook; exact passed75% controller and ordinary lookup sources are copied. Actual native metadata path must pass CPU qualification.',
    cpu_qualification=str(c.ROOT/'agent-token75-timing-cpu-report.json'),
     prior_quality=dict(rejected50=True,only75_considered=True,comparison75=quality['75']['matched'],windows75=quality['75']['windows'],
       fullroute_quality_proven=False,parent_shape_speed_slowing_reasoning_review_pending=True),
     classification='NUMERICAL NEW MODEL; original prompt-prefix invariance fails; generation causality remains required',
     prompt_prefix_invariance_pass=False,tinygrad=old_plan['tinygrad'],
     checkpoint_weight_content_rehashed=False,checkpoint_identity_boundary='Config and shard-index hashes only; existing shared model and FP8 cache, no weight copies',
    native_continuous_Asius_validated=False,parent_review_required_before_launch=True,automatic_adoption_or_fullroute=False,
    isolated_gpu_gather_timing_available=False,
    comparison_command=[str(c.PY),str(c.FEXP/'agent-token75-timing-compare.py'),'--reference',jobs[0]['report'],'--candidate',jobs[1]['report'],
      '--protocol',str(c.PROTOCOL),'--output',str(c.ROOT/'agent-token75-timing-comparison.json')])
  c.write_new(c.PROTOCOL,plan)
  print(json.dumps(dict(prepared=True,protocol=str(c.PROTOCOL),gpu_executed=False,recipe=plan['recipe']),indent=2))


if __name__=='__main__':main()

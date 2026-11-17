"""Read-only CPU accounting for the current complete-pipeline gap. No model/runtime imports."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
LOG=Path('/home/john/asius/tools/alpamayo-car/EXPERIMENT-LOG.md')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):
  return dict(count=len(values),median=statistics.median(values),mean=statistics.mean(values),minimum=min(values),maximum=max(values))
def load(path):
  assert path.stat().st_size<64*1024*1024
  return json.loads(path.read_text())


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,default=WORK/'agent-latency-gap-artifacts/agent-latency-gap-audit.json')
  args=parser.parse_args()
  output=args.output.resolve()
  assert output.is_relative_to(WORK) and output.name.startswith('agent-latency-gap-')
  if output.exists():raise FileExistsError(output)
  paced=WORK/'agent-exact-live-paced'
  summary=load(paced/'report.json')
  entries=[json.loads(line) for line in (paced/'report.jsonl').read_text().splitlines()]
  rows=[entry for entry in entries if entry.get('kind')=='model_completion']
  assert len(rows)==summary['completed_queries']==22
  assert summary['maximum_active_inference']==1 and summary['queued_prediction_snapshots']==0
  assert all(row['paced_consumer']['speculative_record']['accepted'] for row in rows)
  assert all(row['paced_consumer']['speculative_record']['all_targets_finite'] for row in rows)
  assert all(row['prediction']['prompt_tokens']==2027 and row['prediction']['generated_tokens']==6 for row in rows)
  assert all(row['prediction']['stage_timing_kind']=='synchronized stages' for row in rows)
  consumers=[(row['paced_consumer']['consumer_completed_ns']-row['paced_consumer']['consumer_started_ns'])/1e6 for row in rows]
  ages=[row['timing']['query_to_output_ms'] for row in rows]
  misses=[row['paced_consumer']['vision_misses'] for row in rows]
  assert all(len(row['input']['frames'])==10 for row in rows)
  assert all(row['paced_consumer']['uploaded_pixel_bytes']==row['paced_consumer']['vision_misses']*552960 for row in rows)
  assert all(row['input']['query_ns']==row['paced_consumer']['query_ns'] for row in rows)
  stages={name:stats([row['prediction']['timings_ms'][name] for row in rows]) for name in rows[0]['prediction']['timings_ms']}
  wrappers={name:stats([row['input']['live_consumer']['stages_ms'][name] for row in rows])
            for name in rows[0]['input']['live_consumer']['stages_ms']}
  grouped_costs={}
  for name,fields in (
      ('vision',('vision',)),('prefill',('prefill',)),('prefill_expert',('prefill','diffusion')),
      ('vision_prefill',('vision','prefill')),('vision_prefill_expert',('vision','prefill','diffusion'))):
    costs=[sum(row['prediction']['timings_ms'][field] for field in fields) for row in rows]
    remaining=[actual-cost for actual,cost in zip(consumers,costs)]
    grouped_costs[name]=dict(affected_host_interval_ms=stats(costs),
      if_entire_group_free_consumer_ms=stats(remaining),
      if_entire_group_free_capture_to_output_ms=stats([age-cost for age,cost in zip(ages,costs)]),
      if_group_four_times_faster_consumer_ms=stats([actual-.75*cost for actual,cost in zip(consumers,costs)]))
  # A current-shape sensitivity only: retain every observed timestamp and all other work.
  all_costs=[sum(row['prediction']['timings_ms'][field] for field in ('vision','prefill','diffusion')) for row in rows]
  all_remaining=[actual-cost for actual,cost in zip(consumers,all_costs)]
  required={}
  for deadline in (100,80):
    eligible=[cost/(deadline-rest) for cost,rest in zip(all_costs,all_remaining) if rest<deadline]
    required[str(deadline)]=dict(finite_rows=len(eligible),blocked_rows=len(rows)-len(eligible),
      finite_uniform_group_speedup=stats(eligible) if eligible else None)
  earliest_changed=[]
  for previous,current in zip(rows,rows[1:]):
    identity=lambda frame:(frame['view'],frame['capture_ns'],frame['image_calibration_ns'])
    changes=[index for index,(left,right) in enumerate(zip(previous['input']['frames'],current['input']['frames'])) if identity(left)!=identity(right)]
    assert changes
    earliest_changed.append(changes[0])
  profile_path=WORK/'current-profile-dynamic-2048/report.current-profile-kernels.json'
  profile=load(profile_path)['measured']
  assert profile['queries']==2 and profile['valid_timestamp_coverage'] and not profile['invalid_samples']
  assert profile['timings_valid_for_throughput'] is False
  event_stages={name:entry['mean_gpu_events_ms_per_query'] for name,entry in profile['categories'].items()}
  assert abs(sum(event_stages.values())-profile['sum_gpu_events_ms']/2)<1e-8
  prepared_path=WORK/'agent-exact-combined-native-candidate/report.json'
  prepared=load(prepared_path)
  assert prepared['pad_prompt_to']==2048 and prepared['vision_calibration_sha256'] is None
  assert prepared['settings']['diffusion_steps']==3
  measured=prepared['runs']
  assert len(measured)==prepared['summary']['measured_predictions']==24
  assert all(run['vision_misses']==4 for run in measured)
  config_path=BASE/'model/config.json';config=load(config_path)
  assert config['vlm_config']['text_config']['num_hidden_layers']==config['expert_config']['llm_config']['num_hidden_layers']==28
  assert config['vlm_config']['vision_config']['num_hidden_layers']==27
  # One architecture study, not a sweep: group four teacher blocks per same-width student block.
  groups=lambda count:[list(range(start,min(start+4,count))) for start in range(0,count,4)]
  study=dict(vision_teacher_groups=groups(27),language_teacher_groups=groups(28),expert_teacher_groups=groups(28),
    widths_unchanged=True,all_images_and_tokens_unchanged=True,euler_evaluations=3,
    role='Bounded local recoverability study only; fourfold stage scaling still fails the current fixed-dispatch deadline.')
  source_paths=[LOG,Path(__file__),profile_path,WORK/'agent-current-profile-analysis.md',paced/'report.json',paced/'report.jsonl',
    paced/'invocation.json',prepared_path,config_path,BASE/'model/model.safetensors.index.json',BASE/'model/LICENSE',
    BASE/'upstream/src/alpamayo2_edge/models/alpamayo2_edge.py',BASE/'upstream/src/alpamayo2_edge/chat_template.py',
    WORK/'agent-qat.md',WORK/'agent-qkv.md']
  source_paths += [paced/'report.launcher-sources'/name for name in
                   ('model.py','agent-live-consumer.py','agent-paced-replay.py','vision-cache.py','batched-vision.py')]
  result=dict(schema=1,cpu_only=True,gpu_used=False,models_imported=False,source_sha256={str(path):sha(path) for path in source_paths},
    paced=dict(completions=len(rows),recorded_replay_hz=summary['recorded_replay_completion_hz'],
      consumer_ms=stats(consumers),capture_to_output_ms=stats(ages),vision_misses=stats(misses),
      age_before_consumer_ms=stats([age-duration for age,duration in zip(ages,consumers)]),
      runner_stages_ms=stages,consumer_wrapper_stages_ms=wrappers,
      all_nominal_drafts_verified=True,earliest_changed_image_slot_per_transition=earliest_changed,
      source_camera_hz=20,requested_query_ceiling_hz=10,
      critical_limit='PC replay publication/admission timing is not measured Asius sensor/ISP/subscriber latency.'),
    counterfactuals=dict(groups=grouped_costs,required_group_speedups=required,
      limits=['Row-wise subtraction of existing synchronized host intervals; never sums or subtracts stage medians.',
        'No changed dispatch times, cache misses, overlap, capture admission or kernel efficiency are modeled.',
        'These are sensitivity ceilings for the recorded execution, not achievable latency bounds for a new scheduler/model.']),
    prepared=dict(median_input_to_output_ms=prepared['summary']['median_ms'],service_median_ms=prepared['summary']['service_median_ms'],
      throughput_hz=prepared['summary']['throughput_fps'],measured_queries=len(measured),vision_misses_per_query=4),
    four_fresh_image_profile=dict(event_stage_ms=event_stages,event_sum_ms=profile['sum_gpu_events_ms']/2,
      profile_is_not_throughput=True),study=study)
  output.parent.mkdir(parents=True,exist_ok=True)
  output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(json.dumps(dict(output=str(output),consumer_median_ms=result['paced']['consumer_ms']['median'],
    capture_median_ms=result['paced']['capture_to_output_ms']['median'],miss_mean=result['paced']['vision_misses']['mean'],
    grouped_sensitivities={key:{field:value[field]['median'] for field in
      ('if_entire_group_free_consumer_ms','if_entire_group_free_capture_to_output_ms','if_group_four_times_faster_consumer_ms')}
      for key,value in grouped_costs.items()},required_group_speedups=required),indent=2))


if __name__=='__main__':main()

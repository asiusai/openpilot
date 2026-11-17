"""CPU-only completed-pair analysis. Never infer adoption from native timing."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_token75_timing_compare_common',HERE/'agent-token75-timing-common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)


def stats(values):
  values=np.asarray(values,dtype=np.float64)
  common.require(values.size and np.isfinite(values).all(),'Missing/nonfinite timing observations')
  return dict(median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(values.max()),mean=float(values.mean()),sum=float(values.sum()))


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--candidate',type=Path,required=True)
  parser.add_argument('--protocol',type=Path,default=common.PROTOCOL);parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args();plan=common.read(args.protocol);common.verify(plan)
  reference,candidate=map(common.read,(args.reference,args.candidate))
  a,b=(common.validate_output(report,plan,arm) for report,arm in ((reference,'reference'),(candidate,'candidate')))
  fields=('load_ms','native_calibration_ms','preprocess_ms','prepared_queue_ms','cpu_buffers_ms','pipeline_ms','service_ms','gc_ms',
          'vision_hits','vision_misses','calibrated_frame_hits','calibrated_frame_misses','uploaded_pixel_bytes')
  summaries={};differences={}
  for arm,report in (('reference',reference),('candidate',candidate)):
    rows=report['runs']
    summaries[arm]={field:stats([row[field] for row in rows]) for field in fields}
    stages=rows[0]['prediction']['timings_ms']
    summaries[arm]['stages_ms']={stage:stats([row['prediction']['timings_ms'][stage] for row in rows]) for stage in stages}
    summaries[arm]['token_lengths']=stats([row['prediction']['generated_tokens'] for row in rows])
    summaries[arm]['throughput_fps']=report['summary']['throughput_fps']
    summaries[arm]['measured_wall_seconds']=report['summary']['measured_wall_seconds']
  for field in fields: differences[field]=stats([right[field]-left[field] for left,right in zip(reference['runs'],candidate['runs'],strict=True)])
  cache_fields=('vision_hits','vision_misses','calibrated_frame_hits','calibrated_frame_misses','uploaded_pixel_bytes')
  cache_equal={field:all(left[field]==right[field] for left,right in zip(a,b,strict=True)) for field in cache_fields}
  events=candidate['token75_timing_policy_events'];common.require(len(events)==36,'Missing75% decisions')
  measured=events[12:]
  costs=[]
  for event in measured:
    retained=event['retained_prompt_tokens'];rows=event['late_rows'];fallback=event['fallback']
    costs.append(dict(fallback=fallback,late_rows=rows,retained_prompt_tokens=retained,
      gather_output_bytes=0 if fallback else 2*rows*(2048+2*128),
      valid_KV_scatter_write_bytes=0 if fallback else 26*2*8*retained*128*2,
      note='Logical byte counts, not memory traffic measurements or GPU timing'))
  result=dict(classification='Native archival timing pair; numerical candidate; no adoption/fullroute proof',
    reference=str(args.reference),candidate=str(args.candidate),reference_sha256=common.sha(args.reference),candidate_sha256=common.sha(args.candidate),
    warmup=12,measured=24,exact_same36native_inputs=True,measurements=summaries,paired_candidate_minus_reference=differences,
    cache_and_pixel_accounting_equal=cache_equal,
    identical_reasoning_tokens_all36=sum(left['prediction']['generated_token_ids']==right['prediction']['generated_token_ids'] for left,right in zip(a,b,strict=True)),
    measured_fallbacks=sum(event['fallback'] is not None for event in measured),gather_scatter_costs=costs,
    isolated_gpu_gather_time_ms=None,
    gather_cost_boundary='Measured candidate prefill includes early layers, saliency readback fence, CPU selection, mask/index transfers, gather,26late layers, KVscatter; separate gather duration is unavailable in frozen replay.',
    source_input_latency_boundary='pipeline_ms is native file-load-start through output, including queue/CPU overlap; it is not camera-exposure-to-output on Asius.',
    measured_prefill_delta_ms=stats([right['prediction']['timings_ms']['prefill']-left['prediction']['timings_ms']['prefill'] for left,right in zip(reference['runs'],candidate['runs'],strict=True)]),
    current_selected_recipe_comparison='Both arms use ordinary lookup/pad2112/original attention, not current2048/NOMINAL/shuffle-FMA; no cross-recipe gain claim.',
    trajectory_speed_slowing_reasoning_review='Parent-owned; fullroute quality remains mandatory',automatic_adoption_or_fullroute=False)
  common.write_new(args.output,result)
  print(json.dumps({key:value for key,value in result.items() if key not in ('gather_scatter_costs','measurements','paired_candidate_minus_reference')},indent=2))


if __name__=='__main__':main()

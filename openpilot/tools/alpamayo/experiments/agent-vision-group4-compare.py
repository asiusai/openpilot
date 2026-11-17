"""Exact four-query parity gate. Never converts an arithmetic failure to tolerance."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('agent_vision_group4_compare_common',HERE/'agent-vision-group4-common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def compare(before,after):
  checks={}
  checks['completed_four']=all(report.get('completed') and len(report['queries'])==4 for report in (before,after))
  checks['same_frozen_source']=before['source_sha256']==after['source_sha256']
  checks['same_plan']=before['plan_sha256']==after['plan_sha256']
  checks['cleanup']=all(all(report.get('postcleanup',{}).values()) and report.get('postcleanup') for report in (before,after))
  checks['memory_bound']=all(report['peak_live_buffer_bytes']<=report['memory_limit_bytes'] for report in (before,after))
  checks['all_remainder_controls']=all([entry['images'] for entry in report['controls']]==[1,2,3,4] for report in (before,after))
  for a,b in zip(before['controls'],after['controls'],strict=True):
    count=a['images']
    for key in ('images','original','mutated','restored','initial_capture_replay_exact','mutation_replay_exact','earlier_images_independent'):
      checks[f'control{count}.{key}']=a[key]==b[key]
  for a,b in zip(before['queries'],after['queries'],strict=True):
    index=a['index']
    for key in ('index','prepared','packed','cpu','batches','spans','history_before','owner_reused','actual_prompt','padded_prompt',
                'history_after','statistics','spec_record','pixel_bytes','kv_valid','kv_shapes','prediction_bits','prediction'):
      checks[f'query{index}.{key}']=a.get(key)==b.get(key)
    for key in ('keys','features','stitched_embedding','hits','misses','lru_keys'):
      checks[f'query{index}.vision.{key}']=a['vision'][key]==b['vision'][key]
  shared=before['binaries'].keys() & after['binaries'].keys()
  checks['shared_actual_binaries_present']=bool(shared)
  for key in shared:
    a,b=before['binaries'][key],after['binaries'][key]
    checks['binary.'+key]=all(a[field]==b[field] for field in ('target','device','source_sha256','binary_sha256','resources','local_size','global_size'))
  modes=[]
  for report in (before,after):
    families={}
    for entry in report['binaries'].values():
      families.setdefault(entry['name'],set()).add(entry['resources']['workgroup_processor_mode'])
    modes.append(families)
  for name in modes[0].keys() & modes[1].keys():checks['compiler_family_mode.'+name]=modes[0][name]==modes[1][name]
  checks['fp8_tile4x4_actual_family_captured']='alpamayo_fp8_gemm_4_4' in modes[0] and 'alpamayo_fp8_gemm_4_4' in modes[1]
  checks['same_requested_initializer']=before['protected_initializer_ast_sha256']==after['protected_initializer_ast_sha256']
  return dict(exact=all(checks.values()),checks=checks,failed=[key for key,value in checks.items() if not value],
    compared_queries=4,shared_actual_binaries=len(shared),
    actual_binary_counts=dict(group2=len(before['binaries']),group4=len(after['binaries'])),
    actual_wgp_modes={report['arm']:sorted(set(entry['resources']['workgroup_processor_mode'] for entry in report['binaries'].values())) for report in (before,after)},
    compiler_family_modes=[{key:sorted(value) for key,value in families.items()} for families in modes],
    compiler_new_named_shapes=sorted(modes[1].keys()-modes[0].keys()),parent_actual_binary_review_required=True,
    timings_valid_for_throughput=False,
    decision='Exact parity gate only. Parent reviews before any separate timing. No automatic adoption,64/fullroute,or live integration.')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--reference',type=Path,required=True)
  parser.add_argument('--candidate',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  before,after=(c.read(path) for path in (args.reference,args.candidate))
  c.require(before['arm']=='group2' and after['arm']=='group4','Group2 reference and group4 candidate required')
  result=compare(before,after)
  result['evidence_sha256']={str(path):c.sha(path) for path in (args.reference,args.candidate)}
  c.write(args.output,result);print(json.dumps(result,indent=2))
  if not result['exact']:raise SystemExit(1)


if __name__=='__main__':main()

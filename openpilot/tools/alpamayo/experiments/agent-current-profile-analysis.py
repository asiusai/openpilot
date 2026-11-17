"""CPU-only arithmetic and source-cache inspection for the completed diagnostic profile.

Read-only inputs. No tinygrad imports, GPU initialization, compilation, or file writes.
FLOP floors use dense peaks. Logical operand demands are not DRAM measurements.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import struct

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TRIAL = WORK/'current-profile-dynamic-2048'
CHECKPOINT = Path('/tmp/opencode/alpamayo-asius-20261114/model')
CACHE = Path('/tmp/opencode/alpamayo-asius-20261114/cache.db')
FP8, BF16, BANDWIDTH, CACHE_BYTES = 172e12, 86e12, 288e9, 32_000_000


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def headers(checkpoint):
  index = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
  result = {}
  for name in sorted(set(index.values())):
    with (checkpoint/name).open('rb') as stream:
      length = struct.unpack('<Q', stream.read(8))[0]
      if not 0 < length <= 16*1024*1024: raise ValueError('Unexpected safetensors header size')
      result.update(json.loads(stream.read(length)))
  return result


def analyze(trial=TRIAL, checkpoint=CHECKPOINT):
  profile_path = trial/'report.current-profile-kernels.json'
  raw = json.loads((trial/'report.json').read_text())
  aggregate = json.loads(profile_path.read_text())
  measured = aggregate['measured']
  assert measured['queries'] == 2 and measured['valid_timestamp_coverage']
  assert not measured['invalid_samples']
  assert raw['settings']['diffusion_steps'] == 3 and raw['pad_prompt_to'] == 2048
  assert raw['prefill_speculation']['prefill_attention'] == 'shuffle-fma'
  assert raw['vision_calibration_sha256'] is None
  cfg = json.loads((checkpoint/'config.json').read_text())
  tensors = headers(checkpoint)
  restored = {row['name'] for row in raw['precision_restoration']['tensors']}
  rows = []

  def matrix(stage, name, m, count=1):
    n, k = tensors[name+'.weight']['shape']
    precision = 'bf16' if name+'.weight' in restored or k % 16 or n % 16 else 'fp8'
    size = 2 if precision == 'bf16' else 1
    flop = 2*m*n*k*count
    one_touch = (size*(m*k+n*k)+2*m*n)*count
    tile_demand = (size*(m*k*math.ceil(n/64)+n*k*math.ceil(m/64))+2*m*n)*count
    rows.append(dict(stage=stage, name=name, m=m, n=n, k=k, count=count, precision=precision,
                     flops=flop, dense_peak_ms=flop/(BF16 if precision == 'bf16' else FP8)*1000,
                     unique_weights_bytes=n*k*size,
                     logical_one_touch_bytes=one_touch, logical_64tile_operand_bytes=tile_demand))

  vision = cfg['vlm_config']['vision_config']
  language = cfg['vlm_config']['text_config']
  expert = cfg['expert_config']['llm_config']
  fresh, groups, patches, prefill, actions, steps = 4, 2, 720, 2048, 64, 3
  images_per_group = fresh//groups
  matrix('vision', 'vlm.model.visual.embeddings.patch_embedding', patches*images_per_group, groups)
  for i in range(vision['num_hidden_layers']):
    prefix = f'vlm.model.visual.encoder.layers.{i}'
    for suffix in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj', 'mlp.fc1', 'mlp.fc2'):
      matrix('vision', prefix+'.'+suffix, patches*images_per_group, groups)
  for suffix in ('linear_fc1', 'linear_fc2'):
    matrix('vision', 'vlm.model.projector.'+suffix, 180*images_per_group, groups)
  for stage, prefix, layers, m, count in (
      ('prefill', 'vlm.model.language_model.layers', language['num_hidden_layers'], prefill, 1),
      ('diffusion', 'expert.expert.layers', expert['num_hidden_layers'], actions, steps)):
    for i in range(layers):
      for suffix in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.fc1', 'mlp.fc2'):
        matrix(stage, f'{prefix}.{i}.{suffix}', m, count)
  capacity = math.ceil((prefill+raw['settings']['max_tokens'])/256)*256
  keys = capacity+actions
  valid_history = raw['runs'][0]['draft']['prefill_draft']['final_cache_length']
  assert all(e['vision_misses'] == fresh and e['vision_hits'] == 6 for e in raw['runs'])
  assert all(e['draft']['prefill_draft']['accepted'] and e['draft']['prefill_draft']['all_targets_finite'] for e in raw['runs'])
  assert (capacity, keys, valid_history) == (2304, 2368, 2033)
  result = dict(profile_sha256=sha(profile_path), raw_report_sha256=sha(trial/'report.json'),
                summary_plan_reference=aggregate.get('plan'),
                checkpoint_config_sha256=sha(checkpoint/'config.json'),
                hardware=dict(dense_fp8_tflops=172, dense_bf16_tflops=86, bandwidth_decimal_GB_s=288,
                              stated_cache_bytes=CACHE_BYTES),
                shapes=dict(fresh_images=fresh, images_per_group=images_per_group, patches_per_image=patches,
                            original_prompt=2027, padded_prefill=prefill, verified_prefix=valid_history,
                            cache_capacity=capacity, action_queries=actions, physical_expert_keys=keys,
                            expert_valid_keys=valid_history+actions, masked_cache_slots=capacity-valid_history),
                measured_ms_per_query={k:v['mean_gpu_events_ms_per_query'] for k,v in measured['categories'].items()},
                matrices={}, attention={}, generated={}, mappings=[])
  gemm_floor = 0.
  gemm_measured = 0.
  for stage in ('vision', 'prefill', 'diffusion'):
    selected = [row for row in rows if row['stage'] == stage]
    groups_by_type = {}
    for precision in ('fp8', 'bf16'):
      items = [row for row in selected if row['precision'] == precision]
      if not items: continue
      flop = sum(row['flops'] for row in items)
      calls = sum(row['count'] for row in items)
      groups_by_type[precision] = dict(flops=flop, calls=calls,
        dense_peak_ms=sum(row['dense_peak_ms'] for row in items),
        unique_weight_bytes=sum(row['unique_weights_bytes'] for row in items),
        cold_weight_sweep_bytes=sum(row['unique_weights_bytes']*row['count'] for row in items),
        logical_one_touch_bytes=sum(row['logical_one_touch_bytes'] for row in items),
        logical_64tile_operand_bytes=sum(row['logical_64tile_operand_bytes'] for row in items))
    actual = measured['stages'][stage]
    matrix_kernels = [r for r in actual['kernels'] if r['name'].startswith('alpamayo_fp8_gemm')
                      or (stage == 'vision' and r['name'] == 'alpamayo_attention_values')]
    actual_calls = sum(r['count'] for r in matrix_kernels)/2
    assert actual_calls == sum(r['count'] for r in selected)
    actual_ms = sum(r['total_ms'] for r in matrix_kernels)/2
    floor = sum(r['dense_peak_ms'] for r in selected)
    result['matrices'][stage] = dict(types=groups_by_type, measured_ms=actual_ms, peak_floor_ms=floor,
                                   unique_weights_cold_cache_floor_ms=max(0, sum(r['unique_weights_bytes'] for r in selected)-CACHE_BYTES)/BANDWIDTH*1000)
    gemm_floor += floor
    gemm_measured += actual_ms
  assert sum(row['flops'] for row in rows if row['stage']=='prefill') == 2*2048*28*(6*2048**2//2+2*9216*2048)
  attention_flops = dict(
    vision=4*fresh*vision['num_hidden_layers']*16*patches*patches*72,
    prefill=4*language['num_hidden_layers']*16*128*(prefill*(prefill+1)//2),
    diffusion=4*steps*expert['num_hidden_layers']*16*actions*keys*128)
  for stage, flops in attention_flops.items():
    result['attention'][stage] = dict(pair_gemm_flops=flops, dense_bf16_peak_floor_ms=flops/BF16*1000)
  result['attention']['prefill']['128_query_group_pair_gemm_flops'] = 4*28*16*128*sum(128*g*128 for g in range(1,17))
  for stage, value in measured['stages'].items():
    items = [row for row in value['kernels'] if row['family'] == 'generated_E']
    milliseconds = sum(row['total_ms'] for row in items)/2
    count = sum(row['count'] for row in items)/2
    result['generated'][stage] = dict(ms=milliseconds, calls=count, mean_us_per_call=milliseconds*1000/count)
    result['mappings'].extend(dict(stage=stage, name=row['name'], calls=row['count']/2, ms=row['total_ms']/2,
                                   output_elements=math.prod(int(x) for x in row['name'].split('_')[1:]))
                              for row in items[:6])
  by_name = lambda stage, name: next(row['total_ms']/2 for row in measured['stages'][stage]['kernels'] if row['name']==name)
  kv_elements = 8*keys*128
  mask_traffic = 4*16*actions*keys*28*steps
  generated_ms = sum(v['ms'] for v in result['generated'].values())
  total = measured['sum_gpu_events_ms']/2
  attention_floor = sum(v['dense_bf16_peak_floor_ms'] for v in result['attention'].values())
  result['bounds'] = dict(total_event_ms=total, matrix_peak_floor_ms=gemm_floor,
    attention_pair_peak_floor_ms=attention_floor, combined_pair_compute_floor_ms=gemm_floor+attention_floor,
    perfect_matrices_with_everything_else_measured_ms=total-gemm_measured+gemm_floor,
    perfect_matrices_and_free_generated_E_ms=total-gemm_measured+gemm_floor-generated_ms,
    two_fresh_images_linear_vision_scenario_ms=total-result['measured_ms_per_query']['vision']/2,
    note='Dense GEMM floors omit softmax/SFU, casts, layout, launch, inputs, and auxiliary action projections.')
  result['traffic'] = dict(expert_packed_k_or_v_bytes=2*kv_elements, expert_packed_pair_bytes=4*kv_elements,
    expert_mask_logical_read_write_bytes=mask_traffic,
    expert_mask_event_ms=by_name('diffusion','E_128_37_8_16_4'),
    expert_mask_logical_GB_s=mask_traffic/by_name('diffusion','E_128_37_8_16_4')/1e6,
    residual_logical_bytes=56*2048*2048*6,
    residual_event_ms=by_name('prefill','E_32768_32_4'))
  pixel_bytes = {entry['uploaded_pixel_bytes'] for entry in raw['runs']}
  assert pixel_bytes == {2_211_840}
  result['input_link'] = dict(pixel_payload_bytes=pixel_bytes.pop(), link_Gbit_s=5,
    encoding_payload_ceiling_bytes_s=500_000_000, pixel_transfer_floor_ms=2_211_840/500_000_000*1000,
    assumption='Recorded parent USB 5 Gbit/s link, 8b/10b encoding; excludes protocol overhead and other input arrays.')
  result['one_fusion_candidate'] = dict(name='post-attention residual + RMSNorm + fixed-scale packed FP8, language only',
    sites=28, shape=[2048,2048], fixed_scale_key_suffix='/2',
    residual_ms_at_half_aggregate=by_name('prefill','E_32768_32_4')/2,
    norm_ms_at_half_aggregate=by_name('prefill','alpamayo_norm')/2,
    pack_ms_at_half_aggregate=by_name('prefill','E_128_64_8_16_4')/2,
    norm_bf16_roundtrip_bytes_removed=28*2048*2048*4,
    assumption='Each aggregate contains 28 input and 28 post-attention sites; durations are not individually distinguished.')
  fusion = result['one_fusion_candidate']
  fusion['current_chain_ms_estimate'] = sum(fusion[k] for k in
      ('residual_ms_at_half_aggregate', 'norm_ms_at_half_aggregate', 'pack_ms_at_half_aggregate'))
  fusion['removed_separate_E_event_ms_estimate'] = fusion['residual_ms_at_half_aggregate']+fusion['pack_ms_at_half_aggregate']
  auxiliary_bf16_flops = 0
  for name in ('expert.action_in_proj.encoder.trunk.0', 'expert.action_in_proj.encoder.trunk.3',
               'expert.action_in_proj.encoder.trunk.6', 'expert.action_out_proj'):
    n, k = tensors[name+'.weight']['shape']
    auxiliary_bf16_flops += 2*actions*n*k*steps
  head_flops = 2*6*raw['reduced_vocab']['padded_rows']*2048
  result['small_auxiliary_matrices'] = dict(expert_bf16_flops=auxiliary_bf16_flops,
    expert_bf16_dense_peak_ms=auxiliary_bf16_flops/BF16*1000,
    six_vocab_head_fp8_flops=head_flops, six_vocab_head_dense_peak_ms=head_flops/FP8*1000,
    note='Excluded from the primary matrix table; these execute outside its named GEMM families.')
  result['byte_limits'] = ['Logical one-touch bytes count each matrix independently; inter-kernel cache reuse remains possible.',
    'Logical tile operand bytes reuse each operand within a 64x64 tile; they are not external-memory bytes.',
    'Unique-weight cache floors assume 32,000,000 cache bytes and ignore smaller hardware caches.',
    'Biases, scales, padded arithmetic, temporary tensors, and reduction/SFU costs are excluded from the primary matrix estimates.']
  return result


def inspect_llvm(cache):
  names = ('E_32768_32_4', 'E_128_64_8_16_4', 'E_128_288_8_16_4', 'E_64560_32_3', 'E_17280_32_3',
           'E_148_16_2_16_4_2_4', 'E_8_8_74_8_16_4', 'E_128_37_8_16_4',
           'E_1024_4_4_2_16_2_4', 'E_2048_4_8_16_2_2', 'E_2_8_45_6_2_16_3_4',
           'E_45_269_2_4_16_4', 'E_16_15_18_2_16_4_3', 'E_90_36_8_16_4')
  result = []
  with sqlite3.connect('file:'+str(cache)+'?mode=ro', uri=True, timeout=2) as connection:
    for features in ('+cumode', '-cumode'):
      table = 'compile_llvm_gfx1200_'+features+'_opt_24'
      for name in names:
        rows = connection.execute('SELECT key FROM "'+table+'" WHERE instr(key,?)>0 LIMIT 16', ('@'+name+'(',)).fetchall()
        for (text,) in rows:
          if len(text) > 1_000_000: raise ValueError('Unexpected generated LLVM size')
          signature = next(line for line in text.splitlines() if 'define amdgpu_kernel' in line)
          result.append(dict(name=name, cache_table=table, sha256=hashlib.sha256(text.encode()).hexdigest(),
                             signature=signature, source_bytes=len(text),
                             static_ir_counts={op:len(re.findall(re.escape(op),text)) for op in
                                               ('load ', 'store ', 'fadd ', 'fmul ', 'fdiv ', 'exp2', '32767')},
                             limit='Cached source candidates, not a runtime binary identity. Names can collide.'))
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--trial', type=Path, default=TRIAL)
  parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
  parser.add_argument('--llvm', action='store_true')
  args = parser.parse_args()
  result = analyze(args.trial, args.checkpoint)
  if args.llvm: result['cached_llvm_candidates'] = inspect_llvm(CACHE)
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()

"""CPU-only compile and static cost analysis for K128/K256 grouped kernels."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  args.output.mkdir(parents=True,exist_ok=False)
  kernel=local('agent-int4-grouped-kernel');old=local('agent-int4k32-compile');scaled=local('agent-int4-scaled-compile')
  records=[]
  for m,n,k in ((17,48,512),(2112,2048,9216)):
    for group in (128,256):
      for mode in ('quantize','grouped','fp32','raw0'):
        if mode=='quantize':text,binary=kernel.compiled_quantize(m,k,group)
        else:text,binary=kernel.compiled_gemm(m,n,k,group,raw_group=0 if mode=='raw0' else -1,fp32=mode=='fp32')
        asm=old.disassemble(binary);counts=Counter(re.findall(r'^\s*([a-z][a-z0-9_]+)\s',asm,re.M))
        resources=old.metadata(binary);desc=scaled.descriptor(binary)
        assert resources['workgroup_processor_mode']==0 and resources['wavefront_size']==32
        assert desc['fp32_denorm_mode']==3 and desc['fp32_rounding_mode']==0
        if mode!='quantize':
          assert counts['v_wmma_i32_16x16x32_iu4']>0 and counts['v_wmma_i32_16x16x16_iu4']==0
          assert all('neg_lo:[0,1,0]' in line for line in asm.splitlines() if 'v_wmma_i32_16x16x32_iu4 ' in line)
          assert not any('fma' in name for name in counts),'Group multiply and add must remain separate'
        stem=f'{m}-{n}-{k}-g{group}-{mode}'
        (args.output/(stem+'.cpp')).write_text(text);(args.output/(stem+'.s')).write_text(asm);(args.output/(stem+'.hsaco')).write_bytes(binary)
        record=dict(shape=[m,n,k],group=group,mode=mode,resources=resources,descriptor=desc,binary_bytes=len(binary),
          instructions={name:count for name,count in counts.items() if any(token in name for token in
            ('wmma','global_load','global_store','mul_f32','add_f32','cvt_f32_i32','bpermute','rndne','div','rcp','branch'))})
        records.append(record);print(json.dumps(record),flush=True)
  m,n,k=2112,2048,9216
  costs=[]
  for group in (128,256):
    ng=k//group;elements=m*n;blocks=(m//64)*(n//64);waves=blocks*4
    costs.append(dict(group=group,groups=ng,group_int32_bounds=[-120*group,105*group],
      packed_activation_bytes=m*k//2,packed_weight_bytes=n*k//2,activation_scale_bytes=m*ng*4,weight_scale_bytes=n*ng*4,
      offline_weight_group_reductions=n*ng,offline_weight_values_quantized=n*k,offline_weight_preparation_in_timing=False,
      quantize_workgroups=m*k//1024,quantize_waves=m*k//256,activation_full_input_reads=1,
      quantize_input_bytes=m*k*2,quantize_packed_output_bytes=m*k//2,quantize_logical_value_divisions=m*k,
      quantize_group_scale_divisions=m*ng,quantize_group_reductions=m*ng,
      quantize_active_lane_scale_divisions=m*k//8,quantize_vector_shuffle_instructions=(m*k//256)*(4 if group==128 else 5),
      gemm_output_tiles=blocks,gemm_waves=waves,native_k32_wmma_instructions=waves*(k//32)*4,
      element_group_int32_to_fp32=elements*ng,element_group_scale_multiplies=elements*ng,
      element_group_term_multiplies=elements*ng,element_group_sum_adds=elements*ng,
      full_integer_multiply_add_ops=2*m*n*k,
      fp32_output_tile_registers_per_lane=32,int32_partial_registers_per_lane=32))
  files=[Path(__file__),Path(__file__).with_name('agent-int4-grouped-kernel.py'),Path(__file__).with_name('agent-int4-grouped-probe.py'),
         Path(__file__).with_name('agent-int4-scaled-kernel.py')]
  revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/tmp/opencode/alpamayo-asius-20261114/tinygrad',text=True).strip()
  flat_path=Path(__file__).with_name('agent-int4-scaled-compiled-v5')/'report.json'
  flat=json.loads(flat_path.read_text())
  flat_record=next(r for r in flat['records'] if r['shape']==[2112,2048,9216] and r['mode']=='no-bias-cu')
  report=dict(cpu_only=True,complete=True,tinygrad_revision=revision,groups=[128,256],records=records,logical_cost=costs,
              frozen_flat_comparison=dict(source=str(flat_path),sha256=hashlib.sha256(flat_path.read_bytes()).hexdigest(),record=flat_record),
              source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
  (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

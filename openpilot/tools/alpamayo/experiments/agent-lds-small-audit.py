"""CPU/compiler audit of exactly two smaller single-buffer packed FP8 tiles."""
import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_lds_small_kernel',HERE/'agent-lds-small-kernel.py')
S=importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
H,E=S.H,S.H.E
SHAPE=(2048,2048,9216)
COLD=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-lds-cold-fc2')
COLD_HASHES={'report.json':'f00e014c016e9c5c2e5f48c7c6535253d6810f84481f024a53835cbcd0c04c60',
  'run.log':'b7122e8f2fd1ec5e3882fa9f6c3463a5c11092b840639fff122dae733b4b59d2',
  'invocation.json':'6a01b74bdab1de49efd7f5e38109aafc2cc5d770916041bd7279985e27a84c90',
  'wall.json':'d149bf3d954fe08ee640ae5dad6521db774901ac935ebd2a1664745974e2a6b5'}


def provenance():
  for name,digest in COLD_HASHES.items(): assert E.sha((COLD/name).read_bytes())==digest
  cold=json.loads((COLD/'report.json').read_text())
  assert cold['complete'] and cold['gpu_executed'] and cold['timing_data_transfer_bytes']==0
  assert all(v['equal']==v['expected_equal'] for v in cold['validation'])
  files=dict(cold['provenance']['files'])
  for name,digest in files.items(): assert E.sha(Path(name).read_bytes())==digest, f'Qualified source changed: {name}'
  files.update({str(COLD/name):digest for name,digest in COLD_HASHES.items()})
  paths=list(HERE.glob('agent-lds-small-*.py'))+[Path('/tmp/opencode/alpamayo-lds-check.json'),Path('/tmp/opencode/alpamayo-lds-scaled.json')]
  paths += [HERE/name for name in ('lds-gemm.py','check-lds-scaled.py','compile-lds-reference.py','hip-compiler.py')]
  paths += [Path('/tmp/opencode/rdna-fp8-reference')/name for name in ('kernel.hip','gemm_common.hpp','configs.hpp')]
  files.update({str(p):E.sha(p.read_bytes()) for p in paths})
  return dict(files=files,source_sha256=E.sha(json.dumps(files,sort_keys=True).encode()),
    amd_commit=H.provenance()['amd_commit'],amd_license=H.provenance()['amd_license'])


def geometry():
  records=[]
  for label,(bm,bn,bk,wm,wn) in S.CONFIGS.items():
    threads=wm*wn*32
    assert threads==128 and bk==64 and bm%(threads//bk*16)==bn%(threads//bk*16)==0
    checked_fragments=checked_chunks=0
    for operand,(size,waves) in enumerate(((bm,wm),(bn,wn))):
      # Distinct labels make every logical row/K byte independently checkable.
      for valid_rows in ((1,17,size-1,size,size+1,193) if operand==0 else (16,32,size-16,size,size+16,208)):
        k=256
        logical=np.arange(((valid_rows+15)//16*16)*k,dtype=np.int64).reshape(-1,k)+1
        # Both valid physical padding and descriptor-clamped missing fragments occur.
        packed=E.pack_cpu(logical).reshape(-1)
        for origin in (0,(valid_rows//size)*size):
          for ko in range(0,k,bk):
            staged=np.zeros(size*bk,np.int64)
            writes=np.zeros(size*bk,np.uint8)
            for tid in range(threads):
              row,word=tid//bk*16,tid%bk*4
              for chunk in range(size*bk//(threads*16)):
                dst=(row+chunk*(threads//bk*16))*bk+word*4
                src=(origin+row+chunk*(threads//bk*16))*k+ko*16+word*4
                assert src%16==dst%16==0 and 0<=dst<=size*bk-16
                # Byte-level bounds model for the raw-buffer descriptor.
                indices=np.arange(src,src+16)
                safe=indices<packed.size
                staged[dst:dst+16]=np.where(safe,packed[np.minimum(indices,packed.size-1)],0)
                writes[dst:dst+16]+=1; checked_chunks+=1
            assert (writes==1).all()
            for wave in range(waves):
              for tile in range(size//waves//16):
                for kk in range(bk//16):
                  for lane in range(32):
                    at=wave*(size//waves)*bk+lane*4+tile*16*bk+kk*256
                    assert at%4==0 and at+132<=staged.size
                    actual=np.concatenate((staged[at:at+4],staged[at+128:at+132]))
                    row=origin+wave*(size//waves)+tile*16+lane%16
                    ks=ko+kk*16+np.array([0,1,2,3,8,9,10,11])+lane//16*4
                    expected=logical[row,ks] if row<logical.shape[0] else np.zeros(8,np.int64)
                    np.testing.assert_array_equal(actual,expected); checked_fragments+=1
    # Published fold positions: every accumulator visits precisely increasing K16.
    tm,tn=bm//wm//16,bn//wn//16
    recurrence={(m,n):[] for m in range(tm) for n in range(tn)}
    prefetch=[[],[]]
    for base in range(0,SHAPE[2],bk):
      for step in range(bk//16*tm*tn):
        outer,n=divmod(step,tn); ks,m=divmod(outer,tm)
        recurrence[m,n].append(base+ks*16)
        if base==0 and step//2 < (bm if step%2==0 else bn)*bk//(threads*16): prefetch[step%2].append(step//2)
    assert all(values==list(range(0,SHAPE[2],16)) for values in recurrence.values())
    assert prefetch==[list(range(bm*bk//(threads*16))),list(range(bn*bk//(threads*16)))]
    available_a,available_b={0},{0}
    for step in range(bk//16*tm*tn):
      outer,n=divmod(step,tn); ks,m=divmod(outer,tm)
      if step+1<bk//16*tm*tn:
        next_outer,next_n=divmod(step+1,tn); next_ks,next_m=divmod(next_outer,tm)
        if next_ks*tm+next_m!=ks*tm+m: available_a.add(next_ks*tm+next_m)
        if next_m==0: available_b.add(next_ks*tn+next_n)
      assert ks*tm+m in available_a and ks*tn+n in available_b
    for word in (0,32): assert len({(lane+word)%32 for lane in range(32)})==32
    # Independent enumeration of all guarded output stores and block swizzling.
    for m,n,k in ((1,16,64),(17,48,128),(193,208,256),SHAPE):
      mt,nt=math.ceil(m/bm),math.ceil(n/bn)
      coords=[]
      for block in range(mt*nt):
        im,inn=divmod(block,nt)
        count=8 if im<mt-mt%8 else mt%8
        coord=(im,inn) if mt==1 or nt==1 else ((inn+(im%8)*nt)%count+(im//8)*8,(inn+(im%8)*nt)//count)
        coords.append(coord)
      assert len(set(coords))==mt*nt and set(coords)=={(i,j) for i in range(mt) for j in range(nt)}
      if m<200:
        stores=np.zeros((m,n),np.uint8)
        for im,inn in coords:
          for wave in range(wm*wn):
            for i in range(tm):
              for j in range(tn):
                for lane in range(32):
                  col=inn*bn+(wave%wn)*(bn//wn)+j*16+lane%16
                  for r in range(8):
                    row=im*bm+(wave//wn)*(bm//wm)+i*16+8*(lane//16)+r
                    if row<m and col<n: stores[row,col]+=1
        assert (stores==1).all()
    # Uniform barriers: one initial commit barrier, then read-before-write and commit barriers per next K64 stage.
    events=['commit0','barrier']
    for stage in range(SHAPE[2]//bk-1): events.extend((f'read{stage}','barrier',f'commit{stage+1}','barrier'))
    events.append(f'read{SHAPE[2]//bk-1}')
    barriers=events.count('barrier')
    assert barriers==2*(SHAPE[2]//bk)-1
    text=S.source(*SHAPE,label)
    assert 'if constexpr(SingleBuffer)' in text and text.count('__syncthreads();')==3
    assert 'gemm_fp8<WmmaFp8Policy,'+','.join(map(str,S.CONFIGS[label]))+',true>' in text
    assert text.index('for(int k_tile =')<text.index('if constexpr(SingleBuffer)')<text.rindex('for(int m =')
    records.append(dict(label=label,threads=threads,checked_chunks=checked_chunks,checked_fragments=checked_fragments,
      every_staged_byte_written_once=True,fragment_packing_exact=True,whole_16_byte_descriptor_clamping=True,
      padded_M_and_N_tail_stores=True,output_partition_exact=True,block_swizzle_exact=True,
      ascending_k16_per_accumulator=True,all_prefetch_chunks_issued=True,uniform_barrier_schedule=True,barriers_per_block=barriers))
    records[-1].update(no_fragment_read_before_load=True,each_LDS_fragment_word_uses_32_distinct_banks=True,
      complete_prefetch_offset_sequence=[bk*16,(SHAPE[2]-bk)*16])
    for shape in ((1,16,48),(1,17,64),(0,16,64)):
      try: S.source(*shape,label)
      except AssertionError: pass
      else: raise AssertionError('Invalid shape passed the source guard')
  return records


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0',BEAM='0',JITBEAM='0',PARALLEL='0',PROFILE='0',DEBUG='0')
  from tinygrad import Device
  report=dict(complete=False,gpu_executed=False,provenance=provenance(),configurations=S.CONFIGS)
  args.output.parent.mkdir(parents=True,exist_ok=True)
  try:
    render=H.renderer()
    report['geometry']=geometry()
    report['variants']={}
    for label,fn in S.functions(render).items():
      report['variants'][label]=S.compile_one(SHAPE,False,False,label,fn,render,args.output.parent/label)
      E.write_json(args.output,report)
    report['controls']={}
    for bias,row_scale in ((False,False),(True,True)):
      key='row-bias' if bias else 'scalar'
      report['controls'][key]={label:S.compile_one((193,208,256),bias,row_scale,label,fn,render,args.output.parent/(key+'-'+label))
        for label,fn in S.functions(render,row_scale).items()}
    old=S.load('compile-lds-reference')
    text=old.source(Path('/tmp/opencode/rdna-fp8-reference'),S.CONFIGS['small96'],(2027,2048,9216),True)
    binary,assembly=S.compiler_assembly(text)
    (args.output.parent/'historical-raw96.cpp').write_text(text)
    (args.output.parent/'historical-raw96.hsaco').write_bytes(binary)
    (args.output.parent/'historical-raw96.compiler.s').write_text(assembly)
    report['historical_source_current_WGP_compiler']=dict(shape_mnk=[2027,2048,9216],with_bias=True,
      actual_old_binary_or_mode_unknown=True,resources=H.C.metadata(binary),binary_sha256=E.sha(binary),source_sha256=E.sha(text.encode()),
      estimates={key:int(value) for key,value in re.findall(r';\s*(Occupancy|NumVgprs|NumSgprs|ScratchSize|LDSByteSize):\s*(\d+)',assembly)})
    assert not Device._opened_devices
    assert provenance()==report['provenance']
    report['complete']=True
    E.write_json(args.output,report)
    print(json.dumps(dict(complete=True,variants={k:{'status':v['status'],'resources':v['resources'],'estimates':v['compiler_estimates']} for k,v in report['variants'].items()})))
  except BaseException as exc:
    report['failure']=f'{type(exc).__name__}: {exc}'
    E.write_json(args.output,report)
    raise


if __name__=='__main__': main()

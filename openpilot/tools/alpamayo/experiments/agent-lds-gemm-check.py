"""CPU compiler audit by default. GPU mode is reserved for the parent wrapper.

One selected actual shape, two published configurations, no model integration.
Reject each bad candidate independently. Verify six mutations after timing.
"""
import argparse
from collections import Counter
from functools import partial
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from types import FunctionType, SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
SHAPES = {'fc2': (2048,2048,9216), 'fc1': (2048,9216,2048)}
CONTROLS = ('integer','signed-zero','subnormal','outlier','row-bias','finite')


def load(name):
  spec = importlib.util.spec_from_file_location('agent_lds_'+name.replace('-','_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


E, K, T, C = (load(name) for name in ('agent-expert-gemm','agent-lds-gemm-kernel','tiled-gemm','agent-gemm-compile'))


def provenance():
  import tinygrad
  paths = [Path(__file__), HERE/'agent-lds-gemm-kernel.py', HERE/'agent-lds-gemm-host.py', HERE/'agent-expert-gemm.py', HERE/'tiled-gemm.py',
    HERE/'agent-gemm-compile.py', HERE/'agent-matrix-tiles-assembly.py', HERE/'prefill-kernels.py',
    HERE/'wgp_compiler.py', HERE.parent/'amd_gemm.py']
  paths += [K.REFERENCE/name for name in K.HASHES]
  root = Path(tinygrad.__file__).parent
  paths += [root/name for name in ('runtime/support/compiler_amd.py','runtime/support/compiler_llvm.py',
    'runtime/support/hcq2.py','runtime/ops_amd.py','engine/jit.py','codegen/__init__.py','renderer/llvmir.py')]
  hashes = {str(p): E.sha(p.read_bytes()) for p in paths}
  return dict(files=hashes, source_sha256=E.sha(json.dumps(hashes,sort_keys=True).encode()),
    amd_commit='d908d68d5741b0f449d7c31efa733fe8d4865b51',
    amd_license='No license found in inspected source directory or repository root; private experiment only.')


def renderer():
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  value = AMDLLVMRenderer(Target('AMD','LLVM','gfx1200'))
  value.compiler = LLVMCompiler('AMDGPU','gfx1200','-cumode')
  return value


def functions(render, row_scale):
  return {'current': partial(E.bind_renderer(T.kernel,render),tile_m=4,tile_n=4,row_scale=row_scale),
    **{mode:partial(K.program,mode=mode,row_scale=row_scale) for mode in K.CONFIGS}}


def compile_one(shape, bias, row_scale, label, fn, render, artifact=None):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import UOp
  m,n,k = shape
  specs = [((m,n),dtypes.bfloat16), (((m+15)//16*16,k),dtypes.fp8e4m3), ((n,k),dtypes.fp8e4m3),
    ((m if row_scale else 1,),dtypes.float32), ((1,),dtypes.float32)]
  if bias: specs.append(((n,),dtypes.bfloat16))
  arguments = [UOp.placeholder(s,dt,i) for i,(s,dt) in enumerate(specs)]
  program = to_program(fn(*arguments),render)
  text,binary = program.src[2].arg,program.src[3].arg
  assembly,resources = C.disassemble(binary),C.metadata(binary)
  ins = Counter(re.findall(r'^\s*([a-z][a-z0-9_]+)\s',assembly,re.M))
  wmma = [line.strip() for line in assembly.splitlines() if line.strip().startswith('v_wmma_f32_16x16x16_fp8_fp8')]
  zero_c = sum(bool(re.search(r',\s*0\s*//',line)) for line in wmma)
  # Any nonzero private allocation rejects the candidate, even with zero reported spills.
  failures = []
  for field in ('private_segment_fixed_size','vgpr_spill_count','sgpr_spill_count'):
    if resources[field]: failures.append(f'{field}={resources[field]}')
  if resources['wavefront_size'] != 32 or resources['workgroup_processor_mode'] != 1: failures.append('wave32/WGP mismatch')
  if label != 'current' and zero_c:
    # A fully unrolled small control may initialize each accumulator with C=0.
    # Reject a zero reset of an already used accumulator, or any detached FP32 sum.
    seen=set()
    for line in wmma:
      dest=re.search(r'v\[\d+:\d+\]',line).group()
      if re.search(r',\s*0\s*//',line) and dest in seen: failures.append('Repeated zero-C WMMA')
      seen.add(dest)
    if k > 256: failures.append('Zero-C WMMA in long-K loop')
  if label != 'current' and any('add_f32' in op for op in ins): failures.append('Detached FP32 add')
  if label != 'current':
    bm,bn,bk,wm,wn = K.CONFIGS[label]
    assert resources['group_segment_fixed_size'] == (bm+bn)*bk
    assert math.prod(program.arg.local_size) == wm*wn*32
  else: bm,bn,bk,wm,wn = 64,64,16,1,1
  groups = math.ceil(m/bm)*math.ceil(n/bn)
  baseline = math.ceil(m/64)*math.ceil(n/64)*128*k
  params = [UOp.param(slot,arg.dtype,arg.numel(),device='AMD') for slot,arg in zip((5,0,1,2,3,4),arguments)]
  proof = E.prove_cpu_queue([program.call(*params)]*40,device='AMD')
  selected = {op:count for op,count in ins.items() if any(s in op for s in ('load','store','wmma','barrier','setprio','scratch','fadd','add_f32'))}
  if artifact:
    artifact.with_suffix('.source').write_text(text)
    artifact.with_suffix('.s').write_text(assembly)
    artifact.with_suffix('.hsaco').write_bytes(binary)
  estimates={}
  if artifact:
    if label=='current':
      from tinygrad.runtime.autogen import llvm
      llvm.LLVMSetTargetMachineAsmVerbosity(render.compiler.target_machine,True)
      verbose=load('agent-matrix-tiles-assembly').assembly(render.compiler,text)
    else:
      recompiled,verbose=K.compiler_assembly(text)
      assert recompiled==binary, 'Assembly inspection changed compiled binary'
    artifact.with_suffix('.compiler.s').write_text(verbose)
    estimates={key:int(value) for key,value in re.findall(r';\s*(Occupancy|NumVgprs|NumSgprs|ScratchSize|LDSByteSize):\s*(\d+)',verbose)}
  return dict(status='compiler-rejected' if failures else 'compiled',reasons=failures,resources=resources,
    binary_sha256=E.sha(binary),source_sha256=E.sha(text.encode()),static_instructions=selected,
    native_wmma=len(wmma),native_zero_c_wmma=zero_c,compiler_estimates=estimates,
    global_size=program.arg.global_size,local_size=program.arg.local_size,
    nominal_operand_request_bytes=groups*(bm+bn)*k,request_ratio_to_current=groups*(bm+bn)*k/baseline,
    padding_compute_ratio=groups*bm*bn/(m*n),offline_serial_queue_proof=proof,
    occupancy_note='Resource ceilings only. No measured occupancy or throughput; inspect LDS and VGPR limits together.')


def fixture(shape, state, kind='finite', row_scale=False):
  m,n,k = shape
  assert 0 <= state <= 6
  # Six cumulative changes: A, activation scale, W, weight scale, A+bias, then all inputs.
  aseed = 0 if state == 0 else 1 if state < 5 else 2 if state == 5 else 3
  wseed = 0 if state < 3 else 1 if state < 6 else 2
  ar,wr = np.random.default_rng(906000+aseed),np.random.default_rng(906100+wseed)
  def bits(rng, shape): return rng.integers(0,127,shape,dtype=np.uint8) | (rng.integers(0,2,shape,dtype=np.uint8)<<7)
  if kind in ('integer','row-bias'):
    a,w = E.int_fp8(ar.integers(-7,8,(m,k))),E.int_fp8(wr.integers(-7,8,(n,k)))
  elif kind == 'signed-zero':
    a,w = ar.integers(0,2,(m,k),dtype=np.uint8)<<7,wr.integers(0,2,(n,k),dtype=np.uint8)<<7
  elif kind == 'subnormal':
    a = ar.integers(0,8,(m,k),dtype=np.uint8) | (ar.integers(0,2,(m,k),dtype=np.uint8)<<7)
    w = wr.integers(0,8,(n,k),dtype=np.uint8) | (wr.integers(0,2,(n,k),dtype=np.uint8)<<7)
  elif kind == 'outlier':
    a,w = np.zeros((m,k),np.uint8),np.zeros((n,k),np.uint8)
    for target,rng in ((a,ar),(w,wr)):
      target[:,::64] = np.uint8(126)|(rng.integers(0,2,target[:,::64].shape,dtype=np.uint8)<<7)
  else:
    a,w = bits(ar,(m,k)),bits(wr,(n,k))
    for value in (a,w): value.flat[:8] = [0,128,1,129,126,254,56,184]
  finite = kind == 'finite'
  sx = (.125 if state < 2 else .071234569 if state < 6 else .0937129) if finite else (.125 if state < 2 else .25 if state < 6 else .5)
  sw = (.25 if state < 4 else .01933221 if state < 6 else .0291127) if finite else (.25 if state < 4 else .5 if state < 6 else .125)
  scales = np.full(m if row_scale else 1,sx,np.float32)
  if row_scale: scales *= np.exp2(np.arange(m)%3-1).astype(np.float32)
  if kind=='signed-zero' and state in (2,3,6): scales *= np.float32(-1)
  bias_rng = np.random.default_rng(906200+(0 if state < 5 else state-4))
  bias = E.bf16(bias_rng.integers(-8,9,n).astype(np.float32)/8)
  return dict(a=E.pack_cpu(a),w=E.pack_cpu(w),sx=scales,sw=np.array([sw],np.float32),bias=bias,a_unpacked=a,w_unpacked=w)


def independent(data, bias, row_scale):
  a,w = (E.decode_fp8(data[key+'_unpacked']).astype(np.float64) for key in ('a','w'))
  total = a@w.T
  # These controls have exact products, exact sums, and power-of-two scales.
  np.testing.assert_array_equal(total,total.astype(np.float32).astype(np.float64))
  scale = (data['sx'][:,None] if row_scale else data['sx'][0])*data['sw'][0]
  total = total.astype(np.float32)*scale
  if bias: total += E.unpack_bf16(data['bias'])
  return E.bf16(total)


def geometry():
  records = []
  for mode,(bm,bn,bk,wm,wn) in K.CONFIGS.items():
    # Every staged byte and lane fragment, at an interior and a trailing block.
    m,n,k = 193,208,256
    for dimension,waves in ((m,wm),(n,wn)):
      size = bm if dimension == m else bn
      logical = np.arange(dimension*k,dtype=np.int64).reshape(dimension,k)+1
      packed = E.pack_cpu(logical).reshape(-1)
      threads,step = wm*wn*32,wm*wn*32//bk*16
      for origin in (0,(dimension//size)*size):
        for ko in (0,128):
          staged = np.zeros(size*bk,np.int64)
          written = np.zeros(size*bk,np.uint8)
          for tid in range(threads):
            row,word = tid//bk*16,tid%bk*4
            for c in range(size*bk//(threads*16)):
              dst=(row+c*step)*bk+word*4
              src=(origin+row+c*step)*k+ko*16+word*4
              staged[dst:dst+16] = packed[src:src+16] if src+16 <= packed.size else 0
              written[dst:dst+16] += 1
          assert (written == 1).all()
          for wave in range(waves):
            for tile in range(size//waves//16):
              for kk in range(bk//16):
                for lane in range(32):
                  at=wave*(size//waves)*bk+lane*4+tile*16*bk+kk*256
                  actual=np.concatenate((staged[at:at+4],staged[at+128:at+132]))
                  row=origin+wave*(size//waves)+tile*16+lane%16
                  ks=ko+kk*16+np.array([0,1,2,3,8,9,10,11])+lane//16*4
                  expected=logical[row,ks] if row < dimension else np.zeros(8,np.int64)
                  np.testing.assert_array_equal(actual,expected)
    for m,n,k in SHAPES.values():
      mt,nt = math.ceil(m/bm),math.ceil(n/bn)
      coords=[]
      for block in range(mt*nt):
        im,inn=divmod(block,nt)
        count=8 if im < mt-mt%8 else mt%8
        local=inn+(im%8)*nt
        coords.append((local%count+(im//8)*8,local//count))
      assert set(coords) == {(i,j) for i in range(mt) for j in range(nt)} and len(set(coords)) == mt*nt
      assert [base+kk for base in range(0,k,128) for kk in range(0,128,16)] == list(range(0,k,16))
    records.append(dict(mode=mode,every_staged_byte_once=True,every_lane_fragment_exact=True,
      padded_M_and_invalid_N_safe=True,swizzled_actual_output_partition_exact=True,ascending_k16=True))
  for mode in K.CONFIGS:
    try: K.source(17,48,48,mode)
    except AssertionError: pass
    else: raise AssertionError('An unsupported partial K128 stage was admitted')
  return records


def checked(jit,tensors,device):
  E.poison(jit.captured.ret,device)
  value=E.read_output(jit(*tensors)[0],device)
  assert np.isfinite(E.unpack_bf16(value)).all(), 'Nonfinite output'
  return value


def gpu_case(name,shape,kind,bias,row_scale,fns,resources,device,progress=lambda _:None):
  row=dict(case=name,shape_mnk=shape,kind=kind,bias=bias,row_scale=row_scale,variants={
    label:dict(info,checks=[]) for label,info in resources.items()},timing=[])
  def reject(label,stage,state,exc):
    row['variants'][label].update(status='rejected',failure=dict(stage=stage,state=state,error=str(exc)[:2000]))
    # Remove only this candidate's samples. Other accepted results remain reviewable.
    invalid=[sample for sample in row['timing'] if sample['variant']==label]
    if invalid: row.setdefault('invalidated_timing',[]).extend(invalid)
    row['timing']=[sample for sample in row['timing'] if sample['variant'] != label]
    progress(row)
    if label == 'current': raise AssertionError(f'Baseline gate failed at {stage}/{state}') from exc
  def compare(label,jit,expected,stage,state):
    try: np.testing.assert_array_equal(checked(jit,tensors,device),expected)
    except AssertionError as exc: reject(label,stage,state,exc); return False
    row['variants'][label]['checks'].append(dict(stage=stage,state=state,finite_exact_bf16=True,sha256=E.sha(expected.tobytes())))
    return True
  tensors=E.upload(fixture(shape,0,kind,row_scale))
  singles={}
  for label,fn in fns.items():
    if resources[label]['status'] != 'compiled': continue
    jit=E.make_jit(shape,bias,fn,1)
    try:
      for _ in range(3): jit(*tensors)
      assert len(E.captured_kernels(jit)) == 1
    except AssertionError as exc: reject(label,'single_capture',0,exc); continue
    singles[label]=jit
  assert 'current' in singles
  def reference(state,previous=None):
    data=fixture(shape,state,kind,row_scale)
    E.change_inputs(tensors,data,device)
    try:
      value=checked(singles['current'],tensors,device)
      if kind != 'finite': np.testing.assert_array_equal(value,independent(data,bias,row_scale))
      if previous is not None and kind != 'signed-zero': assert not np.array_equal(value,previous), 'Mutation did not affect reference'
    except AssertionError as exc: reject('current','reference',state,exc)
    return value
  expected=reference(0)
  for label,jit in singles.items(): compare(label,jit,expected,'single',0)
  if name.startswith('control'):
    for state in range(1,7):
      expected=reference(state,expected)
      for label,jit in singles.items():
        if row['variants'][label]['status'] != 'rejected': compare(label,jit,expected,'control_mutation',state)
    for info in row['variants'].values():
      if info['status'] == 'compiled': info['status']='passed-control'
    progress(row)
    return row
  batches={}
  for label in singles:
    if row['variants'][label]['status'] == 'rejected': continue
    jit=E.make_jit(shape,bias,fns[label],40)
    try:
      for _ in range(3): jit(*tensors)
      assert len(E.captured_kernels(jit)) == 40 and E.shared_output_slot(jit) == 5
      row['variants'][label]['serial_queue_proof']=jit.proofs[0]
    except AssertionError as exc: reject(label,'queue',0,exc); continue
    batches[label]=jit
    compare(label,jit,expected,'batch',0)
  changed=reference(1,expected)
  for label,jit in batches.items():
    if row['variants'][label]['status'] != 'rejected': compare(label,jit,changed,'changed_batch',1)
  expected=reference(0,changed)
  eligible=[label for label in batches if row['variants'][label]['status'] != 'rejected']
  for label in eligible:
    if label == 'current': continue
    pairs=[]
    for iteration in range(3):
      order=['current',label] if iteration%2 == 0 else [label,'current']
      pair=dict(order=order)
      for key in order:
        device.synchronize(); start=time.perf_counter()
        batches[key](*tensors)
        device.synchronize(); pair[key]=(time.perf_counter()-start)*1000
      pairs.append(pair)
    base,candidate=(statistics.median(pair[key] for pair in pairs)/40 for key in ('current',label))
    row['timing'].append(dict(variant=label,paired_batch_ms=pairs,calls_per_batch=40,current_ms=base,candidate_ms=candidate,
      candidate_over_current=candidate/base,provisional=True,host_inclusive=True,warm_repeated_weights=True,
      packing_uploads_excluded=True,model_streaming_reproduced=False))
  for label in eligible: compare(label,batches[label],expected,'post_timing_original',0)
  # Six full-output gates happen AFTER the timings, through both captured graph sizes.
  for state in range(1,7):
    expected=reference(state,expected)
    for label in eligible:
      if row['variants'][label]['status'] != 'rejected':
        compare(label,singles[label],expected,'post_timing_single_mutation',state)
      if row['variants'][label]['status'] != 'rejected':
        compare(label,batches[label],expected,'post_timing_batch_mutation',state)
    progress(row)
  for info in row['variants'].values():
    if info['status'] == 'compiled': info['status']='passed'
  for sample in row['timing']: sample['provisional']=False
  progress(row)
  return row


def failure_mocks():
  reports=[]
  for failure in ('single','batch','queue','nonfinite','post-timing','blocked','baseline'):
    state,constructed,timed=[0],[],[]
    shape=(17,48,256)
    def expected(): return np.full(shape[:2],0x3f80+state[0],np.uint16)
    class Jit:
      def __init__(self,label,count):
        self.label,self.count,self.calls=label,count,0
        self.captured=SimpleNamespace(ret=())
        self.proofs=[dict(captured_calls=40,ordered_execution_barriers=39)]
        constructed.append((label,count))
      def __call__(self,*inputs): self.calls+=1; return (None,)
    def run(jit,*unused):
      value=expected()
      if jit.label == 'amd3':
        bad=(failure=='single' or failure=='batch' and jit.count==40 or
          failure=='post-timing' and jit.count==40 and state[0]==6)
        if failure=='nonfinite': raise AssertionError('Nonfinite output')
        if bad: value[0,0]+=1
      return value
    def slot(jit):
      if failure=='queue' and jit.label=='amd3': raise AssertionError('Missing WAW barrier')
      return 5
    def independent_fake(*unused):
      value=expected()
      if failure=='baseline': value[0,0]+=1
      return value
    fake=SimpleNamespace(upload=lambda *_:(),make_jit=lambda s,b,fn,c:Jit(fn,c),
      captured_kernels=lambda j:['mock']*j.count,shared_output_slot=slot,
      change_inputs=lambda _,data,device:state.__setitem__(0,data['state']),sha=E.sha)
    namespace=dict(gpu_case.__globals__,E=fake,fixture=lambda shape,s,*_:dict(state=s),checked=run,independent=independent_fake)
    fn=FunctionType(gpu_case.__code__,namespace,argdefs=gpu_case.__defaults__)
    fns={label:label for label in ('current','amd3','amd4')}
    resources={label:dict(status='compiler-rejected' if failure=='blocked' and label=='amd3' else 'compiled') for label in fns}
    try: row=fn('actual',shape,'integer' if failure=='baseline' else 'finite',False,False,fns,resources,SimpleNamespace(synchronize=lambda:None))
    except AssertionError as exc:
      assert failure=='baseline' and 'Baseline gate failed' in str(exc)
      assert not any(count==40 for _,count in constructed)
      reports.append(dict(failure=failure,baseline_failure_fatal=True)); continue
    assert row['variants']['amd4']['status']=='passed'
    assert [v['variant'] for v in row['timing']]==['amd4']
    assert row['variants']['amd3']['status'] in ('rejected','compiler-rejected')
    if failure in ('single','nonfinite','blocked'): assert ('amd3',40) not in constructed
    assert len([c for c in row['variants']['amd4']['checks'] if c['stage']=='post_timing_batch_mutation'])==6
    reports.append(dict(failure=failure,bad_variant_independently_rejected=True,good_variant_passes_six_after_timing=True,mock_timings_discarded=True))
  return reports


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode',choices=('cpu','gpu'),default='cpu')
  parser.add_argument('--case',choices=tuple(SHAPES),default='fc2')
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--source-sha')
  parser.add_argument('--artifacts',type=Path,default=HERE/'agent-lds-gemm-artifacts')
  args=parser.parse_args()
  os.environ.update(BEAM='0',JITBEAM='0',PARALLEL='0',PROFILE='0',DEBUG='0')
  if args.mode=='cpu': os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0')
  from tinygrad import Device
  report=dict(mode=args.mode,provenance=provenance(),gpu_executed=False,complete=False,cases=[],
    scope='Kernel only; unchanged original packed FP8. No model hook, route quality, streaming or full pipeline claim.')
  if args.mode=='gpu': assert args.source_sha==report['provenance']['source_sha256'], 'Source binding missing or changed'
  args.artifacts.mkdir(exist_ok=True,parents=True)
  try:
    if args.mode=='cpu':
      report['geometry']=geometry()
      report['failure_isolation']=failure_mocks()
      render=renderer()
      cases=[('control',(193,208,256),True,True),*[(name,shape,False,False) for name,shape in SHAPES.items()]]
      for name,shape,bias,row_scale in cases:
        fns=functions(render,row_scale)
        results={label:compile_one(shape,bias,row_scale,label,fn,render,args.artifacts/f'{name}-{label}') for label,fn in fns.items()}
        report['cases'].append(dict(case=name,shape_mnk=shape,variants=results))
        E.write_json(args.output,report)
        print(json.dumps(dict(case=name,resources={label:r['resources'] for label,r in results.items()})),flush=True)
      assert not Device._opened_devices
    else:
      assert os.environ.get('TC_OPT')=='2' and os.environ.get('AM_POWER_LIMIT')=='144' and os.environ.get('AM_PERF_LEVEL')=='auto'
      from tinygrad.device import Compiled
      Compiled.wait_timeout_ms=10000
      load('prefill-kernels').fix_dispatch_packet()
      device=Device[Device.DEFAULT]
      assert device.arch=='gfx1200' and device.is_usb
      load('wgp_compiler').install(device)
      report['gpu_executed']=True
      blocked={}
      cases=[('control-'+kind,(193,208,256),kind,kind in ('row-bias','finite'),kind in ('row-bias','finite')) for kind in CONTROLS]
      cases.append((args.case,SHAPES[args.case],'finite',False,False))
      # Audit actual compiled resources before controls. Never launch a known spilling variant.
      actual_fns=functions(device.renderer,False)
      audit={label:compile_one(SHAPES[args.case],False,False,label,fn,device.renderer) for label,fn in actual_fns.items()}
      blocked.update({label:r['reasons'] for label,r in audit.items() if r['status']!='compiled'})
      report['actual_resource_audit']=audit
      for name,shape,kind,bias,row_scale in cases:
        fns=functions(device.renderer,row_scale)
        resources={label:compile_one(shape,bias,row_scale,label,fn,device.renderer) for label,fn in fns.items()}
        for label,reason in blocked.items(): resources[label].update(status='blocked',reasons=reason)
        def progress(row):
          if not any(row is old for old in report['cases']): report['cases'].append(row)
          E.write_json(args.output,report)
        row=gpu_case(name,shape,kind,bias,row_scale,fns,resources,device,progress)
        for label,result in row['variants'].items():
          if result['status'] not in ('passed','passed-control'): blocked[label]=result.get('failure',result.get('reasons'))
    assert provenance()==report['provenance'], 'Source files changed during validation'
    report['complete']=True
    report['summary']=dict(gpu_executed=report['gpu_executed'],cases=len(report['cases']),source_sha256=report['provenance']['source_sha256'],
      final_variants={label:value['status'] for label,value in report['cases'][-1]['variants'].items()},
      accepted_timing_variants=[row['variant'] for row in report['cases'][-1].get('timing',[])])
    E.write_json(args.output,report)
    print(json.dumps(report['summary']))
    if args.mode=='gpu' and any(v['status']=='rejected' for row in report['cases'] for v in row['variants'].values()):
      raise AssertionError('Strict arithmetic or replay gate failed; independently accepted results were retained')
  except BaseException as exc:
    report['failure']=f'{type(exc).__name__}: {exc}'
    E.write_json(args.output,report)
    raise


if __name__=='__main__': main()

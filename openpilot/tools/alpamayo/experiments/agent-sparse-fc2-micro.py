#!/usr/bin/env python3
"""Parent-only arithmetic-first sparse fc2 check. Any mismatch stops all timing."""
import argparse
from functools import partial
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),HERE/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
C=load('agent-sparse-fc2-common')


def bytes_tensor(array,device):
  from tinygrad import Tensor
  return Tensor(np.ascontiguousarray(array).copy(),device='CPU').to(device).realize()


def arithmetic(kernel,device):
  from tinygrad import Tensor,dtypes
  result=[];all_exact=True
  for fixture in C.fixtures():
    a,w=fixture['a'],fixture['w'];m,k=a.shape;n=w.shape[0]
    x=bytes_tensor(C.fmt().pack(a),device);weight=bytes_tensor(C.fmt().pack(w),device)
    xs=bytes_tensor(fixture['xs'],device);ws=bytes_tensor(fixture['ws'],device)
    one=bytes_tensor(np.array([1.],np.float32),device)
    seed=bytes_tensor(fixture['c'],device)
    scratch=kernel.allocate(m,n,k,device);scratch=kernel.compress(x,scratch)
    Tensor.realize(*scratch)
    values,metadata,flags=(s.numpy()for s in scratch)
    ev,em,ef=C.fmt().compress(a)
    np.testing.assert_array_equal(values.view(np.uint8).reshape(ev.shape),ev)
    np.testing.assert_array_equal(metadata.reshape(em.shape),em)
    np.testing.assert_array_equal(flags.reshape(ef.shape),ef)
    checks=[]
    for running in (False,True):
      raw_results={}
      for mode in ('dense','hybrid'):
        out=Tensor.empty(m,n,device=device,dtype=dtypes.float32).realize()
        raw_results[mode]=kernel.gemm(x,weight,one,one,scratch,out,seed,mode=mode,initial=running).realize().numpy()
      checks.append(dict(kind='raw-fp32-running-C' if running else 'raw-fp32-zero-C',
        **C.compare(raw_results['hybrid'],raw_results['dense'])))
      checks.append(dict(kind='finite-fp32-running-C' if running else 'finite-fp32-zero-C',
        exact=all(bool(np.isfinite(value).all())for value in raw_results.values())))
      if not running:
        out=Tensor.empty(m,n,device=device,dtype=dtypes.float32).realize()
        original=kernel.original(x,weight,one,one,out).realize().numpy()
        checks.append(dict(kind='raw-dense-mirror-vs-original-4x4',**C.compare(raw_results['dense'],original)))
        checks.append(dict(kind='raw-hybrid-vs-original-4x4',**C.compare(raw_results['hybrid'],original)))
      bf_results={}
      for mode in ('dense','hybrid'):
        out=Tensor.empty(m,n,device=device,dtype=dtypes.bfloat16).realize()
        bf_results[mode]=kernel.gemm(x,weight,xs,ws,scratch,out,seed,mode=mode,initial=running).realize().bitcast(dtypes.uint16).numpy()
      checks.append(dict(kind='bf16-running-C' if running else 'bf16-zero-C',**C.compare(bf_results['hybrid'],bf_results['dense'])))
      if not running:
        out=Tensor.empty(m,n,device=device,dtype=dtypes.bfloat16).realize()
        original=kernel.original(x,weight,xs,ws,out).realize().bitcast(dtypes.uint16).numpy()
        checks.append(dict(kind='bf16-hybrid-vs-original-4x4',**C.compare(bf_results['hybrid'],original)))
        fp64=C.fmt().decode(a)@C.fmt().decode(w).T
        if fixture['name']=='integer':
          integer_fp32=fp64.astype(np.float32)
          checks.append(dict(kind='independent-integer-fp32',**C.compare(raw_results['hybrid'],integer_fp32)))
          scaled=integer_fp32*np.float32(fixture['xs'][0]*fixture['ws'][0])
          checks.append(dict(kind='independent-integer-bf16',**C.compare(bf_results['hybrid'],C.bf16(scaled))))
        checks.append(dict(kind='fp64-independent-error-report',acceptance_rule='No tolerance; evidence only',
          dense_fp32_max_abs_error=float(np.max(np.abs(raw_results['dense'].astype(np.float64)-fp64))),
          hybrid_fp32_max_abs_error=float(np.max(np.abs(raw_results['hybrid'].astype(np.float64)-fp64)))))
    # An epilogue producing BF16 subnormals must match the actual original.
    tiny=bytes_tensor(np.array([2.**-120],np.float32),device)
    out=Tensor.empty(m,n,device=device,dtype=dtypes.bfloat16).realize()
    expected=kernel.original(x,weight,tiny,one,out).realize().bitcast(dtypes.uint16).numpy()
    out2=Tensor.empty(m,n,device=device,dtype=dtypes.bfloat16).realize()
    actual=kernel.gemm(x,weight,tiny,one,scratch,out2,seed).realize().bitcast(dtypes.uint16).numpy()
    checks.append(dict(kind='bf16-small-scale-epilogue',**C.compare(actual,expected)))
    np.testing.assert_array_equal(x.numpy(),C.fmt().pack(a));np.testing.assert_array_equal(weight.numpy(),C.fmt().pack(w))
    exact=all(c.get('exact',True)for c in checks);all_exact &= exact
    result.append(dict(case=fixture['name'],shape=[m,n,k],checks=checks,all_bits_exact=exact,
      actual_eligible_tiles=int(np.count_nonzero(flags)),total_tiles=int(flags.size),inputs_unchanged=True))
  return all_exact,result


def load_capture(path,plan):
  manifest=json.loads(path.read_text())
  if (not manifest.get('complete') or not manifest.get('paired_prediction_bits_exact') or manifest.get('plan')!=plan
      or manifest.get('sample')!=C.SAMPLE or manifest.get('layers')!=list(C.LAYERS)):
    raise ValueError('Require the completed source-bound one-query original capture')
  if (manifest.get('plan_identity')!=plan['identity_sha256'] or len(manifest['queries'])!=1
      or manifest['queries'][0]['prepared']!=plan['original_query']['prepared']
      or manifest['queries'][0]['prediction']!=plan['original_query']['prediction']
      or manifest.get('raw_payload_bytes')!=C.PAYLOAD):
    raise ValueError('Captured current-query input hashes differ')
  records=manifest['records']
  if [r['layer']for r in records]!=list(C.LAYERS):raise ValueError('Capture layer selection changed')
  arrays=[]
  for record in records:
    if record['shape']!=list(C.SHAPE) or record['actual_prompt_rows']!=2027:raise ValueError('Capture shape/mask changed')
    if set(record['files'])!={'activation_packed','weight_packed','activation_scale','weight_scale','original_bf16'}:
      raise ValueError('Capture operand/output inventory changed')
    data={}
    for key,metadata in record['files'].items():
      p=Path(metadata['path']);C.guard_disk(p)
      if C.sha(p)!=metadata['sha256']:raise ValueError('Original capture bytes changed')
      a=np.load(p,allow_pickle=False,mmap_mode='r')
      if list(a.shape)!=metadata['shape'] or a.dtype.str!=metadata['dtype']:raise ValueError('Capture tensor metadata changed')
      data[key]=a
    shapes={'activation_packed':((2112,9216),np.dtype('uint8')),'weight_packed':((2048,9216),np.dtype('uint8')),
      'activation_scale':((1,),np.dtype('float32')),'weight_scale':((1,),np.dtype('float32')),
      'original_bf16':((2112,2048),np.dtype('uint16'))}
    for key,(shape,dtype)in shapes.items():
      if data[key].shape!=shape or data[key].dtype!=dtype:raise ValueError('Actual capture operand shape/dtype changed: '+key)
    arrays.append((record,data))
  return manifest,arrays


def poison(out):
  from tinygrad import Tensor,dtypes
  out.bitcast(dtypes.uint16).assign(Tensor.full(out.shape,0x7fc1,device=out.device,dtype=dtypes.uint16)).realize()


def graph(kernel,serial,device,x,w,xs,ws,hybrid,expected,expected_format=None):
  from tinygrad import Tensor,TinyJit,dtypes
  m,k=x.shape;n=w.shape[0]
  out=Tensor.empty(m,n,device=device,dtype=dtypes.bfloat16).realize()
  scratch=kernel.allocate(m,n,k,device)if hybrid else tuple(Tensor.empty(1,device=device,dtype=dtypes.uint32).realize()for _ in range(3))
  seed=bytes_tensor(np.array([0.],np.float32),device)
  def execute(a,b,s,t,values,metadata,flags,output,initial):
    for _ in range(40):
      if hybrid:
        values,metadata,flags=kernel.compress(a,(values,metadata,flags))
        output=kernel.gemm(a,b,s,t,(values,metadata,flags),output,initial).realize()
      else:output=kernel.original(a,b,s,t,output).realize()
    return output
  jit=TinyJit(execute)
  args=(x,w,xs,ws,*scratch,out,seed)
  jit(*args)
  phases=('agent_sparse_fc2_compress','agent_sparse_fc2_gemm')if hybrid else('alpamayo_fp8_gemm',)
  with serial.observe(jit,phases):jit(*args)
  proof=serial.inspect(jit,phases)
  poison(out)
  np.testing.assert_array_equal(jit(*args).bitcast(dtypes.uint16).numpy(),expected)
  if expected_format is not None:
    for tensor,wanted in zip(scratch,expected_format,strict=True):
      got=tensor.numpy()
      got=got.view(np.uint8).reshape(wanted.shape)if wanted.dtype==np.uint8 else got.reshape(wanted.shape)
      np.testing.assert_array_equal(got,wanted)
  device_object=__import__('tinygrad').Device[device]
  samples=[]
  device_object.synchronize()
  for _ in range(3):
    start=time.perf_counter_ns();output=jit(*args);device_object.synchronize()
    samples.append((time.perf_counter_ns()-start)/1e6/40)
  return jit,args,output,dict(serial_proof=proof,amortized_ms=samples,median_ms=float(np.median(samples)),
    actual_workspace_bytes=sum(t.numel()*t.dtype.itemsize for t in scratch)if hybrid else 0,
    workspace_bytes=sum(t.numel()*t.dtype.itemsize for t in scratch)if hybrid else 0,
    timed_readback=False,included='Compression/metadata/eligibility + original-byte dense fallback + GEMM + FP32 scale product + BF16 RNE'if hybrid else'Current original packed FP8 4x4 GEMM + scales + BF16 RNE')


def actual(kernel,serial,device,records):
  from tinygrad import Tensor,dtypes
  results=[]
  for record,data in records:
    a=np.array(data['activation_packed'],copy=True);w=np.array(data['weight_packed'],copy=True)
    x=bytes_tensor(a,device);weight=bytes_tensor(w,device)
    xs=bytes_tensor(data['activation_scale'],device);ws=bytes_tensor(data['weight_scale'],device)
    saved=np.asarray(data['original_bf16'])
    logical=C.fmt().unpack(a);ev,em,ef=C.fmt().compress(logical)
    baseline_jit,baseline_args,baseline_out,baseline_timing=graph(kernel,serial,device,x,weight,xs,ws,False,saved)
    original=baseline_out.bitcast(dtypes.uint16).numpy()
    np.testing.assert_array_equal(original,saved)
    candidate_jit,args,out,timing=graph(kernel,serial,device,x,weight,xs,ws,True,saved,(ev,em,ef))
    bits=out.bitcast(dtypes.uint16).numpy()
    np.testing.assert_array_equal(bits,saved)
    # Independent format reference uses EVERY actual byte, including padded rows.
    for tensor,expected in zip(args[4:7],(ev,em,ef),strict=True):
      got=tensor.numpy()
      if expected.dtype==np.uint8:got=got.view(np.uint8).reshape(expected.shape)
      else:got=got.reshape(expected.shape)
      np.testing.assert_array_equal(got,expected)
    mutations=[]
    # A lexicographically first eligible tile is chosen only to test a flag
    # transition. This does not select layers, inputs, scales, or quality policy.
    indices=np.argwhere(ef)
    mr,kt=map(int,indices[0])if len(indices)else(0,0)
    changed=logical.copy()
    if len(indices):changed[mr*16,kt*32:kt*32+4]=[56,64,68,72]
    else:changed[:16,:32]=0
    x.assign(bytes_tensor(C.fmt().pack(changed),device)).realize()
    changed_weight=w.copy()
    col_k=kt*32
    address=(col_k//16)*256+(col_k%16//4)*64+col_k%4
    changed_weight.ravel()[address]=126 if changed_weight.ravel()[address]!=126 else 254
    for label in ('activation-and-eligibility','scales','weights'):
      if label=='scales':xs.assign(bytes_tensor(np.asarray(data['activation_scale'])*np.float32(1.125),device)).realize()
      if label=='weights':weight.assign(bytes_tensor(changed_weight,device)).realize()
      poison(args[7]);mutated=candidate_jit(*args).bitcast(dtypes.uint16).numpy()
      reference=kernel.original(x,weight,xs,ws,baseline_args[7]).realize().bitcast(dtypes.uint16).numpy()
      np.testing.assert_array_equal(mutated,reference)
      actual_flags=args[6].numpy().reshape(ef.shape)
      expected_flag=0 if len(indices)else 1
      assert actual_flags[mr,kt]==expected_flag
      mutations.append(dict(change=label,full_bits_equal=True,current_eligibility_seen=True))
    x.assign(bytes_tensor(a,device)).realize();xs.assign(bytes_tensor(data['activation_scale'],device)).realize()
    weight.assign(bytes_tensor(w,device)).realize()
    poison(args[7]);restored=candidate_jit(*args).bitcast(dtypes.uint16).numpy();np.testing.assert_array_equal(restored,saved)
    np.testing.assert_array_equal(x.numpy(),a);np.testing.assert_array_equal(weight.numpy(),w)
    results.append(dict(layer=record['layer'],shape=list(C.SHAPE),all_original_capture_bits_equal=True,
      physical_tile_eligible_fraction=float(ef.mean()),eligible_tiles=int(ef.sum()),total_tiles=int(ef.size),
      real_complete_tile_eligible_fraction=float(ef[:record['actual_prompt_rows']//16].mean()),
      actual_prompt_rows=record['actual_prompt_rows'],physical_rows_included_in_timing=True,
      original_4x4=baseline_timing,compression_plus_hybrid_4x4=timing,mutations=mutations,
      restored_bits_equal=True,input_and_weight_bytes_unchanged=True))
    del candidate_jit,baseline_jit,args,baseline_args,x,weight
  return results


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
  p.add_argument('--capture',type=Path,required=True);p.add_argument('--parent-gpu',action='store_true')
  a=p.parse_args()
  if not a.parent_gpu:p.error('Only parent GPU serialization may execute this microcheck')
  if a.output.exists():raise FileExistsError('Use a new micro output')
  plan=json.loads(a.plan.read_text());_,runner,root=C.verify_plan(plan)
  import os,sys
  for key,value in plan['baseline_plan']['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  for path in reversed(os.environ['PYTHONPATH'].split(':')):sys.path.insert(0,path)
  guard=runner.local('agent-activation-sparsity-device')
  identity=guard.require_gpu_device()
  from tinygrad import Device
  runner.local('wgp_compiler').install(Device[Device.DEFAULT])
  runner.local('prefill-kernels').fix_dispatch_packet()
  kernel,serial=load('agent-sparse-fc2-kernel'),load('agent-sparse-fc2-serial')
  report=dict(complete=False,arithmetic_exact=False,source_plan=plan['identity_sha256'],gpu_device_identity=identity,
    pure_kernel_or_adoption_claim=False,timings=[],controls=[])
  try:
    exact,controls=arithmetic(kernel,Device.DEFAULT);report['controls']=controls;report['arithmetic_exact']=exact
    if not exact:
      report.update(complete=True,status='NUMERICAL_MISMATCH_STOPPED',timing_skipped=True,
        needed_protocol='Retain raw/BF16 mismatch evidence. Any numerical model proposal needs independent original-decoder comparisons and existing route-quality gates. No exact-kernel adoption.')
      C.write(a.output,report)
      raise ArithmeticError('Sparse accumulation/epilogue differs; full-shape timing and adoption stopped')
    manifest,records=load_capture(a.capture,plan)
    report['capture_sha256']=C.sha(a.capture)
    report['timings']=actual(kernel,serial,Device.DEFAULT,records)
    report.update(complete=True,status='EXACT_CONTROLS_AND_ACTUAL_MATRICES_PASSED',timing_skipped=False,
      quality_claim=False,pipeline_speed_claim=False,readback_outside_timing=True)
  except BaseException as error:
    report['error']=dict(type=type(error).__name__,message=str(error));raise
  finally:C.write(a.output,report)


if __name__=='__main__':main()

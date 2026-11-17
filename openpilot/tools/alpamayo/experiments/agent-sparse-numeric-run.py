#!/usr/bin/env python3
"""Parent-only arithmetic diagnosis on the existing immutable three-layer capture.

No timing or route execution. Sparse results remain numerical evidence, never
an exactness waiver. All aligned dense zero-C results must match the original.
"""
import argparse,importlib.util,json,os,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),HERE/(name+'.py'))
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
C=load('agent-sparse-numeric-common')


def upload(a,device):
  from tinygrad import Tensor
  return Tensor(np.ascontiguousarray(a).copy(),device='CPU').to(device).realize()


def evaluate(kernel,x,w,xs,ws,scratch,seed,mode='hybrid',raw=False,initial=False,out=None):
  from tinygrad import Tensor,dtypes
  if out is None:out=Tensor.empty(x.shape[0],w.shape[0],device=x.device,dtype=dtypes.float32 if raw else dtypes.bfloat16).realize()
  value=kernel.gemm(x,w,xs,ws,scratch,out,seed,mode=mode,initial=initial).realize()
  return value.numpy()if raw else value.bitcast(dtypes.uint16).numpy()


def reference(kernel,x,w,xs,ws,raw=False):
  from tinygrad import Tensor,dtypes
  out=Tensor.empty(x.shape[0],w.shape[0],device=x.device,dtype=dtypes.float32 if raw else dtypes.bfloat16).realize()
  value=kernel.reference(x,w,xs,ws,out).realize()
  return value.numpy()if raw else value.bitcast(dtypes.uint16).numpy()


def check_format(old,scratch,q):
  expected=old.fmt().compress(q)
  for tensor,wanted in zip(scratch,expected,strict=True):
    actual=tensor.numpy();actual=actual.view(np.uint8).reshape(wanted.shape)if wanted.dtype==np.uint8 else actual.reshape(wanted.shape)
    np.testing.assert_array_equal(actual,wanted)
  return expected,C.preservation(q,*expected)


def require_dense(record,dense,original,kind):
  metric=dict(kind=kind,**C.metrics(dense,original));record.append(metric)
  if not metric.get('bits_exact')or not metric['all_finite']:raise ArithmeticError('Aligned dense control differs from original; stop numerical proposal qualification')
  return metric


def controls(report,kernel,device,old):
  from tinygrad import Tensor,dtypes
  for fixture in C.fixtures():
    a,w=fixture['a'],fixture['w'];m,k=a.shape;n=w.shape[0]
    x=upload(old.fmt().pack(a),device);weight=upload(old.fmt().pack(w),device)
    xs,ws=upload(fixture['xs'],device),upload(fixture['ws'],device)
    one=upload(np.array([1.],np.float32),device);seed=upload(fixture['c'],device)
    scratch=kernel.compress(x,kernel.allocate(m,n,k,device));Tensor.realize(*scratch)
    _,proof=check_format(old,scratch,a);dot,absolute,_=C.exact_reference(a,w)
    entry=dict(case=fixture['name'],shape=[m,n,k],format=proof,dense_original_checks=[],cases=[])
    report['controls'].append(entry)
    for initial in (False,True):
      dense=evaluate(kernel,x,weight,one,one,scratch,seed,'dense',True,initial)
      hybrid=evaluate(kernel,x,weight,one,one,scratch,seed,'hybrid',True,initial)
      dense_bf=evaluate(kernel,x,weight,xs,ws,scratch,seed,'dense',False,initial)
      hybrid_bf=evaluate(kernel,x,weight,xs,ws,scratch,seed,'hybrid',False,initial)
      case=dict(nonzero_initial_C=initial,hybrid_vs_aligned_dense_raw=C.metrics(hybrid,dense),
        hybrid_vs_aligned_dense_BF16=C.metrics(hybrid_bf,dense_bf),
        dense_consistency=C.consistency(dense,dot,absolute,k,fixture['c']if initial else None),
        hybrid_consistency=C.consistency(hybrid,dot,absolute,k,fixture['c']if initial else None))
      entry['cases'].append(case)
      if not initial:
        origin=reference(kernel,x,weight,one,one,True);origin_bf=reference(kernel,x,weight,xs,ws)
        require_dense(entry['dense_original_checks'],dense,origin,'raw-FP32-zero-C')
        require_dense(entry['dense_original_checks'],dense_bf,origin_bf,'BF16-zero-C')
        case['hybrid_vs_original_raw']=C.metrics(hybrid,origin);case['hybrid_vs_original_BF16']=C.metrics(hybrid_bf,origin_bf)
        case['original_vs_exact_FP64']=C.metrics(origin,dot);case['aligned_dense_vs_exact_FP64']=C.metrics(dense,dot)
        case['hybrid_vs_exact_FP64']=C.metrics(hybrid,dot)
        previous=evaluate(kernel.original(),x,weight,one,one,scratch,seed,'dense',True,False)
        case['previous_dense_vs_original_raw']=C.metrics(previous,origin)
        if fixture.get('integer_control')or fixture['name']=='integer':
          np.testing.assert_array_equal(hybrid,dot.astype(np.float32))
          ideal=old.bf16(dot.astype(np.float32)*np.float32(fixture['xs'][0]*fixture['ws'][0]))
          np.testing.assert_array_equal(hybrid_bf,ideal);case['independent_signed_integer_exact']=True
      for key in ('dense_consistency','hybrid_consistency'):
        if not case[key]['all_finite']or case[key]['elements_outside_bound']:
          raise ArithmeticError('Nonfinite or independent mathematical consistency bound failed')
    tiny=upload(np.array([2.**-120],np.float32),device)
    original_tiny=reference(kernel,x,weight,tiny,one)
    dense_tiny=evaluate(kernel,x,weight,tiny,one,scratch,seed,'dense')
    hybrid_tiny=evaluate(kernel,x,weight,tiny,one,scratch,seed,'hybrid')
    require_dense(entry['dense_original_checks'],dense_tiny,original_tiny,'BF16-small-scale')
    entry['small_scale_hybrid_vs_original']=C.metrics(hybrid_tiny,original_tiny)
    np.testing.assert_array_equal(x.numpy(),old.fmt().pack(a));np.testing.assert_array_equal(weight.numpy(),old.fmt().pack(w))
    entry['inputs_unchanged']=True


def replay(kernel,device,old,x,w,xs,ws,logical,packed_w):
  from tinygrad import Tensor,TinyJit,dtypes
  m,k=x.shape;n=w.shape[0]
  scratch=kernel.allocate(m,n,k,device);out=Tensor.empty(m,n,device=device,dtype=dtypes.bfloat16).realize()
  seed=upload(np.zeros((m,n),np.float32),device)
  def run(a,b,s,t,v,metadata,flags,output,current_seed):
    compressed=kernel.compress(a,(v,metadata,flags))
    return kernel.gemm(a,b,s,t,compressed,output,current_seed,initial=True).realize()
  jit=TinyJit(run);args=(x,w,xs,ws,*scratch,out,seed)
  jit(*args)
  serial=C.load(C.OLD/'agent-sparse-fc2-serial.py','numeric_scoped_real_HCQ')
  base_prove=serial.prove
  serial.prove=lambda batch,queues,phases,count=1:base_prove(batch,queues,phases,count=1)
  phases=('agent_sparse_fc2_compress','agent_sparse_numeric_gemm')
  with serial.observe(jit,phases):jit(*args)
  proof=serial.inspect(jit,phases)
  assert proof['captured_kernel_calls']==2 and proof['ordered_execution_barriers']==1
  originals=(x.numpy().copy(),w.numpy().copy(),xs.numpy().copy(),ws.numpy().copy())
  flags=old.fmt().compress(logical)[2];eligible=np.argwhere(flags)
  mr,kt=map(int,eligible[0])if len(eligible)else(0,0)
  changed=logical.copy()
  if len(eligible):changed[mr*16,kt*32:kt*32+4]=[56,64,68,72]
  else:changed[:16,:32]=0
  changed_weight=packed_w.copy();offset=(kt*32//16)*256
  changed_weight.ravel()[offset]=126 if changed_weight.ravel()[offset]!=126 else 254
  changes=[];first=None;previous=None
  for epoch,label in enumerate(('unmodified','activation-and-eligibility','weight','scales','running-C','restore')):
    if label=='activation-and-eligibility':x.assign(upload(old.fmt().pack(changed),device)).realize()
    elif label=='weight':w.assign(upload(changed_weight,device)).realize()
    elif label=='scales':xs.assign(upload(originals[2]*np.float32(1.125),device)).realize()
    elif label=='running-C':
      current=np.zeros((m,n),np.float32);current[0,0]=np.float32(2**24);current[-1,-1]=np.float32(-2**24)
      seed.assign(upload(current,device)).realize()
    elif label=='restore':
      for tensor,array in zip((x,w,xs,ws),originals,strict=True):tensor.assign(upload(array,device)).realize()
      seed.assign(upload(np.zeros((m,n),np.float32),device)).realize()
    out.bitcast(dtypes.uint16).assign(Tensor.full(out.shape,0x7fc1,device=device,dtype=dtypes.uint16)).realize()
    actual=jit(*args).bitcast(dtypes.uint16).numpy()
    fresh=kernel.compress(x,kernel.allocate(m,n,k,device));Tensor.realize(*fresh)
    independent=evaluate(kernel,x,w,xs,ws,fresh,seed,'hybrid',initial=True)
    np.testing.assert_array_equal(actual,independent)
    if first is None:first=actual.copy()
    changed_elements=0 if previous is None else int(np.count_nonzero(actual!=previous))
    if label in('scales','running-C')and changed_elements==0:raise AssertionError('Current input mutation has no observable output effect')
    if label=='restore':np.testing.assert_array_equal(actual,first)
    expected_q=logical if label in('unmodified','restore')else changed
    check_format(old,args[4:7],expected_q)
    current_flags=args[6].numpy().reshape(flags.shape)
    if label not in('unmodified','restore'):assert current_flags[mr,kt]==(0 if len(eligible)else 1)
    changes.append(dict(epoch=epoch,change=label,all_bits_equal_fresh_same_candidate=True,current_format_exact=True,poison_replaced=True,
      differing_outputs_from_previous_epoch=changed_elements,output_sha256=C.array_sha(actual)))
    previous=actual.copy()
  for tensor,array in zip((x,w,xs,ws),originals,strict=True):np.testing.assert_array_equal(tensor.numpy(),array)
  return dict(HCQ=proof,replays=changes,input_and_weight_restored=True,not_a_timing_graph=True)


def actual(report,kernel,device,old,records,plan):
  from tinygrad import Tensor,dtypes
  for record,data in records:
    packed=np.array(data['activation_packed'],copy=True);pw=np.array(data['weight_packed'],copy=True)
    a,w=old.fmt().unpack(packed),old.fmt().unpack(pw)
    dot,absolute,identity=C.exact_reference(a,w)
    expected=next(r for r in plan['cpu_reference']if r['layer']==record['layer'])
    if identity['dot_sha256']!=expected['dot_sha256']or identity['absolute_sha256']!=expected['absolute_sha256']:
      raise ValueError('Recomputed independent CPU reference changed')
    x,weight=upload(packed,device),upload(pw,device)
    xs,ws=upload(data['activation_scale'],device),upload(data['weight_scale'],device)
    one=upload(np.array([1.],np.float32),device);seed=upload(np.array([0.],np.float32),device)
    scratch=kernel.compress(x,kernel.allocate(2112,2048,9216,device));Tensor.realize(*scratch)
    _,preservation=check_format(old,scratch,a)
    entry=dict(layer=record['layer'],shape=record['shape'],actual_prompt_rows=record['actual_prompt_rows'],
      source_capture_bytes_unchanged=True,independent_reference=identity,format=preservation,dense_original_checks=[])
    report['actual'].append(entry)
    original_raw=reference(kernel,x,weight,one,one,True);original_bf=reference(kernel,x,weight,xs,ws)
    np.testing.assert_array_equal(original_bf,np.asarray(data['original_bf16']))
    dense_raw=evaluate(kernel,x,weight,one,one,scratch,seed,'dense',True)
    dense_bf=evaluate(kernel,x,weight,xs,ws,scratch,seed,'dense')
    require_dense(entry['dense_original_checks'],dense_raw,original_raw,'actual-raw-FP32-zero-C')
    require_dense(entry['dense_original_checks'],dense_bf,original_bf,'actual-BF16-zero-C')
    hybrid_raw=evaluate(kernel,x,weight,one,one,scratch,seed,'hybrid',True)
    hybrid_bf=evaluate(kernel,x,weight,xs,ws,scratch,seed,'hybrid')
    old_raw=evaluate(kernel.original(),x,weight,one,one,scratch,seed,'hybrid',True)
    old_bf=evaluate(kernel.original(),x,weight,xs,ws,scratch,seed,'hybrid')
    entry.update(hybrid_vs_original_raw=C.metrics(hybrid_raw,original_raw),hybrid_vs_original_BF16=C.metrics(hybrid_bf,original_bf),
      hybrid_vs_previous_hybrid_raw=C.metrics(hybrid_raw,old_raw),hybrid_vs_previous_hybrid_BF16=C.metrics(hybrid_bf,old_bf),
      original_vs_exact_FP64=C.metrics(original_raw,dot),hybrid_vs_exact_FP64=C.metrics(hybrid_raw,dot),
      original_consistency=C.consistency(original_raw,dot,absolute,9216),hybrid_consistency=C.consistency(hybrid_raw,dot,absolute,9216),
      real_prompt_hybrid_vs_original_BF16=C.metrics(hybrid_bf[:record['actual_prompt_rows']],original_bf[:record['actual_prompt_rows']]))
    scale=np.float32(data['activation_scale'][0]*data['weight_scale'][0])
    entry['original_BF16_vs_exact_dot_RNE']=C.metrics(original_bf,old.bf16(dot.astype(np.float32)*scale))
    entry['hybrid_BF16_vs_own_raw_RNE']=C.metrics(hybrid_bf,old.bf16(hybrid_raw*scale))
    # This tests scale association/conversion independently of sum order.
    if not entry['hybrid_BF16_vs_own_raw_RNE']['bits_exact']:raise ArithmeticError('Hybrid BF16 epilogue differs from FP32 scale product/RNE')
    if not entry['hybrid_vs_original_raw']['all_finite']or not entry['hybrid_vs_original_BF16']['all_finite']:
      raise ArithmeticError('Nonfinite full-shape arithmetic')
    for key in('original_consistency','hybrid_consistency'):
      if entry[key]['elements_outside_bound']:raise ArithmeticError('Full-shape independent consistency bound exceeded')
    entry['replay']=replay(kernel,device,old,x,weight,xs,ws,a,pw)
    np.testing.assert_array_equal(x.numpy(),packed);np.testing.assert_array_equal(weight.numpy(),pw)
    print('Full-shape numerical arithmetic characterized: layer',record['layer'],flush=True)


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path)
  p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--parent-gpu',action='store_true');a=p.parse_args()
  if a.output.exists():raise FileExistsError('Use a new numeric diagnostic output')
  if a.prepare:C.write(a.output,C.prepare());print('CPU-only numerical source/capture plan prepared');return
  if a.plan is None:p.error('--plan required')
  plan=json.loads(a.plan.read_text());old,manifest,records,runner,root=C.verify(plan)
  if a.preflight:print('CPU-only numeric source/capture provenance verified');return
  if not a.parent_gpu:p.error('Only parent GPU serialization may execute arithmetic diagnostics')
  if os.environ.get('OPENBLAS_NUM_THREADS')!='4' or os.environ.get('OMP_NUM_THREADS')!='4':raise ValueError('Use four CPU threads for matched exact references')
  for key,value in plan['old_plan']['baseline_plan']['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  os.environ['PYTHONPATH']=C.PYTHONPATH
  for path in reversed(C.PYTHONPATH.split(':')):sys.path.insert(0,path)
  identity=runner.local('agent-activation-sparsity-device').require_gpu_device()
  from tinygrad import Device
  runner.local('wgp_compiler').install(Device[Device.DEFAULT]);runner.local('prefill-kernels').fix_dispatch_packet()
  kernel=load('agent-sparse-numeric-kernel')
  report=dict(complete=False,classification='NUMERICAL_DIAGNOSTIC_ONLY',plan_identity=plan['identity_sha256'],
    source_capture_sha256=C.sha(C.CAPTURE),device=identity,controls=[],actual=[],timings=[],
    exact_adoption_claim=False,quality_claim=False,route_executed=False,old_failed_gate_preserved=True)
  try:
    controls(report,kernel,Device.DEFAULT,old)
    actual(report,kernel,Device.DEFAULT,old,records,plan)
    report.update(complete=True,status='DENSE_ALIGNMENT_QUALIFIED_SPARSE_NUMERICAL_CHARACTERIZED',
      numerical_differences_require_independent_model_and_route_protocol=True)
  except BaseException as error:report['error']=dict(type=type(error).__name__,message=str(error));raise
  finally:C.write(a.output,report)


if __name__=='__main__':main()

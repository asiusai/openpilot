#!/usr/bin/env python3
"""Parent-only, numerical-only, matched complete-boundary sparse fc2 timing."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name):
  s=importlib.util.spec_from_file_location(name.replace('-','_'),HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=load('agent-sparse-timing-common')


def physical(array,device,dtype=None):
  from tinygrad import Tensor,dtypes
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp,Ops
  array=np.ascontiguousarray(array)
  dt=dtype or {np.dtype('uint8'):dtypes.uint8,np.dtype('uint16'):dtypes.uint16,np.dtype('uint32'):dtypes.uint32,np.dtype('float32'):dtypes.float32}[array.dtype]
  buffer=Buffer(device,array.nbytes,initial_value=array.tobytes())
  tensor=Tensor(UOp.from_buffer(buffer,dt)).reshape(array.shape)
  assert tensor.uop.base.op is Ops.BUFFER and not tensor.uop.is_virtual and tensor.uop.is_realized
  return tensor


def write(tensor,array):
  from tinygrad import Device
  from tinygrad.device import Buffer
  array=np.ascontiguousarray(array);buffer=tensor.uop.base.buffer
  assert buffer.is_allocated()and array.nbytes==buffer.nbytes
  source=Buffer('NPY',array.nbytes,opaque=array.copy())
  buffer.copy_from(source)
  Device[tensor.device].synchronize() # outside every timing interval; keep source alive


def output_bits(tensor):
  from tinygrad import dtypes
  return tensor.bitcast(dtypes.uint16).numpy()


def typed_view(tensor,dtype):
  """Reinterpret physical byte storage without a BITCAST copy kernel."""
  from tinygrad import Tensor
  from tinygrad.uop.ops import UOp
  assert tensor.dtype.itemsize==dtype.itemsize
  return Tensor(UOp.from_buffer(tensor.uop.base.buffer,dtype)).reshape(tensor.shape)


def scratch(m,k,device):
  tiles=m//16*(k//32)
  return(physical(np.zeros((tiles,32,2),np.uint32),device),physical(np.zeros((tiles,32),np.uint32),device),physical(np.zeros(tiles,np.uint32),device))


def poison(graph):
  write(graph.args[graph.output_slot],np.full(graph.expected.shape,0x7fc1,np.uint16))
  if graph.hybrid:
    for tensor in graph.args[4:7]:write(tensor,np.full(tensor.shape,0xffffffff,np.uint32))


def binaries(N):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp,Ops
  cpu=json.loads((C.NUMERIC/'agent-sparse-numeric-cpu.json').read_text())
  def find(report,name):return next(r['binary_sha256']for r in report['compiler']if r['name']==name)
  # The actual frozen compressor API always passes N=64. Its old compiler
  # inventory used N=2048. Bind the entire PROGRAM to the previous GPU graph,
  # including source, exact binary, IO metadata and launch geometry.
  kernel=N.load(C.NUMERIC/'agent-sparse-numeric-kernel.py','timing_qualified_compressor')
  tiles=2112//16*(9216//32)
  shapes=[((tiles,32,2),dtypes.uint32),((tiles,32),dtypes.uint32),((tiles,),dtypes.uint32),((2112,9216),dtypes.uint8)]
  params=[UOp.placeholder(shape,dtype,i)for i,(shape,dtype)in enumerate(shapes)]
  program=kernel.original().program(*params,m=2112,n=64,k=9216,mode='compress')
  evidence=json.loads(C.EVIDENCE.read_text())
  assert all(e['replay']['HCQ']['kernel_arguments'][0]['program_key']==program.key.hex()for e in evidence['actual']), 'Actual compressor differs from characterized GPU PROGRAM'
  binary=next(s.arg for s in program.src if s.op is Ops.BINARY)
  import hashlib
  return {'agent_sparse_numeric_gemm':find(cpu,'2112-2048-9216-hybrid-raw0-C0'),
    'alpamayo_fp8_gemm':find(cpu,'2112-2048-9216-original-bf16'),
    'agent_sparse_fc2_compress':hashlib.sha256(binary).hexdigest()}


def make_graph(kernel,serial,shared,expected,hybrid,binary_gate=None,original_kernel=None,count=40):
  from tinygrad import Tensor,TinyJit,dtypes,Device
  x,w,xs,ws=shared;m,k=x.shape;n=w.shape[0]
  out=physical(np.full((m,n),0x7fc1,np.uint16),x.device,dtypes.bfloat16)
  if hybrid:
    workspace=scratch(m,k,x.device);seed=physical(np.array([0.],np.float32),x.device)
    args=(*shared,*workspace,out,seed);output_slot=7
    def execute(a,b,s,t,v,metadata,flags,output,initial):
      for _ in range(count):
        v,metadata,flags=kernel.compress(a,(v,metadata,flags))
        output=kernel.gemm(a,b,s,t,(v,metadata,flags),output,initial).realize()
      return output
  else:
    args=(*shared,out);output_slot=4
    def execute(a,b,s,t,output):
      for _ in range(count):output=(original_kernel or kernel.reference)(a,b,s,t,output).realize()
      return output
  jit=TinyJit(execute);phases=('agent_sparse_fc2_compress','agent_sparse_numeric_gemm')if hybrid else('alpamayo_fp8_gemm',)
  rules=serial.layouts(hybrid);bound=serial.bindings(args)
  graph=SimpleNamespace(jit=jit,args=args,expected=expected,hybrid=hybrid,output_slot=output_slot,
    phases=phases,rules=rules,buffer_ids=[id(b.buffer)for b in bound])
  Device[x.device].synchronize();jit(*args);Device[x.device].synchronize()
  with serial.observe(jit,args,phases,rules,count,binary_gate):jit(*args)
  graph.proof=serial.inspect(jit,args,phases,rules,count,binary_gate)
  assert all(not alias for alias in jit.captured._linears),'Implicit aliased-input staging/copies are forbidden'
  return graph


def check_binding(graph,serial):
  assert [id(b.buffer)for b in serial.bindings(graph.args)]==graph.buffer_ids,'Timing graph buffers changed identity'
  assert all(not alias for alias in graph.jit.captured._linears),'Unexpected replay staging'


def check_output(graph,expected=None):
  result=output_bits(graph.args[graph.output_slot]);expected=graph.expected if expected is None else expected
  np.testing.assert_array_equal(result,expected)
  assert np.isfinite((result.astype(np.uint32)<<16).view(np.float32)).all()
  return result


def check_format(graph,expected_format):
  if not graph.hybrid:return
  for tensor,expected in zip(graph.args[4:7],expected_format,strict=True):
    got=tensor.numpy();got=got.view(np.uint8).reshape(expected.shape)if expected.dtype==np.uint8 else got.reshape(expected.shape)
    np.testing.assert_array_equal(got,expected)


def fresh(kernel,shared,hybrid):
  from tinygrad import Tensor,dtypes
  x,w,xs,ws=shared;m,k=x.shape;n=w.shape[0]
  out=physical(np.full((m,n),0x7fc1,np.uint16),x.device,dtypes.bfloat16)
  if hybrid:
    buffers=kernel.compress(x,scratch(m,k,x.device));Tensor.realize(*buffers)
    seed=physical(np.array([0.],np.float32),x.device)
    result=kernel.gemm(x,w,xs,ws,buffers,out,seed).realize()
  else:result=kernel.reference(x,w,xs,ws,out).realize()
  return output_bits(result)


def mutations(kernel,serial,old,graphs,data,logical,entry):
  shared=graphs['hybrid'].args[:4];x,w,xs,ws=shared
  originals=(np.array(data['activation_packed']),np.array(data['weight_packed']),np.array(data['activation_scale']),np.array(data['weight_scale']))
  eligible=np.argwhere(old.fmt().compress(logical)[2]);assert len(eligible)
  row,kt=map(int,eligible[0]);changed=logical.copy();changed[row*16,kt*32:kt*32+4]=[56,64,68,72]
  changed_weight=originals[1].copy();offset=(kt*32//16)*256
  changed_weight.ravel()[offset]=126 if changed_weight.ravel()[offset]!=126 else 254
  historical={s['change']:s['output_sha256']for s in entry['replay']['replays']}
  records=[];last={}
  for epoch,label in enumerate(('unmodified','activation-and-eligibility','weight','scales','restore')):
    if label=='activation-and-eligibility':write(x,old.fmt().pack(changed))
    elif label=='weight':write(w,changed_weight)
    elif label=='scales':write(xs,originals[2]*np.float32(1.125))
    elif label=='restore':
      for tensor,array in zip(shared,originals,strict=True):write(tensor,array)
    current_q=logical if label in('unmodified','restore')else changed
    expected_format=old.fmt().compress(current_q)
    current=[]
    for name,graph in graphs.items():
      expected=graph.expected if label in('unmodified','restore')else fresh(kernel,shared,graph.hybrid)
      if graph.hybrid:assert C.array_sha(expected)==historical[label],'Current candidate differs from characterized mutation bits'
      poison(graph);check_binding(graph,serial);graph.jit(*graph.args)
      result=check_output(graph,expected);check_format(graph,expected_format)
      delta=0 if name not in last else int(np.count_nonzero(result!=last[name]))
      if label in('activation-and-eligibility','weight','scales'):assert delta>0,'Mutation lacks observable output effect'
      last[name]=result.copy();current.append(dict(candidate=name,all_elements_exact=True,finite=True,
        poison_replaced=True,fresh_current_input_checked=True,format_exact=graph.hybrid,differing_from_previous=delta))
    records.append(dict(epoch=epoch,change=label,candidates=current))
  for tensor,array in zip(shared,originals,strict=True):np.testing.assert_array_equal(tensor.numpy(),array)
  return records


def measure(jit,args,synchronize,clock=time.perf_counter_ns):
  start=clock();jit(*args);synchronize();elapsed=clock()-start
  return dict(batch_ms=elapsed/1e6,amortized_ms=elapsed/1e6/C.CALLS,
    invocations=C.CALLS,completion_fences_inside_interval=1,upload_or_readback_inside_interval=False)


def eligibility(q,flags,valid):
  groups=q.reshape(q.shape[0],q.shape[1]//4,4)
  okay=(((groups&127)!=0).sum(-1)<=2)&~((groups&127)==127).any(-1)
  complete=valid//16;partial=(valid+15)//16
  actual=okay.copy();actual[valid:]=True
  real=actual.reshape(q.shape[0]//16,16,q.shape[1]//32,8).all(axis=(1,3))[:partial]
  return dict(physical_rows=q.shape[0],real_rows=valid,padded_rows=q.shape[0]-valid,
    physical_eligible_tiles=int(flags.sum()),physical_tiles=int(flags.size),physical_fraction=float(flags.mean()),
    complete_real_row_eligible_tiles=int(flags[:complete].sum()),complete_real_row_tiles=int(flags[:complete].size),
    complete_real_row_fraction=float(flags[:complete].mean()),real_rows_only_eligible_tiles=int(real.sum()),
    real_rows_only_tiles=int(real.size),real_rows_only_fraction=float(real.mean()),
    partial_real_tile_rows=valid%16,engine_uses_all_16_physical_rows=True)


def actual(plan,old,records,kernel,serial,device,expected_binaries):
  from tinygrad import Device,Tensor,dtypes
  result=[]
  for record,data,expected,entry in records:
    shared=tuple(physical(data[key],device)for key in('activation_packed','weight_packed','activation_scale','weight_scale'))
    q=old.fmt().unpack(data['activation_packed']);expected_format=old.fmt().compress(q)
    original_shared=(typed_view(shared[0],dtypes.fp8e4m3),typed_view(shared[1],dtypes.fp8e4m3),*shared[2:])
    def original(a,b,s,t,out):return Tensor.custom_kernel(out,a,b,s,t,fxn=kernel.original().original_program)[0]
    graphs={name:make_graph(kernel,serial,shared if hybrid else original_shared,expected if hybrid else np.asarray(data['original_bf16']),hybrid,expected_binaries,original_kernel=original)
      for name,hybrid in(('original',False),('hybrid',True))}
    for graph in graphs.values():check_output(graph);check_format(graph,expected_format)
    checked=mutations(kernel,serial,old,graphs,data,q,entry)
    rounds=[]
    for index,order in enumerate(plan['orders']):
      observations={}
      for name in order:
        graph=graphs[name];poison(graph);check_binding(graph,serial);Device[device].synchronize()
        timing=measure(graph.jit,graph.args,Device[device].synchronize)
        check_output(graph);check_format(graph,expected_format);check_binding(graph,serial)
        observations[name]=dict(**timing,full_element_postcheck=True,finite=True,workspace_postcheck=graph.hybrid)
      rounds.append(dict(round=index,order=order,measurements=observations))
    for tensor,key in zip(shared,('activation_packed','weight_packed','activation_scale','weight_scale'),strict=True):
      np.testing.assert_array_equal(tensor.numpy(),data[key])
    medians={name:float(np.median([r['measurements'][name]['amortized_ms']for r in rounds]))for name in graphs}
    sizes={key:int(data[key].nbytes)for key in data};workspace=[t.numel()*t.dtype.itemsize for t in graphs['hybrid'].args[4:7]]
    assert workspace==[9732096,4866048,152064]
    result.append(dict(layer=record['layer'],shape=list(C.SHAPE),classification='NUMERICAL_DIAGNOSTIC_ONLY',
      eligibility=eligibility(q,expected_format[2],record['actual_prompt_rows']),tensor_bytes=sizes,
      workspace=dict(values=workspace[0],metadata=workspace[1],eligibility=workspace[2],total=sum(workspace)),
      compression_minimum_bytes_per_invocation=sizes['activation_packed']+sum(workspace),
      original_4x4_median_ms=medians['original'],compression_plus_hybrid_median_ms=medians['hybrid'],
      measured_speed_ratio_original_over_hybrid=medians['original']/medians['hybrid'],rounds=rounds,
      mutations=checked,serial_proofs={name:g.proof for name,g in graphs.items()},
      all_timed_outputs_equal_characterized_bits=True,input_and_weight_bytes_restored=True))
    print('Matched complete-boundary timing finished: layer',record['layer'],flush=True)
  return result


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path)
  p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--parent-gpu',action='store_true');a=p.parse_args()
  if a.output.exists():raise FileExistsError('Use a new sparse timing output')
  if a.prepare:C.write(a.output,C.prepare());print('CPU-only sparse timing plan prepared');return
  if a.plan is None:p.error('--plan required')
  plan=json.loads(a.plan.read_text());N,old,_,_,records,runner,root,evidence=C.verify(plan)
  if a.preflight:print('CPU-only sparse timing source/output/capture provenance verified');return
  if not a.parent_gpu:p.error('Only parent GPU serialization may run timing')
  for key,value in plan['numeric_plan']['old_plan']['baseline_plan']['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  os.environ['PROFILE']='0';os.environ['PYTHONPATH']=N.PYTHONPATH
  for path in reversed(N.PYTHONPATH.split(':')):sys.path.insert(0,path)
  identity=runner.local('agent-activation-sparsity-device').require_gpu_device()
  from tinygrad import Device
  runner.local('wgp_compiler').install(Device[Device.DEFAULT]);runner.local('prefill-kernels').fix_dispatch_packet()
  kernel=N.load(C.NUMERIC/'agent-sparse-numeric-kernel.py','timing_unchanged_qualified_kernel')
  serial=load('agent-sparse-timing-serial')
  report=dict(complete=False,classification='NUMERICAL_DIAGNOSTIC_ONLY',source_plan=plan['identity_sha256'],device=identity,
    boundary=plan['boundary'],numeric_evidence_sha256=plan['numeric_evidence_sha256'],layers=[],
    bitexact_to_original_claim=False,route_qualified=False,model_integration=False,pipeline_speed_claim=False)
  try:
    report['layers']=actual(plan,old,records,kernel,serial,Device.DEFAULT,binaries(N))
    report.update(complete=True,status='CHARACTERIZED_NUMERICAL_OUTPUTS_AND_MATCHED_BOUNDARY_TIMING_COMPLETE')
  except BaseException as error:report['error']=dict(type=type(error).__name__,message=str(error));raise
  finally:C.write(a.output,report)


if __name__=='__main__':main()

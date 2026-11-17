"""Matched warm/rotating-weight FC2 trial. CPU-only preparation is the default.

The GEMM functions, original report, and old source files are immutable inputs.
All GPU execution belongs to a later parent-reviewed serialized command.
"""
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import os
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
OLD=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-lds-gemm-fc2-final')
OLD_HASHES={'report.json':'6a4ecc8e0989de4736784b1c75fbdf76daf561d08c6bf912a01aec971813b270',
  'run.log':'09eae8f71462ad256b64b9fcd617e030c18af2e47673541542aab491c61a10e9',
  'invocation.json':'5c664878e4511819698145d202aad7c38069c0044fb2a21203475f0a189c0ec4',
  'wall.json':'6569ddba842ca560ebfa14a84bc2f0b0638f17be62a9616a57f99ad9942252d9'}
SHAPE=(2048,2048,9216)
LABELS=('current','amd4')
ORDERS={'warm':(0,)*40,'rotating':tuple(i%4 for i in range(40))}
KEYS=('a','w','sx','sw')


def local(name):
  spec=importlib.util.spec_from_file_location('agent_lds_cold_'+name.replace('-','_'),HERE/(name+'.py'))
  module=importlib.util.module_from_spec(spec)
  sys.modules[spec.name]=module
  spec.loader.exec_module(module)
  return module


H=local('agent-lds-gemm-check')
E=H.E


def evidence():
  for name,digest in OLD_HASHES.items(): assert E.sha((OLD/name).read_bytes())==digest, f'Changed prior evidence: {name}'
  old=json.loads((OLD/'report.json').read_text())
  assert old['gpu_executed'] and not old['complete'] and old['failure']=='KeyboardInterrupt: '
  assert H.provenance()==old['provenance'], 'Numeric sources or their dependencies changed'
  return old


def provenance():
  old=evidence()
  paths=[Path(__file__),HERE/'agent-lds-cold-check.py',Path('/tmp/opencode/run-alpamayo-speed-trial.py')]
  files={str(p):E.sha(p.read_bytes()) for p in paths}
  files.update(old['provenance']['files'])
  files.update({str(OLD/name):digest for name,digest in OLD_HASHES.items()})
  # The exact full-output validator and CPU execution checks use these helpers.
  import tinygrad
  root=Path(tinygrad.__file__).parent
  for name in ('tensor.py','uop/ops.py','device.py','runtime/ops_cpu.py'):
    p=root/name; files[str(p)]=E.sha(p.read_bytes())
  return dict(files=files,source_sha256=E.sha(json.dumps(files,sort_keys=True).encode()),
    old_source_sha256=old['provenance']['source_sha256'],old_report_sha256=OLD_HASHES['report.json'])


def numeric_functions(render,row_scale=False):
  all_functions=H.functions(render,row_scale)
  return {label:all_functions[label] for label in LABELS}


def audit_evidence(render):
  """Recompile each old calling convention, then bind every reused control."""
  old=evidence()
  results=[]
  cache={}
  for row in old['cases']:
    shape,bias,row_scale=tuple(row['shape_mnk']),row['bias'],row['row_scale']
    assert (shape,bias,row_scale) in (((193,208,256),False,False),((193,208,256),True,True),(SHAPE,False,False))
    fns=numeric_functions(render,row_scale)
    for label in LABELS:
      key=shape,bias,row_scale,label
      if key not in cache: cache[key]=H.compile_one(shape,bias,row_scale,label,fns[label],render)
      new,prior=cache[key],row['variants'][label]
      assert new['status']=='compiled'
      for field in ('binary_sha256','source_sha256','global_size','local_size','resources','offline_serial_queue_proof'):
        assert json.loads(json.dumps(new[field]))==prior[field], (row['case'],label,field)
      expected_args=(0,[1,2,3,4,5] if bias else [1,2,3,4])
      proof=prior['offline_serial_queue_proof']
      assert proof['output_argument_indices']==[expected_args[0]] and proof['input_argument_indices']==expected_args[1]
      if row['case'].startswith('control-'):
        assert row['kind'] in H.CONTROLS and prior['status']=='passed-control'
        assert [(c['stage'],c['state']) for c in prior['checks']]==[('single',0)]+[('control_mutation',i) for i in range(1,7)]
        assert all(c['finite_exact_bf16'] for c in prior['checks'])
        other=row['variants']['current']['checks']
        assert [c['sha256'] for c in prior['checks']]==[c['sha256'] for c in other]
      results.append(dict(case=row['case'],label=label,shape_mnk=shape,bias=bias,row_scale=row_scale,
        binary_sha256=new['binary_sha256'],source_sha256=new['source_sha256'],argument_convention_exact=True,
        saved_controls_reused=row['case'].startswith('control-')))
  actual=next(c for c in old['cases'] if c['case']=='fc2')
  for label in LABELS:
    passed={(c['stage'],c['state']) for c in actual['variants'][label]['checks'] if c['finite_exact_bf16']}
    assert all((stage,state) in passed for state in range(1,5) for stage in ('post_timing_single_mutation','post_timing_batch_mutation'))
  return results,{label:actual['variants'][label]['binary_sha256'] for label in LABELS}


def compare_kernel(flags,reference,candidate):
  """Integer comparison of every BF16 bit, including both finite checks."""
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp,Ops,KernelInfo
  size=reference.numel()
  assert candidate.numel()==size and flags.numel()==(size+1023)//1024
  group,k=UOp.range(flags.numel(),0),UOp.range(1024,1)
  index=group*1024+k
  a=reference.flatten().index(index.valid(index<size)).bitcast(dtypes.uint16)
  b=candidate.flatten().index(index.valid(index<size)).bitcast(dtypes.uint16)
  bad=((a!=b)|(a&0x7f80).eq(0x7f80)|(b&0x7f80).eq(0x7f80))&(index<size)
  value=bad.cast(dtypes.int32).reduce(k,arg=Ops.MAX)
  return flags.flatten().index(group).store(value.cast(dtypes.uint16)).end(group).sink(arg=KernelInfo(name='agent_lds_cold_all_bits'))


def fill_kernel(out,control):
  """Full GPU/CPU fill; control[0] is an exact uint16 bit pattern."""
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp,KernelInfo
  r=UOp.range(out.numel(),0)
  value=control.flatten().index(0).cast(dtypes.uint16).bitcast(dtypes.bfloat16)
  return out.flatten().index(r).store(value).end(r).sink(arg=KernelInfo(name='agent_lds_cold_poison'))


def poke_kernel(out,control):
  from tinygrad import dtypes
  from tinygrad.uop.ops import KernelInfo
  index=control.flatten().index(1).load()
  value=control.flatten().index(0).cast(dtypes.uint16).bitcast(dtypes.bfloat16)
  return out.flatten().index(index).store(value).sink(arg=KernelInfo(name='agent_lds_cold_poke'))


class IO:
  """Track operand/verification transfers. Never called inside measured batches."""
  def __init__(self,device):
    self.device=device
    self.upload_bytes=0
    self.download_bytes=0
  def upload(self,value,dtype):
    from tinygrad import Tensor
    from tinygrad.device import Buffer,BufferSpec
    from tinygrad.uop.ops import UOp
    buffer=Buffer(self.device.device,value.nbytes,options=BufferSpec(cpu_access=True)).ensure_allocated()
    buffer.host.view(size=value.nbytes,fmt='B')[:]=value.tobytes()
    self.upload_bytes+=value.nbytes
    return Tensor(UOp.from_buffer(buffer,dtype)).reshape(value.shape)
  def write(self,tensor,value):
    self.device.synchronize()
    assert value.nbytes==tensor.nbytes()
    tensor.uop.buffer.host.view(size=value.nbytes,fmt='B')[:]=value.tobytes()
    self.upload_bytes+=value.nbytes
  def snapshot(self): return dict(upload_bytes=self.upload_bytes,download_bytes=self.download_bytes)


class Verifier:
  def __init__(self,io):
    from tinygrad import Tensor,TinyJit,dtypes
    self.io,self.device=io,io.device
    self.poison_word=io.upload(np.array([0x7fc1,0],np.uint32),dtypes.uint32)
    self.control=io.upload(np.array([0x3f80,0],np.uint32),dtypes.uint32)
    self.records=[]
    def compare(a,b):
      flags=Tensor.full(((a.numel()+1023)//1024,),1,device=a.device,dtype=dtypes.uint16).contiguous().realize()
      return Tensor.custom_kernel(flags,a,b,fxn=compare_kernel)[0].realize()
    self.compare=TinyJit(compare)
    self.finite=TinyJit(lambda a: compare(a,a))
  def fill(self,out,word=None):
    from tinygrad import Tensor
    control=self.poison_word if word is None else self.control
    if word is not None: self.io.write(control,np.array([word,0],np.uint32))
    Tensor.custom_kernel(out,control,fxn=fill_kernel)[0].realize()
    self.device.synchronize()
  def poke(self,out,index,word):
    from tinygrad import Tensor
    self.io.write(self.control,np.array([word,index],np.uint32))
    Tensor.custom_kernel(out,self.control,fxn=poke_kernel)[0].realize()
    self.device.synchronize()
  def check(self,reference,candidate,stage,expect_equal=True):
    self.device.synchronize()
    # The existing BAR reader reads all 4096 flags: 8 KiB, not an output hash.
    flags=self.finite(reference) if reference is candidate else self.compare(reference,candidate)
    values=E.read_output(flags,self.device)
    self.io.download_bytes+=flags.nbytes()
    equal=bool((values==0).all())
    self.records.append(dict(stage=stage,elements=reference.numel(),all_bits_and_finite=True,
      equal=equal,expected_equal=expect_equal,bad_flag_count=int(np.count_nonzero(values)),
      bad_blocks_first_16=np.flatnonzero(values)[:16].tolist(),flags_initialized_to_failure=True,download_bytes=flags.nbytes()))
    assert equal==expect_equal, f'Full-output validation failed: {stage}'
  def run(self,jit,inputs):
    self.fill(jit.output)
    result=jit(*inputs)[0]
    self.device.synchronize()
    return result


def queue_proof(batch,queues,order,binary_sha=None):
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  from tinygrad.uop.ops import Ops
  assert len(batch)==len(order)==40 and len(queues)==1
  assert len({(d,q) for _,d,q in batch})==1
  spans,mappings,binaries=[],[],[]
  calls=[call for call,_,_ in batch]
  for call in calls:
    assert call.body.op is Ops.PROGRAM
    outs,ins=get_call_outs_ins(call)
    assert list(outs)==[0] and list(ins)==[1,2,3,4]
    args=[unwrap_lane(call.src[1+g]) for g in call.body.arg.globals]
    assert all(base.op is Ops.PARAM and lane is None for base,lane,_ in args)
    mappings.append([base.arg.slot for base,_,_ in args])
    value=call.src[1]
    base,lane,offset=unwrap_lane(value)
    spans.append((base,offset,value.max_numel()*value.dtype.itemsize))
    binaries.append(E.sha(next(u.arg for u in call.body.src if u.op is Ops.BINARY)))
  assert len(set(spans))==1, 'All calls must retain identical output storage'
  assert mappings==[[7,0,3+index,1,2] for index in order], 'Weight rotation lost, reordered, or aliased in captured PARAMs'
  assert len(set(binaries))==1 and (binary_sha is None or binaries[0]==binary_sha), 'Numeric binary changed in captured graph'
  commands=[]
  for cmd in next(iter(queues.values())):
    if cmd.op is Ops.CALL and cmd.body.op is Ops.PROGRAM: commands.append(cmd)
    elif cmd.op is Ops.CALL and cmd.body.op is Ops.CUSTOM_FUNCTION and cmd.body.arg.name=='exec_barrier': commands.append('barrier')
  assert commands==[item for i,call in enumerate(calls) for item in ((call,) if i==0 else ('barrier',call))], 'Missing ordered WAW barrier'
  return dict(calls=40,ordered_execution_barriers=39,output_input_slot=7,output_span_bytes=spans[0][2],
    weight_slots=[mapping[2] for mapping in mappings],argument_slots=mappings,binary_sha256=binaries[0],
    actual_queue_builder_observed=True,all_calls_retained=True)


def offline_queue(calls,order,binary_sha=None,device='AMD'):
  from tinygrad.runtime.support.hcq2 import _build_queues
  batch=[(call,device,'COMPUTE:0') for call in calls]
  ctx=SimpleNamespace(batch=batch,calls={(device,'COMPUTE:0'):list(range(40))},peers={(device,'COMPUTE:0'):set()},
    queues={device:['COMPUTE:0']},stamps=lambda *_:())
  return queue_proof(batch,_build_queues(ctx),order,binary_sha)


@contextmanager
def observe(proofs,order,binary_sha):
  from tinygrad.runtime.support import hcq2
  original=hcq2._build_queues
  def build(ctx):
    queues=original(ctx)
    if len(ctx.batch)==40: proofs.append(queue_proof(ctx.batch,queues,order,binary_sha))
    return queues
  hcq2._build_queues=build
  try: yield
  finally: hcq2._build_queues=original


@contextmanager
def observe_numeric_binary(records,binary_sha):
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops
  original=hcq2._build_queues
  def build(ctx):
    for call,_,_ in ctx.batch:
      assert call.body.op is Ops.PROGRAM
      digest=E.sha(next(u.arg for u in call.body.src if u.op is Ops.BINARY))
      assert digest==binary_sha, 'Mutation capture changed the numeric binary'
      records.append(digest)
    return original(ctx)
  hcq2._build_queues=build
  try: yield
  finally: hcq2._build_queues=original


class RotatingJit:
  def __init__(self,fn,order,binary_sha,shape=SHAPE):
    from tinygrad import Tensor,TinyJit
    self.output,self.proofs,self.order,self.binary_sha=None,[],tuple(order),binary_sha
    self.shape=shape
    def execute(a,sx,sw,w0,w1,w2,w3,out):
      weights=(w0,w1,w2,w3)
      for index in self.order: out=Tensor.custom_kernel(out,a,weights[index],sx,sw,fxn=fn)[0].realize()
      return (out,)
    self.inner=TinyJit(execute)
  @property
  def captured(self): return self.inner.captured
  def __call__(self,*inputs):
    from tinygrad import Tensor,dtypes
    assert len(inputs)==7
    weights=inputs[3:]
    assert len({id(w.uop.buffer.base) for w in weights})==4, 'Weight buffers alias'
    if self.output is None: self.output=Tensor.empty(*self.shape[:2],device=inputs[0].device,dtype=dtypes.bfloat16).realize()
    if self.inner.cnt==1:
      with observe(self.proofs,self.order,self.binary_sha): result=self.inner(*inputs,self.output)
    else: result=self.inner(*inputs,self.output)
    assert result[0].uop.buffer.base is self.output.uop.buffer.base and result[0].uop.buffer.offset==self.output.uop.buffer.offset
    if self.inner.cnt>=2:
      assert not any(t.uop.base in self.captured._jit_bufs for t in (*inputs,self.output)), 'JIT operand staging would alter the measured workload'
    return result
  def audit_capture(self):
    assert len(self.proofs)==1
    proof=self.proofs[0]
    records=[k for call in self.captured._linear.src for k in call.without_after.arg.aux.kernels]
    assert len(records)==40
    assert [list(k[5]) for k in records]==proof['argument_slots']
    assert all(list(k[6][0])==[0] and list(k[6][1])==[1,2,3,4] for k in records)
    return proof


def update_changed(io,tensors,data,known):
  changed=[]
  for tensor,key in zip(tensors,KEYS):
    digest=E.sha(data[key].tobytes())
    if digest!=known[key]: io.write(tensor,data[key]); known[key]=digest; changed.append(key)
  return changed


def weight_data(base):
  """Four resident, distinct synthetic E4M3 matrices. W0 matches the old warm fixture."""
  result=[base['w']]
  for index in range(1,4):
    rng=np.random.default_rng(907300+index)
    logical=rng.integers(0,127,(SHAPE[1],SHAPE[2]),dtype=np.uint8)|(rng.integers(0,2,(SHAPE[1],SHAPE[2]),dtype=np.uint8)<<7)
    logical.flat[:8]=[0,128,1,129,126,254,56,184]
    result.append(E.pack_cpu(logical))
  assert len({E.sha(w.tobytes()) for w in result})==4
  return result


def gpu(report,save,binaries,render):
  from tinygrad import Device,Tensor,dtypes
  from tinygrad.device import Compiled
  assert os.environ.get('TC_OPT')=='2' and os.environ.get('LLVMOPT') in (None,'1')
  assert os.environ.get('AM_POWER_LIMIT')=='144' and os.environ.get('AM_PERF_LEVEL')=='auto'
  Compiled.wait_timeout_ms=10000
  H.load('prefill-kernels').fix_dispatch_packet()
  device=Device[Device.DEFAULT]
  assert device.arch=='gfx1200' and device.is_usb
  H.load('wgp_compiler').install(device)
  fns=numeric_functions(render)
  report['gpu_executed']=True
  io=IO(device); verifier=Verifier(io)
  report['validation']=verifier.records
  # Validate the full-size integer checker before trusting it for GEMM outputs.
  test_a=Tensor.empty(*SHAPE[:2],device=device.device,dtype=dtypes.bfloat16).realize()
  test_b=Tensor.empty(*SHAPE[:2],device=device.device,dtype=dtypes.bfloat16).realize()
  verifier.fill(test_a,0x3f80); verifier.fill(test_b,0x3f80)
  verifier.check(test_a,test_b,'validator-equal')
  for index in (0,1023,1024,SHAPE[0]*SHAPE[1]//2,SHAPE[0]*SHAPE[1]-1):
    verifier.poke(test_b,index,0x3f81)
    verifier.check(test_a,test_b,f'validator-one-bit-{index}',False)
    verifier.poke(test_b,index,0x3f80)
  for word in (0x7f80,0x7fc1):
    verifier.poke(test_a,SHAPE[0]*SHAPE[1]-1,word)
    verifier.poke(test_b,SHAPE[0]*SHAPE[1]-1,word)
    verifier.check(test_a,test_b,f'validator-identical-nonfinite-{word}',False)
  verifier.fill(test_a); verifier.fill(test_b)
  verifier.check(test_a,test_b,'validator-identical-poison',False)
  save()

  # Complete precisely the unfinished states using fresh captured single/40-call graphs.
  data=H.fixture(SHAPE,4)
  tensors=tuple(io.upload(data[key],dtypes.fp8e4m3 if key in ('a','w') else dtypes.float32) for key in KEYS)
  dummy=io.upload(data['bias'],dtypes.bfloat16)
  known={key:E.sha(data[key].tobytes()) for key in KEYS}
  inputs=(*tensors,dummy)
  singles={label:E.make_jit(SHAPE,False,fns[label],1) for label in LABELS}
  serial={label:E.make_jit(SHAPE,False,fns[label],40) for label in LABELS}
  report['remaining_binary_capture_checks']={}
  for label in LABELS:
    observed=[]
    with observe_numeric_binary(observed,binaries[label]):
      for jit in (singles[label],serial[label]):
        for _ in range(3): jit(*inputs)
    assert len(observed)>=41
    report['remaining_binary_capture_checks'][label]=dict(binary_sha256=binaries[label],observed_calls=len(observed))
    assert len(E.captured_kernels(singles[label]))==1 and E.shared_output_slot(serial[label])==5
  report['remaining_actual_mutations']=[]
  report['remaining_serial_proofs']={label:serial[label].proofs[0] for label in LABELS}
  previous=singles['current'].output.clone().realize()
  verifier.check(previous,previous,'resumed-4-reference-finite')
  for state in (5,6):
    data=H.fixture(SHAPE,state)
    changed=update_changed(io,tensors,data,known)
    reference=verifier.run(singles['current'],inputs)
    verifier.check(reference,reference,f'resumed-{state}-reference-finite')
    verifier.check(previous,reference,f'resumed-{state}-reference-changed',False)
    for label,jit in (('current-batch',serial['current']),('amd4-single',singles['amd4']),('amd4-batch',serial['amd4'])):
      verifier.check(reference,verifier.run(jit,inputs),f'resumed-{state}-{label}')
    report['remaining_actual_mutations'].append(dict(state=state,changed_tensors=changed,input_sha256=dict(known),
      fresh_baseline_reference=True,single_and_serial_40_all_bits=True,poison_before_each_run=True))
    report['io']=io.snapshot(); save()
    previous=reference.clone().realize()

  # Restore the original state-zero warm fixture, then add three different resident weights.
  base=H.fixture(SHAPE,0)
  update_changed(io,tensors,base,known)
  arrays=weight_data(base)
  weights=(tensors[1],*(io.upload(w,dtypes.fp8e4m3) for w in arrays[1:]))
  assert len({id(w.uop.buffer.base) for w in weights})==4
  report['weights']=dict(count=4,bytes_each=arrays[0].nbytes,total_bytes=sum(w.nbytes for w in arrays),
    sha256=[E.sha(w.tobytes()) for w in arrays],resident_distinct_allocations=True,
    origin='Synthetic raw finite E4M3 weights; W0 is the old state-zero fixture. Not learned model weights.')
  assert report['weights']['total_bytes']>64*1024**2
  a,_,sx,sw=tensors
  rotating_inputs=(a,sx,sw,*weights)
  references=[]
  for index,w in enumerate(weights):
    args=(a,w,sx,sw,dummy)
    reference=Tensor.empty(*SHAPE[:2],device=device.device,dtype=dtypes.bfloat16).realize()
    verifier.fill(reference)
    reference=E.operation(SHAPE,False,fns['current'],args,reference)
    verifier.check(reference,reference,f'weight-{index}-reference-finite')
    verifier.check(reference,verifier.run(singles['amd4'],args),f'weight-{index}-amd4-single')
    for prior_index,prior in enumerate(references): verifier.check(prior,reference,f'weight-{prior_index}-versus-{index}-different',False)
    references.append(reference)
  batches={workload:{label:RotatingJit(fns[label],order,binaries[label]) for label in LABELS} for workload,order in ORDERS.items()}
  report['queue_proofs']={}
  for workload,variants in batches.items():
    report['queue_proofs'][workload]={}
    for label,jit in variants.items():
      for _ in range(3): jit(*rotating_inputs)
      report['queue_proofs'][workload][label]=jit.audit_capture()
      verifier.check(references[ORDERS[workload][-1]],verifier.run(jit,rotating_inputs),f'pre-{workload}-{label}')
  report['io']=io.snapshot(); save()

  # Conditioning and timed execution have no operand uploads or output readbacks.
  before=io.snapshot()
  report['timing']={workload:dict(provisional=True,rounds=[]) for workload in ORDERS}
  for iteration in range(3):
    workloads=('warm','rotating') if iteration%2==0 else ('rotating','warm')
    for workload in workloads:
      order=LABELS if iteration%2==0 else LABELS[::-1]
      pair=dict(order=order,iteration=iteration)
      for label in order:
        jit=batches[workload][label]
        device.synchronize(); jit(*rotating_inputs); device.synchronize()
        start=time.perf_counter(); jit(*rotating_inputs); device.synchronize()
        pair[label]=(time.perf_counter()-start)*1000
      report['timing'][workload]['rounds'].append(pair)
  assert io.snapshot()==before
  report['timing_data_transfer_bytes']=0
  save()

  # Rebind every final weight in the same captured rotating graph after timing.
  # This checks all four outputs, not only W3, while preserving the 40-call WAW graph.
  for phase in range(4):
    permuted=weights[phase:]+weights[:phase]
    args=(a,sx,sw,*permuted)
    final=(phase+3)%4
    for label in LABELS:
      verifier.check(references[final],verifier.run(batches['rotating'][label],args),f'post-rotating-{label}-final-W{final}')
  for label in LABELS:
    verifier.check(references[0],verifier.run(batches['warm'][label],rotating_inputs),f'post-warm-{label}')
  for workload,item in report['timing'].items():
    current,amd4=(statistics.median(row[label] for row in item['rounds'])/40 for label in LABELS)
    item.update(provisional=False,current_ms=current,amd4_ms=amd4,amd4_over_current=amd4/current,
      calls_per_timed_batch=40,conditioning_calls_per_sample=40,host_inclusive=True,
      no_operand_or_output_transfers_during_timing=True,actual_full_model=False,
      measured_dram_bytes=None,measured_cache_hit_rate=None)
  report['io']=io.snapshot()
  report['complete']=True
  save()


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode',choices=('cpu','gpu'),default='cpu')
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--source-sha')
  args=parser.parse_args()
  os.environ.update(BEAM='0',JITBEAM='0',PARALLEL='0',PROFILE='0',DEBUG='0')
  if args.mode=='cpu': os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0')
  from tinygrad import Device
  report=dict(mode=args.mode,complete=False,gpu_executed=False,provenance=provenance(),
    scope='Current and unchanged AMD4 only. Synthetic rotating weights, not model streaming or a DRAM/cache-counter measurement.')
  assert args.mode!='gpu' or args.source_sha==report['provenance']['source_sha256'], 'Missing or changed source binding'
  def save(): E.write_json(args.output,report)
  try:
    render=H.renderer()
    report['evidence_reuse'],binaries=audit_evidence(render)
    report['numeric_binaries']=binaries
    if args.mode=='cpu':
      report.update(local('agent-lds-cold-check').cpu_audit(render,binaries))
      assert not Device._opened_devices
      report['complete']=True
    else: gpu(report,save,binaries,render)
    assert provenance()==report['provenance'], 'Source or old evidence changed during the trial'
    save()
    print(json.dumps(dict(complete=report['complete'],gpu_executed=report['gpu_executed'],source_sha256=report['provenance']['source_sha256'])))
  except BaseException as exc:
    report['complete']=False
    report['failure']=f'{type(exc).__name__}: {exc}'
    for value in report.get('timing',{}).values(): value['provisional']=True
    save()
    raise


if __name__=='__main__': main()

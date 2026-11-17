"""One CPU-only pre-HCQ schedule cache prototype, with explicit live bindings.

This intentionally rejects AMD, HCQ submit graphs, internal allocations, dynamic
variables, unclassified buffers, and partial/aliased registered buffer views.
It demonstrates a safe narrow serialization boundary, not a model integration.
"""
import argparse
import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import struct
import subprocess
import sys
import time
import numpy as np

ROOT=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
FORMAT='agent-startup-cache-cpu-v1'


def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


@dataclass
class Binding:
  tensor:object
  role:str
  source_id:str


def schema(binding):
  from tinygrad.uop.ops import Ops
  t=binding.tensor
  if binding.role not in ('input','constant','state_zero','output_zero'):raise ValueError('Unclassified binding role')
  if t.device!='CPU':raise ValueError('This proof supports local CPU only; no AMD or remote device is opened')
  if not t.uop.is_realized:raise ValueError('Caller must provide an already resident realized buffer')
  base=t.uop.base
  if base.op is not Ops.BUFFER or t.uop.contiguous_view_offset()!=0 or t.numel()*t.dtype.itemsize!=base.buffer.nbytes:
    raise ValueError('Prototype requires a complete contiguous registered buffer')
  return dict(role=binding.role,source_id=binding.source_id,shape=list(t.shape),dtype=t.dtype.name,device=t.device,
              bytes=base.buffer.nbytes)


def identity(checkpoint_id,calibration_id):
  from tinygrad import Device
  from tinygrad.runtime.autogen import llvm
  from tinygrad import helpers
  d=Device['CPU'];r=d.renderer
  major,minor,patch=ctypes.c_uint(),ctypes.c_uint(),ctypes.c_uint()
  llvm.LLVMGetVersion(ctypes.byref(major),ctypes.byref(minor),ctypes.byref(patch))
  library=Path(llvm.dll._name);stat=library.stat()
  cc=Path(shutil.which(os.environ.get('CC','clang')) or '').resolve()
  if not cc.is_file():raise ValueError('Compiler executable identity unavailable')
  cc_info=dict(path=str(cc),sha256=sha(cc.read_bytes()),version=subprocess.check_output([str(cc),'--version'],text=True).strip(),
               target_arguments=list(getattr(r.compiler,'args',())))
  options={name:str(getattr(helpers,name).value) for name in
           ('NOOPT','BEAM','USE_TC','TC_OPT','TC_SELECT','DEFAULT_FLOAT','DEFAULT_INT')}
  files=['tinygrad/engine/jit.py','tinygrad/uop/ops.py','tinygrad/device.py','tinygrad/engine/realize.py',
         'tinygrad/runtime/support/hcq2.py','tinygrad/runtime/ops_cpu.py','tinygrad/runtime/support/compiler_llvm.py',
         'tinygrad/runtime/support/compiler_cpu.py','tinygrad/runtime/support/compiler_amd.py',
         'tinygrad/codegen/__init__.py','tinygrad/schedule/memory.py']
  cpu={}
  for line in Path('/proc/cpuinfo').read_text().splitlines():
    if ':' in line:
      k,v=line.split(':',1)
      if k.strip() in ('vendor_id','model name','cpu family','model','stepping','flags'):cpu.setdefault(k.strip(),v.strip())
  return dict(checkpoint=checkpoint_id,calibration=calibration_id,
    sources={str(ROOT/f):sha((ROOT/f).read_bytes()) for f in files}|{str(Path(__file__)):sha(Path(__file__).read_bytes())},
    compiler=dict(renderer=type(r).__module__+'.'+type(r).__name__,compiler=type(r.compiler).__module__+'.'+type(r.compiler).__name__,
                  cachekey=r.compiler.cachekey,target=str(r.target),llvm_version=[major.value,minor.value,patch.value],
                  llvm_library=dict(path=str(library),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),cc=cc_info,options=options),
    device=dict(name='CPU',arch=d.arch,machine=platform.machine(),cpu=cpu),
    interpreter=dict(version=list(sys.version_info[:3]),byteorder=sys.byteorder,pickle_protocol=5),
    policy=dict(stage='compiled-pre-HCQ',state='zero-at-each-load',dynamic_variables=False,internal_allocations=False))


def audit(linear,slots):
  from tinygrad.uop.ops import Ops,ProgramInfo,ParamArg
  from tinygrad.runtime.support.hcq2 import HCQInfo,unwrap_lane
  if linear.op is not Ops.LINEAR:raise ValueError('Expected a compiled LINEAR')
  programs=[];written=set();used=set()
  for wrapped in linear.src:
    c=wrapped.without_after
    if isinstance(getattr(c.arg,'aux',None),HCQInfo):raise ValueError('HCQ submit graphs cannot be serialized by this proof')
    if c.op is not Ops.CALL or c.body.op is not Ops.PROGRAM or not isinstance(c.body.arg,ProgramInfo) or c.body.src[-1].op is not Ops.BINARY:
      raise ValueError('Only raw compiled PROGRAM calls are supported')
    if c.body.arg.target.device!='CPU':raise ValueError('Prototype supports CPU code objects only')
    if c.arg.grad_fxn is not None or c.arg.aux is not None:raise ValueError('Call callbacks or runtime context are unsupported')
    io=c.body.arg
    for i in io.globals:
      b,lane,off=unwrap_lane(c.src[1+i])
      if b.op is not Ops.PARAM or lane is not None or not 0<=b.arg.slot<len(slots):raise ValueError('Every call resource needs an explicit named binding')
      used.add(b.arg.slot)
      if i in io.outs:
        if slots[b.arg.slot]['schema']['role'] in ('input','constant'):raise ValueError('A read-only binding is written')
        written.add(b.arg.slot)
      if i in io.ins and slots[b.arg.slot]['schema']['role']=='output_zero' and b.arg.slot not in written:
        raise ValueError('Output data is read before a declared write')
    programs.append(c.body)
  allowed_binary={p.src[-1] for p in programs}
  for u in linear.toposort():
    if u.op in (Ops.BUFFER,Ops.ALLOC,Ops.GETADDR):raise ValueError('Captured storage, scratch allocations, and addresses are forbidden')
    if u.op is Ops.BINARY and u not in allowed_binary:raise ValueError('Only compiler code objects may contain binary data')
    if isinstance(u.arg,ParamArg) and u.arg.buffer is not None:raise ValueError('A live Buffer survived normalization')
    if u.is_variable:raise ValueError('Dynamic variables require a separate validated binding contract')
  if used!=set(range(len(slots))):raise ValueError('Unused or missing binding slots')
  for slot,b in enumerate(slots):
    if b['schema']['role'] in ('state_zero','output_zero') and slot not in written:raise ValueError('Declared state/output lacks a write')
  return dict(program_calls=len(programs),code_object_bytes=sum(len(p.src[-1].arg) for p in set(programs)),
              live_buffers=0,embedded_tensor_bytes=0,address_nodes=0)


class NoStoragePickler(pickle.Pickler):
  def persistent_id(self,obj):
    from tinygrad import Tensor
    from tinygrad.device import Buffer,MultiBuffer
    if isinstance(obj,(Tensor,Buffer,MultiBuffer)):raise ValueError('Tensor/Buffer serialization would capture data')
    return None


class PlanUnpickler(pickle.Unpickler):
  def find_class(self,module,name):
    from tinygrad.uop.ops import UOp,Ops,ProgramInfo,ParamArg,KernelInfo,CallInfo,AxisType
    from tinygrad.dtype import DType,ConstFloat,AddrSpace,InvalidType
    from tinygrad.helpers import Target
    from tinygrad.renderer import Estimates
    from tinygrad.codegen.opt import Opt,OptOps
    allowed={(c.__module__,c.__name__):c for c in (UOp,Ops,ProgramInfo,ParamArg,KernelInfo,CallInfo,AxisType,
      DType,ConstFloat,AddrSpace,InvalidType,Target,Estimates,Opt,OptOps)}
    if (module,name) not in allowed:raise ValueError(f'Unsupported serialized class: {module}.{name}')
    return allowed[module,name]


@contextmanager
def capture_pre_hcq(plans):
  """Intercept the pinned TinyJit boundary before HCQ encoding or address link."""
  import tinygrad.engine.jit as jit_module
  from tinygrad.engine.realize import lower_and_compile
  original=jit_module.compile_linear
  def compile_plan(linear,*args,**kwargs):
    if kwargs.get('input_uops') is None:return original(linear,*args,**kwargs)
    raw=lower_and_compile(linear)
    plans.append((raw,tuple(kwargs.get('input_uops',()))))
    return original(raw,*args,**kwargs)
  jit_module.compile_linear=compile_plan
  try:yield
  finally:jit_module.compile_linear=original


def save(path,plan,input_uops,bindings,current_identity):
  from tinygrad.uop.ops import UOp,Ops
  by_buffer={}
  for name,b in bindings.items():
    schema(b)
    base=b.tensor.uop.base
    if base in by_buffer:raise ValueError('Aliased named bindings are unsupported')
    by_buffer[base]=name
  names=[]
  for base in input_uops:
    if base not in by_buffer:raise ValueError('A JIT argument lacks a named binding')
    names.append(by_buffer[base])
  substitutions={}
  for u in plan.toposort():
    if u.op is Ops.BUFFER:
      if u not in by_buffer:raise ValueError('An unclassified captured buffer would be serialized')
      name=by_buffer[u]
      if name not in names:names.append(name)
      substitutions[u]=UOp.param(names.index(name),u.dtype,u.max_numel(),u.device)
  if set(names)!=set(bindings):raise ValueError('Bindings do not match the complete captured buffer set')
  slots=[dict(name=n,schema=schema(bindings[n])) for n in names]
  portable=plan.substitute(substitutions,walk=True)
  proof=audit(portable,slots)
  stream=io.BytesIO();NoStoragePickler(stream,protocol=5).dump(portable);payload=stream.getvalue()
  path.mkdir(parents=True,exist_ok=False)
  (path/'plan.pkl').write_bytes(payload)
  manifest=dict(format=FORMAT,identity=current_identity,slots=slots,payload_sha256=sha(payload),payload_bytes=len(payload),audit=proof)
  (path/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
  return manifest


class Runner:
  def __init__(self,path,bindings,current_identity):
    from tinygrad.engine.realize import compile_linear,link_linear
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['format']!=FORMAT or canonical(manifest['identity'])!=canonical(current_identity):raise ValueError('Checkpoint/source/compiler/device/calibration identity mismatch')
    if set(bindings)!={s['name'] for s in manifest['slots']}:raise ValueError('Binding name mismatch')
    if any(schema(bindings[s['name']])!=s['schema'] for s in manifest['slots']):raise ValueError('Binding shape/dtype/device/role/source mismatch')
    bases=[bindings[s['name']].tensor.uop.base for s in manifest['slots']]
    if len(set(bases))!=len(bases):raise ValueError('Bindings alias')
    payload=(path/'plan.pkl').read_bytes()
    if len(payload)!=manifest['payload_bytes'] or sha(payload)!=manifest['payload_sha256']:raise ValueError('Plan payload identity mismatch')
    linear=PlanUnpickler(io.BytesIO(payload)).load();proof=audit(linear,manifest['slots'])
    if proof!=manifest['audit']:raise ValueError('Plan audit mismatch')
    self.bindings,self.inputs,self.manifest=dict(bindings),bases,manifest
    # Local CPU proof resets the caller's existing buffers directly. An AMD
    # adapter must supply validated fresh state and an authorized device memset.
    for s in manifest['slots']:
      if s['schema']['role'] in ('state_zero','output_zero'):
        bindings[s['name']].tensor.uop.buffer.host[:]=bytes(s['schema']['bytes'])
    self.linear=link_linear(compile_linear(linear,input_uops=self.inputs,cache=False),input_uops=self.inputs)
    self.calls=0
  def run(self,inputs=None):
    from tinygrad.engine.realize import run_linear
    if inputs is not None:
      wanted={s['name'] for s in self.manifest['slots'] if s['schema']['role']=='input'}
      if set(inputs)!=wanted:raise ValueError('Each current input needs an explicit live binding')
      for i,s in enumerate(self.manifest['slots']):
        if s['name'] not in inputs:continue
        b=Binding(inputs[s['name']],s['schema']['role'],s['schema']['source_id'])
        if schema(b)!=s['schema']:raise ValueError('Current input binding differs from the cached shape')
        self.bindings[s['name']]=b;self.inputs[i]=b.tensor.uop.base
      if len(set(self.inputs))!=len(self.inputs):raise ValueError('Current input aliases another binding')
    if any(self.inputs[i] is not self.bindings[s['name']].tensor.uop.base for i,s in enumerate(self.manifest['slots'])):
      raise ValueError('A live binding changed allocation without an explicit rebind')
    run_linear(self.linear,input_uops=self.inputs,jit=True);self.calls+=1
    return self.bindings['out'].tensor


def state_kernel(kv,x,w):
  from tinygrad.uop.ops import UOp,KernelInfo
  r=UOp.range(kv.numel(),0)
  val=kv.flatten().index(r).load()+x.flatten().index(r).load()*w.flatten().index(r).load()
  return kv.flatten().index(r).store(val).end(r).sink(arg=KernelInfo(name='agent_startup_state'))


def output_kernel(out,kv):
  from tinygrad.uop.ops import UOp,KernelInfo
  r=UOp.range(out.numel(),0)
  return out.flatten().index(r).store(kv.flatten().index(r).load()*2).end(r).sink(arg=KernelInfo(name='agent_startup_output'))


def live(seed):
  from tinygrad import Tensor
  weights=np.full(1<<20,np.float32(19.375),np.float32);weights[:4]=[1,2,4,8]
  def cpu(a):return Tensor(np.asarray(a,np.float32),device='CPU').realize()
  bindings=dict(x=Binding(cpu(np.array([1,2,3,4])+seed),'input','fresh-current-input'),
    kv=Binding(cpu([777,777,777,777]),'state_zero','session-kv'),
    out=Binding(cpu([999,999,999,999]),'output_zero','current-output'),
    weight=Binding(cpu(weights),'constant',sha(weights.tobytes())))
  return bindings,identity(sha(weights.tobytes()),sha(canonical({'toy_scale':1})))


def allocation_info(bindings):
  return {name:dict(object_id=id(b.tensor.uop.buffer),address=b.tensor.uop.buffer._buf,bytes=b.tensor.nbytes()) for name,b in bindings.items()}


def capture(path):
  from tinygrad import TinyJit,Tensor
  bindings,ident=live(1234)
  weight=bindings['weight'].tensor
  def function(x,kv,out):
    kv=Tensor.custom_kernel(kv,x,weight,fxn=state_kernel)[0]
    out=Tensor.custom_kernel(out,kv,fxn=output_kernel)[0]
    Tensor.realize(out,kv);return out,kv
  jit=TinyJit(function);plans=[]
  with capture_pre_hcq(plans):
    for _ in range(3):jit(*(bindings[n].tensor for n in ('x','kv','out')))
  if len(plans)!=1:raise AssertionError(('Expected one TinyJit capture boundary',len(plans)))
  raw,inputs=plans[0];manifest=save(path,raw,inputs,bindings,ident)
  try:save(path.parent/(path.name+'-unclassified'),raw,inputs,{n:b for n,b in bindings.items() if n!='weight'},ident)
  except ValueError:unclassified_rejected=True
  else:raise AssertionError('Unclassified closure constant was accepted')
  # The storage guard proves that an ordinary captured-JIT pickle would include
  # runtime tensor/buffer objects. It refuses before any buffer contents copy.
  try:NoStoragePickler(io.BytesIO(),protocol=5).dump(jit)
  except ValueError:unsafe_rejected=True
  else:raise AssertionError('CapturedJit unexpectedly contained no storage')
  metadata=dict(pid=os.getpid(),allocations=allocation_info(bindings),warm_kv_nonzero=bool(np.all(bindings['kv'].tensor.numpy()!=0)),
                artifact_bytes=manifest['payload_bytes'],weight_bytes=bindings['weight'].tensor.nbytes(),unsafe_pickle_rejected=unsafe_rejected,
                unclassified_constant_rejected=unclassified_rejected)
  payload=(path/'plan.pkl').read_bytes()
  assert all(struct.pack('<Q',v['address']) not in payload for v in metadata['allocations'].values())
  (path/'capture-proof.json').write_text(json.dumps(metadata,indent=2)+'\n');return metadata


@contextmanager
def no_codegen():
  from tinygrad.device import Compiler
  original=Compiler.compile_cached
  def forbidden(*_a,**_kw):raise AssertionError('Replayed schedule attempted compiler work')
  Compiler.compile_cached=forbidden
  try:yield
  finally:Compiler.compile_cached=original


def replay(path):
  from tinygrad import Device
  bindings,ident=live(0);before=allocation_info(bindings)
  with no_codegen():runner=Runner(path,bindings,ident)
  np.testing.assert_array_equal(bindings['kv'].tensor.numpy(),0);np.testing.assert_array_equal(bindings['out'].tensor.numpy(),0)
  assert before==allocation_info(bindings)
  reference=np.zeros(4,np.float32);results=[]
  with no_codegen():
    for value in ([1,2,3,4],[9,7,5,3],[0,1,0,1]):
      x=np.array(value,np.float32);bindings['x'].tensor.uop.buffer.host[:]=x.tobytes()
      reference+=x*np.array([1,2,4,8],np.float32)
      out=runner.run();np.testing.assert_array_equal(out.numpy(),reference*2)
      assert out.uop.buffer is bindings['out'].tensor.uop.buffer
      assert all(runner.inputs[i] is bindings[s['name']].tensor.uop.base for i,s in enumerate(runner.manifest['slots']))
      results.append(out.numpy().tolist())
  assert before==allocation_info(bindings)
  # Replace the input allocation while preserving this session's live KV.
  from tinygrad import Tensor
  next_input=Tensor(np.array([2,4,6,8],np.float32),device='CPU').realize()
  reference+=np.array([2,4,6,8],np.float32)*[1,2,4,8]
  with no_codegen():np.testing.assert_array_equal(runner.run({'x':next_input}).numpy(),reference*2)
  assert runner.bindings['x'].tensor.uop.buffer is next_input.uop.buffer
  rejections=[]
  for key in ('checkpoint','sources','compiler','device','calibration'):
    bad=json.loads(json.dumps(ident));bad[key]={'changed':True}
    try:Runner(path,bindings,bad)
    except ValueError:rejections.append(key)
    else:raise AssertionError('Mismatched identity accepted')
  smaller,ident2=live(10)
  smaller['x']=Binding(bindings['weight'].tensor,'input','fresh-current-input')
  try:Runner(path,smaller,ident2)
  except ValueError:rejections.append('shape')
  else:raise AssertionError('Wrong shape accepted')
  from tinygrad import dtypes
  bad=dict(bindings);bad['x']=Binding(bindings['x'].tensor.bitcast(dtypes.int32),'input','fresh-current-input')
  try:Runner(path,bad,ident)
  except ValueError:rejections.append('dtype')
  else:raise AssertionError('Wrong dtype accepted')
  try:Runner(path,{n:b for n,b in bindings.items() if n!='weight'},ident)
  except ValueError:rejections.append('missing-binding')
  else:raise AssertionError('Missing model constant binding accepted')
  bad=dict(bindings);bad['out']=Binding(bindings['kv'].tensor,'output_zero','current-output')
  try:Runner(path,bad,ident)
  except ValueError:rejections.append('alias')
  else:raise AssertionError('Aliased state/output accepted')
  bad=dict(bindings);bad['weight']=Binding(bindings['weight'].tensor,'constant','wrong-weight-tensor')
  try:Runner(path,bad,ident)
  except ValueError:rejections.append('constant-source-id')
  else:raise AssertionError('Wrong constant source accepted')
  tampered=path.parent/(path.name+'-tampered');tampered.mkdir(exist_ok=False)
  payload=bytearray((path/'plan.pkl').read_bytes());payload[len(payload)//2]^=1
  (tampered/'plan.pkl').write_bytes(payload);(tampered/'manifest.json').write_bytes((path/'manifest.json').read_bytes())
  try:Runner(tampered,bindings,ident)
  except ValueError:rejections.append('payload')
  else:raise AssertionError('Changed payload accepted')
  from dataclasses import replace
  from tinygrad.uop.ops import Ops,UOp
  from tinygrad.runtime.support.hcq2 import HCQInfo
  plan=PlanUnpickler(io.BytesIO((path/'plan.pkl').read_bytes())).load()
  first=plan.src[0].without_after
  forbidden=first.replace(arg=replace(first.arg,aux=HCQInfo(('CPU',))))
  try:audit(plan.replace(src=(forbidden,*plan.src[1:])),runner.manifest['slots'])
  except ValueError:rejections.append('HCQ-submit-state')
  else:raise AssertionError('HCQ state accepted')
  args=list(first.src);v=args[1];args[1]=UOp.alloc(v.max_shape,v.dtype,device='CPU')
  try:audit(plan.replace(src=(first.replace(src=tuple(args)),*plan.src[1:])),runner.manifest['slots'])
  except ValueError:rejections.append('internal-allocation')
  else:raise AssertionError('Private scratch accepted')
  newer,ident3=live(20)
  with no_codegen():other=Runner(path,newer,ident3)
  assert all(newer[n].tensor.uop.buffer is not bindings[n].tensor.uop.buffer for n in bindings)
  np.testing.assert_array_equal(newer['kv'].tensor.numpy(),0)
  with no_codegen():np.testing.assert_array_equal(other.run().numpy(),np.array([21,22,23,24],np.float32)*[1,2,4,8]*2)
  np.testing.assert_array_equal(bindings['kv'].tensor.numpy(),reference)
  parent=json.loads((path/'capture-proof.json').read_text());assert parent['pid']!=os.getpid()
  assert not any(d.startswith(('AMD','NV','CUDA')) for d in Device._opened_devices)
  return dict(pid=os.getpid(),capture_pid=parent['pid'],fresh_process=True,first_session_state_zero=True,
    live_buffers_reused_without_copy=before==allocation_info(bindings),same_buffer_mutations_exact=results,
    independent_reload_allocations=True,second_session_state_zero=True,codegen_calls_on_load_or_replay=0,
    explicit_new_input_allocation_exact=True,captured_address_literals_absent=True,
    observed_addresses_different={n:before[n]['address']!=parent['allocations'][n]['address'] for n in before},
    allocations=before,identity_mismatches_rejected=rejections,
    artifact_bytes=runner.manifest['payload_bytes'],constant_bytes=bindings['weight'].tensor.nbytes(),gpu_opened=False)


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('capture','replay','self-test'));p.add_argument('--artifact',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  if args.mode=='self-test':
    reports=[]
    for mode in ('capture','replay'):
      out=args.artifact.parent/(args.artifact.name+'-'+mode+'.json')
      subprocess.run([sys.executable,__file__,mode,'--artifact',str(args.artifact),'--output',str(out)],check=True)
      reports.append(json.loads(out.read_text()))
    result=dict(cpu_only=True,all_pass=True,capture=reports[0],replay=reports[1])
  else:result=capture(args.artifact) if args.mode=='capture' else replay(args.artifact)
  args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(mode=args.mode,output=str(args.output))),flush=True)


if __name__=='__main__':main()

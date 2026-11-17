"""Exact compiled associations and per-call/dependency templates for the narrow paired phase."""
from contextlib import contextmanager
from dataclasses import asdict

@contextmanager
def guard(common,catalog,report):
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  from tinygrad.engine.realize import get_call_kernel_args
  from tinygrad.uop.ops import UOp
  from tinygrad import dtypes
  from tinygrad.engine import realize
  from tinygrad.codegen import do_to_program
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.helpers import Target
  from tinygrad.uop.ops import Ops,KernelInfo,ProgramInfo
  identity = common.load('agent-ck-source-identity')
  old_build,old_queues,old_args,old_packet = ops_amd.amd_build_program,hcq2._build_queues,ops_amd.AMDComputeQueue.kernargs,ops_amd.dispatch_packet
  old_compile=realize.lower_and_compile
  # This scope contains only the enumerated standalone attention boundary.
  # Keep the model's renderer and its cached compiler instance untouched.
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  def compile_bound(linear,verbose=True):
    replacements={}
    for call in linear.toposort():
      if call.op is not Ops.CALL: continue
      ast=call.body
      if ast.op is Ops.SINK and isinstance(ast.arg,KernelInfo) or ast.op is Ops.PROGRAM:
        p=ast if ast.op is Ops.PROGRAM and isinstance(ast.arg,ProgramInfo) and ast.src[-1].op is Ops.BINARY else do_to_program(ast,renderer)
        identity.authorize(p,catalog)
        if p.src[0].arg.name=='alpamayo_ck_d128' and report.get('gpu_executed'):
          pointers=[]
          for u in call.src[1:]:
            b=u.buffer;assert b.base.is_allocated();b.ensure_allocated()
            address=int(b.get_storage().buf);base=int(b.base.get_storage().buf)
            assert address==base+b.offset
            pointers.append({'address':address,'base_address':base,'offset_bytes':b.offset,'bytes':b.nbytes})
          assert [p['address'] for p in pointers]==report['expected_candidate_pointers']
          assert [p['offset_bytes'] for p in pointers[:2]]==[512,1024]
          report.setdefault('actual_candidate_pointer_bindings',[]).append(pointers)
        replacements[call]=call.replace(src=(p,*call.src[1:]))
      else: raise AssertionError(('Unexpected scoped operation',ast.op))
    return linear.substitute(replacements)
  def build(dev,p,devices):
    aid,item = identity.authorize(p,catalog)
    descriptor,_ = ops_amd._amd_program_image(dev,p.src[3].arg)
    assert asdict(descriptor) == item['descriptor']
    report.setdefault('actual_programs',{})[aid] = item
    return old_build(dev,p,devices)
  def queues(ctx):
    signature = identity.batch_rows(ctx.batch); key = identity.key(signature)
    assert key in catalog['batches'] and signature == catalog['batches'][key], ('Unprepared dependency batch',key,signature)
    report.setdefault('batches',[]).append({'phase':report['phase'],'key':key,'calls':signature})
    return old_queues(ctx)
  def args(queue,call,p,data):
    aid,item = identity.authorize(p,catalog)
    assert asdict(data) == item['descriptor']
    layout = [offset for offset,_ in hcq2.layout_args(get_call_kernel_args(call,p,queue.devs))]
    report.setdefault('argument_records',[]).append({'phase':report['phase'],'association':aid,'offsets':layout,
      'kernarg_bytes':data.kernargs_segment_size,'CU_bit':(data.rsrc1>>29)&1,'dispatch_pointer':data.enable_dispatch_ptr})
    return old_args(queue,call,p,data)
  def packet(*args,**kwargs):
    return [UOp.const(x,dtypes.uint32) if isinstance(x,int) else x for x in old_packet(*args,**kwargs)]
  ops_amd.amd_build_program,hcq2._build_queues,ops_amd.AMDComputeQueue.kernargs,ops_amd.dispatch_packet = build,queues,args,packet
  realize.lower_and_compile=compile_bound
  try: yield
  finally:
    assert ops_amd.amd_build_program is build and hcq2._build_queues is queues and ops_amd.AMDComputeQueue.kernargs is args
    assert realize.lower_and_compile is compile_bound
    realize.lower_and_compile=old_compile
    ops_amd.amd_build_program,hcq2._build_queues,ops_amd.AMDComputeQueue.kernargs,ops_amd.dispatch_packet = old_build,old_queues,old_args,old_packet

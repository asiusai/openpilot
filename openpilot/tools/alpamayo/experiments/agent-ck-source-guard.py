"""Parent runtime observers with strict enumerated source/AST/ABI/call associations."""
from contextlib import contextmanager
from dataclasses import asdict
import importlib.util
from pathlib import Path

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_source_identity',EXP/'agent-ck-source-identity.py')
identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
_folder=None


@contextmanager
def record_programs(report,folder):
  global _folder
  assert _folder is None;_folder=folder
  report['actual_programs']={};report['actual_associations']={}
  try: yield
  finally: _folder=None


@contextmanager
def bindings(helper,plan,report):
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  from tinygrad.engine.realize import get_call_outs_ins,get_call_kernel_args
  from tinygrad.uop.ops import UOp
  from tinygrad import dtypes
  old_build,old_queues,old_args,old_packet=ops_amd.amd_build_program,hcq2._build_queues,ops_amd.AMDComputeQueue.kernargs,ops_amd.dispatch_packet
  def build(dev,program,devices):
    item=identity.identity(program);aid=identity.key(item);digest=item['binary_sha256']
    assert _folder is not None
    if aid not in report['actual_associations']:
      (_folder/(aid+'.source')).write_text(program.src[2].arg)
      (_folder/(aid+'.ast')).write_text(str(program.src[0])+'\n')
      (_folder/(aid+'.linear')).write_text(str(program.src[1])+'\n')
      report['actual_associations'][aid]=item
    if digest not in report['actual_programs']:
      (_folder/(digest+'.hsaco')).write_bytes(program.src[3].arg)
      report['actual_programs'][digest]={'associations':[]}
    if aid not in report['actual_programs'][digest]['associations']:
      report['actual_programs'][digest]['associations'].append(aid)
    identity.authorize(program,plan)
    descriptor,_=ops_amd._amd_program_image(dev,program.src[3].arg)
    assert asdict(descriptor)==item['descriptor']
    return old_build(dev,program,devices)
  def packet(*args,**kwargs):
    return [UOp.const(x,dtypes.uint32) if isinstance(x,int) else x for x in old_packet(*args,**kwargs)]
  def queues(ctx):
    signature=identity.batch_rows(ctx.batch);bid=identity.key(signature)
    rows=[]
    for tag,(call,device,queue) in enumerate(ctx.batch):
      aid,item=identity.authorize(call.body,plan)
      outs,ins=get_call_outs_ins(call)
      rows.append({'tag':tag,'name':call.body.src[0].arg.name,'device':device,'queue':queue,
        'association':aid,'binary_sha256':item['binary_sha256'],'source_sha256':item['source_sha256'],
        'ast_key':item['ast_key'],'linear_key':item['linear_key'],'target':item['target'],
        'outs':outs,'ins':ins,'dependencies':signature[tag]['dependencies'],
        'arguments':[helper.view_info(x) for x in call.src[1:] if not x.is_variable]})
    report['hcq_batches'].append({'phase':report['active_phase'],'batch_association':bid,'canonical':signature,'calls':rows})
    assert bid in plan['batches'], ('Unprepared exact call/dependency binding',bid,signature)
    assert signature==plan['batches'][bid]
    return old_queues(ctx)
  def kernargs(queue,call,program,data):
    aid,item=identity.authorize(program,plan)
    values=get_call_kernel_args(call,program,queue.devs)
    from tinygrad.runtime.support.hcq2 import layout_args
    views=[helper.view_info(x) for x in call.src[1:]]
    name=program.src[0].arg.name
    layout=[offset for offset,_ in layout_args(values)]
    report.setdefault('all_argument_records',[]).append({'phase':report['active_phase'],'association':aid,
      'pointer_offsets':layout,'kernarg_bytes':data.kernargs_segment_size,'arguments':views,
      'dispatch_pointer':data.enable_dispatch_ptr,'CU_mode_bit':(data.rsrc1>>29)&1})
    assert asdict(data)==item['descriptor']
    if name in ('agent_attention_rowsoftmax','alpamayo_ck_d128'):
      assert views[0]['offset_bytes']==512 and views[0]['bytes']==helper.OC*2
      if name=='alpamayo_ck_d128': assert views[1]['offset_bytes']==1024 and views[1]['bytes']==helper.LC*4
      report['native_argument_records'].append({'phase':report['active_phase'],'name':name,'association':aid,
        'pointer_offsets':layout,'kernarg_bytes':data.kernargs_segment_size,
        'dispatch_pointer':data.enable_dispatch_ptr,'CU_mode_bit':(data.rsrc1>>29)&1,'arguments':views,
        'CK_implicit_hex':helper.load('agent-ck-attention-v2-adapter').implicit_args().hex() if name=='alpamayo_ck_d128' else None})
    return old_args(queue,call,program,data)
  ops_amd.amd_build_program,hcq2._build_queues,ops_amd.AMDComputeQueue.kernargs,ops_amd.dispatch_packet=build,queues,kernargs,packet
  try: yield
  finally:
    assert ops_amd.amd_build_program is build and hcq2._build_queues is queues and ops_amd.AMDComputeQueue.kernargs is kernargs
    ops_amd.amd_build_program,hcq2._build_queues,ops_amd.AMDComputeQueue.kernargs,ops_amd.dispatch_packet=old_build,old_queues,old_args,old_packet

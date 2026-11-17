"""CPU/offline compiler checks of actual graphs and deliberate queue-proof mutations."""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def load(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_')+'_check',path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--artifacts',type=Path,default=HERE/'agent-residual-norm-queue-cpu')
  args=parser.parse_args();args.artifacts.mkdir(parents=True,exist_ok=True)
  capture=load(HERE/'agent-residual-norm-queue-capture.py');proof=load(HERE/'agent-residual-norm-queue-proof.py')
  result,contexts=capture.capture(args.artifacts,proof)
  runner=load(HERE/'agent-residual-norm-queue-run.py')
  retry,buffers=runner.dependencies();binding,samples,plan,old=retry.verify_capture()
  _,adaptation=runner.build_micro(retry,buffers,old,binding,[],proof,{})
  from tinygrad import Device,dtypes
  from tinygrad.uop.ops import Ops,UOp
  from tinygrad.runtime.support import hcq2
  from tinygrad.engine.jit import _prepare_jit_inputs
  mutations=[]
  def rejects(name,function):
    try:function()
    except AssertionError as error:mutations.append(dict(case=name,rejected=True,message=str(error)))
    else:raise AssertionError('Mutation was accepted: '+name)
  for ctx,queues,label,inputs in contexts:
    qkey=next(iter(queues))
    # Every barrier in these actual graphs protects at least one required hazard.
    removed=0
    for index,command in enumerate(queues[qkey]):
      if command.op is Ops.CALL and command.body.op is Ops.CUSTOM_FUNCTION and command.body.arg.name=='exec_barrier':
        altered={qkey:queues[qkey][:index]+queues[qkey][index+1:]}
        rejects(label+'/remove-barrier-'+str(removed),lambda:proof.audit(ctx,altered,label,inputs));removed+=1
    assert removed==(15 if label=='candidate' else 46)
    def altered_batch(calls):
      changed=replace(ctx,batch=[(call,device,queue) for call,(_,device,queue) in zip(calls,ctx.batch)])
      return proof.audit(changed,hcq2._build_queues(changed),label,inputs)
    calls=[call for call,_,_ in ctx.batch]
    # Preserve the real programs and all other actual arguments in each mutation.
    changed=calls[:];call=changed[1 if label=='candidate' else 3]
    sources=list(call.src);sources[3 if label=='candidate' else 2]=calls[0].src[3 if label=='candidate' else 2]
    changed[1 if label=='candidate' else 3]=call.replace(src=tuple(sources))
    rejects(label+'/break-running-residual',lambda:altered_batch(changed))
    changed=calls[:];tag=1 if label=='candidate' else 5;call=changed[tag]
    sources=list(call.src);sources[2 if label=='candidate' else 1]=calls[0 if label=='candidate' else 2].src[2 if label=='candidate' else 1]
    changed[tag]=call.replace(src=tuple(sources))
    rejects(label+'/alias-retained-packed-outputs',lambda:altered_batch(changed))
    rejects(label+'/missing-program',lambda:proof.audit(replace(ctx,batch=ctx.batch[:-1]),queues,label,inputs))
    changed_queue={qkey:list(queues[qkey])};programs=[i for i,c in enumerate(changed_queue[qkey]) if c.op is Ops.CALL and c.body.op is Ops.PROGRAM]
    first,second=programs[:2];changed_queue[qkey][first],changed_queue[qkey][second]=changed_queue[qkey][second],changed_queue[qkey][first]
    rejects(label+'/reordered-programs',lambda:proof.audit(ctx,changed_queue,label,inputs))
    if label=='candidate':
      changed=calls[:];call=changed[0]
      body=call.body.replace(arg=replace(call.body.arg,outs=(0,)))
      changed[0]=call.replace(src=(body,*call.src[1:]))
      rejects('candidate/missing-second-output-write',lambda:altered_batch(changed))
  # Independently check physical Buffer views, not only identical UOp roots.
  inputs=contexts[0][3];root=_prepare_jit_inputs(inputs,{})[0][0].buffer
  view=root.view(64,256)
  resolve=proof.resolver([])
  full=resolve(UOp.from_buffer(root,dtypes.uint8,root.device))
  part=resolve(UOp.from_buffer(view,dtypes.uint8,root.device))
  assert part==(full[0],256,64) and proof.overlaps(full,part)
  assert not proof.overlaps((full[0],0,256),part)
  assert proof.overlaps((full[0],255,2),part)
  rejects('unsupported-constant-root',lambda:resolve(UOp.const(1,dtypes.float32)))
  original=hcq2._build_queues
  try:
    with proof.observe([], 'candidate',contexts[1][3]):raise RuntimeError('injected failure')
  except RuntimeError:pass
  assert hcq2._build_queues is original
  assert not any(name.startswith(('AMD','USB')) for name in Device._opened_devices)
  result.update(negative_controls=mutations,physical_buffer_alias_views=True,hook_restored_after_failure=True,
    launcher_adapter=adaptation,
    tinygrad_source_sha256={name:hashlib.sha256((capture.TINY/name).read_bytes()).hexdigest() for name in
      ('tinygrad/runtime/support/hcq2.py','tinygrad/engine/jit.py','tinygrad/engine/realize.py',
       'tinygrad/schedule/memory.py','tinygrad/uop/ops.py')},
    source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
      runner.FILES})
  path=args.artifacts/'agent-residual-norm-queue-check.json'
  path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(json.dumps(dict(output=str(path),gpu_executed=False,numeric_execution=False,negative_controls=len(mutations),
    graphs=[dict(label=r['label'],kernel_calls=r['kernel_calls'],hazards=r['proof']['hazard_counts'],
      barriers=len(r['proof']['barrier_before_tags'])) for r in result['records']]),indent=2))


if __name__=='__main__':main()

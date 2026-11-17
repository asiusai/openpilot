"""Full-width tinygrad CPU forward/backward and independent causal/padding controls. No updates."""
import argparse
import importlib
import json
from pathlib import Path
import resource
import sys
import time

C=importlib.import_module('agent-distill-tinygrad-common')

def execute(device,controls=True):
  import numpy as np
  from tinygrad import Tensor,Device
  from tinygrad.helpers import GlobalCounters,USE_TC,ALLOW_TF32,TRAINING
  assert USE_TC.value==0 and ALLOW_TF32.value==0 and TRAINING.value==0
  oracle,values=C.arrays()
  block=C.Block(values,oracle['config'],device)
  tensor=lambda a:Tensor(a.copy(),device=device)
  x=tensor(values['x']);cos=tensor(values['cos']);sin=tensor(values['sin'])
  targets=tuple(tensor(values['target:'+k]) for k in ('y','k','v'))
  # Keep weights as stable leaf buffers, then derive gradients before realizing the output graph.
  params=list(block.weights.values());params[0].realize(*params[1:],x,cos,sin,*targets)
  outputs,activation=block(x,cos,sin);loss=C.recovery_loss(outputs,targets)
  grads=loss.gradient(*params,x)
  # One explicit reverse-mode request for all eight parameter tensors plus input; no optimizer.
  loss.realize(*outputs,activation,*grads)
  result={'forward':{},'gradients':{},'checks':{},'opened_devices':[]}
  for name,t in [*zip(('y','k','v'),outputs),('activation',activation),('loss',loss)]:
    expected=values['out:'+name] if name in ('y','k','v') else values[name]
    result['forward'][name]=C.compare(t.numpy(),expected,'forward')
  names=[*block.weights,'x'];actual_grads={}
  for name,t in zip(names,grads,strict=True):
    a=t.numpy();actual_grads[name]=a
    result['gradients'][name]=C.compare(a,values['grad:'+name],'gradient')
    assert np.isfinite(a).all() and np.any(a!=0),name
  delta=(outputs[0].numpy()[:,:3]-values['target:y'][:,:3]).reshape(3,2048)
  derivative=(np.float32(2/delta.size)*delta.T)@activation.numpy()[:,:3].reshape(3,9216)
  result['fc2_independent_derivative']=C.compare(actual_grads['mlp.fc2.weight'],derivative,'fc2_derivative')
  result['checks']['padding_input_gradient_zero']=bool(np.array_equal(actual_grads['x'][:,3:],np.zeros_like(actual_grads['x'][:,3:])))
  result['checks']['weights_unchanged']=all(np.array_equal(p.numpy(),values['w:'+name]) for name,p in block.weights.items())
  if controls:
    from tinygrad.codegen import to_program_cache
    def forward_only(array,causal=True):
      inp=tensor(array).realize()
      out,act=block(inp,cos,sin,causal=causal)
      out[0].realize(*out[1:],act)
      return out[0].numpy()
    # Compare identical forward-only realization boundaries. The jointly realized
    # backward above can require different fusion boundaries without changing causality.
    baseline=forward_only(values['x'])
    prior_programs={p.key.hex() for p in to_program_cache.values()}
    future=values['x'].copy();future[:,3:]+=1
    changed=forward_only(future)
    result['checks']['causal_control_same_compiled_programs']=prior_programs=={p.key.hex() for p in to_program_cache.values()}
    result['checks']['causal_future_exact']=bool(np.array_equal(changed[:,:3],baseline[:,:3]))
    result['causal_control_max_absolute']=float(np.max(np.abs(changed[:,:3]-baseline[:,:3])))
    bad=forward_only(values['x'],causal=False)
    bad_changed=forward_only(future,causal=False)
    result['checks']['noncausal_negative_detected']=bool(np.max(np.abs(bad_changed[:,:3]-bad[:,:3]))>1e-4)
    padded=[]
    for key in ('y','k','v'):
      a=values['target:'+key].copy()
      if key=='y':a[:,3:]+=999
      else:a[:,:,3:]+=999
      padded.append(tensor(a))
    result['checks']['padding_target_exact']=bool(np.array_equal(C.recovery_loss(outputs,padded).numpy(),loss.numpy()))
    result['checks']['included_padding_negative_detected']=bool(float(C.recovery_loss(outputs,padded,4).item())>float(loss.item())+1)
    wrong_rope=block(x,tensor(np.ones_like(values['cos'])),tensor(np.zeros_like(values['sin'])))[0][0].numpy()
    result['checks']['omitted_rope_negative_detected']=not C.compare(wrong_rope,values['out:y'],'forward')['passed']
    result['checks']['zero_gradient_negative_detected']=not C.compare(np.zeros_like(actual_grads['mlp.fc2.weight']),values['grad:mlp.fc2.weight'],'gradient')['passed']
  result['all_passed']=all(x['passed'] for g in ('forward','gradients') for x in result[g].values()) and result['fc2_independent_derivative']['passed'] and all(result['checks'].values())
  result['opened_devices']=sorted(Device._opened_devices)
  result['tinygrad_live_bytes']=dict(GlobalCounters.mem_used_per_device)
  result['parameter_count']=50335744;result['optimizer_updates']=0;result['controls_executed']=controls
  return result

def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  assert a.output.is_relative_to(C.ROOT) and not a.output.exists();C.no_gpu_fds()
  def deny_hardware_open(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      path=args[0].decode() if isinstance(args[0],bytes) else args[0]
      assert not path.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/')),path
  sys.addaudithook(deny_hardware_open)
  from tinygrad.helpers import DEV
  assert DEV.device=='CPU' and DEV.renderer=='LLVM'
  start=time.monotonic();result=execute('CPU')
  C.no_gpu_fds();assert all(k.split(':')[0] in ('CPU','NPY','DISK') for k in result['opened_devices'])
  result.update(complete=True,gpu_executed=False,seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    tolerances=C.TOLERANCES,oracle_sha256=C.file_sha(C.ROOT/'oracle.json'),source_sha256={str(p):C.file_sha(p) for p in
      (Path(__file__),C.EXP/'agent-distill-tinygrad-common.py')})
  C.write(a.output,result)
  print(json.dumps(result,indent=2));assert result['all_passed'],'Predeclared numeric or independent control gate failed'

if __name__=='__main__':main()

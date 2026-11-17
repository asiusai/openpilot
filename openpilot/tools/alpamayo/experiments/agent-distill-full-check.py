"""One full L2048 reverse-mode diagnostic, two matched causal forwards, and no optimizer."""
import importlib
import time

C=importlib.import_module('agent-distill-full-common')

def execute(device,stage):
  import numpy as np
  from tinygrad import Tensor,Device
  from tinygrad.helpers import GlobalCounters,USE_TC,ALLOW_TF32,TRAINING,NO_MEMORY_PLANNER
  from tinygrad.codegen import to_program_cache
  assert USE_TC.value==ALLOW_TF32.value==TRAINING.value==0 and NO_MEMORY_PLANNER.value==1
  oracle,values=C.array_manifest();cfg=oracle['config'];assert oracle['tolerances']==C.TOLERANCES
  assert (cfg['hidden_size'],cfg['intermediate_size'],cfg['num_attention_heads'],cfg['num_key_value_heads'],cfg['head_dim'])==(2048,9216,16,8,128)
  assert cfg['hidden_act']=='relu2' and cfg['attention_dropout']==0 and not cfg['attention_bias'] and not cfg['mlp_bias']
  assert values['x'].shape==(1,C.LENGTH,2048) and values['cos'].shape==values['sin'].shape==(1,C.LENGTH,128)
  block=C.block(values,device);tensor=lambda a:Tensor(a.copy(),device=device)
  x=tensor(values['x']);cos=tensor(values['cos']);sin=tensor(values['sin'])
  targets=tuple(tensor(values['target:'+name]) for name in ('y','k','v'))
  params=list(block.weights.values());params[0].realize(*params[1:],x,cos,sin,*targets)
  stage('inputs-resident');begin=time.monotonic()
  outputs,activation=block(x,cos,sin);objective=C.loss(outputs,targets)
  gradients=objective.gradient(*params,x)
  objective.realize(*outputs,activation,*gradients);Device[device].synchronize()
  result={'forward':{},'gradients':{},'checks':{},'forward_backward_compile_and_execute_seconds':time.monotonic()-begin}
  stage('forward-backward-complete')
  actual_outputs=[t.numpy() for t in outputs];actual_activation=activation.numpy();actual_loss=objective.numpy()
  for name,a in [*zip(('y','k','v'),actual_outputs),('activation',actual_activation),('loss',actual_loss)]:
    expected=values['out:'+name] if name in ('y','k','v') else values[name]
    result['forward'][name]=C.B.compare(a,expected,'forward')
  actual_grads={}
  for name,t in zip([*block.weights,'x'],gradients,strict=True):
    a=t.numpy();assert np.isfinite(a).all() and np.any(a!=0),name
    actual_grads[name]=a;result['gradients'][name]=C.B.compare(a,values['grad:'+name],'gradient')
  delta=(actual_outputs[0][:,:C.VALID_ROWS]-values['target:y'][:,:C.VALID_ROWS]).reshape(C.VALID_ROWS,2048)
  derivative=(np.float32(2/delta.size)*delta.T)@actual_activation[:,:C.VALID_ROWS].reshape(C.VALID_ROWS,9216)
  result['fc2_independent_derivative']=C.B.compare(actual_grads['mlp.fc2.weight'],derivative,'fc2_derivative')
  result['checks']['padding_input_gradient_exact_zero']=bool(np.array_equal(actual_grads['x'][:,C.VALID_ROWS:],np.zeros_like(actual_grads['x'][:,C.VALID_ROWS:])))
  result['checks']['zero_gradient_negative_detected']=not C.B.compare(np.zeros_like(actual_grads['mlp.fc2.weight']),values['grad:mlp.fc2.weight'],'gradient')['passed']
  assert all(v['passed'] for g in ('forward','gradients') for v in result[g].values()),result
  assert result['fc2_independent_derivative']['passed'] and all(result['checks'].values()),result
  stage('all-full-gradient-comparisons-passed')
  # Recompute both scalar losses from identical already-realized outputs. This avoids
  # comparing different forward/backward fusion boundaries for the exact padding check.
  padded=[]
  for name in ('y','k','v'):
    a=values['target:'+name].copy()
    if name=='y':a[:,C.VALID_ROWS:]+=999
    else:a[:,:,C.VALID_ROWS:]+=999
    padded.append(tensor(a))
  base_loss=C.loss(outputs,targets).numpy();changed_loss=C.loss(outputs,padded).numpy()
  result['checks']['padding_target_exact']=bool(np.array_equal(base_loss,changed_loss))
  result['checks']['included_padding_negative_detected']=bool(float(C.loss(outputs,padded,C.LENGTH).item())>float(base_loss)+1)
  def forward_only(array):
    inp=tensor(array).realize();out,act=block(inp,cos,sin);out[0].realize(*out[1:],act)
    return out[0].numpy()
  baseline=forward_only(values['x']);prior_programs={p.key.hex() for p in to_program_cache.values()}
  future=values['x'].copy();future[:,C.VALID_ROWS:]+=1
  changed=forward_only(future)
  result['checks']['causal_same_compiled_programs']=prior_programs=={p.key.hex() for p in to_program_cache.values()}
  result['checks']['causal_future_exact']=bool(np.array_equal(changed[:,:C.VALID_ROWS],baseline[:,:C.VALID_ROWS]))
  result['checks']['future_mutation_reached_output']=bool(not np.array_equal(changed[:,C.VALID_ROWS:],baseline[:,C.VALID_ROWS:]))
  result['causal_max_absolute']=float(np.max(np.abs(changed[:,:C.VALID_ROWS]-baseline[:,:C.VALID_ROWS])))
  # The native CPU full-length oracle independently rejects removal of the causal mask.
  result['checks']['native_full_length_noncausal_negative']=oracle['noncausal_negative_max_absolute']>1e-4
  result['checks']['weights_bitwise_unchanged']=all(np.array_equal(p.numpy().view(np.uint32),values['w:'+name].view(np.uint32)) for name,p in block.weights.items())
  stage('all-controls-and-parameter-readbacks-complete')
  result.update(all_passed=all(result['checks'].values()),optimizer_updates=0,length=C.LENGTH,valid_rows=C.VALID_ROWS,
    parameter_count=50335744,parameter_gradient_tensors=len(params),input_gradient_tensors=1,
    tinygrad_live_bytes=dict(GlobalCounters.mem_used_per_device),opened_devices=sorted(Device._opened_devices),
    scope='Synthetic FP32 native-teacher plumbing; no route quality, BF16 training, optimizer, or steady-state throughput claim')
  assert result['all_passed'];return result

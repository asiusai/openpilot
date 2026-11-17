"""CPU connected-suffix numerical/gradient proof, actual capture entrypoint mock, and full-shape schedule accounting."""
import argparse
import inspect
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode=True;HERE=Path(__file__).resolve().parent;BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
os.environ.update(DEV='CPU',TC='0',ALLOW_TF32='0',NO_MEMORY_PLANNER='1',SCACHE='0',LRU='0',
  CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and (os.fsdecode(args[0]).startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')) or
      any(s in os.fsdecode(args[0]) for s in ('0000008e--321b2ddc03','00000090--1cf3f7d926'))):raise RuntimeError('CPU motion qualification prohibited access')
sys.addaudithook(guard);sys.path.insert(0,str(BASE/'tinygrad'))
import numpy as np
import torch
torch.set_num_threads(4);torch.set_default_device('cpu')
from tinygrad import Tensor,Device,dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
M=runpy.run_path(str(HERE/'agent-distill-motion-objective-core.py'))
old=M['module']('motion_prior_math_test',HERE/'agent-distill-objective-check.py')


def numerical():
  rng=np.random.default_rng(1128);weights=lambda:old.weights(rng,8,16)
  student_np=weights();frozen_np={i:weights() for i in M['SOURCES']}
  t=lambda value:Tensor(value.copy())
  student={k:t(v).is_param_(True) for k,v in student_np.items()}
  frozen=dict(layers={i:{k:t(v) for k,v in w.items()} for i,w in frozen_np.items()},norm=Tensor.ones(8),
    head_weight=t(rng.normal(0,.1,(2,8)).astype(np.float32)),head_bias=t(rng.normal(0,.03,(2,)).astype(np.float32)))
  random=lambda shape:t(rng.normal(0,.2,shape).astype(np.float32))
  capacity=24;valid=17;rows=4;contexts={slot:(random((1,2,capacity,4)),random((1,2,capacity,4))) for slot in M['SLOTS']}
  mask=Tensor(np.r_[np.zeros(valid),np.full(capacity-valid,-np.inf),np.zeros(rows)].astype(np.float32));steps=[]
  for i in range(3):
    angles=rng.normal(0,.2,(1,1,rows,4)).astype(np.float32)
    steps.append(dict(x=random((1,rows,8)),cos=t(np.cos(angles)),sin=t(np.sin(angles)),mask=mask,target_velocity=random((1,rows,2))))
  loss,out=M['objective'](student,contexts,frozen,steps,head_dim=4)
  grads=loss.gradient(*student.values());actual=[g.numpy() for g in grads]
  ts={k:torch.tensor(v,requires_grad=True) for k,v in student_np.items()};ref=lambda value:torch.tensor(value.numpy())
  tf={i:{k:torch.tensor(v) for k,v in w.items()} for i,w in frozen_np.items()};tc={i:tuple(ref(v) for v in pair) for i,pair in contexts.items()}
  velocities=[];losses=[]
  for step in steps:
    c,s,m=(ref(step[k]) for k in ('cos','sin','mask'))
    x=old.torch_block(ts,ref(step['x']),c,s,tc[12],m)[0]
    for source in M['SOURCES']:x=old.torch_block(tf[source],x,c,s,tc[source-3],m)[0]
    h=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*ref(frozen['norm'])
    velocity=h@ref(frozen['head_weight']).T+ref(frozen['head_bias']);velocities.append(velocity)
    losses.append((velocity-ref(step['target_velocity'])).square().mean())
  expected=torch.stack(losses).mean();expected.backward()
  np.testing.assert_allclose(loss.item(),float(expected.detach()),rtol=3e-6,atol=2e-7)
  for a,b in zip(out['velocity'],velocities,strict=True):np.testing.assert_allclose(a.numpy(),b.detach().numpy(),rtol=5e-5,atol=3e-7)
  errors={}
  for (name,value),gradient in zip(ts.items(),actual,strict=True):
    np.testing.assert_allclose(gradient,value.grad.numpy(),atol=3e-7,rtol=4e-4)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient)>0
    errors[name]=float(np.linalg.norm(gradient-value.grad.numpy())/max(np.linalg.norm(value.grad.numpy()),1e-30))
  # Removing the path through the frozen suffix must remove all student gradients.
  detached=Tensor.stack(*[(M['suffix'](y.detach(),step,contexts,frozen,4)-step['target_velocity']).square().mean()
    for y,step in zip(out['student_outputs'],steps,strict=True)]).mean()
  assert all(np.count_nonzero(g.numpy())==0 for g in detached.gradient(*student.values()))
  # Frozen constants are detached, but the student values passed through them are not.
  assert all(np.count_nonzero(g.numpy())==0 for g in loss.gradient(frozen['head_weight'],frozen['layers'][27]['mlp.fc2.weight']))
  for g in loss.gradient(*contexts[24]):assert np.count_nonzero(g.numpy()[:,:,valid:])==0 and np.linalg.norm(g.numpy()[:,:,:valid])>0
  name='mlp.fc2.weight';index=np.unravel_index(np.argmax(np.abs(actual[list(student).index(name)])),student_np[name].shape)
  delta=1e-3;values=[]
  for sign in (-1,1):
    arrays={k:v.copy() for k,v in student_np.items()};arrays[name][index]+=sign*delta
    value,_=M['objective']({k:t(v).is_param_(True) for k,v in arrays.items()},contexts,frozen,steps,head_dim=4);values.append(float(value.item()))
  difference=(values[1]-values[0])/(2*delta);analytic=float(actual[list(student).index(name)][index])
  np.testing.assert_allclose(difference,analytic,rtol=.02,atol=2e-5)
  # Native final normalization/projection formulas, including the actual bias path.
  runtime=Path(old.C['read'](old.C['PLAN'])['runtime_root']);sys.path.insert(0,str(runtime))
  Native=M['module']('motion_native_head',runtime/'model.py').EdgeModel;native=Native.__new__(Native)
  native.fp8_wmma=native.cuda_kernels=False;native.weight_scales={}
  native.weights={'expert.expert.norm.weight':frozen['norm'],'expert.action_out_proj.weight':frozen['head_weight'],'expert.action_out_proj.bias':frozen['head_bias']}
  x=random((1,4,8));actual_head=native.linear(native.norm(x,'expert.expert.norm',rms=True),'expert.action_out_proj')
  expected_head=(x*(x.square().mean(-1,keepdim=True)+1e-5).rsqrt()*frozen['norm'])@frozen['head_weight'].T+frozen['head_bias']
  np.testing.assert_allclose(actual_head.numpy(),expected_head.numpy(),atol=2e-7,rtol=3e-6)
  return dict(final_flow_loss=float(loss.item()),torch_loss=float(expected.detach()),frozen_suffix_layers=12,
    all8_student_gradients_finite_nonzero=True,max_gradient_relative_l2=max(errors.values()),finite_difference=difference,analytic=analytic,
    detached_student_output_negative_zero=True,frozen_weight_gradient_zero=True,masked_context_gradient_zero=True,
    native_final_norm_head_with_bias_matches=True,gradient_reaches_language=False,optimizer_updates=0)


def schedule():
  from tinygrad.helpers import NO_MEMORY_PLANNER
  def empty(shape):return Tensor.empty(shape)
  student={k:empty(shape).is_param_(True) for k,shape in M['SHAPES'].items()}
  frozen=dict(layers={i:{k:empty(shape) for k,shape in M['SHAPES'].items()} for i in M['SOURCES']},
    norm=empty((512,)),head_weight=empty((2,512)),head_bias=empty((2,)))
  contexts={i:(empty((1,8,2304,128)),empty((1,8,2304,128))) for i in M['SLOTS']}
  steps=[dict(x=empty((1,64,512)),cos=empty((1,1,64,128)),sin=empty((1,1,64,128)),mask=empty((2368,)),target_velocity=empty((1,64,2))) for _ in range(3)]
  with Context(TRAINING=1,NO_MEMORY_PLANNER=1):
    loss,out=M['objective'](student,contexts,frozen,steps);gradients=loss.gradient(*student.values())
    assert [t.shape for t in out['velocity']]==[(1,64,2)]*3
    linear=loss.schedule_linear(*out['velocity'],*gradients)
  buffers={}
  for call in linear.src:
    assert call.op is Ops.CALL
    for value in call.src[1:]:
      base=value.buf_uop
      if base.op is Ops.BUFFER:buffers[base]=base.max_numel()*base.dtype.itemsize
  return dict(full64_full2304_three_teacher_states=True,kernels_executed=0,scheduled_kernels=len(linear.src),
    distinct_logical_buffers=len(buffers),total_distinct_logical_buffer_bytes=sum(buffers.values()),largest_logical_buffer_bytes=max(buffers.values()),
    largest_ten_logical_buffers=sorted(buffers.values(),reverse=True)[:10],no_memory_planner=1,
    actual_AMD_peak_cumulative_and_scratch_unmeasured=True,analytic=M['estimate']())


def with_frozen(initialize):
  def wrapped(*args,**kwargs):
    components=initialize(*args,**kwargs);model=components.model
    for source in M['SOURCES']:
      for name,shape in M['SHAPES'].items():model.weights[f'expert.expert.layers.{source}.'+name]=Tensor.full(shape,1 if 'layernorm' in name else .01)
    full='expert.expert.layers.16.self_attn.q_proj.weight'
    model.weights[full]=Tensor.full(M['SHAPES']['self_attn.q_proj.weight'],.25,dtype=dtypes.fp8e4m3)
    model.weight_scales[full]=Tensor([.5])
    model.weights.update({'expert.expert.norm.weight':Tensor.ones(512),
      'expert.action_out_proj.weight':Tensor.full((2,512),.01),'expert.action_out_proj.bias':Tensor.zeros(2)})
    return components
  return wrapped


def integration(root):
  # Reuse the qualified token-control entrypoint fixture; redirect only to the new capture wrapper and add frozen constants.
  check=M['module']('motion_token_fixture',HERE/'agent-distill-token-control-check.py')
  t=check.T;paired,receipt=t['E'].selected_joint(t['E'].ACTUAL_ROOT,t['E'].ACTUAL_REPORT_SHA)
  plan=t['read'](t['C'].PLAN);row=t['P'].selected_rows(t['P'].INPUTS,['train-command-0'])[0]
  refs=t['references'](plan,row,receipt)
  source=inspect.getsource(check.integration)
  before="namespace=runpy.run_path(str(HERE/'agent-distill-token-control-run.py'))"
  assert source.count(before)==1;source=source.replace(before,"namespace=runpy.run_path(str(HERE/'agent-distill-motion-objective-capture.py'))")
  before='initialize=old.initializer(plan,ready[1],audit)';assert source.count(before)==1
  source=source.replace(before,'initialize=with_frozen(old.initializer(plan,ready[1],audit))')
  check.with_frozen=with_frozen;exec(compile(source,str(Path(__file__))+':capture-entrypoint','exec'),check.__dict__)
  result=check.integration(root,paired,refs);report=M['read'](root/'report.json')
  capture=report['motion_capture'];assert capture['complete'] and len(capture['tensors'])==139
  assert all(step['compact_layer_order']==list(range(25)) for step in capture['steps'])
  assert len(report['forced']['rollout']['steps'])==3 and capture['fixed_teacher_actions'] and not capture['free_running_training']
  contexts,frozen,steps,_=M['load_capture'](root/'report.json',M['sha'](root/'report.json'),'CPU',fixture=True)
  assert len(contexts)==13 and len(frozen['layers'])==12 and len(steps)==3
  np.testing.assert_array_equal(frozen['layers'][16]['self_attn.q_proj.weight'].numpy(),np.full((2048,512),.125,np.float32))
  assert capture['weight_storage']['layer.16.self_attn.q_proj.weight']['dequantized']
  try:M['load_capture'](root/'report.json',M['sha'](root/'report.json'),'CPU')
  except ValueError:pass
  else:raise AssertionError('Mock capture admitted for real qualification')
  result.update(new_capture_entrypoint=True,all139_arrays_loaded_with_shape_hash_finite_checks=True,three_fixed_conditions_separate_from_own_Euler=True,
    actual_selected_checkpoint_loaded=True,mocked_neural_weights_and_arithmetic=True,
    deployed_fp8_weight_times_scale_capture_checked=True,mock_rejected_for_real_qualification=True)
  return result


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(M['ROOT']):raise ValueError('Fresh owned CPU directory required')
  root.mkdir(parents=True)
  result=dict(gpu_executed=False,optimizer_updates=0,numerical=numerical(),full_shape_schedule=schedule())
  result['integration']=integration(root/'integration')
  assert all(name.startswith(('CPU','NPY','PYTHON','DISK')) for name in Device._opened_devices)
  result.update(all_passed=True,opened_devices=list(Device._opened_devices),source_sha256={str(p):M['sha'](p) for p in HERE.glob('agent-distill-motion-objective-*.py')})
  (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

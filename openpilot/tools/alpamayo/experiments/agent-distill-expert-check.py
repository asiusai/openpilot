"""Full-shape CPU block/gradient checks and synthetic-only optimizer recovery controls."""
import argparse
import copy
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir',type=Path,required=True)
  parser.add_argument('--completed-train-report',type=Path)
  parser.add_argument('--completed-train-sha')
  parser.add_argument('--completed-validation-report',type=Path)
  parser.add_argument('--completed-validation-sha')
  args=parser.parse_args();root=args.output_dir.resolve()
  assert root.is_relative_to(WORK/'agent-distill-expert-artifacts');root.mkdir(parents=True,exist_ok=False)
  os.environ.update(DEV='CPU:LLVM',TC='0',ALLOW_TF32='0',TRAINING='0',PROFILE='0',PMC='',SQTT='',VIZ='0',
    OPTIM_DTYPE='float32',FUSE_OPTIM='0',CONST_LR='0',PYTHONDONTWRITEBYTECODE='1',
    CACHEDB=str(root/'cpu-cache.db'),TMPDIR=str(root),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',
    OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
  def guard(event,values):
    if event!='open' or not isinstance(values[0],(str,bytes)):return
    path=os.fsdecode(values[0])
    assert not path.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb'))
    assert not any(word in path for word in ('0000008e--321b2ddc03','00000090--1cf3f7d926','sealed_test'))
  sys.addaudithook(guard)
  sys.path.insert(0,str(BASE/'tinygrad'))
  import numpy as np
  import torch
  torch.set_num_threads(4);torch.set_default_device('cpu')
  from tinygrad import Tensor,Device,dtypes
  from tinygrad.helpers import Context
  B=runpy.run_path(str(HERE/'agent-distill-expert-block.py'))
  T=B['module']('agent_distill_expert_train_check',HERE/'agent-distill-expert-train.py')
  block_module=T.B
  native_root=WORK/'agent-distill-command-artifacts/capture-v1/runtime'
  sys.path.insert(0,str(native_root))
  Native=runpy.run_path(str(native_root/'model.py'))['EdgeModel']
  started=time.monotonic();weights=B['seed']();block=B['Block'](weights,'CPU')
  rng=np.random.default_rng(20261117)
  def synthetic(seed):
    rng=np.random.default_rng(seed)
    angles=np.repeat(rng.normal(0,.1,(1,1,64,64)).astype(np.float32),2,axis=-1)
    values={k:rng.normal(0,.1,shape).astype(np.float32) for k,shape in B['SHAPES'].items()}
    values.update(cos=np.cos(angles),sin=np.sin(angles),mask=np.full(2368,-np.inf,np.float32),timestep=np.zeros(1,np.float32))
    values['mask'][:2033]=0;values['mask'][2304:]=0
    return values
  arrays=synthetic(1);data=B['inputs'](arrays,'CPU')
  original=Native.__new__(Native);original.dtype=dtypes.float32;original.cuda_kernels=original.fp8_wmma=False
  original.weight_scales={};prefix='expert.expert.layers.12'
  original.weights={prefix+'.'+name:value for name,value in block.weights.items()}
  native=original.text_layer(data['x'],data['cos'],data['sin'],prefix,[data['context_k'],data['context_v']],data['mask']).numpy()
  output=B['forward'](block,data).numpy()
  np.testing.assert_allclose(output,native,rtol=5e-4,atol=3e-5)
  result=dict(gpu_executed=False,all_passed=False,actual_data_optimizer_updates=0,
    native_forward=dict(shape=list(output.shape),max_absolute=float(np.max(np.abs(output-native)))),
    parameter_count=sum(value.size for value in weights.values()),context_keys=2304,action_rows=64)
  print('Native full-shape CPU forward passed',flush=True)
  # Independent PyTorch FP32 forward and all parameter gradients on the same full shapes.
  tw={name:torch.tensor(value,requires_grad=True) for name,value in sorted(weights.items())}
  td={name:torch.tensor(value,requires_grad=name=='x') for name,value in arrays.items()}
  def linear(x,name):return x@tw[name+'.weight'].T
  def norm(x,name):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*tw[name+'.weight']
  h=norm(td['x'],'input_layernorm')
  q,k,v=[linear(h,'self_attn.'+name+'_proj').reshape(1,64,heads,128).transpose(1,2) for name,heads in (('q',16),('k',8),('v',8))]
  def rotate(t):return torch.cat((-t[...,64:],t[...,:64]),dim=-1)
  q,k=q*td['cos']+rotate(q)*td['sin'],k*td['cos']+rotate(k)*td['sin']
  k=torch.cat((td['context_k'],k),dim=2).repeat_interleave(2,dim=1)
  v=torch.cat((td['context_v'],v),dim=2).repeat_interleave(2,dim=1)
  probabilities=torch.softmax(q@k.transpose(-2,-1)*(128**-.5)+td['mask'],dim=-1)
  residual=td['x']+linear((probabilities@v).transpose(1,2).reshape(1,64,2048),'self_attn.o_proj')
  ty=residual+linear(torch.relu(linear(norm(residual,'post_attention_layernorm'),'mlp.fc1')).square(),'mlp.fc2')
  tloss=(ty-td['y']).square().mean();tloss.backward()
  np.testing.assert_allclose(output,ty.detach().numpy(),rtol=5e-4,atol=3e-5)
  objective=B['loss'](block,data)
  targets=[*block.weights.values(),data['x'],data['context_k'],data['context_v'],data['y']]
  gradients=objective.gradient(*targets);Tensor.realize(*gradients)
  stats={}
  for name,actual,expected in zip([*block.weights,'x'],gradients,[*(value.grad.numpy() for value in tw.values()),td['x'].grad.numpy()]):
    a=actual.numpy();np.testing.assert_allclose(a,expected,rtol=3e-3,atol=3e-7)
    relative=float(np.linalg.norm((a-expected).ravel().astype(np.float64))/max(np.linalg.norm(expected.ravel().astype(np.float64)),1e-30))
    assert relative<2e-4
    stats[name]=dict(max_absolute=float(np.abs(a-expected).max()),relative_l2=relative)
  assert all(np.count_nonzero(g.numpy())==0 for g in gradients[-3:])
  result['torch_forward_max_absolute']=float(np.abs(output-ty.detach().numpy()).max())
  result['gradient_comparison']=stats;result['teacher_context_and_target_gradients_zero']=True
  print('Independent PyTorch full-shape forward/gradient passed',flush=True)
  # Masked content cannot affect results. A late visible key must still affect output.
  masked={k:v.copy() for k,v in arrays.items()};masked['context_k'][:,:,2033:]=100;masked['context_v'][:,:,2033:]=-200
  altered=B['forward'](block,B['inputs'](masked,'CPU')).numpy();np.testing.assert_array_equal(output,altered)
  visible={k:v.copy() for k,v in arrays.items()};visible['context_v'][:,:,2032]+=50
  changed=B['forward'](block,B['inputs'](visible,'CPU')).numpy()
  assert float(np.abs(output-changed).max())>1e-5
  result['masked_context_exact']=True;result['late_visible_key_preserved']=True
  failures=[]
  def reject(name,fn):
    try:fn()
    except (ValueError,AssertionError,KeyError,FileNotFoundError):failures.append(name)
    else:raise AssertionError('Negative control accepted: '+name)
  reject('missing independent validation',lambda:B['target_headers']([Path('unused')],[]))
  shortened=dict(data,context_k=data['context_k'][:,:,:2303],context_v=data['context_v'][:,:,:2303])
  reject('truncated context',lambda:B['forward'](block,shortened))
  reject('changed action row count',lambda:B['forward'](block,dict(data,x=data['x'][:,:63])))
  if args.completed_train_report:
    assert args.completed_train_sha and B['sha'](args.completed_train_report)==args.completed_train_sha
    val=[]
    if args.completed_validation_report:
      assert args.completed_validation_sha and B['sha'](args.completed_validation_report)==args.completed_validation_sha
      val=[args.completed_validation_report]
    rows,manifest=B['target_headers']([args.completed_train_report],val,allow_train_only=not val)
    B['validate_targets'](rows)
    actual=[]
    for row in rows:
      values=B['inputs'](B['load'](row),'CPU');prediction=B['forward'](block,values)
      loss=B['loss'](block,values);grads=loss.gradient(*block.weights.values());Tensor.realize(*grads)
      assert np.isfinite(prediction.numpy()).all() and np.isfinite(float(loss.item()))
      assert all(bool(g.isfinite().all().item()) for g in grads)
      actual.append(dict(id=row['id'],split=row['split'],context_length=row['context_length'],loss=float(loss.item()),
                         all_parameter_gradients_finite=True))
    result['completed_target_cpu_checks']=dict(train_report_sha256=args.completed_train_sha,
      validation_report_sha256=args.completed_validation_sha,manifest=manifest,rows=actual,optimizer_updates=0,
      teacher_kv_only=True,independent_validation_targets_checked=bool(val))
    print('Completed captures: CPU forward/gradient checks passed; no actual-data fitting',flush=True)
  # Four synthetic updates exercise the actual loop, all moments and durable resumption.
  synthetic_rows=[dict(id='synthetic-train',split='train',route_id='synthetic-drive-a'),
                  dict(id='synthetic-validation',split='validation',route_id='synthetic-drive-b')]
  fixtures={'synthetic-train':synthetic(7),'synthetic-validation':synthetic(8)}
  loader=lambda row:{k:v.copy() for k,v in fixtures[row['id']].items()}
  context=dict(synthetic_only=True,hyperparameters=B['HYPER'],test='full-shape-expert-recovery')
  one=root/'synthetic-one';two=root/'synthetic-two';resumed=root/'synthetic-resumed'
  T.fit(B['Block'](weights,'CPU'),synthetic_rows,loader,one,context,'CPU',stop_after=1)
  T.fit(B['Block'](weights,'CPU'),synthetic_rows,loader,two,context,'CPU',stop_after=2)
  T.fit(B['Block'](weights,'CPU'),synthetic_rows,loader,resumed,context,'CPU',resume=one,stop_after=2)
  shapes={k:v.shape for k,v in weights.items()}
  uninterrupted,ma=T.S.load(two,context,shapes);restored,mb=T.S.load(resumed,context,shapes)
  for name in uninterrupted:np.testing.assert_array_equal(uninterrupted[name],restored[name])
  assert ma==mb and T.S.sha(two/'best.npz')==T.S.sha(resumed/'best.npz')
  # First synthetic AdamW update also matches a separate NumPy formula for every weight.
  first,_=T.S.load(one,context,shapes)
  synthetic_block=B['Block'](weights,'CPU');sdata=B['inputs'](fixtures['synthetic-train'],'CPU')
  sgrad=B['loss'](synthetic_block,sdata).gradient(*synthetic_block.weights.values());Tensor.realize(*sgrad)
  max_adam_error=0.0
  for name,g in zip(synthetic_block.weights,sgrad,strict=True):
    grad=g.numpy();m=np.float32(.1)*grad;v=np.float32(.001)*grad*grad
    expected=weights[name]-np.float32(3.5e-6)*(m/(1-np.float32(.9))/(np.sqrt(v/(1-np.float32(.999)))+np.float32(1e-8))+
                                           np.float32(.01)*weights[name])
    np.testing.assert_allclose(first['weight/'+name],expected,rtol=0,atol=2e-7)
    max_adam_error=max(max_adam_error,float(np.abs(first['weight/'+name]-expected).max()))
  corrupt=root/'corrupt';shutil.copytree(one,corrupt)
  latest=T.S.load(corrupt,context,shapes)
  T.S.intent(corrupt,1);reject('uncommitted update replay',lambda:T.S.load(corrupt,context,shapes))
  reject('different resume context',lambda:T.S.load(one,dict(wrong=True),shapes))
  checkpoint=root/'bad-checkpoint';shutil.copytree(one,checkpoint)
  pointer=B['read'](checkpoint/'latest.json')
  with (checkpoint/pointer['file']).open('ab') as stream:stream.write(b'corruption')
  reject('modified optimizer checkpoint',lambda:T.S.load(checkpoint,context,shapes))
  # Reach the actual update25 selection branch with a small synthetic parameter.
  # Opposite validation targets must retain the initial checkpoint despite lower training loss.
  class SelectionBlock:
    def __init__(self):self.weights={'gain':Tensor(np.ones(1,np.float32),device='CPU').is_param_(True).realize()}
    def __call__(self,x,*unused):return x*self.weights['gain']
  selection=[]
  for opposite in (False,True):
    def selection_loader(row):
      values=loader(row)
      values['y']=values['x']*np.float32(2 if opposite and row['split']=='validation' else 0)
      return values
    location=root/('selection-retain' if opposite else 'selection-improve')
    selected=T.fit(SelectionBlock(),synthetic_rows,selection_loader,location,
                   dict(synthetic_only=True,opposite_validation=opposite),'CPU')
    expected_step=0 if opposite else 25
    assert selected['complete'] and selected['metadata']['best_step']==expected_step
    assert [item['step'] for item in selected['metadata']['validation']]==[0,25]
    selection.append(dict(opposite_validation=opposite,best_step=expected_step,complete_updates=25))
  result.update(optimizer_resume_bit_exact=True,numpy_adamw_max_absolute=max_adam_error,
                full_shape_synthetic_optimizer_updates=4,selection_toy_optimizer_updates=50,
                validation_only_selection=selection,real_data_optimizer_updates=0,negative_controls_rejected=failures,
                opened_devices=sorted(Device._opened_devices),elapsed_seconds=time.monotonic()-started)
  assert set(Device._opened_devices)<={'CPU','NPY','PYTHON'} and not torch.cuda.is_initialized()
  result['source_sha256']={str(path):B['sha'](path) for path in [*HERE.glob('agent-distill-expert-*.py'),
    HERE/'agent-distill-fit-state.py',HERE/'agent-distill-continue-admit.py',HERE/'agent-distill-fit-train.py',
    HERE/'agent-distill-kfd1-common.py',HERE/'agent-distill-joint-runtime.py',native_root/'model.py']}
  result['all_passed']=True
  (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(json.dumps(result,indent=2))


if __name__=='__main__':main()

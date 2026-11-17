"""CPU numerical and capture-alignment qualification. Zero optimizer updates."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys
from types import MethodType

os.environ.update(DEV='CPU',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
  CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
sys.path.insert(0,str(BASE/'tinygrad'))
import numpy as np
import torch
torch.set_num_threads(4);torch.set_default_device('cpu')
from tinygrad import Tensor,TinyJit,Device,dtypes
from tinygrad.helpers import Context
from tinygrad.uop.ops import UOp

C=runpy.run_path(str(HERE/'agent-distill-objective-capture.py'))
O=C['module']('objective_checked_core',HERE/'agent-distill-objective-core.py')


def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)):
    name=os.fsdecode(args[0])
    if name.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/')) or any(k in name for k in C['FORBIDDEN']):
      raise RuntimeError('CPU objective check prohibited access: '+name)


sys.addaudithook(guard)


def weights(rng,width,hidden,head_dim=4,qheads=4,kvheads=2):
  shapes={'input_layernorm.weight':(width,), 'post_attention_layernorm.weight':(width,),
    'self_attn.q_proj.weight':(qheads*head_dim,width),'self_attn.k_proj.weight':(kvheads*head_dim,width),
    'self_attn.v_proj.weight':(kvheads*head_dim,width),'self_attn.o_proj.weight':(width,qheads*head_dim),
    'mlp.fc1.weight':(hidden,width),'mlp.fc2.weight':(width,hidden)}
  return {k:(np.ones(v,np.float32) if 'layernorm' in k else rng.normal(0,.07,v).astype(np.float32)) for k,v in shapes.items()}


def torch_block(w,x,c,s,context=None,mask=None):
  def lin(x,n):return x@w[n+'.weight'].T
  def norm(x,n):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*w[n+'.weight']
  h=norm(x,'input_layernorm');b,n,_=x.shape
  q,k,v=[lin(h,'self_attn.'+name+'_proj').reshape(b,n,w['self_attn.'+name+'_proj.weight'].shape[0]//4,4).transpose(1,2) for name in ('q','k','v')]
  rot=lambda t:torch.cat((-t[...,2:],t[...,:2]),-1)
  q,k=q*c+rot(q)*s,k*c+rot(k)*s
  key,val=(torch.cat((context[0],k),2),torch.cat((context[1],v),2)) if context else (k,v)
  key,val=(t.repeat_interleave(2,1) for t in (key,val))
  if mask is None:mask=torch.triu(torch.full((n,n),float('-inf')),diagonal=1)
  attention=torch.softmax(q@key.transpose(-2,-1)*.5+mask,-1)@val
  residual=x+lin(attention.transpose(1,2).reshape(b,n,-1),'self_attn.o_proj')
  y=residual+lin(torch.relu(lin(norm(residual,'post_attention_layernorm'),'mlp.fc1')).square(),'mlp.fc2')
  return y,k,v


def numerical():
  rng=np.random.default_rng(1127);lw=weights(rng,16,24);ew=weights(rng,8,16)
  l={k:Tensor(v.copy()).is_param_(True) for k,v in lw.items()};e={k:Tensor(v.copy()).is_param_(True) for k,v in ew.items()}
  capacity=40;prompt=21;valid=29;rows=5
  make=lambda shape:Tensor(rng.normal(0,.2,shape).astype(np.float32))
  angles=rng.normal(0,.2,(1,1,capacity,4)).astype(np.float32)
  lang=dict(x=make((1,capacity,16)),y=make((1,capacity,16)),k=make((1,2,capacity,4)),v=make((1,2,capacity,4)),
            cos=Tensor(np.cos(angles)),sin=Tensor(np.sin(angles)))
  mask=np.full(capacity+rows,-np.inf,np.float32);mask[:valid]=0;mask[capacity:]=0
  data=dict(prompt_length=prompt,valid_length=valid,capacity=capacity,language=lang,expert=[])
  for i in range(3):
    a=rng.normal(0,.2,(1,1,rows,4)).astype(np.float32)
    data['expert'].append(dict(x=make((1,rows,8)),y=make((1,rows,8)),cos=Tensor(np.cos(a)),sin=Tensor(np.sin(a)),mask=Tensor(mask.copy())))
  losses,out=O.objective(l,e,data,head_dim=4,query_chunk=7,minimum_prefill=32)
  dense,_=O.objective(l,e,data,head_dim=4,query_chunk=40,minimum_prefill=32)
  np.testing.assert_allclose(losses['total'].numpy(),dense['total'].numpy(),atol=2e-7,rtol=2e-6)
  tl={k:torch.tensor(v,requires_grad=True) for k,v in lw.items()};te={k:torch.tensor(v,requires_grad=True) for k,v in ew.items()}
  ref=lambda t:torch.tensor(t.numpy())
  extent=out['language_extent'];ly,lk,lv=torch_block(tl,ref(lang['x'])[:,:extent],ref(lang['cos'])[:,:,:extent],ref(lang['sin'])[:,:,:extent])
  def rec(start,stop):return (ly[:,start:stop]-ref(lang['y'])[:,start:stop]).square().mean()+.1*((lk[:,:,start:stop]-ref(lang['k'])[:,:,start:stop]).square().mean()+(lv[:,:,start:stop]-ref(lang['v'])[:,:,start:stop]).square().mean())
  lk,lv=(torch.nn.functional.pad(t,(0,0,0,capacity-extent)) for t in (lk,lv))
  els=[(torch_block(te,ref(s['x']),ref(s['cos']),ref(s['sin']),(lk,lv),ref(s['mask']))[0]-ref(s['y'])).square().mean() for s in data['expert']]
  # rec uses only valid rows, so the appended invalid context tail changes no target slice.
  expected=.5*(rec(0,prompt)+rec(prompt,valid))+torch.stack(els).mean();expected.backward()
  actual=float(losses['total'].item());np.testing.assert_allclose(actual,float(expected.detach()),atol=2e-7,rtol=2e-6)
  params=list(l.values())+list(e.values());gradients=losses['total'].gradient(*params)
  errors=[]
  for a,b in zip(gradients,list(tl.values())+list(te.values()),strict=True):
    x,y=a.numpy(),b.grad.numpy();np.testing.assert_allclose(x,y,atol=3e-7,rtol=3e-4)
    errors.append(float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)))
  expert_only=losses['expert'].gradient(*list(l.values()),*list(e.values()))
  language_norms={name:float(np.linalg.norm(g.numpy())) for name,g in zip(l,expert_only[:len(l)],strict=True)}
  assert language_norms['self_attn.k_proj.weight']>0 and language_norms['self_attn.v_proj.weight']>0
  assert language_norms['self_attn.q_proj.weight']==language_norms['mlp.fc1.weight']==0
  assert all(float(np.linalg.norm(g.numpy()))>0 for g in expert_only[len(l):])
  stopped=[]
  for s in data['expert']:
    y=O.expert(e,s['x'],s['cos'],s['sin'],out['context_k'].detach(),out['context_v'].detach(),s['mask'],head_dim=4)
    stopped.append((y-s['y']).square().mean())
  detached=Tensor.stack(*stopped).mean()
  assert all(np.count_nonzero(g.numpy())==0 for g in detached.gradient(*l.values()))
  targets=[*lang.values(),*(v for s in data['expert'] for v in s.values())]
  assert all(np.count_nonzero(g.numpy())==0 for g in losses['total'].gradient(*targets))
  # One finite difference directly verifies the expert-to-language KV path.
  name='self_attn.v_proj.weight';original=lw[name].copy();delta=1e-3;values=[]
  for sign in (-1,1):
    changed={k:v.copy() for k,v in lw.items()};changed[name].flat[0]+=sign*delta
    ll={k:Tensor(v).is_param_(True) for k,v in changed.items()}
    objective,_=O.objective(ll,e,data,head_dim=4,query_chunk=7,minimum_prefill=32);values.append(float(objective['expert'].item()))
  finite_difference=(values[1]-values[0])/(2*delta)
  analytic=float(expert_only[list(l).index(name)].numpy().flat[0])
  np.testing.assert_allclose(finite_difference,analytic,atol=2e-5,rtol=.02)
  return dict(total=actual,torch_total=float(expected.detach()),max_parameter_gradient_relative_l2=max(errors),
    expert_only_language_gradient_norms=language_norms,expert_parameters_receive_gradients=True,
    detached_context_negative_control_zero=True,targets_detached=True,chunked_matches_dense=True,
    expert_to_language_v_finite_difference=finite_difference,expert_to_language_v_analytic=analytic)


def full_shapes():
  l=weights(np.random.default_rng(1),2048,9216,128,16,8)
  e=weights(np.random.default_rng(2),512,3840,128,16,8)
  # Construct the actual-shape forward graph without executing large matrix products.
  l={k:Tensor.empty(v.shape).is_param_(True) for k,v in l.items()};e={k:Tensor.empty(v.shape).is_param_(True) for k,v in e.items()}
  t=lambda *s:Tensor.zeros(*s)
  data=dict(prompt_length=2027,valid_length=2049,capacity=2304,
    language=dict(x=t(1,2304,2048),y=t(1,2304,2048),k=t(1,8,2304,128),v=t(1,8,2304,128),cos=t(1,1,2304,128),sin=t(1,1,2304,128)),
    expert=[dict(x=t(1,64,512),y=t(1,64,512),cos=t(1,1,64,128),sin=t(1,1,64,128),mask=t(2368)) for _ in range(3)])
  losses,output=O.objective(l,e,data)
  assert output['context_k'].shape==output['context_v'].shape==(1,8,2304,128)
  assert output['language'][0].shape==(1,2176,2048) and all(t.shape==(1,64,512) for t in output['expert'])
  assert losses['total'].shape==()
  return dict(actual_shape_forward_graph_constructed=True,large_matmuls_executed=False,
    cases=[O.memory_assessment(n) for n in (2033,2049,2091,2304)])


def actual_expert():
  B=C['module']('objective_native_expert',HERE/'agent-distill-expert-block.py')
  checkpoint=WORK/'agent-distill-expert-artifacts/pilot-v2/best.npz'
  digest='0206e0e9f3458173da1468a99c6aa2fa7942481ac00639c179fd0e0fd4c5fc3d'
  assert C['sha'](checkpoint)==digest
  with np.load(checkpoint,allow_pickle=False) as archive:values={k:archive[k].copy() for k in archive.files}
  block=B.Block(values,'CPU')
  rows,_=B.target_headers([WORK/'agent-distill-joint-teacher-v1/report.json'],[WORK/'agent-distill-joint-validation-v1/report.json'])
  row=next(r for r in rows if r['split']=='validation' and r['step']==0);data=B.inputs(B.load(row),'CPU')
  old=B.forward(block,data)
  new=O.expert(block.weights,*(data[k] for k in ('x','cos','sin','context_k','context_v','mask')))
  np.testing.assert_array_equal(old.numpy(),new.numpy())
  loss=(new-data['y']).square().mean();gradients=loss.gradient(data['context_k'],data['context_v'])
  norms=[]
  for gradient in gradients:
    array=gradient.numpy();assert np.isfinite(array).all() and np.count_nonzero(array[:,:,:row['context_length']])>0
    assert np.count_nonzero(array[:,:,row['context_length']:])==0;norms.append(float(np.linalg.norm(array)))
  return dict(selected_expert_checkpoint_sha256=digest,sample=row['id'],actual_context_shape=[1,8,2304,128],action_rows=64,
    original_qualified_forward_bit_exact=True,residual_mse=float(loss.item()),context_gradient_norms=norms,
    invalid_context_gradient_exactly_zero=True,optimizer_updates=0)


def alignment(root):
  runtime=Path(C['read'](C['PLAN'])['runtime_root']);sys.path.insert(0,str(runtime))
  Native=C['module']('objective_capture_native',runtime/'model.py').EdgeModel
  model=Native.__new__(Native);model.config=C['read'](BASE/'model/config.json');model.dtype=dtypes.bfloat16
  model.fp8_wmma=True;model.cuda_kernels=False;model.trajectory_jits={}
  model.cache=[Tensor.zeros(2,1,8,2304,128,dtype=dtypes.bfloat16).realize() for _ in range(28)]
  model.norm=lambda x,*a,**kw:x
  model.linear=lambda x,name,*a,**kw:x[:,:,:2]*.01 if name=='expert.action_out_proj' else x
  model.action_projection=lambda a,t:(a+t.reshape(1,1,1)).pad(((0,0),(0,0),(0,510))).cast(model.dtype)
  def layer(self,x,c,s,prefix,cache=None,mask=None,cache_pos=None):
    if prefix.startswith('vlm.'):
      start=cache_pos if isinstance(cache_pos,int) else cache_pos.unbind()[1]
      amount=float(x[0,0,0].float().item())
      cache[:,:,:,start:start+x.shape[1]].assign(Tensor.full((2,1,8,x.shape[1],128),amount,dtype=model.dtype)).realize()
      return x
    return (x.float()+.001).cast(model.dtype).realize()
  model.text_layer=MethodType(layer,model)
  saved={};root.mkdir()
  def save(name,array,bf16):
    path=root/(name+'.npy');np.save(path,array,allow_pickle=False)
    item=dict(path=str(path),sha256=C['sha'](path),shape=list(array.shape),dtype=str(array.dtype),bf16_bits=bf16)
    saved[name]=item;return item
  original=model.text_layer
  with Context(JIT=0),C['aligned_targets'](model,save) as captured:
    for start,rows,value in ((0,2048,.25),(2027,6,.5),(2033,16,.75),(2049,4,1.)):
      position=start if start==0 else UOp.variable('objective_capture_position',0,2250).bind(start)
      model._language(Tensor.full((1,rows,2048),value,dtype=model.dtype),Tensor.ones(1,1,rows,128,dtype=model.dtype),
                      Tensor.zeros(1,1,rows,128,dtype=model.dtype),position).realize()
    actions=model._trajectory(Tensor.zeros(1,64,2),Tensor.ones(1,1,128,128,dtype=model.dtype),
      Tensor.zeros(1,1,128,128,dtype=model.dtype),2049,3,22).numpy()
    prediction=dict(generated_token_ids=list(range(21))+[model.config['traj_ids']['future_start']],generated_tokens=22,prefill_tokens=2048,prompt_tokens=2027)
    captured['finish'](prediction,dict(actual_prompt_length=2027),(np.zeros((1,2048),np.int32),))
  assert model.text_layer==original and captured['complete'] and np.isfinite(actions).all()
  assert captured['valid_rows_last_write'][2027:2033]==[1]*6 and captured['valid_rows_last_write'][2033:]==[2]*16
  x=np.load(saved['language.x']['path']);numeric=(x.astype(np.uint32)<<16).view(np.float32)
  assert np.all(numeric[:,2027:2033]==.5) and np.all(numeric[:,2033:2049]==.75) and np.all(numeric[:,2049:2053]==1.)
  record=dict(synthetic_fixture=True,split='train',route_id=C['ROUTES']['train'],targets=captured)
  loaded=C['load_record'](record,fixture=True)
  assert loaded['valid_length']==2049 and loaded['language']['x'].shape==(1,2304,2048)
  for i,step in enumerate(loaded['expert']):
    np.testing.assert_allclose(step['timestep'].numpy(),[i/3],atol=1e-7)
    if i:np.testing.assert_allclose(step['actions'].numpy(),loaded['expert'][i-1]['actions'].numpy()+loaded['expert'][i-1]['velocity'].numpy()/3,atol=2e-6)
  try:C['load_record'](record)
  except ValueError:pass
  else:raise AssertionError('Unmarked training admission accepted fixture')
  return dict(actual_native_language_and_euler_loops=True,full_2048_prefill=True,full_2304_cache=True,all_64_action_rows=True,
    overwritten_draft_rows_replaced=True,reasoning_crosses_prefill_boundary=True,unaccepted_future_rows_excluded=True,
    teacher_actions_and_three_timesteps_preserved=True,valid_context_length=2049,layer_calls=[len(w['layer_order']) for w in captured['language_writes']],
    tensors=len(saved),target_bytes=sum(Path(v['path']).stat().st_size for v in saved.values()),synthetic_fixture_rejected_for_real_training=True)


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
  root=args.output_dir.resolve()
  if root.exists() or not root.is_relative_to(WORK/'agent-distill-objective-artifacts'):raise ValueError('Fresh owned CPU output required')
  root.mkdir(parents=True)
  result=dict(gpu_executed=False,optimizer_updates=0,quality_qualified=False,numerical=numerical(),full_shapes=full_shapes(),
    actual_expert=actual_expert(),alignment=alignment(root/'capture-fixture'))
  assert all(name.startswith(('CPU','NPY')) for name in Device._opened_devices)
  result.update(all_passed=True,opened_devices=list(Device._opened_devices),source_sha256={str(p):C['sha'](p) for p in HERE.glob('agent-distill-objective-*.py')})
  (root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

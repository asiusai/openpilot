"""Bounded CPU native Torch L2048 oracle. Synthetic inputs; BF16-derived FP32 masters; zero updates."""
import gc
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time

C=importlib.import_module('agent-distill-full-common')

def main():
  assert not (C.ROOT/'oracle.json').exists();C.no_gpu_fds();started=time.monotonic()
  def deny(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      assert not os.fsdecode(args[0]).startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/'))
  sys.addaudithook(deny)
  native=C.B.native();torch=native.torch_cpu()
  import numpy as np
  from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeTextRotaryEmbedding
  student,config=native.make(12);masters=native.initialize(student,12);old=C.read(C.OLD/'oracle.json')
  assert masters==old['student_bf16_master_sha256'];assert not student.training
  files={}
  def save(name,array):
    a=np.asarray(array);assert np.isfinite(a).all(),name
    path=C.ROOT/'oracle-arrays'/f'{len(files):02d}.npy';path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists();np.save(path,a,allow_pickle=False)
    files[name]={'path':str(path),'sha256':C.file_sha(path),'shape':list(a.shape),'dtype':str(a.dtype)}
  for name,p in student.named_parameters():
    item=old['arrays']['w:'+name];path=C.OLD/item['file'];a=np.load(path,allow_pickle=False)
    assert C.file_sha(path)==item['sha256'] and np.array_equal(p.detach().numpy(),a)
    files['w:'+name]={k:v for k,v in item.items() if k!='file'};files['w:'+name]['path']=str(path)
  rng=np.random.default_rng(20261117)
  x=torch.tensor((rng.standard_normal((1,C.LENGTH,2048))*.05).astype(np.float32),requires_grad=True)
  t=torch.arange(C.LENGTH,dtype=torch.int64)
  positions=torch.stack((17+t,3+t//17,8+(t*3)//11))[:,None,:]
  cos,sin=Cosmos3EdgeTextRotaryEmbedding(config)(x.detach(),positions)
  for name,a in (('x',x.detach().numpy()),('position_ids',positions.numpy()),('cos',cos.numpy()),('sin',sin.numpy())):save(name,a)
  teacher_hashes={};hidden=x.detach();first=None
  with torch.no_grad():
    for layer in (12,13,14,15):
      block,cfg=native.make(layer);teacher_hashes.update(native.initialize(block,layer));block.requires_grad_(False)
      hidden,key,value=native.forward(block,cfg,hidden,cos,sin)
      if first is None:first=key.detach().clone(),value.detach().clone()
      assert all(p.grad is None for p in block.parameters());del block;gc.collect()
      print(json.dumps({'cpu_teacher_layer_complete':layer,'seconds':time.monotonic()-started}),flush=True)
  targets=(hidden.detach(),first[0]+.01,first[1]-.01)
  for name,a in zip(('y','k','v'),targets):save('target:'+name,a.numpy())
  versions=[p._version for p in student.parameters()];capture=[]
  handle=student.mlp.fc2.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach()))
  outputs=native.forward(student,config,x,cos,sin);handle.remove()
  loss,terms=native.recovery_loss(outputs,targets,C.VALID_ROWS);loss.backward()
  for name,a in zip(('y','k','v'),outputs):save('out:'+name,a.detach().numpy())
  save('activation',capture[0].numpy());save('loss',np.array(float(loss.detach()),dtype=np.float32))
  for name,p in student.named_parameters():
    assert torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0
    save('grad:'+name,p.grad.numpy())
  assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)>0;save('grad:x',x.grad.numpy())
  delta=(outputs[0][:,:C.VALID_ROWS]-targets[0][:,:C.VALID_ROWS]).detach().reshape(C.VALID_ROWS,2048)
  expected=(2/delta.numel())*(delta.T@capture[0][:,:C.VALID_ROWS].reshape(C.VALID_ROWS,9216))
  torch.testing.assert_close(student.mlp.fc2.weight.grad,expected,rtol=2e-5,atol=1e-8)
  fc2max=float((expected-student.mlp.fc2.weight.grad).abs().max());del expected
  assert torch.count_nonzero(x.grad[:,C.VALID_ROWS:])==0
  assert [p._version for p in student.parameters()]==versions
  for name,p in student.named_parameters():assert np.array_equal(p.detach().numpy(),np.load(files['w:'+name]['path'],allow_pickle=False))
  for p in student.parameters():p.grad=None
  # The same full-length forward realization checks causal isolation. An unmasked
  # full-length native forward supplies an independent negative control.
  with torch.no_grad():
    padded=tuple(a.clone() for a in targets)
    padded[0][:,C.VALID_ROWS:]+=999
    for a in padded[1:]:a[:,:,C.VALID_ROWS:]+=999
    original_loss=native.recovery_loss(outputs,targets,C.VALID_ROWS)[0]
    assert torch.equal(native.recovery_loss(outputs,padded,C.VALID_ROWS)[0],original_loss)
    assert native.recovery_loss(outputs,padded,C.LENGTH)[0]>original_loss+1
    future=x.detach().clone();future[:,C.VALID_ROWS:]+=1
    baseline=native.forward(student,config,x.detach(),cos,sin)[0]
    changed=native.forward(student,config,future,cos,sin)[0]
    assert torch.equal(baseline[:,:C.VALID_ROWS],changed[:,:C.VALID_ROWS])
    assert not torch.equal(baseline[:,C.VALID_ROWS:],changed[:,C.VALID_ROWS:])
    bad=native.forward(student,config,x.detach(),cos,sin,causal=False)[0]
    bad_changed=native.forward(student,config,future,cos,sin,causal=False)[0]
    bad_delta=float((bad_changed[:,:C.VALID_ROWS]-bad[:,:C.VALID_ROWS]).abs().max());assert bad_delta>1e-4
  native.assert_cpu_only();C.no_gpu_fds()
  sources=[Path(__file__),C.EXP/'agent-distill-full-common.py',C.EXP/'agent-distill-feasibility-block.py',
    native.NATIVE,native.CHECKPOINT/'config.json',native.CHECKPOINT/'model.safetensors.index.json',C.OLD/'oracle.json']
  report={'complete':True,'gpu_executed':False,'optimizer_updates':0,'length':C.LENGTH,'valid_rows':C.VALID_ROWS,
    'tolerances':C.TOLERANCES,'arrays':files,'source_sha256':{str(p):C.file_sha(p) for p in sources},
    'student_bf16_master_sha256':masters,'teacher_bf16_master_sha256':teacher_hashes,'config':config.to_dict(),
    'parameter_count':50335744,'input_scope':'Synthetic Gaussian hidden states, seed20261117; native merged three-axis RoPE',
    'target_scope':'Native FP32 checkpoint layers12-15 on synthetic input; K/V offset +0.01/-0.01 exercises derivatives',
    'loss':float(loss.detach()),'loss_terms':{k:float(v.detach()) for k,v in terms.items()},
    'independent_fc2_max_absolute':fc2max,'padding_input_gradient_zero':True,'padding_target_exact':True,
    'included_padding_negative_detected':True,'causal_future_exact':True,'noncausal_negative_max_absolute':bad_delta,
    'student_weights_unchanged':True,'frozen_teacher':True,'seconds':time.monotonic()-started,
    'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  C.write(C.ROOT/'oracle.json',report)
  print(json.dumps({k:report[k] for k in ('complete','gpu_executed','loss','seconds','max_rss_kib')},indent=2))

if __name__=='__main__':main()

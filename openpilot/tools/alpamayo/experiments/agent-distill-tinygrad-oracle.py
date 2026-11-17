"""Build a private native Torch CPU oracle with actual BF16 masters and synthetic positions."""
import gc
import importlib
import json
from pathlib import Path
import resource
import time

C=importlib.import_module('agent-distill-tinygrad-common')

def main():
  assert not (C.ROOT/'oracle.json').exists()
  started=time.monotonic();C.no_gpu_fds()
  native=C.native();torch=native.torch_cpu()
  import numpy as np
  from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeTextRotaryEmbedding
  student,config=native.make(12);masters=native.initialize(student,12)
  old=C.read(C.WORK/'agent-distill-feasibility-artifacts/agent-distill-feasibility-cpu-gradient.json')
  for name,digest in masters.items():assert old['checkpoint_tensor_sha256'][name]==digest
  values={'w:'+name:p.detach().numpy().copy() for name,p in student.named_parameters()}
  rng=np.random.default_rng(20261117)
  x=torch.tensor((rng.standard_normal((1,4,2048))*.05).astype(np.float32),requires_grad=True)
  positions=torch.tensor([[[17,18,21,27]],[[3,3,4,5]],[[8,11,11,14]]],dtype=torch.int64)
  # Actual installed native mRoPE recomposition, with explicit synthetic three-axis positions.
  cos,sin=Cosmos3EdgeTextRotaryEmbedding(config)(x.detach(),positions)
  values.update(x=x.detach().numpy().copy(),position_ids=positions.numpy().copy(),cos=cos.numpy().copy(),sin=sin.numpy().copy())
  teacher_hashes={};hidden=x.detach();first=None
  with torch.no_grad():
    for layer in (12,13,14,15):
      block,cfg=native.make(layer);teacher_hashes.update(native.initialize(block,layer));block.requires_grad_(False)
      hidden,key,value=native.forward(block,cfg,hidden,cos,sin)
      if first is None:first=key.detach().clone(),value.detach().clone()
      assert all(p.grad is None for p in block.parameters());del block;gc.collect()
  # A fixed offset exercises auxiliary K/V derivatives that would otherwise be zero at initialization.
  targets=(hidden.detach(),first[0]+.01,first[1]-.01)
  values.update({'target:'+name:t.detach().numpy().copy() for name,t in zip(('y','k','v'),targets)})
  versions=[p._version for p in student.parameters()];capture=[]
  handle=student.mlp.fc2.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach()))
  outputs=native.forward(student,config,x,cos,sin);handle.remove()
  loss,terms=native.recovery_loss(outputs,targets,3);loss.backward()
  values.update({'out:'+name:t.detach().numpy().copy() for name,t in zip(('y','k','v'),outputs)})
  values['activation']=capture[0].numpy().copy();values['loss']=np.array(float(loss.detach()),dtype=np.float32)
  values.update({'grad:'+name:p.grad.numpy().copy() for name,p in student.named_parameters()});values['grad:x']=x.grad.numpy().copy()
  delta=(outputs[0][:,:3]-targets[0][:,:3]).detach().reshape(3,2048)
  expected=(2/delta.numel())*(delta.T@capture[0][:,:3].reshape(3,9216))
  torch.testing.assert_close(student.mlp.fc2.weight.grad,expected,rtol=2e-5,atol=1e-8)
  assert [p._version for p in student.parameters()]==versions
  assert torch.count_nonzero(x.grad[:,3:])==0
  assert all(np.isfinite(a).all() for a in values.values())
  files={}
  for i,(name,array) in enumerate(values.items()):
    path=C.ROOT/'oracle-arrays'/f'{i:02d}.npy';path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,array,allow_pickle=False)
    files[name]={'file':str(path.relative_to(C.ROOT)),'sha256':C.file_sha(path),'shape':list(array.shape),'dtype':str(array.dtype)}
  sources=[Path(__file__),C.EXP/'agent-distill-tinygrad-common.py',C.EXP/'agent-distill-feasibility-block.py',
    native.NATIVE,native.CHECKPOINT/'config.json',native.CHECKPOINT/'model.safetensors.index.json',
    C.WORK/'agent-distill-feasibility-artifacts/agent-distill-feasibility-cpu-gradient.json']
  native.assert_cpu_only()
  report={'complete':True,'gpu_executed':False,'optimizer_updates':0,'tolerances':C.TOLERANCES,'arrays':files,
    'source_sha256':{str(p):C.file_sha(p) for p in sources},'student_bf16_master_sha256':masters,
    'teacher_bf16_master_sha256':teacher_hashes,'config':config.to_dict(),'parameter_count':50335744,
    'positions':'Explicit synthetic nonuniform T/H/W positions; native merged mRoPE cos/sin, no route inputs',
    'target_scope':'Native FP32 layers12-15 on synthetic input; K/V targets offset +0.01/-0.01 solely for nonzero derivative controls',
    'loss':float(loss.detach()),'loss_terms':{k:float(v.detach()) for k,v in terms.items()},
    'independent_fc2_max_absolute':float((expected-student.mlp.fc2.weight.grad).abs().max()),
    'padding_input_gradient_zero':True,'student_weights_unchanged':True,'frozen_teacher':True,
    'seconds':time.monotonic()-started,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  C.write(C.ROOT/'oracle.json',report)
  print(json.dumps({k:report[k] for k in ('complete','gpu_executed','optimizer_updates','loss','seconds','max_rss_kib')}))

if __name__=='__main__':main()

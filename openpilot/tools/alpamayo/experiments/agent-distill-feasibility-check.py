"""Native full-width CPU backward controls. Zero optimizer updates and no GPU initialization."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def load(path):
  spec=importlib.util.spec_from_file_location('agent_distill_block',path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,default=WORK/'agent-distill-feasibility-artifacts/agent-distill-feasibility-cpu-gradient.json')
  args=parser.parse_args();assert args.output.resolve().is_relative_to(WORK)
  if args.output.exists():raise FileExistsError(args.output)
  implementation=load(HERE/'agent-distill-feasibility-block.py');torch=implementation.torch_cpu()
  torch.manual_seed(20261117)
  student,config=implementation.make(12);weights=implementation.initialize(student,12)
  teachers=[]
  for index in (12,13,14,15):
    block,cfg=implementation.make(index);implementation.initialize(block,index)
    block.requires_grad_(False);teachers.append((block,cfg))
  assert config.hidden_size==2048 and config.intermediate_size==9216
  assert config.num_attention_heads==16 and config.num_key_value_heads==8 and config.head_dim==128
  length=4
  inv=1/(1e8**(torch.arange(0,128,2,device='cpu',dtype=torch.float32)/128))
  angles=torch.arange(length,device='cpu',dtype=torch.float32)[:,None]*inv[None,:]
  angles=torch.cat((angles,angles),dim=-1)[None]
  cos,sin=angles.cos(),angles.sin()
  x=(torch.randn(1,length,2048,device='cpu')*.05).requires_grad_()
  with torch.no_grad():
    hidden=x.detach();first_kv=None
    for block,cfg in teachers:
      hidden,key,value=implementation.forward(block,cfg,hidden,cos,sin)
      if first_kv is None:first_kv=(key.detach(),value.detach())
    targets=(hidden.detach(),*first_kv)
  versions=[parameter._version for parameter in student.parameters()]
  captured=[]
  handle=student.mlp.fc2.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach()))
  outputs=implementation.forward(student,config,x,cos,sin)
  handle.remove()
  loss,terms=implementation.recovery_loss(outputs,targets,3)
  loss.backward()
  assert torch.isfinite(loss)
  assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in student.parameters())
  assert sum(float(parameter.grad.abs().sum()) for parameter in student.parameters())>0
  # Independent linear/MSE derivative for the complete native FC2 matrix.
  delta=(outputs[0][:,:3]-targets[0][:,:3]).detach().reshape(3,2048)
  expected=(2/delta.numel())*(delta.T@captured[0][:,:3].reshape(3,9216))
  torch.testing.assert_close(student.mlp.fc2.weight.grad,expected,rtol=2e-5,atol=1e-8)
  derivative_error=float((student.mlp.fc2.weight.grad-expected).abs().max())
  assert [parameter._version for parameter in student.parameters()]==versions
  assert all(parameter.grad is None for block,_ in teachers for parameter in block.parameters())
  # Input padding is excluded from fitting even when capture tensors retain physical padding rows.
  changed=tuple(value.clone() for value in targets)
  changed[0][:,3:]+=999;changed[1][:,:,3:]+=999;changed[2][:,:,3:]+=999
  repeated,_=implementation.recovery_loss(outputs,changed,3)
  torch.testing.assert_close(repeated,loss.detach(),rtol=0,atol=0)
  with torch.no_grad():
    future=x.detach().clone();future[:,-1]+=1
    altered=implementation.forward(student,config,future,cos,sin)[0]
    torch.testing.assert_close(altered[:,:-1],outputs[0].detach()[:,:-1],rtol=0,atol=0)
  student.zero_grad(set_to_none=True)
  fresh=implementation.forward(student,config,x.detach(),cos,sin)
  kv_loss=(fresh[1]-(targets[1]+.01)).square().mean()+(fresh[2]-(targets[2]-.01)).square().mean()
  kv_loss.backward()
  assert student.self_attn.k_proj.weight.grad.abs().sum()>0 and student.self_attn.v_proj.weight.grad.abs().sum()>0
  assert student.self_attn.q_proj.weight.grad is None and student.mlp.fc2.weight.grad is None
  # Actual native expert geometry with detached language conditioning.
  student.zero_grad(set_to_none=True)
  language=implementation.forward(student,config,x.detach(),cos,sin)
  expert,expert_config=implementation.make(12,expert=True);expert_weights=implementation.initialize(expert,12,expert=True)
  assert expert_config.hidden_size==512 and expert_config.intermediate_size==3840
  actions=torch.randn(1,64,512,device='cpu')*.05
  action_angles=torch.arange(length,length+64,device='cpu',dtype=torch.float32)[:,None]*inv[None,:]
  action_angles=torch.cat((action_angles,action_angles),dim=-1)[None]
  expert_out=implementation.forward(expert,expert_config,actions,action_angles.cos(),action_angles.sin(),False,
                                    tuple(value.detach() for value in language[1:]))
  expert_out[0].square().mean().backward()
  assert all(parameter.grad is None for parameter in student.parameters())
  assert expert.self_attn.k_proj.weight.grad.abs().sum()>0
  assert [parameter._version for parameter in student.parameters()]==versions
  implementation.assert_cpu_only()
  result=dict(schema=1,cpu_backward_executed=True,training_executed=False,optimizer_updates=0,gpu_initialized=False,
    teacher_probe='Four actual checkpoint BF16-master blocks evaluated in CPU FP32; gradient plumbing only, not selected FP8 teacher parity.',
    group=[12,13,14,15],student_initialized_from=12,language_shape=[1,4,2048],expert_shape=[1,64,512],
    language_heads=[16,8,128],teacher_weights_frozen=True,student_weights_unchanged=True,
    finite_all_student_gradients=True,independent_fc2_derivative_max_absolute_error=derivative_error,
    kv_auxiliary_gradient=True,expert_detach_blocks_language_gradient=True,causal_future_mutation_exact=True,
    padding_target_mutation_exact=True,initial_loss=float(loss.detach()),
    loss_terms={key:float(value.detach()) for key,value in terms.items()},
    checkpoint_tensor_sha256={**weights,**expert_weights},
    source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
      (Path(__file__),HERE/'agent-distill-feasibility-block.py',implementation.NATIVE,
       implementation.CHECKPOINT/'config.json',implementation.CHECKPOINT/'model.safetensors.index.json')})
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(json.dumps(result,indent=2))


if __name__=='__main__':main()

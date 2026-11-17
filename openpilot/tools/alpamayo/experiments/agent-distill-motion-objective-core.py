"""Final flow-velocity fidelity through student expert12 and frozen expert16..27.

First qualification freezes the selected language model and uses its captured own K/V.
Teacher action states are fixed conditioning. This is not free-running rollout training.
Frozen deployed weights are dequantized constants; activations use an explicit FP32 surrogate.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-motion-objective-artifacts'
SOURCES=list(range(16,28));SLOTS=list(range(12,25))
SELECTED_SHA='8073b781a84b8f27dc9a64b8ffe7fe1aa82fbbead91ab431918396716414e29a'
CONTROL_SHA='85eeb535c1199ca41dec06d15a2d730467e5c4aca98172e7715fd43dfcde0606'
SHAPES={'input_layernorm.weight':(512,),'post_attention_layernorm.weight':(512,),
  'self_attn.q_proj.weight':(2048,512),'self_attn.k_proj.weight':(1024,512),'self_attn.v_proj.weight':(1024,512),
  'self_attn.o_proj.weight':(512,2048),'mlp.fc1.weight':(3840,512),'mlp.fc2.weight':(512,3840)}
BOUNDS=dict(timeout_seconds=900,live_allocation_limit=12*1024**3,cumulative_allocation_limit=256*1024**3,
  max_single_buffer=320*1024**2,free_reserve=5*1024**3)

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value
if sha(HERE/'agent-distill-objective-core.py')!='3ad258bf1236b31b397314b7caab9fb0b941e60d8afe3db3750939a6dec39a84':raise ValueError('Original block math changed')
O=module('motion_existing_block_math',HERE/'agent-distill-objective-core.py')


def suffix(x,step,contexts,frozen,head_dim=128):
  """Frozen weights still propagate derivatives to x and any differentiable context."""
  if set(frozen['layers'])!=set(SOURCES) or set(contexts)!=set(SLOTS):raise ValueError('Incomplete paired suffix mapping')
  for source in SOURCES:
    slot=source-3
    weights={name:value.detach() for name,value in frozen['layers'][source].items()}
    x=O.expert(weights,x,step['cos'],step['sin'],*contexts[slot],step['mask'],head_dim=head_dim)
  h=x*(x.square().mean(-1,keepdim=True)+1e-5).rsqrt()*frozen['norm'].detach()
  velocity=h@frozen['head_weight'].detach().T+frozen['head_bias'].detach()
  return velocity


def objective(student,contexts,frozen,steps,*,head_dim=128):
  """One mean final-flow MSE over three fixed teacher action/time states; all64 action rows.

  No local residual MSE is included. No student value or suffix intermediate is detached.
  The first captured-context experiment trains expert12 only; language is deliberately fixed.
  """
  from tinygrad import Tensor
  if len(steps)!=3:raise ValueError('Require all three actual teacher Euler conditions')
  losses=[];outputs=[];student_outputs=[]
  for step in steps:
    y=O.expert(student,step['x'].detach(),step['cos'].detach(),step['sin'].detach(),*contexts[12],step['mask'],head_dim=head_dim)
    velocity=suffix(y,step,contexts,frozen,head_dim)
    losses.append((velocity-step['target_velocity'].detach()).square().mean())
    student_outputs.append(y);outputs.append(velocity)
  return Tensor.stack(*losses).mean(),dict(per_step=losses,velocity=outputs,student_outputs=student_outputs)


def selected_expert():
  import numpy as np
  path=WORK/'agent-distill-objective-pilot-artifacts/live-v1/best.npz'
  if sha(path)!=SELECTED_SHA:raise ValueError('Selected joint25 checkpoint changed')
  with np.load(path,allow_pickle=False) as archive:values={k.removeprefix('expert.'):archive[k].copy() for k in archive.files if k.startswith('expert.')}
  if {k:v.shape for k,v in values.items()}!=SHAPES or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in values.values()):raise ValueError('Invalid expert masters')
  return values


def load_capture(path,digest,device='CPU',*,fixture=False):
  import numpy as np
  from tinygrad import Tensor
  path=Path(path).resolve()
  if not path.is_relative_to(WORK) or sha(path)!=digest:raise ValueError('Explicit completed motion capture checksum required')
  report=read(path);capture=report['motion_capture']
  if not report.get('complete') or not capture.get('complete') or (not fixture and (not report.get('gpu_executed') or report.get('cpu_integration_mock'))):raise ValueError('Completed actual motion capture required')
  if capture['selected_checkpoint_sha256']!=SELECTED_SHA or capture['token_control_report_sha256']!=CONTROL_SHA:raise ValueError('Changed selected language/expert conditioning')
  if capture['sample_id']!='train-command-0' or capture['valid_length']!=2047 or capture['context_slots']!=SLOTS or capture['suffix_sources']!=SOURCES:
    raise ValueError('Unexpected first resource-control geometry')
  if not capture['fixed_teacher_actions'] or capture['free_running_training'] or capture['trainable_roles']!=['expert12']:raise ValueError('Changed objective meaning')
  arrays={}
  for name,item in capture['tensors'].items():
    target=Path(item['path']).resolve()
    if not target.is_relative_to(path.parent) or sha(target)!=item['sha256']:raise ValueError('Changed owned capture array')
    value=np.load(target,allow_pickle=False)
    if list(value.shape)!=item['shape'] or str(value.dtype)!=item['dtype']:raise ValueError('Changed capture type/shape')
    if item['bf16_bits']:
      if value.dtype!=np.uint16:raise ValueError('Expected BF16 bits')
      value=(value.astype(np.uint32)<<16).view(np.float32)
    elif value.dtype!=np.float32:raise ValueError('Expected FP32 tensor')
    if name.endswith('.mask'):
      mask=np.r_[np.zeros(2047),np.full(257,-np.inf),np.zeros(64)].astype(np.float32);np.testing.assert_array_equal(value,mask)
    elif not np.isfinite(value).all():raise ValueError('Nonfinite motion tensor')
    arrays[name]=value
  expected={f'context.{slot}':(2,1,8,2304,128) for slot in SLOTS}
  expected.update({f'layer.{source}.{name}':shape for source in SOURCES for name,shape in SHAPES.items()})
  expected.update(norm=(512,),head_weight=(2,512),head_bias=(2,))
  fields=dict(x=(1,64,512),student_y=(1,64,512),cos=(1,1,64,128),sin=(1,1,64,128),mask=(2368,),
    actions=(1,64,2),timestep=(1,),target_velocity=(1,64,2),native_velocity=(1,64,2))
  expected.update({f'step.{i}.{name}':shape for i in range(3) for name,shape in fields.items()})
  if {k:v.shape for k,v in arrays.items()}!=expected:raise ValueError('Missing full-shape motion targets/weights/context')
  for i in range(3):
    np.testing.assert_allclose(arrays[f'step.{i}.timestep'],[i/3],rtol=0,atol=1e-7)
    if i:np.testing.assert_allclose(arrays[f'step.{i}.actions'],arrays[f'step.{i-1}.actions']+arrays[f'step.{i-1}.target_velocity']/3,rtol=2e-6,atol=2e-6)
  values={k:Tensor(v,device=device).is_param_(False) for k,v in arrays.items()}
  contexts={slot:(values[f'context.{slot}'][0],values[f'context.{slot}'][1]) for slot in SLOTS}
  frozen=dict(layers={source:{name:values[f'layer.{source}.{name}'] for name in SHAPES} for source in SOURCES},
    norm=values['norm'],head_weight=values['head_weight'],head_bias=values['head_bias'])
  steps=[{name:values[f'step.{i}.{name}'] for name in fields} for i in range(3)]
  return contexts,frozen,steps,capture


def qualify(path,digest,device,phase=lambda name:None):
  """One forward/backward, no update. Call only inside the existing qualified device/ledger lifecycle.

  This function is intentionally not a standalone GPU launcher. No new device admission path.
  """
  from tinygrad import Tensor
  from tinygrad.helpers import Context
  import numpy as np
  phase('motion-inputs-and-selected-expert');contexts,frozen,steps,capture=load_capture(path,digest,device)
  student={k:Tensor(v,device=device).is_param_(True) for k,v in selected_expert().items()}
  phase('connected-final-flow-forward-backward')
  with Context(TRAINING=1):
    loss,out=objective(student,contexts,frozen,steps);gradients=loss.gradient(*student.values())
    Tensor.realize(loss,*out['velocity'],*gradients)
    norms={k:float(g.square().sum().sqrt().item()) for k,g in zip(student,gradients,strict=True)}
    if any(not bool(g.isfinite().all().item()) for g in gradients) or not all(np.isfinite(list(norms.values()))) or not all(v>0 for v in norms.values()):raise ValueError('Missing/invalid student gradient')
  phase('surrogate-versus-deployed-frozen-suffix')
  diagnostics=[]
  with Context(TRAINING=0):
    for step,pred in zip(steps,out['velocity'],strict=True):
      isolated=suffix(step['student_y'],step,contexts,frozen)
      diagnostics.append(dict(final_flow_mse=float((pred-step['target_velocity']).square().mean().item()),
        full_surrogate_vs_native_mse=float((pred-step['native_velocity']).square().mean().item()),
        frozen_suffix_only_surrogate_vs_native_mse=float((isolated-step['native_velocity']).square().mean().item())))
  return dict(complete=True,optimizer_updates=0,checkpoint_selection_allowed=False,quality_qualified=False,
    loss=float(loss.item()),gradient_norms=norms,per_teacher_state=diagnostics,trainable_roles=['expert12'],
    language_gradients_claimed=False,free_running_rollout_training=False,capture_sha256=digest)


def estimate():
  parameters=sum(__import__('math').prod(shape) for shape in SHAPES.values());weights=parameters*4
  frozen=12*weights+(512+1024+2)*4;contexts=13*2*1*8*2304*128*4
  score=16*64*2368*4
  return dict(trainable_tensors=8,trainable_parameters=parameters,trainable_master_bytes=weights,
    frozen_suffix_layers=12,frozen_weight_bytes=frozen,own_context_fp32_bytes=contexts,own_context_capture_bf16_bytes=contexts//2,
    action_rows=64,context_capacity=2304,teacher_states=3,expert_attention_score_bytes=score,
    largest_weight_bytes=3840*512*4,forward_backward_named_attention_subtotal=3*13*score*4,
    capture_payload_approximately_bytes=frozen+contexts//2+2*1024**2,
    optimizer_updates=0,checkpoint_output_bytes=0,workspace_free_required_bytes=2*1024**3,
    named_subtotals_are_not_peak=True,scratch_measured=False,
    scratch_formula_gfx1100='round_up(max(private_bytes,128),4) * 64 * 32 * 96',
    max_private_bytes_under_320MiB_cap=1704,
    full_shape_AMD_compiler_and_allocator_qualification_required=True,bounds=BOUNDS)

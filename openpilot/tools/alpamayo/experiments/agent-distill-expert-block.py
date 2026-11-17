"""One FP32 expert student, original widths, full current teacher context, residual loss only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-distill-expert-artifacts'
ROUTES = {'train': '00000071--858232e52d', 'validation': '00000073--41474e402a'}
FORBIDDEN = ('0000008e--321b2ddc03', '00000090--1cf3f7d926', 'sealed_test')
HYPER = dict(updates=25, validate_every=25, learning_rate=3.5e-6, beta1=.9, beta2=.999,
             epsilon=1e-8, weight_decay=.01, fused=False, optimizer='tinygrad.AdamW',
             loss='MSE(group.y), all 64 action rows', conditioning='detached current full teacher KV',
             dtype='float32', rows=64, context_keys=2304, seed_layer=12,
             order='explicit report/sample order, then Euler steps 0,1,2; cyclic')
BOUNDS = dict(timeout_seconds=900, live_allocation_limit=12*1024**3, cumulative_allocation_limit=256*1024**3,
              max_single_buffer=256*1024**2, free_reserve=5*1024**3)
SHAPES = dict(x=(1,64,512), y=(1,64,512), cos=(1,1,64,128), sin=(1,1,64,128),
              context_k=(1,8,2304,128), context_v=(1,8,2304,128), mask=(2368,),
              actions=(1,64,2), timestep=(1,), velocity=(1,64,2))


def module(name, path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def seed():
  helper=module('agent_distill_expert_joint_source',HERE/'agent-distill-joint-runtime.py')
  return helper.seed_block(BASE/'model','expert.expert.layers.12.')


class Block:
  """Differentiable original native expert math. No action projection or solver change."""
  def __init__(self, weights, device='CPU'):
    from tinygrad import Tensor
    self.weights={name:Tensor(value.copy(),device=device).is_param_(True).realize() for name,value in sorted(weights.items())}
    if sum(w.numel() for w in self.weights.values()) != 7078912:raise ValueError('Wrong original expert geometry')
    expected={'input_layernorm.weight':(512,), 'post_attention_layernorm.weight':(512,),
              'self_attn.q_proj.weight':(2048,512),'self_attn.k_proj.weight':(1024,512),
              'self_attn.v_proj.weight':(1024,512),'self_attn.o_proj.weight':(512,2048),
              'mlp.fc1.weight':(3840,512),'mlp.fc2.weight':(512,3840)}
    if {k:v.shape for k,v in self.weights.items()} != expected:raise ValueError('Wrong expert parameters')
  def norm(self,x,name):return x*(x.square().mean(-1,keepdim=True)+1e-5).rsqrt()*self.weights[name+'.weight']
  def linear(self,x,name):return x@self.weights[name+'.weight'].T
  def __call__(self,x,cos,sin,context_k,context_v,mask):
    if x.shape != SHAPES['x'] or context_k.shape != SHAPES['context_k'] or context_v.shape != context_k.shape:
      raise ValueError('Keep all 64 rows and all 2304 context keys')
    if cos.shape != SHAPES['cos'] or sin.shape != cos.shape or mask.shape != SHAPES['mask']:
      raise ValueError('Keep actual action mRoPE and full context/action mask')
    h=self.norm(x,'input_layernorm')
    q,k,v=[self.linear(h,'self_attn.'+name+'_proj').reshape(1,64,heads,128).transpose(1,2)
           for name,heads in (('q',16),('k',8),('v',8))]
    def rotate(t):return (-t[...,64:]).cat(t[...,:64],dim=-1)
    q,k=q*cos+rotate(q)*sin,k*cos+rotate(k)*sin
    # Detach teacher conditioning explicitly. No flow loss is propagated into language.
    k=context_k.detach().cat(k,dim=2).repeat_interleave(2,dim=1)
    v=context_v.detach().cat(v,dim=2).repeat_interleave(2,dim=1)
    probabilities=(q.matmul(k.transpose(-2,-1))*(128**-.5)+mask.detach()).softmax(-1)
    h=probabilities.matmul(v).transpose(1,2).reshape(1,64,2048)
    residual=x+self.linear(h,'self_attn.o_proj')
    h=self.linear(self.norm(residual,'post_attention_layernorm'),'mlp.fc1').relu().square()
    return residual+self.linear(h,'mlp.fc2')


def forward(block,data):return block(*(data[k] for k in ('x','cos','sin','context_k','context_v','mask')))
def loss(block,data):return (forward(block,data)-data['y'].detach()).square().mean()


def tensor_array(item, expected_shape, mask=False):
  import numpy as np
  path=Path(item['path']).resolve()
  if not path.is_relative_to(WORK) or any(word in str(path) for word in FORBIDDEN):raise ValueError('Forbidden target path')
  if item['shape'] != list(expected_shape) or sha(path) != item['sha256']:raise ValueError('Target shape/hash mismatch')
  array=np.load(path,allow_pickle=False)
  if array.shape != expected_shape or str(array.dtype) != item['dtype']:raise ValueError('Target type mismatch')
  if item['bf16_bits']:
    if array.dtype != np.uint16:raise ValueError('Expected original BF16 bits')
    array=(array.astype(np.uint32)<<16).view(np.float32)
  elif array.dtype != np.float32:raise ValueError('Expected original FP32 values')
  if mask:
    if not np.all((array==0)|np.isneginf(array)):raise ValueError('Invalid attention mask')
  elif not np.isfinite(array).all():raise ValueError('Nonfinite teacher target')
  return array


def target_headers(train_reports, validation_reports, *, allow_train_only=False):
  """Only explicit completed reports. Validate split independence before tensor reads."""
  if not train_reports or (not validation_reports and not allow_train_only):
    raise ValueError('Distinct completed TRAIN and VAL reports are required')
  records=[];reports=[];contract=None;seen=set()
  for split,paths in (('train',train_reports),('validation',validation_reports)):
    for path in paths:
      path=Path(path).resolve()
      if not path.is_relative_to(WORK) or any(word in str(path) for word in FORBIDDEN):raise ValueError('Forbidden report')
      report=read(path)
      if not report.get('complete') or not report.get('gpu_executed') or report.get('mode')!='capture':
        raise ValueError('Only completed unchanged full-teacher captures are eligible')
      current=dict(settings=report['settings'], model_sources={k:v for k,v in report['source_sha256'].items() if k.endswith('/model.py')})
      if contract is None:contract=current
      if not current['model_sources'] or current != contract:raise ValueError('Teacher settings/source changed between splits')
      reports.append(dict(path=str(path),sha256=sha(path),split=split))
      if not report['rows']:raise ValueError('Empty target report')
      for row in report['rows']:
        if row['split'] != split or row['route_id'] != ROUTES[split]:raise ValueError('Wrong or overlapping drive split')
        if any(word in json.dumps(row) for word in FORBIDDEN):raise ValueError('Forbidden target record')
        key=(split,row['id'])
        if key in seen:raise ValueError('Duplicate example');
        seen.add(key)
        capture=row['targets']
        if len(capture['expert_steps'])!=3 or any(step['layer_order']!=list(range(28)) for step in capture['expert_steps']):
          raise ValueError('Need all three actual full-teacher Euler evaluations')
        if any(order!=list(range(28)) for order in capture['language_orders']):raise ValueError('Changed teacher language depth')
        if not capture['language_orders'] or any(item['layer_order']!=list(range(27)) for item in capture['vision_batches']):
          raise ValueError('Changed full teacher execution')
        if len(row['vision_input_keys'])!=10:raise ValueError('Expected all ten input image slots')
        length=row['teacher']['prompt_tokens']+row['teacher']['generated_tokens']
        if not 0<length<=2304:raise ValueError('Invalid complete reasoning cache length')
        for step in range(3):
          items={}
          for name in SHAPES:
            suffix=name if name in ('actions','timestep','velocity') else 'group12.'+name
            item=capture['tensors'][f'expert{step}.'+suffix]
            if item['shape']!=list(SHAPES[name]):raise ValueError('Full teacher target shape differs')
            if any(word in item['path'] for word in FORBIDDEN):raise ValueError('Forbidden tensor')
            items[name]=item
          records.append(dict(id=row['id']+f'/euler{step}',sample=row['id'],split=split,route_id=row['route_id'],
                              step=step,context_length=length,tensors=items,report_sha256=reports[-1]['sha256']))
  fingerprint=hashlib.sha256(json.dumps(dict(reports=reports,records=records),sort_keys=True,separators=(',',':')).encode()).hexdigest()
  return records,dict(reports=reports,manifest_sha256=fingerprint,teacher_contract=contract)


def load(record):
  import numpy as np
  arrays={name:tensor_array(record['tensors'][name],shape,name=='mask') for name,shape in SHAPES.items()}
  mask=np.full(2368,-np.inf,np.float32);mask[:record['context_length']]=0;mask[2304:]=0
  np.testing.assert_array_equal(arrays['mask'],mask)
  np.testing.assert_allclose(arrays['timestep'],[record['step']/3],atol=1e-7,rtol=0)
  return arrays


def validate_targets(records):
  import numpy as np
  prior=None
  for record in records:
    arrays=load(record)
    if record['step']:
      if prior is None or prior[0]['sample']!=record['sample']:raise ValueError('Missing adjacent actual Euler state')
      old=prior[1]
      np.testing.assert_allclose(arrays['actions'],old['actions']+old['velocity']/3,atol=2e-6,rtol=2e-6)
      for name in ('context_k','context_v','mask','cos','sin'):np.testing.assert_array_equal(arrays[name],old[name])
    prior=(record,arrays)
  return True


def inputs(arrays,device):
  from tinygrad import Tensor
  return {name:Tensor(value.copy(),device=device).is_param_(False).realize() for name,value in arrays.items()}

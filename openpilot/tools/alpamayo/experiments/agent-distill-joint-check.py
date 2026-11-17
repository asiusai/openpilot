"""CPU numerical checks of actual native signatures, compact caches, and rollout capture."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys
from types import MethodType

os.environ.update(DEV='CPU', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                  OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
HERE = Path(__file__).resolve().parent
ROOT = WORK/'agent-distill-command-artifacts/capture-v1/runtime'
sys.path[:0] = [str(BASE/'tinygrad'), str(ROOT)]
import numpy as np
from tinygrad import Tensor, TinyJit, Device, dtypes

R = runpy.run_path(str(HERE/'agent-distill-joint-runtime.py'))
Native = runpy.run_path(str(ROOT/'model.py'))['EdgeModel']
CONFIG = json.loads((BASE/'model/config.json').read_text())


def model():
  value = Native.__new__(Native)
  value.config = deepcopy(CONFIG); value.dtype = dtypes.float32
  value.fp8_wmma = value.cuda_kernels = False
  value.cache = None; value.prompt_length = value.greedy_shape = None
  value.weights = {}; value.weight_scales = {}; value.trajectory_jits = {}; value.encode_jits = {}
  value.prefill_jit = TinyJit(value._language); value.decode_jit = TinyJit(value._language)
  value.greedy_decode_jit = TinyJit(value._greedy_decode)
  return value


def compact_mapping():
  value = model(); calls = []; expert_seen = []
  for p in (R['LANGUAGE'], R['EXPERT']): value.weights[p+'12.input_layernorm.weight'] = Tensor.ones(1)
  def layer(self, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
    source = int(prefix.rsplit('.', 1)[1])
    is_expert = prefix.startswith(('expert.', 'agent_joint.expert.'))
    if is_expert:
      context = cache[0].numpy()
      assert np.all(context == source+1), (prefix, context)
      expert_seen.append(source)
      return (x + cache[0].mean()).realize()
    calls.append(source)
    cache.assign(Tensor.full(cache.shape, source+1, dtype=cache.dtype)).realize()
    return (x+source+1).realize()
  value.text_layer = MethodType(layer, value)
  value.norm = lambda x, *a, **kw: x
  value.linear = lambda x, *a, **kw: x
  value.action_projection = lambda actions, timestep: actions
  arrays = {'input_layernorm.weight': np.ones(1, dtype=np.float32)}
  original_config = value.config
  with R['joint_pilot'](value, arrays, arrays, allow_untrained_expert=True) as receipt:
    n = len(receipt['mapping']); assert n == 25
    value.cache = [Tensor.zeros(2,1,8,8,128).realize() for _ in range(n)]
    x = Tensor.zeros(1,2,2048); c = Tensor.ones(1,1,2,128); s = c*0
    y = value._language(x,c,s,0).numpy()
    sources = list(range(13))+list(range(16,28))
    assert calls == sources and np.all(y == sum(i+1 for i in sources))
    # Invoke the actual native expert loop and actual three-Euler trajectory loop.
    a = Tensor.zeros(1,64,2)
    output = value._trajectory(a,Tensor.ones(1,1,64,128),Tensor.zeros(1,1,64,128),2,3,0).numpy()
    expected = 0.0
    for _ in range(3): expected += (expected+sum(i+1 for i in sources))/3
    assert expert_seen == sources*3 and np.allclose(output,expected,atol=1e-4)
    # Reused allocation gets only current student writes on the next query.
    calls.clear(); value._language(x+7,c,s,0).realize()
    assert calls == sources
    assert all(np.all(cache.numpy() == src+1) for cache,src in zip(value.cache,sources,strict=True))
  assert value.config is original_config and value.cache is None
  assert not any(k.startswith('agent_joint.') for k in value.weights)
  assert R['mapping'](R['GROUPS'])[-1]['teacher_group'] == [24,25,26,27]
  assert R['VISION_GROUPS'][-1] == (24,25,26)
  return dict(compact_cache_count=25, language_sources=sources, expert_sources=sources,
              actual_native_three_euler_loop=True, teacher_cache_slots_13_14_15_absent=True, restoration=True)


def numerical_language():
  weights = R['load_pilot'](WORK/'agent-distill-continue-artifacts/pilot-100-v1/best.npz')
  value = model(); prefix = 'agent_joint.language.12'
  value.weights = {prefix+'.'+k: Tensor(v, device='CPU').realize() for k,v in weights.items()}
  graph = runpy.run_path(str(HERE/'agent-distill-tinygrad-common.py'))
  block = graph['Block'].__new__(graph['Block']); block.eps = 1e-5
  block.weights = {k: value.weights[prefix+'.'+k] for k in weights}
  rng = np.random.default_rng(20261117)
  x = Tensor(rng.normal(0,.1,(1,3,2048)).astype(np.float32))
  angles = rng.normal(0,.1,(1,3,128)).astype(np.float32)
  cos, sin = Tensor(np.cos(angles)), Tensor(np.sin(angles))
  cache = Tensor.zeros(2,1,8,3,128).realize()
  actual = value.text_layer(x,cos.unsqueeze(1),sin.unsqueeze(1),prefix,cache,None,0)
  expected, _ = block(x,cos,sin)
  errors = {}
  for name,a,b in zip(('residual','k','v'),(actual,cache[0],cache[1]),expected,strict=True):
    aa,bb = a.numpy(),b.numpy(); np.testing.assert_allclose(aa,bb,atol=3e-5,rtol=5e-4)
    errors[name] = float(np.abs(aa-bb).max())
  # The opt-in runtime rounds masters to BF16. Check that path independently.
  value.weights = {name: tensor.cast(dtypes.bfloat16).realize() for name,tensor in value.weights.items()}
  value.dtype = dtypes.bfloat16
  cache_bf16 = Tensor.zeros(2,1,8,3,128,dtype=dtypes.bfloat16).realize()
  bf16 = value.text_layer(x.cast(dtypes.bfloat16),cos.unsqueeze(1).cast(dtypes.bfloat16),
                         sin.unsqueeze(1).cast(dtypes.bfloat16),prefix,cache_bf16,None,0).float().numpy()
  assert bf16.shape == (1,3,2048) and np.isfinite(bf16).all()
  return dict(checkpoint_sha256=R['PILOT_SHA'], actual_width=2048, actual_checkpoint_cpu_forward=True,
              fp32_training_graph_max_errors=errors, bf16_finite=True,
              bf16_vs_fp32_max=float(np.abs(bf16-expected[0].numpy()).max()), quality_qualified=False)


def numerical_expert():
  value = model(); prefix = 'agent_joint.expert.12'
  weights = R['seed_block'](BASE/'model', R['EXPERT']+'12.')
  value.weights = {prefix+'.'+k: Tensor(v,device='CPU').realize() for k,v in weights.items()}
  rng = np.random.default_rng(13)
  x = Tensor(rng.normal(0,.1,(1,2,512)).astype(np.float32))
  cos = Tensor.ones(1,1,2,128); sin = Tensor.zeros(1,1,2,128)
  k,v = [rng.normal(0,.1,(1,8,5,128)).astype(np.float32) for _ in range(2)]
  mask = Tensor(np.array([0,0,-np.inf,-np.inf,-np.inf,0,0],dtype=np.float32))
  y = value.text_layer(x,cos,sin,prefix,[Tensor(k),Tensor(v)],mask).numpy()
  changed_k,changed_v = k.copy(),v.copy(); changed_k[:,:,2:] = 400; changed_v[:,:,2:] = -500
  masked = value.text_layer(x,cos,sin,prefix,[Tensor(changed_k),Tensor(changed_v)],mask).numpy()
  np.testing.assert_array_equal(y,masked)
  changed_v[:,:,:2] += 1
  current = value.text_layer(x,cos,sin,prefix,[Tensor(k),Tensor(changed_v)],mask).numpy()
  assert np.isfinite(y).all() and np.max(np.abs(y-current)) > 1e-5
  value.dtype = dtypes.bfloat16
  value.weights = {name:t.cast(dtypes.bfloat16).realize() for name,t in value.weights.items()}
  full = value.text_layer(x.repeat(1,32,1).cast(dtypes.bfloat16),Tensor.ones(1,1,64,128,dtype=dtypes.bfloat16),
    Tensor.zeros(1,1,64,128,dtype=dtypes.bfloat16),prefix,[Tensor(k).cast(dtypes.bfloat16),Tensor(v).cast(dtypes.bfloat16)],
    Tensor(np.array([0,0,-np.inf,-np.inf,-np.inf]+[0]*64,dtype=np.float32))).float().numpy()
  assert full.shape == (1,64,512) and np.isfinite(full).all()
  return dict(residual_width=512, kv_shape=list(k.shape), masked_padding_exact=True,
              valid_current_context_changes_output=True, maximum_context_delta=float(np.abs(y-current).max()),
              actual_bf16_64_action_rows_finite=True)


def capture_rollout():
  value = model()
  value.cache = [Tensor.full((2,1,8,4,128),i+1,dtype=dtypes.float32).realize() for i in range(28)]
  value.text_layer = MethodType(lambda self,x,c,s,prefix,cache=None,mask=None,cache_pos=None: (x+.001).realize(),value)
  value.vision_layer = MethodType(lambda self,x,i: (x+.01).realize(),value)
  value.linear = MethodType(lambda self,x,name,quantized_input=None,heads=None: x,value)
  value.batched_projector = lambda x: tuple(x[i,:180].pad(((0,0),(0,896))) for i in range(x.shape[0]))
  value.norm = lambda x,*a,**kw: x
  value.action_projection = lambda a,t: a+t.reshape(1,1,1)
  saved = {}
  def save(name,tensor): saved[name] = tensor.numpy().copy(); return dict(shape=list(tensor.shape))
  before = value.text_layer
  with R['teacher_targets'](value,save) as state:
    vision = Tensor.zeros(2,720,1152)
    for i in range(27): vision=value.vision_layer(vision,i)
    value.batched_projector(vision)
    c=Tensor.ones(1,1,4,128);s=c*0
    value._language(Tensor.zeros(1,4,2048),c,s,0)
    output=value._trajectory(Tensor.zeros(1,64,2),Tensor.ones(1,1,64,128),Tensor.zeros(1,1,64,128),2,3,0).numpy()
  R['validate_targets'](state)
  assert value.text_layer == before
  for i in range(3):
    np.testing.assert_allclose(saved[f'expert{i}.timestep'],[i/3],atol=1e-7,rtol=0)
    if i:
      np.testing.assert_allclose(saved[f'expert{i}.actions'],saved[f'expert{i-1}.actions']+saved[f'expert{i-1}.velocity']/3,
                                 atol=1e-7,rtol=1e-6)
  assert not np.array_equal(saved['expert0.actions'],saved['expert2.actions'])
  assert saved['expert0.group12.context_k'].shape == (1,8,4,128)
  assert saved['vision0.group12.x'].shape == (2,720,1152)
  assert state['projector_calls'] == 2 and saved['projector0.features'].shape == (180,2048)
  reference = value._trajectory(Tensor.zeros(1,64,2),Tensor.ones(1,1,64,128),Tensor.zeros(1,1,64,128),2,3,0).numpy()
  np.testing.assert_array_equal(reference,output)
  return dict(actual_teacher_euler_recurrence=True, captured_steps=3, tensors=len(saved), output_finite=bool(np.isfinite(output).all()),
              observation_preserves_output_exactly=True)


def main():
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  result=dict(gpu_executed=False, mapping=compact_mapping(), language=numerical_language(),
              expert=numerical_expert(), capture=capture_rollout())
  assert all(name.startswith(('CPU','DISK','NPY')) for name in Device._opened_devices),Device._opened_devices
  result['opened_devices']=list(Device._opened_devices)
  result['source_sha256']={str(path):R['sha'](path) for path in HERE.glob('agent-distill-joint-*.py')}
  result['all_passed']=True
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()

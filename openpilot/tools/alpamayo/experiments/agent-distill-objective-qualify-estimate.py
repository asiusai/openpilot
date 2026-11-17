"""CPU schedule-only full-shape buffer estimate. No kernels, optimizer updates, or target reads."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
os.environ.update(DEV='CPU',NO_MEMORY_PLANNER='1',LRU='0',TC='0',ALLOW_TF32='0',PYTHONDONTWRITEBYTECODE='1',SCACHE='0')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
sys.path.insert(0,str(BASE/'tinygrad'))
from tinygrad import Tensor,Device
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
Q=runpy.run_path(str(HERE/'agent-distill-objective-qualify-core.py'))


def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')):
    raise RuntimeError('Schedule-only CPU estimate')
sys.addaudithook(guard)


def buffers(*linears):
  found={}
  for linear in linears:
    for call in linear.src:
      assert call.op is Ops.CALL
      for value in call.src[1:]:
        base=value.buf_uop
        if base.op is Ops.BUFFER:found[base]=base.max_numel()*base.dtype.itemsize
  return dict(kernels=sum(len(linear.src) for linear in linears),distinct_logical_buffers=len(found),
    total_distinct_buffer_bytes=sum(found.values()),largest_buffer_bytes=max(found.values()),
    largest_buffer_sizes=sorted(found.values(),reverse=True)[:10])


def main():
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  if args.output.exists() or not args.output.resolve().is_relative_to(Q['ROOT']):raise ValueError('Fresh owned estimate output required')
  args.output.parent.mkdir(parents=True,exist_ok=True)
  weights={}
  for role,width,hidden in (('language',2048,9216),('expert',512,3840)):
    shapes={'input_layernorm.weight':(width,),'post_attention_layernorm.weight':(width,),
      'self_attn.q_proj.weight':(2048,width),'self_attn.k_proj.weight':(1024,width),'self_attn.v_proj.weight':(1024,width),
      'self_attn.o_proj.weight':(width,2048),'mlp.fc1.weight':(hidden,width),'mlp.fc2.weight':(width,hidden)}
    weights.update({role+'.'+k:Tensor.empty(shape).is_param_(True) for k,shape in shapes.items()})
  empty=lambda shape:Tensor.empty(shape)
  data=dict(prompt_length=2027,valid_length=2047,capacity=2304,
    language={k:empty(shape) for k,shape in dict(x=(1,2304,2048),y=(1,2304,2048),k=(1,8,2304,128),v=(1,8,2304,128),
      cos=(1,1,2304,128),sin=(1,1,2304,128)).items()},expert=[])
  for _ in range(3):data['expert'].append({k:empty(shape) for k,shape in dict(x=(1,64,512),y=(1,64,512),
    cos=(1,1,64,128),sin=(1,1,64,128),mask=(2368,)).items()})
  l={k.removeprefix('language.'):v for k,v in weights.items() if k.startswith('language.')}
  e={k.removeprefix('expert.'):v for k,v in weights.items() if k.startswith('expert.')}
  with Context(TRAINING=1):
    losses,out=Q['O'].objective(l,e,data)
    gradients=losses['total'].gradient(*weights.values(),out['context_k'],out['context_v'])
    linear=losses['total'].schedule_linear(*losses.values(),*out['language'],*out['expert'],*gradients)
    estimate=buffers(linear)
    for weight,gradient in zip(weights.values(),gradients[:len(weights)],strict=True):weight.grad=gradient
    optimizer=Q['optimizer'](weights);updates=optimizer.schedule_step()
    optimizer_linear=updates[0].schedule_linear(*updates[1:])
  assert all(name.startswith(('CPU','NPY','PYTHON')) for name in Device._opened_devices)
  result=dict(gpu_executed=False,kernels_executed=0,optimizer_updates=0,full_2048_2304_three64_graph=True,
    no_memory_planner=1,joint_forward_backward_schedule=estimate,
    optimizer_with_zero_state_initialization_schedule=buffers(optimizer_linear),
    both_schedules_unique_buffers=buffers(linear,optimizer_linear),analytical_state_and_io=Q['estimate'](),
    interpretation='Distinct logical global buffers in CPU schedules, not an AMD allocator peak or cumulative allocation measurement. They exclude AMD compiler scratch, queue storage, host-map lifetime, finite/norm readbacks and checkpoint/recovery transfers. No scheduled kernel was executed.',
    source_sha256={str(p):Q['sha'](p) for p in (Path(__file__),HERE/'agent-distill-objective-qualify-core.py',HERE/'agent-distill-objective-core.py')})
  args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()

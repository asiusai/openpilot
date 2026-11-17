"""CPU source/grouping/full720-layout contracts; no original model inference."""
import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('agent_vision_group4_check_common',HERE/'agent-vision-group4-common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def main():
  c.require(os.environ.get('DEV')=='CPU','Set DEV=CPU')
  sys.path[:0]=[str(c.BASE),str(c.BASE/'experiments')]
  from tinygrad import Device,Tensor,dtypes
  from tinygrad.helpers import Context
  assert Device.DEFAULT=='CPU'
  cache=c.module(c.BASE/'experiments/vision-cache.py','agent_vision_group4_cpu_cache')
  batched=c.module(c.BASE/'experiments/batched-vision.py','agent_vision_group4_cpu_batched')
  source=textwrap.dedent(inspect.getsource(cache.VisionCache.encode))
  prefix=source.split('    for chunk in chunks:',1)[0]
  assert prefix!=source
  prefix+='    return chunks, missing\n'
  namespace=dict(cache.__dict__);exec(compile(prefix,str(Path(__file__)), 'exec'),namespace)
  grouping=namespace['encode'];group_checks=[]
  for limit in (2,4):
    for misses in range(11):
      owner=SimpleNamespace(grouped=True,keys=[((1,20,36),bytes([i])) for i in range(10)],
        cache={((1,20,36),bytes([i])):None for i in range(10-misses)},batch_encoder=True,group_limit=limit)
      batches=[(i*720,(i+1)*720,1) for i in range(10)]
      chunks,missing=grouping(owner,None,None,None,batches,None)
      expected=[min(limit,misses-begin) for begin in range(0,misses,limit)]
      assert [len(chunk) for chunk in chunks]==expected and len(missing)==misses
      assert [item[1] for chunk in chunks for item in chunk]==[(a,b) for a,b,_ in batches[10-misses:]]
      group_checks.append(dict(group_limit=limit,misses=misses,chunks=expected))
  # Actual buffer-view helper, full native patch widths, with distinct boundaries.
  pixels=np.arange(10*720*768,dtype=np.uint32).reshape(7200,768)%251
  host=Tensor(pixels.astype(np.uint8),device='CPU').realize()
  for i in range(10):
    view=cache.VisionCache.image_view(host,i*720,(i+1)*720)
    np.testing.assert_array_equal(view.numpy(),pixels[None,i*720:(i+1)*720].astype(np.uint8))
  quant_records=[]
  for images in (1,2,3,4):
    captured=[];norm_shapes=[]
    class Model:
      config={'vlm_config':{'vision_config':{'num_hidden_layers':1}}}
      weight_scales={}
      weights={}
      dtype=dtypes.bfloat16
      def linear(self,x,name,*args,**kwargs):return x
      def quantize(self,x):raise AssertionError('Grouped branch must stay active')
      def norm(self,x,name,*args,**kwargs):norm_shapes.append((name,x.shape));return x
      def vision_layer(self,x,index):
        x=self.norm(x,'vlm.model.visual.encoder.layers.0.layer_norm1')
        q,scale=self.quantize(x);captured.append((q.realize(),scale.realize()))
        return x
    model=Model();encode=batched.install(model)
    rng=np.random.default_rng(4400+images)
    values=rng.normal(size=(images,720,1152)).astype(np.float32)*np.arange(1,images+1,dtype=np.float32)[:,None,None]
    pixels=Tensor(values,device='CPU').cast(dtypes.bfloat16).realize();positions=Tensor.zeros(1,720,1152,device='CPU',dtype=dtypes.bfloat16).realize()
    pairs=[value for i in range(images) for value in (pixels[i:i+1].clone().realize(),positions)]
    with Context(BEAM=0):encode(*pairs)
    q,scale=captured[-1];bits=q.bitcast(dtypes.uint8).numpy().reshape(images,720,1152);scales=scale.numpy().reshape(images,720)
    bf=pixels.float().numpy()
    # Pinned tinygrad lowers constant division to FP32 reciprocal multiplication.
    # The failed mathematical-division oracle remains in replay02.log (one ULP).
    expected=np.maximum(np.abs(bf).max(axis=(1,2))*np.float32(1/448),np.float32(1e-12)).astype(np.float32)
    np.testing.assert_array_equal(scales,np.broadcast_to(expected[:,None],scales.shape))
    assert all(shape[0]==1 for name,shape in norm_shapes if 'encoder.layers.' in name)
    changed=values.copy();changed[-1]*=4
    modified=Tensor(changed,device='CPU').cast(dtypes.bfloat16).realize()
    pairs=[value for i in range(images) for value in (modified[i:i+1].clone().realize(),positions)]
    encode(*pairs);newq,newscale=captured[-1]
    np.testing.assert_array_equal(newq.bitcast(dtypes.uint8).numpy().reshape(images,720,1152)[:-1],bits[:-1])
    np.testing.assert_array_equal(newscale.numpy().reshape(images,720)[:-1],scales[:-1])
    quant_records.append(dict(images=images,rows=720,per_image_scales_independently_verified=True,
      changing_last_image_cannot_change_earlier_quantized_bytes=True,actual_wrapper_encoder_norm_calls_keep_one_image=True,
      scale_oracle='FP32 maximum times FP32(1/448); pinned constant-division lowering'))
  for path in HERE.glob('agent-vision-group4-*.py'):ast.parse(path.read_text(),filename=str(path))
  assert all(device.split(':')[0] in ('CPU','NPY','PYTHON') for device in Device._opened_devices)
  out=c.WORK/'agent-vision-group4-cpu-check-native.json'
  c.write(out,dict(cpu_only=True,gpu_executed=False,original_model_inference=False,grouping_checks=group_checks,
    actual_full720_image_views=10,patch_width=768,quantization_controls=quant_records,opened_devices=sorted(Device._opened_devices),
    limits='Actual wrapper and view/grouping methods; identity projection/normalization CPU fixture. Actual grouped HIP reductions/projector/network/JIT parity remain parent GPU gates.',
    source_sha256={str(path):c.sha(path) for path in (Path(__file__),c.BASE/'experiments/batched-vision.py',c.BASE/'experiments/vision-cache.py',c.BASE/'experiments/projector-kernels.py')}))
  print(json.dumps(dict(cpu_passed=True,grouping_cases=len(group_checks),per_image_controls=len(quant_records),gpu_executed=False),indent=2))


if __name__=='__main__':main()

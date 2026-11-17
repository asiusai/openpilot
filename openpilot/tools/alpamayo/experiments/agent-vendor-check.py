"""CPU checks for packed layout, fixed-scale dispatch, numerical reports, and cleanup."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location('agent_vendor_common',Path(__file__).with_name('agent-vendor-common.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  args = parser.parse_args()
  if os.environ.get('DEV') != 'CPU':
    parser.error('Set DEV=CPU explicitly')
  from tinygrad import Tensor,Device,TinyJit,dtypes
  from tinygrad.helpers import GlobalCounters
  from tinygrad.runtime.ops_amd import AMDComputeQueue
  from amd_gemm import pack,unpack
  quantizer = common.module('quantize-packed')
  route = common.module('agent-vendor-route')
  bench = common.module('agent-vendor-bench')
  # Treat layout copies as bytes. All codes, including signed zero and subnormals, must survive.
  codes = (np.arange(512,dtype=np.uint16)%256).astype(np.uint8).reshape(16,32)
  fp8 = common.byte_layout(Tensor(codes,device='CPU').realize())
  np.testing.assert_array_equal(common.unpack_activation(common.pack_bytes(fp8),16).bitcast(dtypes.uint8).numpy(),codes)
  packed = common.pack_bytes(fp8)
  byte_replay = TinyJit(lambda source:common.unpack_activation(source,16))
  for _ in range(2):
    byte_replay(packed)
  for shift in (1,17,63):
    current_codes = np.roll(codes,shift,axis=1).copy()
    current_fp8 = common.byte_layout(Tensor(current_codes,device='CPU').realize())
    common.assign_bytes(packed,common.pack_bytes(current_fp8))
    np.testing.assert_array_equal(byte_replay(packed).bitcast(dtypes.uint8).numpy(),current_codes)
  import functools
  parameters = [Tensor(np.arange(4,dtype=np.float32)+index,device='CPU').realize() for index in range(40)]
  weight = Tensor(np.full(4,2,dtype=np.float32),device='CPU').realize()
  sx = Tensor([.5],device='CPU',dtype=dtypes.float32).realize()
  sw = Tensor([1.0],device='CPU',dtype=dtypes.float32).realize()
  forty = TinyJit(functools.partial(bench.forty_calls,lambda a,b,s,t:(a*b*s+t).realize()))
  for _ in range(2):
    forty(*parameters,weight,sx,sw)
  before = GlobalCounters.kernel_count
  output = forty(*parameters,weight,sx,sw)
  assert GlobalCounters.kernel_count-before == 40
  np.testing.assert_array_equal(output.numpy(),np.arange(4,dtype=np.float32)+40)
  common.assign_bytes(parameters[-1],Tensor(np.arange(4,dtype=np.float32)+99,device='CPU').realize())
  np.testing.assert_array_equal(forty(*parameters,weight,sx,sw).numpy(),np.arange(4,dtype=np.float32)+100)
  prefix = 'vlm.model.language_model.layers.0'
  class Model:
    def __init__(self):
      self.weights,self.weight_scales,self.host_weights = {},{},{}
      for suffix,(rows,width) in (('fc1',(9216,2048)),('fc2',(2048,9216))):
        values = ((np.arange(rows,dtype=np.uint16)[:,None]+3*np.arange(width,dtype=np.uint16)[None,:])%127).astype(np.uint8)
        self.host_weights[suffix] = values
        logical = common.byte_layout(Tensor(values,device='CPU').realize())
        self.weights[prefix+'.mlp.'+suffix+'.weight'] = unpack(common.pack_bytes(logical))
        self.weight_scales[prefix+'.mlp.'+suffix+'.weight'] = Tensor([.5],device='CPU',dtype=dtypes.float32).realize()
      self.original_calls = []
    def quantize(self,value):
      raise AssertionError('Prefill must use the installed fixed packed quantizer')
    def linear(self,value,name,quantized_input=None,heads=None):
      self.original_calls.append(name)
      return {'original_path':name}
    def text_layer(self,value,cos,sin,name,cache=None,mask=None,cache_pos=None):
      if value.shape[1] < 128:
        return self.linear(value,name+'.mlp.fc1')
      # Match the real prefill quantizer sequence: shared QKV, output projection, fc1, fc2.
      self.quantize(value)
      self.quantize(value)
      h = self.linear(value,name+'.mlp.fc1')
      return self.linear(h,name+'.mlp.fc2')
  class Vendor:
    def __init__(self,model):
      self.model,self.calls,self.fail = model,[],False
    def gemm(self,x,w,scales,ones,bias,zero,index):
      if self.fail:
        raise RuntimeError('Expected vendor fixture failure')
      suffix = 'fc1' if w.shape == (9216,2048) else 'fc2'
      expected_x = 8.0 if suffix == 'fc1' else 13.0
      expected_scale = .125 if suffix == 'fc1' else .25
      sample = x[:2,:32].float().numpy()
      np.testing.assert_array_equal(sample,np.full_like(sample,expected_x))
      assert float(scales[0].item()) == expected_scale
      assert float(scales[1].item()) == .5
      np.testing.assert_array_equal(w.bitcast(dtypes.uint8).numpy(),self.model.host_weights[suffix])
      self.calls.append({'suffix':suffix,'solution':index,'activation_scale':expected_scale,'weight_scale':.5})
      return Tensor.full((2112,w.shape[0]),3.25,device='CPU',dtype=dtypes.bfloat16).realize()
  model = Model()
  fixed = {prefix+'/'+str(index):value for index,value in enumerate((.03125,.0625,.125,.25))}
  quantizer.install(model,fixed_scales=fixed,capture_scales=True)
  original_linear,original_layer,original_quantize = model.linear,model.text_layer,model.quantize
  vendor = Vendor(model)
  current = Tensor.ones(1,2112,2048,device='CPU',dtype=dtypes.bfloat16).contiguous().realize()
  with route.install_prefill(model,quantizer.PackedActivation,vendor) as stats:
    result = model.text_layer(current,None,None,prefix,cache_pos=0)
    np.testing.assert_array_equal(result[:,:2,:8].float().numpy(),np.full((1,2,8),3.25,dtype=np.float32))
    assert model.quantize == original_quantize
    assert len(vendor.calls) == 2
    assert set(model.prefill_activation_scales) == set(fixed)
    # These requests must retain the previous implementation and never reach the vendor.
    model.linear(current[:,:1],'vlm.model.visual.encoder.layers.0.mlp.fc1')
    model.text_layer(current[:,:1],None,None,prefix,cache_pos=17)
    model.text_layer(current[:,:64],None,None,'expert.expert.layers.0')
    assert len(vendor.calls) == 2
  assert model.linear == original_linear and model.text_layer == original_layer
  vendor.fail = True
  try:
    with route.install_prefill(model,quantizer.PackedActivation,vendor):
      model.text_layer(current,None,None,prefix,cache_pos=0)
  except RuntimeError as error:
    assert str(error) == 'Expected vendor fixture failure'
  else:
    raise AssertionError('Expected fixture failure did not occur')
  assert model.linear == original_linear and model.text_layer == original_layer

  # Exercise the reviewed ABI context's successful and exceptional cleanup without a GPU.
  previous = os.environ.get('ALPAMAYO_VENDOR_SOLUTIONS')
  os.environ['ALPAMAYO_VENDOR_SOLUTIONS'] = str(common.METADATA)
  try:
    binary_vendor = common.module('vendor-gemm')
    queue_method = AMDComputeQueue.kernargs
    fake = SimpleNamespace(is_usb=True,arch='gfx1200',synchronize=lambda:None)
    with binary_vendor.argument_adapter(fake):
      assert binary_vendor.ACTIVE and AMDComputeQueue.kernargs is not queue_method
    assert not binary_vendor.ACTIVE and AMDComputeQueue.kernargs is queue_method
    try:
      with binary_vendor.argument_adapter(fake):
        raise LookupError('Expected ABI fixture failure')
    except LookupError:
      pass
    assert not binary_vendor.ACTIVE and AMDComputeQueue.kernargs is queue_method
    for index in common.SOLUTIONS:
      values = binary_vendor.argument_values(index,(2112,9216,2048),tuple(range(1,9)),32)
      assert (values['A'],values['B'],values['D'],values['C']) == (3,2,1,8)
      assert (values['AddressScaleA'],values['AddressScaleB']) == (5,4)
      assert values['SizesFree0'] == 9216 and values['SizesFree1'] == 2112 and values['SizesSum0'] == 2048
  finally:
    if previous is None:
      os.environ.pop('ALPAMAYO_VENDOR_SOLUTIONS',None)
    else:
      os.environ['ALPAMAYO_VENDOR_SOLUTIONS'] = previous

  # Independently known midpoint tests for BF16 ties-to-even and deviation reporting.
  known = np.array([1+2**-8,1+3*2**-8,-1-2**-8,0],dtype=np.float32)
  np.testing.assert_array_equal(common.bf16_round(known),np.array([0x3f80,0x3f82,0xbf80,0],dtype=np.uint16))
  original = common.bf16_round(np.array([[1,2,0]],dtype=np.float32))
  changed = original.copy(); changed[0,0] += 1
  deviation = common.differences(original,changed)
  assert not deviation['exact_vs_original'] and deviation['differing_bf16_values'] == 1
  assert deviation['bf16_ulp_delta_changed']['max'] == 1
  oracle = common.fp64_samples(original,changed,np.array([[1,2]],dtype=np.float32),
                               np.array([[1,0],[0,1],[0,0]],dtype=np.float32),1,1,np.zeros(3,dtype=np.float32),3)
  assert oracle['differing_values']['candidate_worse'] == 1
  opened = sorted(Device._opened_devices)
  assert all(name.split(':')[0] in ('CPU','PYTHON','NPY') for name in opened),opened
  report = {'status':'passed','opened_devices':opened,'vendor_dispatch_fixture':vendor.calls,
            'raw_fp8_codes_preserved':256,'byte_layout_jit_mutations':3,
            'cpu_kernels_per_forty_call_graph':40,'forty_call_mutation_checked':True,
            'fixed_scale_sequence_preserved':True,'packed_weight_and_activation_layout_checked':True,
            'no_output_rescaling':True,'original_vision_expert_decode_dispatch_preserved':True,
            'method_cleanup_on_success_and_failure':True,'abi_adapter_cleanup_on_success_and_failure':True,
            'independent_bf16_midpoints_checked':True,'deviations_are_reported_not_rejected':True,
            'limit':'CPU fixture only. No vendor binary executed and no actual model or route quality validated.'}
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
  main()

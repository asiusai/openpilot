"""Check cached rotary uploads through capture, replay, and changed input bytes."""
import argparse
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from tinygrad import Device,dtypes


def module(name):
  spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
  result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--checkpoint',type=Path,required=True)
  parser.add_argument('--sample',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  torch.set_num_threads(4)
  adapter=module('fast-inputs').FastInputs(args.checkpoint,camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True)
  data=module('nvidia-reference').load_raw(args.sample)
  config,_,prepared=adapter.prepare(data);packed=adapter.native_inputs(prepared)
  cls=module('compact-inputs').CompactInputs
  settings=SimpleNamespace(max_tokens=64,vision_batch=1,byte_pixels=True,cache_rotary_inputs=False)
  arrays=cls.arrays(settings,prepared,packed)
  ordinary=cls(settings,config,prepared,packed,arrays=arrays)
  settings.cache_rotary_inputs=True
  cached=cls(settings,config,prepared,packed,arrays=arrays)
  initial=cached.rotary_uploaded_bytes
  keys=list(range(10));cache=SimpleNamespace(keys=keys,verify=False,cache={})
  ordinary.selective_cache=cached.selective_cache=cache
  rng=np.random.default_rng(9060)
  for i in range(8):
    updated=[a.copy() for a in arrays]
    if i in (3,4): updated[3].flat[0]^=1
    assert ordinary.update(settings,config,prepared,packed,updated)
    assert cached.update(settings,config,prepared,packed,updated)
    if i<3: assert cached.rotary_uploaded_bytes==initial
    noise=rng.standard_normal((1,64,2)).astype(np.float32)
    expected=ordinary.upload(noise);actual=cached.upload(noise)
    Device[Device.DEFAULT].synchronize()
    for a,b in zip(expected,actual,strict=True):
      if a.dtype==dtypes.bfloat16: a,b=a.bitcast(dtypes.uint16),b.bitcast(dtypes.uint16)
      np.testing.assert_array_equal(a.numpy(),b.numpy())
    cache.cache={key:True for key in keys if i>=2 or key not in (4,9)}
  assert cached.rotary_uploaded_bytes==initial+2*arrays[3].nbytes
  args.output.write_text(json.dumps({'checks':8,'exact':True,'initial_rotary_bytes':initial,
    'total_rotary_bytes':cached.rotary_uploaded_bytes,'device':Device.DEFAULT})+'\n')
  print('PASS: cached rotary arrays match all uploads through replay, mutation, restoration, and selective-image changes')


if __name__=='__main__': main()

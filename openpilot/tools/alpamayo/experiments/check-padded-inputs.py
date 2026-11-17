"""Verify padding leaves actual token positions and every future rotary input exact."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def module(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--checkpoint',required=True)
  p.add_argument('--samples',type=Path,nargs='+',required=True)
  args=p.parse_args();torch.set_num_threads(4)
  adapter=module('fast-inputs').FastInputs(args.checkpoint,camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True)
  compact=module('compact-inputs').CompactInputs
  settings=SimpleNamespace(max_tokens=64,byte_pixels=True)
  for sample in args.samples:
    data=module('nvidia-reference').load_raw(sample)
    _,_,prepared=adapter.prepare(data);packed=adapter.native_inputs(prepared)
    actual=packed[0].shape[1]
    padded,newpacked=module('padded-prefill').pad_inputs(prepared,packed,2112)
    before,after=(compact.arrays(settings,a,b) for a,b in ((prepared,packed),(padded,newpacked)))
    for i,(a,b) in enumerate(zip(before,after,strict=True)):
      if i==2:b=b[:,:actual]
      elif i in (3,4):b=b[:,:,:actual]
      np.testing.assert_array_equal(a,b)
    assert padded['actual_prompt_length']==actual
    assert compact.signature(SimpleNamespace(max_tokens=64,byte_pixels=True,vision_batch=1),adapter.config,prepared,packed)==compact.signature(SimpleNamespace(max_tokens=64,byte_pixels=True,vision_batch=1),adapter.config,padded,newpacked)
    print(f'{actual} -> 2112: real prompt, camera inputs, decode and action rotary arrays exact',flush=True)


if __name__=='__main__':main()

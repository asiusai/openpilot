"""Verify temporal thinning retains exact selected full-resolution pixels and ego history."""
import argparse
import importlib.util
from pathlib import Path
import torch


def module(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--checkpoint',required=True)
  p.add_argument('--sample',type=Path,required=True)
  args=p.parse_args();torch.set_num_threads(4)
  fast=module('fast-inputs').FastInputs
  data=module('nvidia-reference').load_raw(args.sample)
  kwargs=dict(camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True)
  _,_,full=fast(args.checkpoint,**kwargs).prepare(data)
  pixels=full['tokenized_data']['pixel_values'];grids=full['tokenized_data']['image_grid_thw']
  offsets=[0,*grids.prod(dim=1).cumsum(dim=0).tolist()]
  for indices in ((0,1,4),(1,4),(4,)):
    adapter=fast(args.checkpoint,**kwargs,front_history_indices=indices)
    _,_,reduced=adapter.prepare(data)
    selected=[i+base for base in (0,5) for i in indices]
    expected=torch.cat([pixels[offsets[i]:offsets[i+1]] for i in selected])
    assert torch.equal(reduced['tokenized_data']['pixel_values'],expected)
    assert torch.equal(reduced['tokenized_data']['image_grid_thw'],grids[selected])
    for name in ('ego_history_xyz','ego_history_rot'):assert torch.equal(reduced[name],full[name])
    packed=adapter.native_inputs(reduced)
    assert packed[0].shape==reduced['tokenized_data']['input_ids'].shape
    assert packed[3].shape[0]==expected.shape[0]
    print(f'{indices}: selected full-resolution pixel patches and ego history exact; {packed[0].shape[1]} prompt tokens',flush=True)


if __name__=='__main__':main()

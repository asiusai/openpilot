"""Independent CPU baseline comparisons and paired chronological input timing."""
import argparse
from collections import Counter
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace
import numpy as np
import torch


def local(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),Path(__file__).with_name(name+'.py'))
  value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def load(path):
  return {file.stem:[str(np.load(file,allow_pickle=False).item())] if file.stem=='nav_text'
          else torch.from_numpy(np.load(file,allow_pickle=False)) for file in Path(path).glob('*.npy')}


def exact(a,b,label):
  if isinstance(a,torch.Tensor):
    assert isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype,label
    if not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
      raise AssertionError(label)
  elif isinstance(a,np.ndarray):
    assert isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype,label
    np.testing.assert_array_equal(a.view(np.uint8).reshape(-1),b.view(np.uint8).reshape(-1),err_msg=label)
  elif isinstance(a,dict):
    assert set(a)==set(b),label
    for key in a:exact(a[key],b[key],label+'/'+key)
  else:assert a==b,label


def compare_outputs(left,right,label):
  lp,lk,la=left;rp,rk,ra=right
  exact(lp,rp,label+'/prepared')
  for index,(a,b) in enumerate(zip(lk,rk,strict=True)):exact(a,b,label+f'/packed/{index}')
  for index,(a,b) in enumerate(zip(la,ra,strict=True)):exact(a,b,label+f'/upload/{index}')


def prepare(adapter,data,compact,settings):
  started=time.perf_counter_ns();_,_,prepared=adapter.prepare(data);after_prepare=time.perf_counter_ns()
  packed=adapter.native_inputs(prepared);after_native=time.perf_counter_ns()
  arrays=compact.arrays(settings,prepared,packed);finished=time.perf_counter_ns()
  timing=dict(prepare_ms=(after_prepare-started)/1e6,native_ms=(after_native-after_prepare)/1e6,
              arrays_ms=(finished-after_native)/1e6,total_ms=(finished-started)/1e6)
  return (prepared,packed,arrays),timing


def summary(timings):
  return {key:dict(median=float(np.median([row[key] for row in timings])),p95=float(np.percentile([row[key] for row in timings],95)))
          for key in timings[0]} if timings else {}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--checkpoint',required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--work',type=Path,default=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116'))
  parser.add_argument('--limit',type=int,default=0)
  parser.add_argument('--warmup',type=int,default=12)
  parser.add_argument('--only',choices=('ring','floor','quality','all'),default='all')
  parser.add_argument('--adversarial-only',action='store_true')
  args=parser.parse_args();torch.set_num_threads(4)
  baseline,candidate=local('fast-inputs'),local('agent-cpu-inputs-adapter')
  compact=local('compact-inputs').CompactInputs
  settings=SimpleNamespace(max_tokens=64,byte_pixels=True)
  options=dict(camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True,cache_resize=True)
  groups={name:[row['sample'] for row in json.loads((args.work/'stream-10hz'/name/'manifest.json').read_text())['schedule']]
          for name in ('ring','floor')}
  groups['quality']=json.loads((args.work/'quality-samples.json').read_text())
  if args.only!='all':groups={args.only:groups[args.only]}
  if args.adversarial_only:groups={}
  report=dict(cpu_only=True,torch_threads=torch.get_num_threads(),python=platform.python_version(),torch_version=torch.__version__,
              cpu_affinity=sorted(os.sched_getaffinity(0)),load_average=os.getloadavg(),groups={},adversarial=[],options=options,
              timing_boundary='adapter.prepare + native_inputs + CompactInputs.arrays; source file loading and native RGB calibration excluded')
  first_data=load(json.loads((args.work/'quality-samples.json').read_text())[0]) if args.adversarial_only else None
  for name,paths in groups.items():
    if args.limit:paths=paths[:args.limit]
    adapters={'baseline':baseline.FastInputs(args.checkpoint,**options),'candidate':candidate.FastInputs(args.checkpoint,**options)}
    timings={'baseline':[],'candidate':[]};rows=[]
    for index,path in enumerate(paths):
      data=load(path)
      if first_data is None:first_data={key:value.clone() if isinstance(value,torch.Tensor) else value[:] for key,value in data.items()}
      outputs={};per_call={}
      for key in (('baseline','candidate') if index%2==0 else ('candidate','baseline')):
        outputs[key],per_call[key]=prepare(adapters[key],data,compact,settings)
      compare_outputs(outputs['baseline'],outputs['candidate'],path)
      if index>=args.warmup:
        for key in timings:timings[key].append(per_call[key])
      rows.append(dict(sample=path,measured=index>=args.warmup,timings=per_call))
      if index%40==0:print(json.dumps(dict(group=name,index=index,exact=True,timings=per_call)),flush=True)
    report['groups'][name]=dict(checked=len(paths),exact=True,warmup=args.warmup,measured=len(timings['baseline']),
                               timing={key:summary(value) for key,value in timings.items()},
                               cache=adapters['candidate'].cache_state(),rows=rows)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(group=name,result=report['groups'][name]['timing'])),flush=True)
    del adapters,outputs;gc.collect()
  # Use independent adapter objects. Each comparison includes raw patch bytes,
  # all tokenized fields, learned-history IDs, positional values, and upload arrays.
  adapters=[baseline.FastInputs(args.checkpoint,**options),candidate.FastInputs(args.checkpoint,**options)]
  def check(label,data):
    outputs=[prepare(adapter,data,compact,settings)[0] for adapter in adapters]
    compare_outputs(*outputs,label)
    report['adversarial'].append(dict(case=label,exact=True))
    return outputs
  warm=check('warm',first_data)
  grids=warm[1][0]['tokenized_data']['image_grid_thw']
  assert grids.shape==(10,3) and torch.all(grids.prod(dim=1)==720)
  report['latest_full_resolution_patch_rows']=720
  changed=dict(first_data,image_frames=first_data['image_frames'].clone())
  changed['image_frames'][1,0,0,20,21]^=127
  changed['image_frames'][-1,0,2,27,32]^=255
  check('same camera and timestamp; history and latest pixel mutations',changed)
  changed=dict(first_data,image_frames=first_data['image_frames'].clone())
  changed['image_frames'].numpy()[-1,0,1,25,33]^=173
  check('same timestamp pixel mutation through NumPy without Tensor version change',changed)
  changed=dict(first_data,nav_text=['Turn left at the next intersection, then continue for 200 meters.'])
  check('navigation changes with unchanged images',changed)
  permutation=torch.tensor([9,0,7,2,5,4,3,6,1,8])
  changed=dict(first_data,**{key:first_data[key][permutation].contiguous() for key in ('image_frames','camera_indices','absolute_timestamps')})
  check('input camera and history reordering',changed)
  changed=dict(first_data,absolute_timestamps=first_data['absolute_timestamps'].clone())
  changed['absolute_timestamps'][0]+=60000
  check('one actual timestamp changes its relative-time label',changed)
  changed=dict(first_data,absolute_timestamps=first_data['absolute_timestamps']+17)
  check('all actual timestamps change without relative-time label changes',changed)
  changed=dict(first_data,ego_history_xyz=first_data['ego_history_xyz'].clone())
  changed['ego_history_xyz'][0,-1,1]+=.4
  check('learned ego history changes with unchanged image and text structure',changed)
  outputs=check('cache warm before output mutation',first_data)
  outputs[1][0]['tokenized_data']['pixel_values'][0,0]^=255
  outputs[1][0]['tokenized_data']['input_ids'][0,0]^=1
  outputs[1][1][1][0,0,0]+=1000
  check('mutated returned patches and RoPE cannot poison caches',first_data)
  adapters=[baseline.FastInputs(args.checkpoint,**options),candidate.FastInputs(args.checkpoint,patch_cache_entries=3,patch_cache_bytes=4*1024*1024,rope_cache_entries=1,text_cache_entries=1,**options)]
  for repeat in range(3):
    data=first_data if repeat!=1 else dict(first_data,nav_text=['Stop at the next safe location and wait for further instructions.'])
    check(f'bounded cache eviction {repeat}',data)
  assert len(adapters[1].patch_cache)<=3 and adapters[1].patch_bytes<=4*1024*1024
  assert adapters[1].cpu_input_statistics['patch_evictions']>0
  assert adapters[1].cpu_input_statistics['text_evictions']>0 and adapters[1].cpu_input_statistics['rope_evictions']>0
  report['eviction_cache']=adapters[1].cache_state()
  for extra in ({'history_pixel_scale':.5},{'history_feature_scale':.5},{'front_history_indices':(0,2,4)},
                {'narrow_crop':.9},{'pixel_scale':.75},
                {'camera_pixel_scales':(1.0,.75,1.0,1.0,1.0,1.0)}):
    adapters=[kind.FastInputs(args.checkpoint,**options,**extra) for kind in (baseline,candidate)]
    check('existing option '+repr(extra),first_data)
    check('existing option warmed '+repr(extra),first_data)
  errors=[]
  for kind in (baseline,candidate):
    adapter=kind.FastInputs(args.checkpoint,**options,wide_equidistant_focal=120.0)
    try:prepare(adapter,first_data,compact,settings)
    except ValueError as error:errors.append(str(error))
    else:raise AssertionError('The current baseline wide projection unexpectedly accepted this 4-D frame')
  assert errors==['Expected a CPU RGB uint8 image']*2
  report['adversarial'].append(dict(case='existing wide projection rejects 4-D fixture equally',exact=True,errors=errors))
  report['complete']=True;report['load_average_end']=os.getloadavg()
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(dict(complete=True,groups={key:value['checked'] for key,value in report['groups'].items()},
                       adversarial_cases=len(report['adversarial']))),flush=True)


if __name__=='__main__':main()

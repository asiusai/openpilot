"""Bounded CPU profiling and exact comparisons on the current recorded live recipe."""
import argparse
from collections import OrderedDict
from contextlib import ExitStack
import copy
import cProfile
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pstats
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
EXP=SOURCE/'experiments'
HERE=Path(__file__).resolve().parent
ARTIFACTS=WORK/'agent-live-cpu-fast-artifacts'
CHECKPOINT=Path('/tmp/opencode/alpamayo-asius-20261114/model')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def load(path,name=None):
  path=Path(path);name=name or path.stem.replace('-','_')
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
  sys.modules[name]=module;spec.loader.exec_module(module);return module


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def configure():
  ARTIFACTS.mkdir(exist_ok=True)
  for name in ('tmp','cache'):(ARTIFACTS/name).mkdir(exist_ok=True)
  os.environ.update(DEV='CPU',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    PYTHONDONTWRITEBYTECODE='1',CACHEDB=str(ARTIFACTS/'compiler-cache.db'),TMPDIR=str(ARTIFACTS/'tmp'),
    XDG_CACHE_HOME=str(ARTIFACTS/'cache'))
  sys.path[:0]=[str(TINY),str(EXP),str(SOURCE),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']


def exact(a,b,label='value'):
  import numpy as np
  import torch
  if isinstance(a,torch.Tensor):
    assert isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype,label
    assert np.array_equal(a.contiguous().view(torch.uint8).numpy(),b.contiguous().view(torch.uint8).numpy()),label
  elif isinstance(a,np.ndarray):
    assert isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype,label
    assert a.tobytes()==b.tobytes(),label
  elif isinstance(a,dict):
    assert a.keys()==b.keys(),label
    for key in a:exact(a[key],b[key],label+'/'+str(key))
  elif isinstance(a,(tuple,list)):
    assert type(a) is type(b) and len(a)==len(b),label
    for i,(x,y) in enumerate(zip(a,b,strict=True)):exact(x,y,label+'/'+str(i))
  else:assert a==b,label


def summary(rows):
  import numpy as np
  return {key:dict(median_ms=float(np.median([r[key] for r in rows])),p95_ms=float(np.percentile([r[key] for r in rows],95)))
          for key in rows[0]}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--queries',type=int,default=20)
  parser.add_argument('--preparer-queries',type=int,default=6)
  args=parser.parse_args()
  assert 8<=args.queries<=24 and 0<=args.preparer_queries<=6 and not args.output.exists()
  assert args.output.resolve().is_relative_to(WORK)
  configure()
  import numpy as np
  import torch
  from tinygrad import Device,Tensor
  torch.set_num_threads(4)
  opened=[];original_device=type(Device).__getitem__
  def cpu_device(self,name):
    if str(name).split(':')[0] not in ('CPU','NPY'):raise AssertionError('GPU device access forbidden: '+str(name))
    opened.append(str(name));return original_device(self,name)
  def no_network(*a,**kw):raise AssertionError('Network forbidden')
  with ExitStack() as stack:
    stack.enter_context(patch.object(type(Device),'__getitem__',cpu_device))
    stack.enter_context(patch.object(socket.socket,'connect',no_network))
    A=load(EXP/'agent-cpu-inputs-adapter.py')
    K=load(EXP/'compact-inputs.py');L=load(EXP/'agent-live-consumer.py')
    policy=load(EXP/'agent-speculation-policy.py')
    fast=load(HERE/'agent-live-cpu-fast-hook.py')
    settings=SimpleNamespace(max_tokens=64,vision_batch=1,byte_pixels=True,cache_rotary_inputs=False)
    options=dict(camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True,cache_resize=True)
    adapters=[A.FastInputs(CHECKPOINT,**options) for _ in range(2)]
    class CPUStorage(K.CompactInputs):
      """Only initial GPU allocation is excluded. CPU Tensors and update are canonical."""
      def __init__(self,args,config,prepared,packed,arrays=None):
        self._signature=self.signature(args,config,prepared,packed)
        self.cpu=[Tensor(a,device='CPU').realize() for a in arrays]
        self.cache_rotary_inputs=False
        self.selective_cache=None
    owners=[]
    for adapter in adapters:
      owner=L.Consumer.__new__(L.Consumer)
      owner.adapter=adapter;owner.config=adapter.config;owner.settings=settings;owner.input_class=CPUStorage
      owner.inputs=None;owner.position_hashes=OrderedDict();owner.budgets=L.Budgets();owner.vision_cache=object()
      owner.constructed_inputs=owner.reused_inputs=0;owners.append(owner)
    handle=fast.install(owners[1]);stack.callback(handle.close)
    draft=policy.CHECKPOINT_DRAFT_IDS;anchor=adapters[0].config.traj_ids['future_start']
    source_files=[EXP/name for name in ('agent-live-consumer.py','agent-cpu-inputs-adapter.py','fast-inputs.py',
      'compact-inputs.py','agent-speculation-policy.py','agent-live-bridge.py','agent-paced-replay.py')]+[SOURCE/'inputs.py',HERE/'agent-live-cpu-fast-hook.py',Path(__file__)]
    report=dict(complete=False,cpu_only=True,gpu_executed=False,torch_threads=4,numpy_version=np.__version__,torch_version=torch.__version__,rows=[],controls=[],
      source_sha256={str(p):sha(p) for p in source_files},overlay=handle.provenance,
      boundary='Measured: adapter.prepare, native_inputs, nominal pad2048, CompactInputs.arrays, Consumer._bind_inputs including real CPU buffer update. Excluded: source file IO, exact comparisons, GPU upload/initial allocation/inference.',
      cpu_storage='Actual tinygrad CPU Tensor buffers and original CompactInputs.update; initial GPU allocation omitted equally in both arms.',
      load_average=os.getloadavg(),affinity=sorted(os.sched_getaffinity(0)))
    def run(arm,data):
      adapter=adapters[arm];owner=owners[arm]
      start=time.perf_counter_ns();_,_,prepared=adapter.prepare(data);p=time.perf_counter_ns()
      packed=adapter.native_inputs(prepared);n=time.perf_counter_ns()
      prepared,packed=policy.padded_inputs(prepared,packed,2048,draft,anchor)
      arrays=K.CompactInputs.arrays(settings,prepared,packed);a=time.perf_counter_ns()
      bound=owner._bind_inputs(prepared,packed,arrays);b=time.perf_counter_ns()
      return (prepared,packed,arrays,bound),dict(prepare=(p-start)/1e6,native=(n-p)/1e6,pad_arrays=(a-n)/1e6,
        bind=(b-a)/1e6,total=(b-start)/1e6)
    def compare(left,right):
      exact(left,right)
      for i,(a,b) in enumerate(zip(owners[0].inputs.cpu,owners[1].inputs.cpu,strict=True)):
        exact(a.numpy(),b.numpy(),'actual_cpu_buffer/'+str(i))
      assert owners[0].position_hashes==owners[1].position_hashes
      assert owners[0].constructed_inputs==owners[1].constructed_inputs and owners[0].reused_inputs==owners[1].reused_inputs
    manifest_path=WORK/'stream-10hz/floor/manifest.json'
    schedule=json.loads(manifest_path.read_text())['schedule'][:args.queries*4:4]
    report['recorded_manifest_sha256']=sha(manifest_path)
    first=None
    for index,row in enumerate(schedule):
      begin=time.perf_counter_ns()
      data={p.stem:[str(np.load(p,allow_pickle=False).item())] if p.stem=='nav_text' else torch.from_numpy(np.load(p,allow_pickle=False))
        for p in Path(row['sample']).glob('*.npy')}
      io_ms=(time.perf_counter_ns()-begin)/1e6
      assert data['image_frames'].shape==(10,1,3,432,768)
      if first is None:first=data
      outputs={};timing={}
      for arm in ([0,1] if index%2==0 else [1,0]):outputs[arm],timing[str(arm)]=run(arm,data)
      compare(outputs[0],outputs[1])
      assert outputs[0][1][3].shape==fast.POSITION_SHAPE
      report['rows'].append(dict(sample=row['sample'],query_ns=row['query_ns'],actual_timestamps=data['absolute_timestamps'].tolist(),
        measured=index>=4,source_load_ms=io_ms,exact=True,timing=timing))
    report['timing']={str(arm):summary([row['timing'][str(arm)] for row in report['rows'] if row['measured']]) for arm in (0,1)}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Profile separately: profiler work is never included in paired timing.
    profile=cProfile.Profile();profile.enable();profiled,_=run(0,first);profile.disable()
    paired,_=run(1,first);compare(profiled,paired)
    text=io.StringIO();pstats.Stats(profile,stream=text).strip_dirs().sort_stats('cumulative').print_stats(20)
    report['baseline_cpu_profile']=text.getvalue()
    def control(label,data):
      outputs=[run(arm,data)[0] for arm in (0,1)];compare(*outputs)
      report['controls'].append(dict(case=label,exact=True));return outputs
    control('restored original query',first)
    changed=dict(first,nav_text=['Turn left at the next junction.']);control('navigation mutation',changed)
    changed=dict(first,ego_history_xyz=first['ego_history_xyz'].clone());changed['ego_history_xyz'][0,-1,1]+=.4
    control('learned history mutation',changed)
    changed=dict(first,ego_history_rot=first['ego_history_rot'].clone());changed['ego_history_rot'][0,-1,0,1]+=.001
    control('history rotation mutation',changed)
    changed=dict(first,image_frames=first['image_frames'].clone());changed['image_frames'].numpy()[4,0,1,23,17]^=127
    control('current image NumPy mutation without version shortcut',changed)
    changed=dict(first,image_frames=first['image_frames'].clone());changed['image_frames'][0,0,2,31,19]^=255
    control('old image mutation',changed)
    changed=dict(first,absolute_timestamps=first['absolute_timestamps'].clone());changed['absolute_timestamps'][0]+=17000
    control('actual timestamp mutation',changed)
    outputs=control('prepare position guard controls',first)
    prepared,packed,arrays,_=outputs[0]
    for label,index,bits in [('first-byte mutation',0,1),('last-byte mutation',packed[3].size-1,1),
        ('signed zero',33,0x80000000),('subnormal',34,1),('NaN payload',35,0x7fc00001)]:
      positions=packed[3].copy();words=positions.view(np.uint32).reshape(-1)
      if label in ('first-byte mutation','last-byte mutation'):words[index]^=bits
      else:words[index]=bits
      changed=(*packed[:3],positions);errors=[]
      for owner in owners:
        try:owner._bind_inputs(prepared,changed,arrays)
        except ValueError as error:errors.append(str(error))
        else:raise AssertionError('Changed fixed positions accepted: '+label)
      assert errors==['Fixed vision-position contents changed under an existing cached layout']*2
      report['controls'].append(dict(case=label,rejected_exactly=True))
      exact(owners[0]._bind_inputs(prepared,packed,arrays),owners[1]._bind_inputs(prepared,packed,arrays))
    control('changed then restored positions and input arrays',first)
    words=packed[3].view(np.uint32).reshape(-1);saved=int(words[17]);words[17]^=1
    try:
      for owner in owners:
        try:owner._bind_inputs(prepared,packed,arrays)
        except ValueError as error:assert str(error)=='Fixed vision-position contents changed under an existing cached layout'
        else:raise AssertionError('In-place position mutation bypassed the guard')
    finally:words[17]=saved
    exact(owners[0]._bind_inputs(prepared,packed,arrays),owners[1]._bind_inputs(prepared,packed,arrays))
    report['controls'].append(dict(case='in-place mutation of cached position object then exact restoration',rejected_exactly=True))
    # Equal new arrays may hit; altered contents never rely on object identity or versions.
    cloned=(*packed[:3],packed[3].copy())
    exact(owners[0]._bind_inputs(prepared,cloned,arrays),owners[1]._bind_inputs(prepared,cloned,arrays))
    report['controls'].append(dict(case='equal new position object',exact=True))
    try:fast.install(owners[1])
    except ValueError:report['controls'].append(dict(case='duplicate installation rejected',rejected_exactly=True))
    else:raise AssertionError('Duplicate binding hook accepted')
    assert handle.guard.counts['full_content_hits']>0 and handle.guard.counts['full_content_misses']>5
    if args.preparer_queries:report['live_preparer']=preparer_profile(args.preparer_queries,run,compare)
    report['guard']=handle.guard.state()
    original=handle.original;handle.close();assert owners[1]._bind_inputs==original and '_bind_inputs' not in vars(owners[1])
    report['hooks_restored']=True
    assert all(str(name).split(':')[0] in ('CPU','NPY') for name in Device._opened_devices)
    report.update(complete=True,gpu_executed=False,cpu_devices=sorted(set(opened)),load_average_end=os.getloadavg(),
      exact_queries=len(report['rows']),no_inference_or_live_hz_claim=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,gpu_executed=False,queries=len(report['rows']),timing=report['timing'],
      controls=len(report['controls']),guard=report['guard'],preparer_queries=args.preparer_queries)),flush=True)


def preparer_profile(count,run,compare):
  import numpy as np
  import torch
  paced=load(EXP/'agent-paced-replay.py')
  trace=paced.Trace(paced.SOURCE/'manifest.json',paced.LOGS)
  sampler_root=WORK/'warp-fast-pathfix-source'
  sample=load(sampler_root/'agent-warp-fast-sampler.py','cpu_fast_existing_sampler')
  sample.ARTIFACTS=ARTIFACTS/'sampler'
  cfg=json.loads((sampler_root/'agent-warp-fast-validated-portable-v6.json').read_text())['variants'][0]
  sampler=sample.Sampler(cfg['kind'],cfg['workers'],Path(cfg['provenance']['command'][0]).name,
    cfg['provenance']['native'],cfg['max_maps'],cfg['max_bytes'])
  assert sampler.lib.provenance['library_sha256']==cfg['provenance']['library_sha256']
  # Check payloads only as they arrive, rather than reading future RGB during setup.
  root=WORK/'agent-paced-replay-cache-35-43/rgb'
  if not root.exists():
    raise ValueError('Resolve the original qualified RGB cache path before CPU profiling')
  rgb=paced.RGB(root,trace.frames,verify=False)
  inventory=json.loads((root.parent/'paced-cache.json').read_text())
  assert inventory['frame_metadata_sha256']==paced.json_hash(trace.frames)
  class Clock:
    now=trace.start;anchor=None
    def __call__(self):return self.now
  clock=Clock();verified=set()
  def arrived(ref):
    assert ref['release_ns']<=clock()
    path=rgb.path(ref);key=str(path.relative_to(root))
    if key not in verified:assert sha(path)==inventory['files'][key];verified.add(key)
    return rgb(ref)
  arrived.report=rgb.report
  store=paced.bridge.Store(frame_limit=64,byte_limit=384*1024**2)
  producer=paced.Producer(trace,store,arrived,clock)
  prep=paced.bridge.Preparer(trace.manifest['source_camera_geometry'],trace.manifest['target_camera_geometry'],warp=sampler.calibrated_image)
  rows=[];last=0;current={}
  try:
    producer.prime_history()
    for i in range(count):
      clock.now=trace.start+i*400_000_000
      start=time.perf_counter_ns();producer.pump();ingest=(time.perf_counter_ns()-start)/1e6
      selection=store.select(clock(),last,current,250_000_000,False)
      assert all(f.ready_ns<=clock() for f in selection.frames)
      start=time.perf_counter_ns();arrays,detail=prep.prepare(selection);elapsed=(time.perf_counter_ns()-start)/1e6
      arrays['nav_text']=[trace.nav(selection.query_ns)]
      data={k:v if k=='nav_text' else torch.from_numpy(v) for k,v in arrays.items()}
      outputs=[run(arm,data)[0] for arm in (0,1)];compare(*outputs)
      last=selection.query_ns;current={c:max(f.capture_ns for f in selection.frames if f.camera==c) for c in (1,6)}
      rows.append(dict(query_ns=last,preparer_ms=elapsed,ingest_io_ms=ingest,history_ms=detail['history_build_ms'],
        warp_stack_ms=detail['color_warp_stack_ms'],image_hits=detail['image_cache_hits'],
        actual_timestamps=arrays['absolute_timestamps'].tolist(),downstream_arrays_exact=True))
  finally:sampler.close();producer.close()
  return dict(rows=rows,trace=trace.provenance,sampler=sampler.state(),rgb=rgb.report(),
    checked_arrived_files=len(verified),future_pixels_read=False,preparer_unchanged=True,
    timing_boundary='Original live Preparer including measured/learned history, selected existing CPU sampler and stack. Source arrival IO is reported separately.',
    median_ms=float(np.median([row['preparer_ms'] for row in rows])),priming=producer.prime)


if __name__=='__main__':main()

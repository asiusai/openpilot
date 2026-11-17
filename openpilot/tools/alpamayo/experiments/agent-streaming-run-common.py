"""Frozen-source loading and the one transactional image-bank policy."""
from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE/'agent-streaming-plan-parent-trial'
CORE=ROOT/'alpamayo'
EXP=CORE/'experiments'
POLICY='image-bank-48-refresh-2s-all-text-current-latest-image-current'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(name, directory=HERE):
  namespace='agent_streaming_private_'+name.replace('-','_')
  if namespace in sys.modules:return sys.modules[namespace]
  spec=importlib.util.spec_from_file_location(namespace,Path(directory)/(name+'.py'))
  value=importlib.util.module_from_spec(spec);sys.modules[namespace]=value;spec.loader.exec_module(value)
  return value

def verify_sources():
  record=json.loads((ROOT/'sources.json').read_text())
  for item in record['records']:
    if sha(ROOT/item['copy'])!=item['sha256']:raise ValueError('Frozen dependency changed: '+item['copy'])
  for name in ('configuration','smoke32'):
    if sha(ROOT/(name+'.json'))!=record[name+'_sha256']:raise ValueError('Frozen '+name+' changed')
  return record

def sources():
  return {p.name:sha(p) for p in sorted(HERE.glob('agent-streaming-run-*.py'))}

def read_sample(path):
  import numpy as np
  import torch
  # Explicit current-input allowlist: never load future labels or saved outputs.
  names=('image_frames','absolute_timestamps','camera_indices','ego_history_xyz','ego_history_rot','nav_text')
  return {name:([str(np.load(Path(path)/(name+'.npy'),allow_pickle=False).item())] if name=='nav_text' else
    torch.from_numpy(np.load(Path(path)/(name+'.npy'),allow_pickle=False,mmap_mode='c'))) for name in names}

def prepare_cpu_environment():
  import os
  os.environ.update(DEV='CPU',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CACHEDB=str(ROOT/'cpu-cache.db'))
  config=json.loads((ROOT/'configuration.json').read_text())
  for key,value in config['environment'].items():
    if key.startswith('ALPAMAYO_') and value is not None:os.environ[key]=value
  for path in (CORE,EXP,Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')):sys.path.insert(0,str(path))

def validate_report(path,kind):
  report=json.loads(Path(path).read_text())
  if report.get('kind')!=kind or report.get('passed') is not True:raise ValueError('Missing passing '+kind+' report')
  if report.get('source_sha256')!=sources():raise ValueError('Prototype changed since '+kind+' validation')
  if report.get('frozen_sources_sha256')!=sha(ROOT/'sources.json'):raise ValueError('Frozen sources changed')
  if report.get('external_provenance_sha256')!=sha(ROOT/'external-provenance.json'):raise ValueError('External provenance changed')
  return report

def parent_gpu_environment(*,require_micro=True):
  import os,fcntl
  verify_sources()
  validate_report(ROOT/'cpu-checks.json','cpu-controls')
  validate_report(ROOT/'offline-compile.json','offline-gfx1200')
  if require_micro:validate_report(ROOT/'micro-gpu.json','parent-gpu-micro')
  load('agent-streaming-run-provenance').verify()
  # An enclosing parent flock must hold the same global lock as other trials.
  with open('/tmp/opencode/alpamayo-chestnut-speed.lock','a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:pass
    else:raise RuntimeError('Parent must hold global flock before GPU access')
  if os.environ.get('AGENT_STREAMING_PARENT_TIMEOUT')!='1':raise RuntimeError('Use documented timeout command')
  for key,value in json.loads((ROOT/'configuration.json').read_text())['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  os.environ.update(HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
  for path in (CORE,EXP,Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')):sys.path.insert(0,str(path))

def report_base(kind):
  return dict(kind=kind,source_sha256=sources(),frozen_sources_sha256=sha(ROOT/'sources.json'),
    external_provenance_sha256=sha(ROOT/'external-provenance.json'))

class Planner:
  def __init__(self):
    self.bank=OrderedDict();self.epoch=self.previous=self.nav=self.provenance=None;self.pending=None

  def abort(self):
    self.bank.clear();self.epoch=self.previous=self.nav=self.provenance=None;self.pending=None

  def begin(self, meta, spans, prompt, *, force_full=False):
    import numpy as np
    if self.pending is not None:raise RuntimeError('Uncommitted prediction')
    identities=meta['identities'];now=meta['timestamp_us'];nav=meta['nav'];provenance=meta['provenance']
    if len(identities)!=10 or len(spans)!=10:raise ValueError('Ten complete image records required')
    reason=('forced_full_control' if force_full else 'initial' if self.epoch is None else
      'time_reset' if now<=self.previous else 'input_gap' if now-self.previous>500000 else
      'navigation_change' if nav!=self.nav else 'provenance_change' if provenance!=self.provenance else
      'periodic_2s' if now-self.epoch>=2000000 else None)
    prior=OrderedDict() if reason else self.bank.copy()
    newest={camera:max(k[1] for k in identities if k[0]==camera) for camera in (1,6)}
    reused=[i for i,k in enumerate(identities) if k in prior and k[1]<newest[k[0]]]
    image_mask=np.zeros(prompt,dtype=bool);bank_read=np.full(2048,-1,dtype=np.int32);image_slots=[]
    for i,((a,b),key) in enumerate(zip(spans,identities,strict=True)):
      if not 0<=a<b<prompt or b-a!=180 or image_mask[a:b].any():raise ValueError('Full 180-token spans required')
      image_mask[a:b]=True;image_slots.extend(range(a,b))
      if i in reused:bank_read[a:b]=prior[key][0]*180+np.arange(180)
    selected=np.flatnonzero(bank_read[:prompt]<0).astype(np.int32)
    rows=((len(selected)+63)//64)*64
    if not 128<=rows<=2048 or selected[-1]!=prompt-1:raise ValueError('Invalid selected query rows')
    query_slots=np.pad(selected,(0,rows-len(selected)),constant_values=-1)
    query_read=np.full(2048,-1,dtype=np.int32);query_read[selected]=np.arange(len(selected))
    if not np.all((query_read[:prompt]>=0)^(bank_read[:prompt]>=0)):raise AssertionError('Every active slot has one source')
    after=prior.copy()
    for i,key in enumerate(identities):
      if i not in reused:after[key]=(-1,meta['query_us']) if key not in after else (after[key][0],meta['query_us'])
      after.move_to_end(key)
    while len(after)>48:after.popitem(last=False)
    free=iter(sorted(set(range(48))-{slot for slot,_ in after.values() if slot>=0}))
    for key,(slot,built) in list(after.items()):
      if slot<0:after[key]=(next(free),built)
    destinations=np.concatenate([np.arange(after[key][0]*180,(after[key][0]+1)*180,dtype=np.int32)
      if i not in reused else np.full(180,-1,dtype=np.int32) for i,key in enumerate(identities)])
    event=dict(policy=POLICY,classification='approximate recurrent candidate',query_timestamp_us=meta['query_us'],
      image_clock_us=now,reset=reason,reused_images=reused,refreshed_images=[i for i in range(10) if i not in reused],
      query_tokens=len(selected),query_rows=rows,original_prompt_tokens=prompt,prefill_bound=2048,
      selected_slots_sha256=hashlib.sha256(query_slots.tobytes()).hexdigest(),bank_entries=len(after),
      state_build_timestamps_us=[prior[k][1] if i in reused else meta['query_us'] for i,k in enumerate(identities)],
      capture_timestamps_us=[k[1] for k in identities],synchronous_full_refresh=bool(reason),
      reasoning_and_actions='fresh current computation; approximate language history',quality_passed=None)
    plan=dict(query_slots=query_slots,query_read=query_read,bank_read=bank_read,
      image_slots=np.asarray(image_slots,dtype=np.int32),destinations=destinations,last=len(selected)-1,
      prompt=prompt,rows=rows,event=event,after=after,epoch=now if reason else self.epoch,meta=meta)
    self.pending=plan
    return plan

  def commit(self):
    if self.pending is None:raise RuntimeError('No completed prediction to commit')
    plan=self.pending;meta=plan['meta'];self.bank=plan['after'];self.epoch=plan['epoch']
    self.previous,self.nav,self.provenance=meta['timestamp_us'],meta['nav'],meta['provenance']
    self.pending=None

def metadata(data, prepared, sample, vision_keys, provenance):
  import numpy as np
  name=Path(sample).name
  if not name.startswith('sample-') or not name[7:].isdigit():raise ValueError('Explicit query timestamp required')
  query=int(name[7:]);cameras=data['camera_indices'].reshape(-1).tolist();times=data['absolute_timestamps'].reshape(-1).tolist()
  if cameras!=[1]*5+[6]*5 or len(vision_keys)!=10:raise ValueError('Both complete camera histories required')
  for begin in (0,5):
    if any(a>=b for a,b in zip(times[begin:begin+4],times[begin+1:begin+5])):raise ValueError('Unordered history')
    for stamp,offset in zip(times[begin:begin+5],(-2000000,-1000000,-500000,-100000,0),strict=True):
      if not 0<=query+offset-stamp<=150000:raise ValueError('History capture unavailable or stale for requested target')
  for key,shape in (('ego_history_xyz',(1,20,3)),('ego_history_rot',(1,20,3,3))):
    value=data[key]
    if tuple(value.shape)!=shape or not np.isfinite(value.numpy()).all():raise ValueError('Unusable current motion history')
  nav=data['nav_text'][0]
  if not isinstance(nav,str) or not nav.strip():raise ValueError('Current navigation required')
  grids=prepared['tokenized_data']['image_grid_thw'].tolist()
  if grids!=[[1,20,36]]*10:raise ValueError('Full original image grids required')
  identities=tuple((int(camera),int(stamp),tuple(grid),digest.hex()) for camera,stamp,(grid,digest)
                   in zip(cameras,times,vision_keys,strict=True))
  return dict(query_us=query,timestamp_us=max(times),nav=nav,identities=identities,provenance=provenance)

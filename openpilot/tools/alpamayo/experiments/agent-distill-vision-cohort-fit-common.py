"""Admission and streaming for the fixed160 vision fit. No implicit input discovery."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-vision-cohort-fit-artifacts'
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
PY=BASE/'upstream/.venv/bin/python'
TINY=BASE/'tinygrad'
OLD_PLAN=WORK/'agent-distill-vision-artifacts/plan.json'
OLD_SHA='47c8d9c8b7224d0524185cf6a7d640a81d536518c6dc7fcd39ede644ae4734a3'
LEGACY=WORK/'agent-distill-vision-cohort-artifacts/legacy-v1/report.json'
LEGACY_SHA='d543c7582f6c833bbace1e921384edd4a2aadd79b52420f8a171bd743edea416'
LEDGER=WORK/'agent-distill-continue-artifacts/live-v1/frozen/agent-distill-continue-ledger.py'
LEDGER_SHA='f070ca77dedbc6f22027a2dcb063189b6c66647ae1436d022d0cbf7386b1663d'
# The canonical ledger digest is checked against the qualified old runner's identical ledger below.
STATE_BASE=WORK/'agent-distill-fit-artifacts/live-v4/frozen/agent-distill-fit-state.py'
KFD=HERE/'agent-distill-kfd1-common.py'
TARGET='KFD:1+AMD:LLVM:gfx1100'
LIMITS=dict(target=TARGET,selected_gpu_id=27211,selected_pci='0000:0c:00.0',live_allocation_limit=12*1024**3,
  cumulative_allocation_limit=256*1024**3,max_single_buffer=256*1024**2,free_reserve=5*1024**3)
HYPER=dict(updates=160,lr=3.5e-5,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,microbatch_images=1,
  loss='FP32 mean squared error over all720x1152 outputs',selection_steps=[0,160],
  order='scene=(step-1)%16; logical_image=(step-1)//16',validation='mean10images per scene; mean8scenes')


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path); value=importlib.util.module_from_spec(spec)
  sys.modules[name]=value;spec.loader.exec_module(value);return value
def original():return module('vision_cohort_fit_original',HERE/'agent-distill-vision-common.py')
def capture_adapter():return module('vision_cohort_fit_capture',HERE/'agent-distill-vision-cohort-capture.py')
def no_devices():original().no_devices()
def storage(path):
  path=Path(path).resolve()
  if not path.is_relative_to(ROOT):raise ValueError('Use the owned500GB fit artifact directory')
  while not path.exists():path=path.parent
  if subprocess.check_output(['findmnt','-n','-o','TARGET','-T',str(path)],text=True,timeout=5).strip()!='/mnt/nvme-500':
    raise ValueError('500GB workspace is not mounted')
  if __import__('shutil').disk_usage(path).free<5*1024**3:raise ValueError('Preserve5GiB free storage')


def source_pins():
  if sha(OLD_PLAN)!=OLD_SHA:raise ValueError('Changed qualified original vision plan')
  old=read(OLD_PLAN)
  for name,digest in old['sources'].items():
    if sha(name)!=digest:raise ValueError('Changed qualified source '+name)
  if sha(LEDGER)!=LEDGER_SHA or sha(LEDGER)!=old['sources'][str(original().LEDGER)]:raise ValueError('Canonical ledger differs from qualified complete ledger')
  C=capture_adapter();C.checked_plan()
  if sha(LEGACY)!=LEGACY_SHA:raise ValueError('Changed completed GPU legacy control')
  reference=read(LEGACY)
  if not reference['complete'] or not reference['gpu_executed'] or reference['cpu_integration_mock']:
    raise ValueError('Need the completed actual GPU control')
  if reference['rows'][0]['legacy_comparison']['exact_group12_arrays']!=10:raise ValueError('Legacy equality control failed')
  files=[*sorted(HERE.glob('agent-distill-vision-cohort-fit-*.py')),HERE/'agent-distill-vision-cohort-capture.py',LEDGER,STATE_BASE,KFD]
  return {str(path):sha(path) for path in files}


def real_report(report):
  if (report.get('synthetic_cpu_fixture') or report.get('cpu_integration_mock') is not False or report.get('gpu_executed') is not True
      or report.get('complete') is not True or report.get('legacy_control') is not None or report.get('mode')!='vision-target-only'):
    raise ValueError('Only complete actual cohort captures are admitted')


def verify_tensor(info,report_path,scene,name):
  import numpy as np
  filename=Path(info['path']).resolve()
  if filename.parent!=Path(report_path).resolve().parent/scene or filename.name!=name+'.npy':raise ValueError('Target escaped its declared scene directory')
  if sha(filename)!=info['sha256']:raise ValueError('Changed target '+str(filename))
  array=np.load(filename,mmap_mode='r',allow_pickle=False)
  if list(array.shape)!=info['shape'] or array.dtype!=np.uint16 or np.any((array&0x7f80)==0x7f80):raise ValueError('Invalid BF16 target')


def admit(bundle_path,bundle_sha,*,all_arrays=True):
  """Read only explicitly supplied, SHA-bound complete reports. Return descriptors, never bulk GPU data."""
  C=capture_adapter()
  if sha(bundle_path)!=bundle_sha:raise ValueError('Capture-set checksum mismatch')
  bundle=read(bundle_path)
  if bundle.get('synthetic_cpu_fixture'):raise ValueError('CPU fixtures cannot enter training')
  if bundle['schema']!=1 or not 12<=len(bundle['reports'])<=24:raise ValueError('Expected one/two-scene reports covering24 scenes')
  pin=bundle['inputs_manifest']
  if sha(pin['path'])!=pin['sha256']:raise ValueError('Changed completed producer manifest')
  manifest=read(pin['path']);anchors=C.read(C.COVERAGE)['anchors'];ids=[r['id'] for r in anchors]
  for i in range(0,24,2):C.validate_manifest(manifest,ids[i:i+2],verify_files=False)
  expected={r['id']:r for r in manifest['samples']}; reference=read(LEGACY)
  if sha(LEGACY)!=LEGACY_SHA:raise ValueError('Changed verified GPU control')
  rows={};seen_reports=set();seen_targets=set();split_keys={'train':set(),'validation':set()}
  for report_pin in bundle['reports']:
    path=Path(report_pin['path']).resolve()
    if path in seen_reports or sha(path)!=report_pin['sha256']:raise ValueError('Repeated or changed explicit report')
    seen_reports.add(path);report=read(path);real_report(report)
    if (report['coverage_sha256']!=C.COVERAGE_SHA or report['inputs_manifest_sha256']!=pin['sha256']
        or report['groups']!=[list(g) for g in C.GROUPS] or report['settings']!=reference['settings']
        or report['source_sha256']!=reference['source_sha256'] or not 1<=len(report['rows'])<=2):
      raise ValueError('Capture recipe, source, manifest, or scene count changed')
    for row in report['rows']:
      if row['id'] not in expected or row['id'] in rows or row.get('complete') is not True:raise ValueError('Unexpected, repeated, or incomplete scene')
      wanted=expected[row['id']]
      if any(row[k]!=wanted[k] for k in ('id','split','route_id','query_ns','input_file_sha256','actual_prompt_tokens')):
        raise ValueError('Capture does not match the admitted scene')
      C.validate_targets(row['targets'],row['images'])
      keys=row['vision_input_keys'];unique={};descriptors=[]
      if len(keys)!=10:raise ValueError('Expected ten logical history images')
      for i,(key,image) in enumerate(zip(keys,row['images'],strict=True)):
        digest=key['pixel_sha256']
        if key['grid']!=[1,20,36] or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid original image key')
        rank=unique.setdefault(digest,len(unique));batch,member=divmod(rank,2)
        if (image['logical_image']!=i or image['pixel_sha256']!=digest or image['grid']!=key['grid']
            or (image['batch'],image['member'],image['projector'])!=(batch,member,rank)):
          raise ValueError('Changed deduplication or logical image mapping')
        descriptors.append(dict(id=f'{row["id"]}/image{i}',scene=row['id'],split=row['split'],image_index=i,
          batch=batch,member=member,pixel_sha256=digest,report_sha256=report_pin['sha256'],
          x=row['targets']['tensors'][f'vision{batch}.group12.x'],y=row['targets']['tensors'][f'vision{batch}.group12.y']))
      counts=[min(2,len(unique)-i) for i in range(0,len(unique),2)]
      if [b['images'] for b in row['targets']['vision_batches']]!=counts:raise ValueError('Unexpected encoded batch grouping')
      for name,info in row['targets']['tensors'].items():
        if info['path'] in seen_targets:raise ValueError('Different scene targets alias the same file')
        seen_targets.add(info['path'])
        if all_arrays or '.group12.' in name:verify_tensor(info,path,row['id'],name)
      split_keys[row['split']].update(unique);rows[row['id']]=descriptors
  if set(rows)!=set(expected):raise ValueError('Missing cohort scene')
  if split_keys['train']&split_keys['validation']:raise ValueError('Exact image content crosses the train/validation split')
  train=[rows[f'train-scene-{scene:02d}'][image] for image in range(10) for scene in range(16)]
  validation=[r for scene in range(8) for r in rows[f'validation-scene-{scene:02d}']]
  return dict(train=train,validation=validation,receipt=dict(bundle=dict(path=str(Path(bundle_path).resolve()),sha256=bundle_sha),
    inputs_manifest=pin,reports=bundle['reports'],coverage_sha256=C.COVERAGE_SHA,legacy_control_sha256=LEGACY_SHA,
    logical_images=dict(train=160,validation=80),independent_anchors=dict(train=16,validation=8),
    physical_drives=2,unique_pixel_keys={k:len(v) for k,v in split_keys.items()}))


def image_arrays(row):
  import numpy as np
  result={}
  for name in ('x','y'):
    info=row[name]
    if sha(info['path'])!=info['sha256']:raise ValueError('Target changed after admission')
    bits=np.load(info['path'],mmap_mode='r',allow_pickle=False)
    if list(bits.shape)!=info['shape'] or bits.dtype!=np.uint16:raise ValueError('Changed target geometry')
    selected=bits[row['member']:row['member']+1]
    if selected.shape!=(1,720,1152) or np.any((selected&0x7f80)==0x7f80):raise ValueError('Invalid individual image target')
    result[name]=(selected.astype(np.uint32)<<16).view(np.float32)
  return result


def prepare(bundle,bundle_sha,destination):
  storage(destination);pins=source_pins();dataset=admit(bundle,bundle_sha)
  arrays,seed=original().seeds()
  plan=dict(schema=1,protocol='fixed160-original-seed-vision-group12',hyperparameters=HYPER,limits=LIMITS,
    original_qualified_plan_sha256=OLD_SHA,sources=pins,seed=seed,parameter_count=sum(v.size for v in arrays.values()),
    geometry=dict(patches=720,width=1152,mlp=4304,heads=16,head_dim=72,layernorm_eps=1e-6),
    dataset=dataset['receipt'],gpu_executed=False,optimizer_updates=0)
  write(destination,plan);no_devices();return plan


def verify_plan(path,digest,*,all_arrays=True):
  if sha(path)!=digest:raise ValueError('Plan checksum mismatch')
  plan=read(path)
  if plan['protocol']!='fixed160-original-seed-vision-group12' or plan['hyperparameters']!=HYPER or plan['limits']!=LIMITS:
    raise ValueError('Changed fixed training protocol')
  if source_pins()!=plan['sources']:raise ValueError('Changed fit sources')
  arrays,seed=original().seeds()
  if seed!=plan['seed'] or sum(v.size for v in arrays.values())!=plan['parameter_count']:raise ValueError('Original seed changed')
  bundle=plan['dataset']['bundle'];dataset=admit(bundle['path'],bundle['sha256'],all_arrays=all_arrays)
  if dataset['receipt']!=plan['dataset']:raise ValueError('Capture-set receipt changed')
  return plan,dataset

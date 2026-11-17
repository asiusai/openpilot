"""Source-bound common helpers and predeclared arithmetic controls. CPU at import."""
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OVERLAY=EXP/'agent-activation-sparsity-closurefix-source'
BASELINE=WORK/'activation-sparsity-devicefix-source'
LAYERS=(0,14,27)
SHAPE=(2112,2048,9216)
SAMPLE='/mnt/nvme-500/tmp/alpamayo-calibrated-two-six-20261116/samples/sample-3179527641'
PAYLOAD=3*(2112*9216+2048*9216+2112*2048*2+8)


def sha(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as stream:
    while b:=stream.read(8*1024*1024):h.update(b)
  return h.hexdigest()


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def local(name):return load(HERE/(name+'.py'),name.replace('-','_'))
def write(path,data):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


@lru_cache(None)
def fmt():
  path=EXP/'agent-sparse-format.py'
  if sha(path)!='e6c856ce18984ed8f68827d56acf89864136fff1ce6596124bc93bfbcc6f7abc':raise ValueError('Verified sparse format source changed')
  return load(path,'sparse_fc2_independent_format')


def bf16(a):
  a=np.asarray(a,np.float32);u=a.view(np.uint32)
  regular=u+((u>>16)&1)+np.uint32(0x7fff)
  special=u|((u&0xffff)!=0).astype(np.uint32)<<16
  mask=(np.negative(u,dtype=np.uint32)&0x7f800000)!=0
  return (np.where(mask,regular,special)>>16).astype(np.uint16)


def guard_disk(path):
  """Require the mounted 500GB filesystem; never use root or the footage disk."""
  path=Path(path).resolve();mount=Path('/mnt/nvme-500')
  if not path.is_relative_to(mount) or not os.path.ismount(mount):raise ValueError('Capture must use mounted /mnt/nvme-500')
  if mount.stat().st_dev==Path('/').stat().st_dev:raise ValueError('Capture mount unexpectedly uses the root filesystem')
  existing=path
  while not existing.exists():existing=existing.parent
  if existing.stat().st_dev!=mount.stat().st_dev:raise ValueError('Capture output crosses filesystem boundaries')
  st=os.statvfs(mount)
  if not 400_000_000_000<st.f_blocks*st.f_frsize<600_000_000_000 or st.f_bavail*st.f_frsize<PAYLOAD+64*1024*1024:
    raise ValueError('Unexpected 500GB mount capacity or insufficient capture space')
  return dict(mount=str(mount),device_id=mount.stat().st_dev,payload_bound_bytes=PAYLOAD,
              free_bytes=st.f_bavail*st.f_frsize)


def dependency_files():
  paths=[EXP/n for n in ('agent-sparse-kernel.py','agent-sparse-format.py','agent-sparse-compile.py')]
  paths += [OVERLAY/n for n in ('agent-activation-sparsity-closurefix.py','agent-activation-sparsity-overlay-plan.json')]
  paths += [BASELINE/'plan.json',WORK/'activation-sparsity-closurefix-observe/report.json',WORK/'agent-sparse-format-micro/report.json']
  paths += list(HERE.glob('agent-sparse-fc2-*.py'))
  paths += [HERE/'agent-sparse-fc2-cpu.json']
  paths += [p for p in (HERE/'agent-sparse-fc2-cpu-compiled').glob('*')if p.is_file()]
  return {str(p):sha(p) for p in sorted(set(paths))}


def prepare_plan():
  cpu=json.loads((HERE/'agent-sparse-fc2-cpu.json').read_text())
  if (not cpu.get('cpu_only') or not cpu.get('complete') or not cpu.get('all_pass') or not cpu.get('compiler')
      or cpu['source_hashes']!={p.name:sha(p)for p in HERE.glob('agent-sparse-fc2-*.py')}):
    raise ValueError('Require completed CPU controls/compiler report matching every current sparse-fc2 source')
  overlay=load(OVERLAY/'agent-activation-sparsity-closurefix.py','sparse_fc2_passed_observer')
  manifest=json.loads((OVERLAY/'agent-activation-sparsity-overlay-plan.json').read_text())
  runner,baseline,root=overlay.verify_overlay(manifest)
  if '--vision-calibration' in baseline['numerical_options'] or '--prefill-calibration' not in baseline['numerical_options']:
    raise ValueError('Require original dynamic vision and fixed language quantization')
  report=json.loads((WORK/'activation-sparsity-closurefix-observe/report.json').read_text())
  if (not report.get('complete') or not report['observer_overlay']['execution_completed']
      or report['observer_overlay']['plan']!=manifest or report['plan']!=baseline):raise ValueError('Require the completed source-bound actual sparsity diagnostic')
  if len(report['queries'])!=10 or any(not q.get('paired_output_bits_exact') or len(q['layers'])!=28 for q in report['queries']):
    raise ValueError('Sparsity diagnostic pair or layer count incomplete')
  first=report['queries'][0]
  if first['sample']!=SAMPLE or first['valid_rows']!=2027:raise ValueError('Predeclared first query changed')
  result=dict(schema='agent-sparse-fc2-v1',sources=dependency_files(),baseline_plan=baseline,
    observer_overlay=manifest,original_query=first,predeclared_layers=list(LAYERS),shape=list(SHAPE),
    sample=SAMPLE,one_query_only=True,no_holdout_or_fitted_selector=True,
    numerical_policy='UNVERIFIED: require bitwise raw FP32 and BF16 equality; stop timing on any mismatch',
    candidate='one wave, four-by-four fragments, 64x64 output, per16x32 sparse/dense fallback',
    measurement_boundary='existing packed FP8 + immutable original weights -> compression/metadata/eligibility + GEMM + BF16 RNE')
  result['identity_sha256']=digest(result)
  return result


def verify_plan(plan):
  if plan['schema']!='agent-sparse-fc2-v1' or digest({k:v for k,v in plan.items()if k!='identity_sha256'})!=plan['identity_sha256']:
    raise ValueError('Sparse fc2 plan identity mismatch')
  for path,expected in plan['sources'].items():
    if sha(path)!=expected:raise ValueError('Frozen sparse fc2 dependency changed: '+path)
  if str(Path(__file__).resolve()) not in plan['sources']:raise ValueError('Run the frozen sparse fc2 source tree')
  overlay=load(OVERLAY/'agent-activation-sparsity-closurefix.py','sparse_fc2_passed_observer')
  runner,baseline,root=overlay.verify_overlay(plan['observer_overlay'])
  if baseline!=plan['baseline_plan']:raise ValueError('Original model protocol changed')
  return overlay,runner,root


def fixtures():
  """Finite FP8 cases fixed before capture, including possible K16 boundary loss."""
  rng=np.random.default_rng(260117)
  for name,k,kind in [('integer',64,'integer'),('finite-k32',32,'finite'),('finite-k256',256,'finite'),
      ('finite-running-k9216',9216,'finite'),('subnormal',64,'subnormal'),('signed-zero',64,'zero'),
      ('k16-rounding-boundary',32,'boundary'),('mixed-dense-fallback',64,'overflow')]:
    a=np.zeros((64,k),np.uint8)
    if kind=='zero':a[::2]=128
    elif kind=='boundary':a[:,0]=56;a[:,16]=56
    else:
      for r in range(64):
        for g in range(k//4):
          positions=rng.choice(4,int(rng.integers(1,3)),replace=False)
          for p in positions:
            code=int(rng.choice([56,184])) if kind=='integer' else int(rng.integers(1,8 if kind=='subnormal' else 127))|int(rng.integers(2))*128
            a[r,g*4+p]=code
    if kind=='subnormal':
      a[:]=0
      a[:,::4]=(1+np.arange(64,dtype=np.uint8)%7)[:,None]
      a[:,1::4]=a[:,::4]
      a[1::2]|=128
    if kind=='integer':w=rng.choice(np.array([0,56,184],np.uint8),(64,k))
    elif kind=='subnormal':w=np.broadcast_to((1+np.arange(64,dtype=np.uint8)%7)[:,None],(64,k)).copy()
    elif kind=='boundary':w=np.zeros((64,k),np.uint8);w[:,0]=56;w[:,16]=56
    else:w=rng.integers(0,127,(64,k),dtype=np.uint8)|(rng.integers(0,2,(64,k),dtype=np.uint8)<<7)
    if kind=='overflow':a[0,:4]=[56,64,68,72]
    seeds=np.array([0,0x80000000,1,0x80000001,0x3f800000,0xbf800000,0x4b800000,0xcb800000],np.uint32)
    c=np.resize(seeds,(64,64)).view(np.float32).copy()
    if kind=='boundary':c[:]=np.float32(2**24)
    yield dict(name=name,a=a,w=w,c=c,xs=np.array([.071234569],np.float32),ws=np.array([.01933221],np.float32))


def compare(a,b):
  a,b=np.asarray(a),np.asarray(b)
  if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Output shape/dtype mismatch')
  bits_a=a.view(np.uint32 if a.dtype==np.float32 else np.uint16)
  bits_b=b.view(bits_a.dtype)
  return dict(exact=bool(np.array_equal(bits_a,bits_b)),differing=int(np.count_nonzero(bits_a!=bits_b)),elements=int(a.size))

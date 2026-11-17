"""Freeze or verify read-only runtime inputs; never initialize a GPU."""
import argparse,hashlib,importlib.metadata,importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE/'agent-streaming-plan-parent-trial'
MANIFEST=ROOT/'external-provenance.json'
PACKAGES=('torch','torchvision','transformers','tokenizers','numpy','safetensors','triton','huggingface-hub')

def digest(path):
  value=hashlib.sha256()
  with Path(path).open('rb') as source:
    while chunk:=source.read(8*1024*1024):value.update(chunk)
  return value.hexdigest()

def versions():
  result={}
  for name in PACKAGES:
    try:result[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:result[name]=None
  return result

def create():
  if MANIFEST.exists():raise FileExistsError('Provenance is already frozen')
  paths=set();checkpoint=Path('/tmp/opencode/alpamayo-asius-20261114/model')
  paths.update(p for p in checkpoint.iterdir() if p.suffix in ('.json','.jinja','.safetensors'))
  paths.add(Path('/tmp/opencode/alpamayo-nvidia-reference-full/initial_noise.npy'))
  bulk=Path('/dev/shm/alpamayo-9060-fp8');paths.update((bulk/'index.json',bulk/'weights.bin'))
  for sample in json.loads((ROOT/'smoke32.json').read_text()):
    paths.update(Path(sample)/(n+'.npy') for n in ('image_frames','absolute_timestamps','camera_indices','ego_history_xyz','ego_history_rot','nav_text'))
  paths.update(Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad/tinygrad').rglob('*.py'))
  # Installed package source locations, without importing or executing models.
  for name in ('alpamayo2_edge','transformers'):
    spec=importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:raise ValueError('Missing '+name)
    for folder in spec.submodule_search_locations:paths.update(Path(folder).rglob('*.py'))
  records=[]
  for path in sorted(paths):
    before=path.stat();value=digest(path);after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed during hashing: '+str(path))
    records.append(dict(path=str(path),size=before.st_size,sha256=value))
  record=dict(records=records,package_versions=versions(),python=sys.version,
    boundary='All checkpoint assets, FP8 bulk cache, noise, 32 current-input fixtures, tinygrad Python, Alpamayo Python, Transformers Python')
  MANIFEST.write_text(json.dumps(record,indent=2)+'\n');print('Frozen external files:',len(records),flush=True)

def verify():
  record=json.loads(MANIFEST.read_text())
  if record['package_versions']!=versions() or record['python']!=sys.version:raise ValueError('Runtime environment changed')
  for item in record['records']:
    if Path(item['path']).stat().st_size!=item['size'] or digest(item['path'])!=item['sha256']:
      raise ValueError('External runtime input changed: '+item['path'])
  return record

if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
  if args.freeze:create()
  else:verify();print('External provenance verified')

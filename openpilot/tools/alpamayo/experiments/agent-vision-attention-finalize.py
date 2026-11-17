"""Seal the small parent protocol and copy the existing verified firmware cache only."""
import hashlib, importlib, json, sys
from pathlib import Path

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-vision-attention-artifacts')
sys.path.insert(0,str(EXP));C=importlib.import_module('agent-vision-attention-common')

def main():
  path=ROOT/'protocol-final-v2.json';assert not path.exists()
  plan=json.loads((ROOT/'prepared/protocol.json').read_text())
  # Only host-side allocation checks/cache isolation changed after graph preparation.
  # All generated programs, dependencies, numerical files, and candidate source remain pinned.
  for p,h in plan['source_sha256'].items():
    if Path(p)==EXP/'agent-vision-attention-run.py':continue
    assert C.sha(p)==h,('changed prepared source',p)
  plan['source_sha256'][str(EXP/'agent-vision-attention-run.py')]=C.sha(EXP/'agent-vision-attention-run.py')
  seed=ROOT.parent/'agent-ck-selected-recovery-work/parent-firmware-cache-seed-v1.json'
  assert C.sha(seed)=='6fa69ff5134f830183781264d8077dbda8c0335c5780a3113926e1967c79ba12'
  cache=ROOT/'runtime-cache/tinygrad/downloads/fw';cache.mkdir(parents=True,exist_ok=True)
  copies=[]
  for row in json.loads(seed.read_text())['files']:
    src=Path(row['destination']);assert src.stat().st_size==row['bytes'] and C.sha(src)==row['sha256']
    dest=cache/src.name
    if dest.exists(): assert C.sha(dest)==row['sha256']
    else: dest.write_bytes(src.read_bytes())
    plan['source_sha256'][str(dest)]=C.sha(dest);copies.append({'name':row['name'],'source':str(src),'destination':str(dest),'sha256':C.sha(dest)})
  plan['firmware_cache']={'seed_receipt':str(seed),'seed_sha256':C.sha(seed),'files':copies,'new_runtime_firmware':False}
  plan['source_sha256'][str(EXP/'agent-vision-attention-finalize.py')]=C.sha(EXP/'agent-vision-attention-finalize.py')
  plan['historical_preparation_sha256']=C.sha(ROOT/'prepared/protocol.json')
  path.write_text(json.dumps(plan,indent=2)+'\n')
  print(json.dumps({'protocol':str(path),'sha256':C.sha(path),'gpu_executed':False}))

if __name__=='__main__':main()

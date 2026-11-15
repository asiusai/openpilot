"""Extract only gfx1200 code/data from a pinned AMD package; never install it."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import urllib.request

root=Path('/tmp/opencode/alpamayo-hipblaslt-7.0.2')
root.mkdir(exist_ok=True)
url='https://repo.radeon.com/rocm/apt/7.0.2/pool/main/h/hipblaslt/hipblaslt_1.0.0.70002-56~24.04_amd64.deb'
expected='82367f5b7ee8615c673faa117e2975991d7e075e5202146541bd210349124144'
package=root/'hipblaslt_1.0.0.70002-56~24.04_amd64.deb'
if not package.exists():
  temporary=package.with_suffix('.download')
  with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as output:
    shutil.copyfileobj(response,output,1024*1024)
  temporary.rename(package)
with package.open('rb') as source:
  actual=hashlib.file_digest(source,'sha256').hexdigest()
assert actual==expected,(actual,expected)
control=root/'control'
subprocess.run(['dpkg-deb','--control',str(package),str(control)],check=True)
scripts=[p.name for p in control.iterdir() if p.name in ('preinst','postinst','prerm','postrm','config')]
print(json.dumps({'sha256':actual,'package_bytes':package.stat().st_size,'maintainer_scripts_not_executed':scripts}),flush=True)
process=subprocess.Popen(['dpkg-deb','--fsys-tarfile',str(package)],stdout=subprocess.PIPE)
selected=[]
with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
  for member in archive:
    if not member.isfile(): continue
    relative=PurePosixPath(member.name)
    assert not relative.is_absolute() and '..' not in relative.parts
    if 'gfx1200' not in relative.name and relative.name.lower() not in ('copyright','license','license.txt'):
      continue
    target=root/'extracted'/str(relative)
    target.parent.mkdir(parents=True,exist_ok=True)
    with archive.extractfile(member) as source,target.open('wb') as output:
      shutil.copyfileobj(source,output,1024*1024)
    with target.open('rb') as source:
      digest=hashlib.file_digest(source,'sha256').hexdigest()
    record=dict(path=str(target),bytes=member.size,sha256=digest)
    selected.append(record)
    print(json.dumps(record),flush=True)
assert process.wait()==0
assert selected
(root/'manifest.json').write_text(json.dumps(dict(url=url,version='1.0.0.70002-56~24.04',sha256=expected,
  maintainer_scripts_not_executed=scripts,files=selected),indent=2)+'\n')

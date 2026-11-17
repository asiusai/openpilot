"""Seal the minimal new import repair, retaining the failed sources and evidence unchanged."""
import difflib
import importlib
import json
from pathlib import Path
import shutil

C=importlib.import_module('agent-distill-kfd1-import-common')

def main():
  C.no_gpu_fds();assert not C.ROOT.exists()
  assert C.file_sha(C.PRIOR/'protocol.json')==C.PRIOR_PROTOCOL_SHA
  p=C.read(C.PRIOR/'protocol.json')
  for path,digest in p['source_sha256'].items():assert C.file_sha(path)==digest,path
  assert C.file_sha(C.OLD/'oracle.json')==p['oracle_sha256']
  assert C.file_sha(C.OLD/'cpu-check-matched.json')==p['cpu_evidence_sha256']
  C.ROOT.mkdir();(C.ROOT/'tmp').mkdir();frozen=C.ROOT/'frozen';frozen.mkdir()
  sources=p['source_sha256'].copy()
  sources[str(C.PRIOR/'protocol.json')]=C.PRIOR_PROTOCOL_SHA
  failure=C.PRIOR/'parent-gfx1100-kfd1/run.log'
  sources[str(failure)]=C.file_sha(failure)
  for path in C.EXP.glob('agent-distill-kfd1-import-*.py'):
    dest=frozen/path.name;shutil.copyfile(path,dest);sources[str(path)]=sources[str(dest)]=C.file_sha(path)
  p.update(source_sha256=sources,prior_kfd1_protocol_sha256=C.PRIOR_PROTOCOL_SHA,
    import_repair='PMC and SQTT imported from pinned tinygrad.runtime.ops_amd; helpers retains DEV, PROFILE, MAX_BUFFER_SIZE',
    numeric_change='None; original frozen complete numerical diagnostic reused',
    new_storage_root=str(C.ROOT),gpu_executed_during_preparation=False)
  C.write(C.ROOT/'protocol.json',p);digest=C.file_sha(C.ROOT/'protocol.json')
  old=(C.PRIOR/'frozen/agent-distill-kfd1-parent.py').read_text().splitlines(keepends=True)
  new=(frozen/'agent-distill-kfd1-import-parent.py').read_text().splitlines(keepends=True)
  (C.ROOT/'parent-source.diff').write_text(''.join(difflib.unified_diff(old,new,fromfile='frozen-prior-parent',tofile='frozen-import-repair-parent')))
  base=[str(C.PY),str(frozen/'agent-distill-kfd1-import-parent.py'),'--protocol-sha',digest]
  C.write(C.ROOT/'commands.json',{'protocol_sha256':digest,'passive_preflight':base,'parent_only':base+['--execute-reviewed'],'executed':False})
  for path in [*frozen.iterdir(),C.ROOT/'protocol.json',C.ROOT/'commands.json',C.ROOT/'parent-source.diff']:path.chmod(0o444)
  print(json.dumps({'sealed':True,'gpu_executed':False,'protocol_sha256':digest,'parent_only':base+['--execute-reviewed']},indent=2))

if __name__=='__main__':main()

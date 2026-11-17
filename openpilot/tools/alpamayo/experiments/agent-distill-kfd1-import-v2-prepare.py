"""Generate new frozen sources from the immutable import repair, correcting pinned boolean parsing."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
FIRST=WORK/'agent-distill-kfd1-import-artifacts'
PRIOR=WORK/'agent-distill-kfd1-artifacts'
ROOT=WORK/'agent-distill-kfd1-import-v2-artifacts'
PY='/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def replace_once(text,before,after):
  assert text.count(before)==1,before
  return text.replace(before,after)

def main():
  assert not ROOT.exists()
  assert sha(FIRST/'protocol.json')=='c664920ff11e0db9af7157d6c4d71e668d84fc8da24656a7115f527ab9a83fc9'
  original=json.loads((FIRST/'protocol.json').read_text())
  for path,digest in original['source_sha256'].items():assert sha(path)==digest,path
  ROOT.mkdir();(ROOT/'tmp').mkdir();frozen=ROOT/'frozen';frozen.mkdir()
  sources=original['source_sha256'].copy();sources[str(FIRST/'protocol.json')]=sha(FIRST/'protocol.json')
  sources[str(Path(__file__).resolve())]=sha(__file__)
  for suffix in ('common','parent','check'):
    prior=FIRST/'frozen'/f'agent-distill-kfd1-import-{suffix}.py';text=prior.read_text()
    text=text.replace('agent-distill-kfd1-import-common','agent-distill-kfd1-import-v2-common')
    text=text.replace('agent-distill-kfd1-import-parent','agent-distill-kfd1-import-v2-parent')
    if suffix=='common':
      text=replace_once(text,"ROOT=WORK/'agent-distill-kfd1-import-artifacts'","ROOT=WORK/'agent-distill-kfd1-import-v2-artifacts'")
    if suffix=='parent':
      text=replace_once(text,"PROFILE='0',PMC='0',SQTT='0',LRU='0'","PROFILE='0',PMC='',SQTT='',VIZ='0',LRU='0'")
    if suffix=='check':
      text=replace_once(text,"  new.body[0]=copy.deepcopy(old.body[0])",'''  new.body[0]=copy.deepcopy(old.body[0])
  for call in ast.walk(new):
    if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='update':
      kws={k.arg:k for k in call.keywords}
      if {'PROFILE','PMC','SQTT','VIZ'}<=kws.keys():
        assert kws['PMC'].value.value==kws['SQTT'].value.value=='' and kws['VIZ'].value.value=='0'
        kws['PMC'].value.value=kws['SQTT'].value.value='0'
        call.keywords=[k for k in call.keywords if k.arg!='VIZ']''')
      text=replace_once(text,"  assert not hasattr(helpers,'PMC') and not hasattr(helpers,'SQTT')",'''  assert not hasattr(helpers,'PMC') and not hasattr(helpers,'SQTT')
  # Real pinned getenv casts a nonempty string to True for boolean defaults.
  for key in ('PMC','SQTT'):
    previous=os.environ.get(key)
    try:
      os.environ[key]='0';assert helpers.getenv(key,False) is True
      os.environ[key]='';assert helpers.getenv(key,False) is False
    finally:
      if previous is None:os.environ.pop(key,None)
      else:os.environ[key]=previous
  env=P.environment()
  assert env['PMC']==env['SQTT']=='' and env['VIZ']==env['PROFILE']=='0'
  assert PMC.value is False and SQTT.value is False''')
      text=replace_once(text,"'parent_change_limited_to_imports_and_module_path':True", "'parent_change_limited_to_imports_module_path_boolean_environment':True,'boolean_zero_string_pitfall_verified':True")
    path=frozen/f'agent-distill-kfd1-import-v2-{suffix}.py'
    ast.parse(text,filename=str(path));path.write_text(text);sources[str(path)]=sha(path)
  original.update(source_sha256=sources,new_storage_root=str(ROOT),
    import_repair='PMC/SQTT from runtime.ops_amd. Empty-string boolean env values disable them; VIZ=0 and PROFILE=0. All disabled assertions retained.',
    supersedes_unqualified_import_protocol_sha256=sha(FIRST/'protocol.json'),
    production_changes=['private common-module path','correct PMC/SQTT import owner','boolean env values parsed false by pinned getenv'],
    gpu_executed_during_preparation=False)
  write(ROOT/'protocol.json',original);digest=sha(ROOT/'protocol.json')
  base=[PY,str(frozen/'agent-distill-kfd1-import-v2-parent.py'),'--protocol-sha',digest]
  write(ROOT/'commands.json',{'protocol_sha256':digest,'passive_preflight':base,'parent_only':base+['--execute-reviewed'],'executed':False})
  old=(PRIOR/'frozen/agent-distill-kfd1-parent.py').read_text().splitlines(keepends=True)
  new=(frozen/'agent-distill-kfd1-import-v2-parent.py').read_text().splitlines(keepends=True)
  (ROOT/'parent-source.diff').write_text(''.join(difflib.unified_diff(old,new,fromfile='frozen-failed-kfd1-parent',tofile='frozen-corrected-import-v2-parent')))
  for path in [*frozen.iterdir(),ROOT/'protocol.json',ROOT/'commands.json',ROOT/'parent-source.diff']:path.chmod(0o444)
  print(json.dumps({'sealed':True,'gpu_executed':False,'protocol_sha256':digest,'parent_only':base+['--execute-reviewed']},indent=2))

if __name__=='__main__':main()

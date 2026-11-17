"""Freeze only this new full-length preparation; never modify earlier evidence."""
import importlib
import json
from pathlib import Path
import shutil

C=importlib.import_module('agent-distill-full-common')

def main():
  assert not (C.ROOT/'protocol.json').exists() and not (C.ROOT/'frozen').exists();C.no_gpu_fds()
  sources={}
  for name in ('common','budget','oracle','check','parent','compile','cpu-check','seal'):
    path=C.EXP/f'agent-distill-full-{name}.py';destination=C.ROOT/'frozen'/path.name
    compile(path.read_text(),str(path),'exec');destination.parent.mkdir(exist_ok=True);shutil.copyfile(path,destination)
    sources[str(path)]=sources[str(destination)]=C.file_sha(path)
  old=C.read(C.OLD/'protocol.json')
  for path,digest in old['source_sha256'].items():
    if Path(path).is_relative_to(C.TINY):
      assert C.file_sha(path)==digest,path;sources[path]=digest
  extras=[Path(C.K.__file__),Path(C.B.__file__),C.OLD/'oracle.json',C.OLD/'protocol.json',
    C.SUCCESS/'parent-gfx1100-kfd1/report.json',C.SUCCESS/'protocol.json',C.SUCCESS/'cpu-import-check.json',
    C.ROOT/'oracle.json',C.ROOT/'budget.json',C.ROOT/'offline-compile.json']
  oracle=C.read(C.ROOT/'oracle.json')
  extras.extend(Path(v['path']) for v in oracle['arrays'].values())
  extras.extend(Path(p) for p in oracle['source_sha256'])
  extras.extend((C.ROOT/'offline-programs').iterdir())
  for path in extras:sources[str(path)]=C.file_sha(path)
  licenses={}
  for name in ('tinygrad','upstream','model'):
    source=C.OLD/'frozen'/f'LICENSE-{name}.txt';target=C.ROOT/'frozen'/source.name;shutil.copyfile(source,target)
    sources[str(source)]=sources[str(target)]=C.file_sha(source);licenses[name]={'path':str(target),'sha256':C.file_sha(target)}
  topology=C.K.passive_topology();C.K.selected_node(topology)
  budget=C.read(C.ROOT/'budget.json');offline=C.read(C.ROOT/'offline-compile.json')
  assert oracle['complete'] and offline['complete'] and not oracle['gpu_executed'] and not offline['gpu_executed']
  assert budget['schedules_by_NO_MEMORY_PLANNER']['1']['maximum_buffer_bytes']==C.MAX_BUFFER
  protocol={'schema':1,'target':C.TARGET,'length':C.LENGTH,'valid_rows':C.VALID_ROWS,'timeout_seconds':300,
    'optimizer_updates':0,'limits':{'live':C.LIVE_LIMIT,'cumulative':C.TOTAL_LIMIT,'single_buffer':C.MAX_BUFFER,'free_reserve':C.FREE_RESERVE},
    'tolerances':C.TOLERANCES,'passive_topology':topology,'source_sha256':sources,'licenses':licenses,
    'tinygrad_revision':C.B.REVISION,'precision':'FP32 from source-bound actual BF16 checkpoint layer12 masters; TC0 and TF32off',
    'memory_layout':'NO_MEMORY_PLANNER=1 avoids the 897589248-byte arena; max raw tensor268435456 bytes',
    'scope':'Synthetic native FP32 layers12-15 teacher, full2048 forward/backward; no actual route targets, optimizer, or quality claim',
    'nonblocking_lock':'/tmp/opencode/alpamayo-chestnut-speed.lock','gpu_executed':False,
    'parent_requirements':'Fresh availability review; selected KFD owner rejection; no reset, unbind, privilege, driver fallback or GPU0 access',
    'program_identity_note':'Generic native AMDLLVMRenderer +cumode; no inference PROGRAM constants or previous WGP experiments'}
  C.write(C.ROOT/'protocol.json',protocol)
  print(json.dumps({'protocol':str(C.ROOT/'protocol.json'),'sha256':C.file_sha(C.ROOT/'protocol.json'),'source_files':len(sources)},indent=2))

if __name__=='__main__':main()

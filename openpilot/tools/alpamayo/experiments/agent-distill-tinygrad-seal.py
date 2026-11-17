"""CPU-only freeze and source/identity checks for one optional parent KFD diagnostic."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess

C=importlib.import_module('agent-distill-tinygrad-common')

def main():
  C.no_gpu_fds();assert not (C.ROOT/'protocol.json').exists()
  cpu=C.read(C.ROOT/'cpu-check-matched.json');assert cpu['complete'] and cpu['all_passed'] and not cpu['gpu_executed']
  assert cpu['tolerances']==C.TOLERANCES
  for path,digest in cpu['source_sha256'].items():assert C.file_sha(path)==digest,path
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=C.TINY,text=True).strip()==C.REVISION
  oracle,arrays=C.arrays();del arrays
  dest=C.ROOT/'frozen';dest.mkdir(exist_ok=False);sources={}
  for path in C.EXP.glob('agent-distill-tinygrad-*.py'):
    target=dest/path.name;shutil.copyfile(path,target)
    sources[str(path)]=sources[str(target)]=C.file_sha(path)
  for path in (C.TINY/'LICENSE',C.BASE/'upstream/LICENSE',C.BASE/'model/LICENSE'):
    target=dest/('LICENSE-'+path.parent.name+'.txt');shutil.copyfile(path,target)
    sources[str(path)]=sources[str(target)]=C.file_sha(path)
  # Pin the complete runtime Python tree, including generated definitions and backend selection.
  for path in (C.TINY/'tinygrad').rglob('*.py'):sources[str(path)]=C.file_sha(path)
  sources.update(oracle['source_sha256'])
  topology=C.passive_topology();selected=topology['nodes'][0]
  assert selected['pci']==C.SELECTED_PCI and selected['gfx_target_version']==110000
  assert selected['render_rw_access'] and topology['kfd_rw_access']
  protocol={'schema':1,'source_sha256':sources,'tinygrad_revision':C.REVISION,
    'oracle_sha256':C.file_sha(C.ROOT/'oracle.json'),'cpu_evidence_sha256':C.file_sha(C.ROOT/'cpu-check-matched.json'),
    'target':C.TARGET,'selected_pci':C.SELECTED_PCI,'passive_topology':topology,
    'timeout_seconds':300,'cumulative_kfd_allocation_limit':2*1024**3,'max_single_buffer_bytes':128*1024**2,
    'shape':[1,4,2048],'parameter_count':50335744,'tolerances':C.TOLERANCES,'optimizer_updates':0,
    'arithmetic':'FP32, TC=0, ALLOW_TF32=0, dropout off; no BF16 training or selected FP8 runtime equivalence',
    'parent_status':'Prepared only. Availability and actual KFD allocator remain unverified.',
    'gpu_executed_during_preparation':False,'input_scope':oracle['positions'],'target_scope':oracle['target_scope']}
  C.write(C.ROOT/'protocol.json',protocol);digest=C.file_sha(C.ROOT/'protocol.json')
  command=[str(C.PY),str(dest/'agent-distill-tinygrad-parent.py'),'--protocol-sha',digest,'--execute-reviewed']
  C.write(C.ROOT/'commands.json',{'protocol_sha256':digest,'parent_only':command,'executed':False})
  for path in [*dest.iterdir(),C.ROOT/'protocol.json',C.ROOT/'commands.json',C.ROOT/'oracle.json',*list((C.ROOT/'oracle-arrays').iterdir())]:path.chmod(0o444)
  print(json.dumps({'sealed':True,'gpu_executed':False,'protocol_sha256':digest,'command':command},indent=2))

if __name__=='__main__':main()

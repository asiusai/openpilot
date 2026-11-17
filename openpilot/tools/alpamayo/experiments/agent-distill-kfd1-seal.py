"""Freeze a new ordinal1 wrapper, retaining every original numerical source and evidence byte."""
import importlib
import json
from pathlib import Path
import shutil

C=importlib.import_module('agent-distill-kfd1-common')

def main():
  C.no_gpu_fds();assert not C.ROOT.exists()
  assert C.file_sha(C.OLD/'protocol.json')==C.OLD_PROTOCOL_SHA
  original=C.read(C.OLD/'protocol.json')
  for path,digest in original['source_sha256'].items():assert C.file_sha(path)==digest,path
  assert C.file_sha(C.OLD/'oracle.json')==original['oracle_sha256']
  assert C.file_sha(C.OLD/'cpu-check-matched.json')==original['cpu_evidence_sha256']
  cpu=C.read(C.OLD/'cpu-check-matched.json');assert cpu['all_passed'] and not cpu['gpu_executed']
  oracle=C.read(C.OLD/'oracle.json')
  for row in oracle['arrays'].values():assert C.file_sha(C.OLD/row['file'])==row['sha256']
  C.ROOT.mkdir();(C.ROOT/'tmp').mkdir();frozen=C.ROOT/'frozen';frozen.mkdir()
  sources=dict(original['source_sha256'])
  sources[str(C.OLD/'protocol.json')]=C.OLD_PROTOCOL_SHA
  for path in C.EXP.glob('agent-distill-kfd1-*.py'):
    dest=frozen/path.name;shutil.copyfile(path,dest)
    sources[str(path)]=sources[str(dest)]=C.file_sha(path)
  topology=C.passive_topology();selected=C.selected_node(topology);availability=C.passive_availability()
  C.write(C.ROOT/'availability-at-seal.json',availability)
  p={'schema':1,'source_sha256':sources,'original_protocol_sha256':C.OLD_PROTOCOL_SHA,
    'oracle_sha256':original['oracle_sha256'],'cpu_evidence_sha256':original['cpu_evidence_sha256'],
    'numerical_common_sha256':C.NUMERIC_COMMON_SHA,'numerical_check_sha256':C.NUMERIC_CHECK_SHA,
    'target':C.TARGET,'selected_kfd_index':C.SELECTED_INDEX,'selected_pci':C.SELECTED_PCI,
    'selected_gpu_id':C.SELECTED_GPU_ID,'selected_render_minor':C.SELECTED_RENDER_MINOR,'selected_node':C.SELECTED_NODE,
    'passive_topology':topology,'timeout_seconds':300,'cumulative_kfd_allocation_limit':C.LIMIT,
    'max_single_buffer_bytes':128*1024**2,'tolerances':C.TOLERANCES,'optimizer_updates':0,
    'numeric_change':'None: exact original frozen execute(), Block, losses, oracle and full-array comparisons reused read-only',
    'wrapper_changes':['KFD ordinal1/PCI0c/GPU27211/renderD129 binding before and after open',
      'new output/cache/tmp paths on nvme500','passive KFD queue/VRAM snapshots and observed-conflict guard'],
    'unqualified':'Availability, actual allocation, and GPU arithmetic remain unverified.',
    'gpu_executed_during_preparation':False,'parent_review_required_before_execution':True}
  C.write(C.ROOT/'protocol.json',p);digest=C.file_sha(C.ROOT/'protocol.json')
  base=[str(C.PY),str(frozen/'agent-distill-kfd1-parent.py'),'--protocol-sha',digest]
  C.write(C.ROOT/'commands.json',{'protocol_sha256':digest,'passive_preflight':base,'parent_only':base+['--execute-reviewed'],'executed':False})
  for path in [*frozen.iterdir(),C.ROOT/'protocol.json',C.ROOT/'commands.json',C.ROOT/'availability-at-seal.json']:path.chmod(0o444)
  print(json.dumps({'sealed':True,'gpu_executed':False,'protocol_sha256':digest,'preflight':base,'selected':selected},indent=2))

if __name__=='__main__':main()

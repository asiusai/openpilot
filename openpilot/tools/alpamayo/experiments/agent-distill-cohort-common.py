"""Fixed 24-scene cohort identities and CPU preparation helpers; no model initialization."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-cohort-artifacts'
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
ARCHIVE=Path('/mnt/nvme-256/footage/f9ad9a96')
COVERAGE=WORK/'agent-distill-coverage-artifacts/metadata-v1/agent-distill-coverage-manifest.json'
COVERAGE_SHA='965fa3875e5b26e1138a05ae08c64ab9c0b64cd982f6e25d4111ee847a53ee78'
OLD_PLAN=WORK/'agent-distill-command-artifacts/capture-v1/agent-distill-command-capture-plan.json'
GEOMETRY=Path('/mnt/nvme-500/tmp/alpamayo-camera-crops-20261116/source-camera-geometry.json')
EXCLUDED=('0000008e--321b2ddc03','00000090--1cf3f7d926')
PROTECTED=EXCLUDED[0]
COMMANDS=('Follow the road.',)
INPUTS=ROOT/'inputs-v1/agent-distill-cohort-inputs.json'

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value
def storage(path,upcoming=0):
  parent=Path(path).resolve()
  while not parent.exists():parent=parent.parent
  assert subprocess.check_output(['findmnt','-n','-o','TARGET','-T',str(parent)],text=True,timeout=5).strip()=='/mnt/nvme-500'
  fs=os.statvfs(parent);assert fs.f_bavail*fs.f_frsize>=4*1024**3+upcoming,'Preserve 4GiB free on task storage'
def source_mount():
  assert subprocess.check_output(['findmnt','-n','-o','UUID','-T',str(ARCHIVE)],text=True,timeout=5).strip()=='aac28ecf-db48-4b4f-8a99-bde6a3726ab9'
def cpu_guard(event,values):
  if event!='open' or not isinstance(values[0],(str,bytes)):return
  path=os.fsdecode(values[0]);assert not path.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb'))
  assert not any(route in path for route in EXCLUDED)
  assert not ('sealed_test' in path and path.endswith('.npz'))
  assert 'agent-distill-continue-artifacts/pilot-' not in path
def assert_no_devices():
  if 'torch' in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
  for path in Path('/proc/self/fd').iterdir():
    try:target=str(path.readlink())
    except OSError:continue
    assert not target.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb'))
def old_common():return module('agent_distill_cohort_original_cpu_helpers',Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments/agent-distill-command-common.py'))
def cpu_imports():
  value=old_common();torch,helper=value.cpu_imports()
  runtime=Path(read(OLD_PLAN)['runtime_root']);sys.path[:0]=[str(runtime/'experiments'),str(runtime)]
  return torch,helper
def fingerprint(value):return old_common().fingerprint(value)
def coverage():
  assert sha(COVERAGE)==COVERAGE_SHA;value=read(COVERAGE)
  assert value['training_anchors']==16 and value['validation_anchors']==8 and value['commands_per_anchor']==1
  assert len(value['anchors'])==24
  return value
def protocol():
  old=read(OLD_PLAN)
  return dict(schema=1,protocol='owned-scene-cohort-layer-reconstruction-v1',coverage=dict(path=str(COVERAGE),sha256=COVERAGE_SHA),
    scenes=24,training_scenes=16,validation_scenes=8,commands=list(COMMANDS),
    command_provenance='One predeclared experimental Follow the road. condition per scene; not recorded intent or safe/available route evidence.',
    protected_and_sealed_routes=list(EXCLUDED),teacher_contract=read(COVERAGE)['teacher_recipe'],
    original_capture_plan=dict(path=str(OLD_PLAN),sha256=sha(OLD_PLAN)),targets=old['target_semantics'],
    qualification='Internal-layer reconstruction only. Two physical drives; no driving/navigation accuracy or latency qualification.')
def validate_rows(manifest):
  coverage_value=coverage();assert manifest['protocol']==protocol() and len(manifest['samples'])==24
  expected={row['id']:row for row in coverage_value['anchors']};ids=[]
  for row in manifest['samples']:
    original=expected[row['id']];ids.append(row['id'])
    assert all(row[key]==original[key] for key in ('id','split','drive_id','route_id','query_ns','conditioning'))
    assert row['split'] in ('train','validation') and row['command_index']==0 and row['conditioning']['text']==COMMANDS[0]
    assert row['actual_prompt_tokens']<=2042 and row['actual_prompt_tokens']>1800
    assert row['complete_input'] and row['official_selected_tokens_and_image_normalization_exact']
    sample=Path(row['sample']).resolve();assert sample.is_relative_to(ROOT/'inputs-v1/samples') and sample.name==row['id']
    assert set(row['input_file_sha256'])=={'image_frames.npy','camera_indices.npy','absolute_timestamps.npy','ego_history_xyz.npy','ego_history_rot.npy','nav_text.npy'}
    for name,digest in row['input_file_sha256'].items():assert sha(sample/name)==digest
  assert ids==list(expected) and len(set(ids))==24
  return manifest

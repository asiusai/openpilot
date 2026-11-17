"""One fixed internal-layer pilot. Data/protocol checks import no GPU backend."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode=True
EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-fit-artifacts'
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
TARGETS=WORK/'agent-distill-command-gpu/report.agent-distill-command-targets'
MANIFEST=TARGETS/'agent-distill-command-train-validation.json'
MANIFEST_SHA='237391db1b8fff0d0d8dd78d62e5c50fac5d38d21dd89bbdbcae180a95a9ef36'
INPUTS=WORK/'agent-distill-command-artifacts/inputs-v1/agent-distill-command-inputs.json'
INPUTS_SHA='83da7c2dc65a50185d052de07c30698ade2b1818103d90d69ab394f8f600f3b6'
GRAPH=WORK/'agent-distill-tinygrad-artifacts/frozen/agent-distill-tinygrad-common.py'
GRAPH_SHA='4ed60b0ffc6d40ccf6bdf4cc55965465249aae40bb34878be95b3b3a138fbee0'
KFD=WORK/'agent-distill-kfd1-artifacts/frozen/agent-distill-kfd1-common.py'
PROTECTED='0000008e--321b2ddc03'
COMMANDS=('Follow the road.','Turn left at the next intersection.','Turn right at the next intersection.')
HYPER=dict(updates=25,absolute_update_cap=200,validate_every=25,learning_rate=3.5e-6,
  optimizer='tinygrad.AdamW',beta1=.9,beta2=.999,epsilon=1e-8,weight_decay=.01,fused=False,
  loss='MSE(h16)+0.1*(MSE(K12)+MSE(V12)); actual prompt positions only',seed_layer=12,
  dtype='float32',length=2048,batch_size=1,order='chronological drive then fixed command index; cyclic training order')
SHAPES=dict(x=(1,2048,2048),y=(1,2048,2048),k=(1,8,2048,128),v=(1,8,2048,128),cos=(1,2048,128),sin=(1,2048,128))
BOUNDS=dict(timeout_seconds=900,live_allocation_limit=16*1024**3,max_single_buffer=256*1024**2,
  free_reserve=5*1024**3,cumulative_allocation_limit=512*1024**3)

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
  with Path(path).open('x') as stream:stream.write(json.dumps(data,indent=2,allow_nan=False)+'\n')
def module(name,path):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def no_devices():
  if 'torch' in sys.modules:assert not sys.modules['torch'].cuda.is_initialized()
  for p in Path('/proc/self/fd').iterdir():
    try:value=str(p.readlink())
    except OSError:continue
    assert not value.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb')),value

def validate_manifest(value,canonical):
  assert value==canonical,'Only the exact six original captured train/validation records are admitted'
  rows=value['samples'];assert len(rows)==6
  drives={};commands={}
  for r in rows:
    assert r['split'] in ('train','validation') and PROTECTED not in json.dumps(r)
    assert 'sealed_test' not in json.dumps(r) and r['actual_prompt_tokens'] in (2027,2030)
    assert drives.setdefault(r['drive_id'],r['split'])==r['split']
    assert r['capture_path']==str(TARGETS/r['split']/(r['id']+'.npz'))
    assert r['layer_order']==list(range(16)) and r['original_layer_calls']==16 and r['finite_bf16_bits'] is True
    assert r['conditioning']['basis']=='experimental command'
    assert r['conditioning']['recorded_available_ns'] is None and r['conditioning']['recorded_navigation_source'] is None
    assert not r['conditioning']['recorded_driver_intent']
    i=int(r['id'].rsplit('-',1)[1]);assert r['conditioning']['text']==COMMANDS[i]
    assert r['actual_prompt_tokens']==(2027 if i==0 else 2030)
    commands.setdefault(r['split'],[]).append(i)
    assert r['capture_shapes']=={k:list(v) for k,v in SHAPES.items()}
  assert commands=={'train':[0,1,2],'validation':[0,1,2]} and len(drives)==2
  assert len({r['capture_sha256'] for r in rows})==6
  return rows

def records():
  assert sha(MANIFEST)==MANIFEST_SHA and sha(INPUTS)==INPUTS_SHA
  value=read(MANIFEST);rows=validate_manifest(value,value)
  originals={r['id']:r for r in read(INPUTS)['samples'] if r['split'] in ('train','validation')}
  for r in rows:
    old=originals[r['id']]
    assert all(r[k]==old[k] for k in ('drive_id','query_ns','conditioning','input_file_sha256','actual_prompt_tokens'))
    assert r['route_id']==old['route_id']
  return rows

def bf16(bits):
  import numpy as np
  assert bits.dtype==np.dtype('uint16') and not np.any((bits&0x7f80)==0x7f80),'Expected finite original BF16 bit arrays'
  return (bits.astype(np.uint32)<<16).view(np.float32)

def load_target(row,allowed):
  import numpy as np
  assert row in allowed and row['split'] in ('train','validation') and 'sealed_test' not in json.dumps(row)
  assert PROTECTED not in json.dumps(row)
  path=Path(row['capture_path']);assert path==TARGETS/row['split']/(row['id']+'.npz')
  assert path.stat().st_size==26215834 and sha(path)==row['capture_sha256']
  with zipfile.ZipFile(path) as archive:
    assert set(archive.namelist())=={k+'.npy' for k in SHAPES}
    assert sum(i.file_size for i in archive.infolist())<26*1024**2
  with np.load(path,allow_pickle=False) as archive:
    result={k:bf16(archive[k]) for k in SHAPES}
  assert all(v.shape==SHAPES[k] and np.isfinite(v).all() for k,v in result.items())
  return result

def graph():
  assert sha(GRAPH)==GRAPH_SHA
  return module('agent_distill_fit_qualified_graph',GRAPH)

def loss(B,outputs,targets,valid):
  assert isinstance(valid,int) and not isinstance(valid,bool) and len(outputs)==len(targets)==3
  y,k,v=outputs;assert 0<valid<=y.shape[1]
  assert y.ndim==3 and k.ndim==v.ndim==4 and k.shape==v.shape and k.shape[2]==y.shape[1]
  assert all(a.shape==b.shape for a,b in zip(outputs,targets,strict=True))
  assert all(str(a.dtype)=='dtypes.float32' or str(a.dtype)=='dtypes.f32' for a in (*outputs,*targets))
  return B.recovery_loss(outputs,targets,valid)

def protocol(path):
  p=read(path)
  assert not sys.flags.optimize and p['schema']==1 and p['hyperparameters']==HYPER
  assert p['protected_routes']==[PROTECTED] and p['timeout_seconds']==900
  assert all(p[k]==v for k,v in BOUNDS.items())
  assert p['target']=='KFD:1+AMD:LLVM:gfx1100' and p['selected_pci']=='0000:0c:00.0'
  assert p['selected_gpu_id']==27211 and p['training_manifest_sha256']==MANIFEST_SHA
  assert p['graph_sha256']==GRAPH_SHA and p['sealed_targets_opened'] is False
  assert {str(f.resolve()) for f in EXP.glob('agent-distill-fit-*.py')}<=set(p['source_sha256'])
  assert p['source_sha256'][str(GRAPH)]==GRAPH_SHA and p['source_sha256'][str(INPUTS)]==INPUTS_SHA
  oracle=read(WORK/'agent-distill-tinygrad-artifacts/oracle.json');prefix='vlm.model.language_model.layers.12.'
  expected={name.removeprefix(prefix):dict(bf16_sha256=digest,shape=oracle['arrays']['w:'+name.removeprefix(prefix)]['shape'])
    for name,digest in oracle['student_bf16_master_sha256'].items()}
  assert p['master_weights']==expected and p['block_config']==read(BASE/'model/config.json')['vlm_config']['text_config']
  assert p['teacher_contract']==read(INPUTS)['protocol']['teacher_contract'] and p['teacher_frozen'] is True
  for f,digest in p['source_sha256'].items():assert sha(f)==digest,f
  return p

def qualification(p,receipt_path,reviewed_sha):
  """No availability or GPU access here. Parent supplies a reviewed evidence envelope."""
  assert receipt_path is not None,'Missing reviewed full-length/optimizer feasibility admission'
  receipt_path=Path(receipt_path).resolve()
  assert receipt_path.is_relative_to(ROOT) and receipt_path.name.startswith('agent-distill-fit-')
  assert sha(receipt_path)==reviewed_sha and len(reviewed_sha)==64
  r=read(receipt_path);assert r['schema']==1 and r['pilot_protocol_sha256']==p['_sha256']
  assert r['parent_reviewed'] is True and r['no_reset_or_owner_displacement'] is True
  evidence=r['full_length']
  assert evidence['path'].startswith(str(WORK/'agent-distill-full-artifacts')+'/')
  assert sha(evidence['path'])==evidence['sha256']
  report=read(evidence['path']);assert report['complete'] and report['gpu_executed'] and report['optimizer_updates']==0
  # Exact JSON pointers let the full-length owner retain its own report schema.
  def at(value,pointer):
    assert pointer.startswith('/')
    for key in pointer[1:].split('/'):
      key=key.replace('~1','/').replace('~0','~')
      value=value[int(key)] if isinstance(value,list) else value[key]
    return value
  for key,expected in dict(length=2048,all_passed=True,target=p['target'],pci=p['selected_pci'],graph_sha256=GRAPH_SHA).items():
    assert at(report,evidence['fields'][key])==expected,key
  assert str(at(report,evidence['fields']['no_memory_planner']))==str(r['no_memory_planner'])
  budget=r['optimizer_resource_evidence']
  assert budget['path'].startswith(str(ROOT)+'/') and sha(budget['path'])==budget['sha256']
  b=read(budget['path'])
  assert b['complete'] and b['gpu_executed'] and b['optimizer_updates']==0
  assert b['parameter_count']==50335744 and b['optimizer']==HYPER['optimizer'] and b['fused'] is False
  assert b['target']==p['target'] and b['source_protocol_sha256']==p['_sha256']
  assert b['full_length_report_sha256']==evidence['sha256'] and b['optimizer_state_allocated_and_dummy_update_verified'] is True
  assert b['live_bytes_upper']<=p['live_allocation_limit'] and b['single_buffer_bytes_upper']<=p['max_single_buffer']
  upper=25*b['train_update_with_checkpoint_seconds_upper']+6*b['validation_example_seconds_upper']+b['startup_seconds_upper']
  assert 0<upper<=p['timeout_seconds']-30
  assert all(0<b[k]<=p['timeout_seconds'] for k in ('train_update_with_checkpoint_seconds_upper','validation_example_seconds_upper','startup_seconds_upper'))
  assert b['no_real_data_fitting'] is True
  return r

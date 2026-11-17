"""Fresh immutable pilot binding after successful full-length qualification. Never runs training."""
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-fit-artifacts'
LIVE=ROOT/'live-v4'
FULL=WORK/'agent-distill-full-artifacts'

def sha(p):
  with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
  with p.open('x') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n')
def replace(text,old,new):
  assert text.count(old)==1,old[:120]
  return text.replace(old,new)

def main():
  prior=ROOT/'agent-distill-fit-protocol.json';assert sha(prior)=='53b4ff252d92d4b3b57dd5cb279632ea24e27b504fcee047faa7ee97f9836ef8'
  p=json.loads(prior.read_text())
  for f,digest in p['source_sha256'].items():assert sha(f)==digest,f
  report=FULL/'parent-gfx1100-kfd1/report.json';assert sha(report)=='9d6125b6b34b071f33977bfa465351e989daad7e986dd94546e7cefe0db565d8'
  assert sha(FULL/'protocol.json')=='73804ee0ca981c59c6e051bb0069ef8261eae3a89e6966b1deb64556f4be803f'
  r=json.loads(report.read_text());assert r['complete'] and r['numerical']['all_passed'] and r['length']==2048 and r['optimizer_updates']==0
  fp=json.loads((FULL/'protocol.json').read_text())
  for f,digest in fp['source_sha256'].items():assert sha(f)==digest,f
  assert not LIVE.exists();LIVE.mkdir();frozen=LIVE/'frozen';frozen.mkdir()
  common=(HERE/'agent-distill-fit-common.py').read_text()
  common=replace(common,'live_allocation_limit=16*1024**3','live_allocation_limit=12*1024**3')
  common=replace(common,'cumulative_allocation_limit=512*1024**3','cumulative_allocation_limit=256*1024**3')
  start=common.index('def qualification(')
  common=common[:start]+'''FULL_REPORT=WORK/'agent-distill-full-artifacts/parent-gfx1100-kfd1/report.json'
FULL_PROTOCOL=WORK/'agent-distill-full-artifacts/protocol.json'
FULL_REPORT_SHA='9d6125b6b34b071f33977bfa465351e989daad7e986dd94546e7cefe0db565d8'
FULL_PROTOCOL_SHA='73804ee0ca981c59c6e051bb0069ef8261eae3a89e6966b1deb64556f4be803f'
def verify_full():
  assert sha(FULL_REPORT)==FULL_REPORT_SHA and sha(FULL_PROTOCOL)==FULL_PROTOCOL_SHA
  r=read(FULL_REPORT);q=read(FULL_PROTOCOL)
  assert r['complete'] and r['gpu_executed'] and r['optimizer_updates']==0 and r['length']==2048
  assert r['protocol_sha256']==FULL_PROTOCOL_SHA and r['target']=='KFD:1+AMD:LLVM:gfx1100'
  n=r['numerical'];assert n['all_passed'] and n['optimizer_updates']==0 and n['length']==2048
  assert n['parameter_count']==50335744 and n['parameter_gradient_tensors']==8 and n['input_gradient_tensors']==1
  assert n['fc2_independent_derivative']['passed'] and all(n['checks'].values())
  a=r['actual_device'];assert a['gpu_id']==27211 and a['pci']=='0000:0c:00.0' and a['render_minor']==129
  assert a['requested_kfd_index']==1 and a['logical_device_id']==0 and a['filtered_kfd_nodes']==['2']
  assert a['renderer_target']==r['target'] and not(a['can_recover'] or a['is_usb'] or a['native_pci'])
  assert q['source_sha256'][str(GRAPH)]==GRAPH_SHA
  for f,digest in q['source_sha256'].items():assert sha(f)==digest,f
  return r
def qualification(p,availability_path,reviewed_sha):
  verify_full()
  assert p['full_qualification']==dict(path=str(FULL_REPORT),sha256=FULL_REPORT_SHA,protocol_sha256=FULL_PROTOCOL_SHA)
  assert availability_path is not None,'Fresh parent-reviewed KFD1 availability snapshot required'
  path=Path(availability_path).resolve()
  assert path.is_relative_to(ROOT) and path.name.startswith('agent-distill-fit-')
  assert reviewed_sha and sha(path)==reviewed_sha
  return dict(no_memory_planner=1,availability_review=dict(path=str(path),sha256=reviewed_sha))
'''
  train=(HERE/'agent-distill-fit-train.py').read_text()
  train=replace(train,"HERE/'agent-distill-fit-common.py'","HERE/'agent-distill-fit-live-common.py'")
  train=replace(train,"C.ROOT/'agent-distill-fit-protocol.json'","C.ROOT/'live-v4/agent-distill-fit-live-protocol.json'")
  train=replace(train,"C.ROOT/'pilot-v1'","C.ROOT/'pilot-live-v1'")
  train=replace(train,"parser.add_argument('--admission',type=Path)","parser.add_argument('--admission','--availability-snapshot',dest='admission',type=Path)")
  train=replace(train,"parser.add_argument('--reviewed-admission-sha')","parser.add_argument('--reviewed-admission-sha','--reviewed-availability-sha',dest='reviewed_admission_sha')")
  train=replace(train,'from tinygrad.helpers import Context,DEV,MAX_BUFFER_SIZE,PROFILE',
    'from tinygrad.helpers import Context,DEV,MAX_BUFFER_SIZE,PROFILE,USE_TC,ALLOW_TF32,NO_MEMORY_PLANNER,LRU')
  train=replace(train,"  assert not PMC.value and not SQTT.value and not PROFILE.value and MAX_BUFFER_SIZE.value==p['max_single_buffer']",
    "  assert not PMC.value and not SQTT.value and not PROFILE.value and MAX_BUFFER_SIZE.value==p['max_single_buffer']\n  assert USE_TC.value==ALLOW_TF32.value==0 and NO_MEMORY_PLANNER.value==1 and LRU.value==0")
  train=replace(train,"      previous=C.read(args.resume_from.resolve()/'best.json');assert C.sha(args.resume_from.resolve()/previous['file'])==previous['sha256']",
    "      previous=C.read(args.resume_from.resolve()/'best.json');assert previous['file']=='best.npz'\n      assert C.sha(args.resume_from.resolve()/previous['file'])==previous['sha256']")
  begin=train.index('  original_alloc,original_free=KFDIface.alloc,KFDIface.free')
  end=train.index("  try:\n    dev=Device['AMD']",begin)
  train=train[:begin]+'''  L=C.module('agent_distill_fit_live_ledger',HERE/'agent-distill-fit-ledger.py')
  ledger=L.Ledger(p,K.passive_availability)
  original_alloc,original_free,original_unmap=KFDIface.alloc,KFDIface.free,KFDIface.unmap
  KFDIface.alloc=ledger.alloc(original_alloc);KFDIface.free=ledger.free(original_free);KFDIface.unmap=ledger.unmap(original_unmap)
  report=dict(complete=False,gpu_executed=False,real_optimizer_updates=0,protocol_sha256=args.protocol_sha,
    sealed_targets_opened=False,driving_model_modified=False,allocation_budget=ledger.report,quality_qualified=False,
    full_length_qualification=p['full_qualification'],
    optimizer_logical_persistent_bytes=402685964,parameter_gradient_and_adam_bytes=805371904,
    optimizer_runtime_previously_measured=False)
  S=C.module('agent_distill_fit_state_child',HERE/'agent-distill-fit-state.py')
  ledger.phase('device-and-seed')
'''+train[end:]
  train=replace(train,"    def validate(step):\n      losses=[]","    def validate(step):\n      ledger.phase('validation-'+str(step));losses=[]")
  train=replace(train,"    S.save(root,S.snapshot(block.weights,opt),metadata,context,shapes,initial_recovery=bool(args.resume_from))",
    "    ledger.phase('initial-optimizer-state-checkpoint');S.save(root,S.snapshot(block.weights,opt),metadata,context,shapes,initial_recovery=bool(args.resume_from))")
  train=replace(train,"      row=training[(step-1)%3];S.intent(root,step-1)","      ledger.phase('update-'+str(step));row=training[(step-1)%3];S.intent(root,step-1)")
  train=replace(train,"      S.save(root,S.snapshot(block.weights,opt),metadata,context,shapes)","      ledger.phase('checkpoint-'+str(step));output_storage(root);S.save(root,S.snapshot(block.weights,opt),metadata,context,shapes)")
  train=replace(train,"    KFDIface.alloc,KFDIface.free=original_alloc,original_free\n    report['allocator_hooks_restored']=KFDIface.alloc is original_alloc and KFDIface.free is original_free",'''    # Child-process guards stay active through native buffer finalizers.
    report['allocator_guards_retained_through_finalizers']=True''')
  train=replace(train,"  C.no_devices();return p,rows,None","  C.verify_full();C.no_devices();return p,rows,None")
  train=replace(train,'def preflight(args):','''def output_storage(path):
  parent=Path(path).resolve()
  while not parent.exists():parent=parent.parent
  assert subprocess.check_output(['findmnt','-n','-o','TARGET','-T',str(parent)],text=True,timeout=5).strip()=='/mnt/nvme-500'
  fs=os.statvfs(parent);assert fs.f_bavail*fs.f_frsize>=4*1024**3,'Need 4GiB free for atomic private checkpoints'

def preflight(args):
  output_storage(C.ROOT)''')
  train=replace(train,"      blocked=['Parent-reviewed successful full L2048 zero-update qualification.','Reviewed full-parameter AdamW dummy-update resource/time envelope.','Fresh parent-reviewed KFD1 no-reset availability.']","      blocked=['Fresh parent-reviewed KFD1 no-reset availability.'],full_length_qualification_passed=True")
  main_start=train.index('def main():')
  snapshot='''def snapshot_availability():
  parser=argparse.ArgumentParser();parser.add_argument('--snapshot-availability',action='store_true')
  parser.add_argument('--protocol-sha',required=True);parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args();path=C.ROOT/'live-v4/agent-distill-fit-live-protocol.json'
  assert C.sha(path)==args.protocol_sha;C.protocol(path);C.verify_full();C.no_devices()
  output=args.output.resolve();assert output.is_relative_to(C.ROOT) and output.name.startswith('agent-distill-fit-') and not output.exists()
  K=C.module('agent_distill_fit_availability_only',C.KFD);value=K.passive_availability();C.no_devices()
  output_storage(output.parent);C.write(output,value);print(json.dumps(dict(path=str(output),sha256=C.sha(output),gpu_executed=False,parent_review_required=True),indent=2))

'''
  train=train[:main_start]+snapshot+train[main_start:]
  train=replace(train,"if __name__=='__main__':main()","if __name__=='__main__':\n  if '--snapshot-availability' in sys.argv:snapshot_availability()\n  else:main()")
  for name,text in [('agent-distill-fit-live-common.py',common),('agent-distill-fit-live-train.py',train)]:
    ast.parse(text);(frozen/name).write_text(text)
  for name in ('agent-distill-fit-state.py','agent-distill-fit-ledger.py'):(frozen/name).write_bytes((HERE/name).read_bytes())
  # Preserve the established CPU data/loss/AdamW/recovery evidence and exact math.
  p.update(live_allocation_limit=12*1024**3,cumulative_allocation_limit=256*1024**3,
    full_qualification=dict(path=str(report),sha256=sha(report),protocol_sha256=sha(FULL/'protocol.json')),
    admission_status='Full-length prerequisite passed. Actual parent pilot requires fresh reviewed KFD1 availability; optimizer cost remains unmeasured.',
    required_admission=dict(availability='Fresh parent-reviewed snapshot plus immediate per-allocation conflict/reserve checks; no reset or displacement.'),
    cumulative_bound_basis='256GiB for 25 updates, six validation forwards, optimizer temporaries and checkpoint readbacks. Cap is not an allocation forecast.',
    prior_fit_protocol_sha256=sha(prior))
  sources=dict(p['source_sha256']);sources.update(fp['source_sha256'])
  sources.update({str(f.resolve()):sha(f) for f in [*HERE.glob('agent-distill-fit-*.py'),*frozen.glob('*.py'),report,FULL/'protocol.json',
    ROOT/'cpu-controls-v2/agent-distill-fit-cpu-check.json',ROOT/'guard-controls-v1/agent-distill-fit-guardcheck.json']})
  p['source_sha256']=sources
  protocol=LIVE/'agent-distill-fit-live-protocol.json';write(protocol,p);digest=sha(protocol)
  base=[str(Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')),str(frozen/'agent-distill-fit-live-train.py'),'--protocol-sha',digest]
  write(LIVE/'agent-distill-fit-live-commands.json',dict(executed=False,protocol_sha256=digest,preflight=base,
    parent_snapshot=base+['--snapshot-availability','--output',str(ROOT/'agent-distill-fit-availability-001.json')],
    parent_pilot=base+['--availability-snapshot',str(ROOT/'agent-distill-fit-availability-001.json'),'--reviewed-availability-sha','<fresh reviewed snapshot SHA256>','--execute-training'],
    timeout_seconds=900,updates=25,absolute_update_cap=200))
  print(json.dumps(dict(protocol=str(protocol),sha256=digest,full_length_qualification_bound=True,gpu_executed=False,actual_optimizer_updates=0),indent=2))

if __name__=='__main__':main()

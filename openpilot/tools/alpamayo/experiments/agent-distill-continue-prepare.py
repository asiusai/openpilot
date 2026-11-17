"""Create one separately pinned 25-to-100 continuation. No training or device access."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OLD=WORK/'agent-distill-fit-artifacts'
ROOT=WORK/'agent-distill-continue-artifacts'
LIVE=ROOT/'live-v1'
OLD_PROTOCOL=OLD/'live-v4/agent-distill-fit-live-protocol.json'
OLD_SHA='b60849f66ff169ea1dffc0530ae221be3e03cb38b7f27e229149c6d10274ad04'
FIXED={'report.json':'0978ce8d1fe080ece9f3c220d5996e89464aae27d595fe7085c859ecdc0fa8e6',
  'latest.json':'a47b333c1a2af80a546a4659c959de8e3e02635a10372e33478685443e1eece4',
  'checkpoint-0025.npz':'6fd65991ecd412e16b597b2d18e52806457c9c9686840b22951ee7b27099e7fe',
  'best.npz':'59a8e064a4e92f9744edc5c8d0037785a941e3beb9def2120a0c92c3bcc5effd'}

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def replace(text,old,new):
  assert text.count(old)==1,(old[:160],text.count(old));return text.replace(old,new)

def projection(report):
  phases=report['allocation_budget']['phases'];measurements=[]
  for i,phase in enumerate(phases[:-1]):
    if phase['name'].startswith('update-') and int(phase['name'][7:])<25:
      # Full update plus its following recovery checkpoint, bounded by the next update.
      nxt=phases[i+2];assert nxt['name'].startswith('update-')
      measurements.append(nxt['requested_bytes_at_start']-phase['requested_bytes_at_start'])
  old_total=report['allocation_budget']['requested_bytes']
  # Three complete old runs conservatively retain repeated seed/validation/readback overhead.
  # Add four full raw recovery payloads for restoration and context-transition copies.
  raw=604028940;estimate=3*old_total+4*raw
  assert estimate<256*1024**3
  return dict(measured_first_run_cumulative_bytes=old_total,measured_complete_update_checkpoint_intervals=len(measurements),
    measured_interval_min_bytes=min(measurements),measured_interval_max_bytes=max(measurements),
    additional_updates=75,projection_formula='3 * predecessor all-KFD cumulative + 4 * raw recovery payload bytes',
    conservative_new_process_projection_bytes=estimate,limit_bytes=256*1024**3,
    historical_plus_projected_lineage_bytes=old_total+estimate,
    basis='New process counter starts once. Never reset within phases or updates. Historical predecessor is separately reported, not discarded.')

def main():
  assert sha(OLD_PROTOCOL)==OLD_SHA;p=read(OLD_PROTOCOL)
  for path,digest in p['source_sha256'].items():assert sha(path)==digest,path
  predecessor=OLD/'pilot-live-v1'
  for name,digest in FIXED.items():assert sha(predecessor/name)==digest,name
  report=read(predecessor/'report.json');pointer=read(predecessor/'latest.json')
  assert report['complete'] and report['committed_step']==report['real_optimizer_updates']==25
  assert read(predecessor/'journal.json')==dict(committed=25,intended=None)
  assert not LIVE.exists();ROOT.mkdir(exist_ok=True);LIVE.mkdir();frozen=LIVE/'frozen';frozen.mkdir()
  old_frozen=OLD/'live-v4/frozen'
  common=(old_frozen/'agent-distill-fit-live-common.py').read_text()
  common=replace(common,"ROOT=WORK/'agent-distill-fit-artifacts'","ROOT=WORK/'agent-distill-continue-artifacts'")
  common=replace(common,'HYPER=dict(updates=25,','HYPER=dict(updates=100,')
  common=replace(common,"EXP.glob('agent-distill-fit-*.py')","EXP.glob('agent-distill-continue-*.py')")
  common=replace(common,"path.name.startswith('agent-distill-fit-')","path.name.startswith('agent-distill-continue-')")
  common=replace(common,"  return p\n\nFULL_REPORT",'''  assert p['continuation']==dict(starting_total_updates=25,additional_updates=75,target_total_updates=100,next_command_index=1)
  assert p['predecessor']['protocol_sha256']=='b60849f66ff169ea1dffc0530ae221be3e03cb38b7f27e229149c6d10274ad04'
  assert p['predecessor']['files']['checkpoint-0025.npz']=='6fd65991ecd412e16b597b2d18e52806457c9c9686840b22951ee7b27099e7fe'
  return p

FULL_REPORT''')
  state=(old_frozen/'agent-distill-fit-state.py').read_text()
  state=replace(state,'assert 0<=step<=25','assert 0<=step<=100')
  state=replace(state,'0<=step<=25','0<=step<=100')
  train=(old_frozen/'agent-distill-fit-live-train.py').read_text()
  train=replace(train,'"""Fixed 25-update pilot. Default OFF; reviewed full-length/resource and fresh ownership evidence are mandatory."""',
    '"""Fixed 25-to-100 continuation. Default OFF; predecessor and fresh availability gates are mandatory."""')
  train=replace(train,"HERE/'agent-distill-fit-live-common.py'","HERE/'agent-distill-continue-common.py'")
  train=train.replace("C.ROOT/'live-v4/agent-distill-fit-live-protocol.json'","C.ROOT/'live-v1/agent-distill-continue-protocol.json'")
  assert train.count("C.ROOT/'live-v1/agent-distill-continue-protocol.json'")==2
  train=replace(train,"C.ROOT/'pilot-live-v1'","C.ROOT/'pilot-100-v1'")
  train=replace(train,"HERE/'agent-distill-fit-ledger.py'","HERE/'agent-distill-continue-ledger.py'")
  train=replace(train,"HERE/'agent-distill-fit-state.py'","HERE/'agent-distill-continue-state.py'")
  train=replace(train,"output.name.startswith('agent-distill-fit-')","output.name.startswith('agent-distill-continue-')")
  train=replace(train,"  if args.execute_training or args.child:",'''  A=C.module('agent_distill_continue_admission_preflight',HERE/'agent-distill-continue-admit.py')
  old_state=C.module('agent_distill_continue_predecessor_state',Path(p['predecessor']['state_source']))
  arrays,metadata,transition=A.predecessor(p,old_state)
  del arrays
  if args.resume_from:
    prior=args.resume_from.resolve();assert prior.is_relative_to(C.ROOT) and prior.name.startswith('pilot-')
    S=C.module('agent_distill_continue_resume_preflight',HERE/'agent-distill-continue-state.py')
    shapes={name:tuple(value['shape']) for name,value in p['master_weights'].items()}
    resumed,resumed_metadata=S.load(prior,A.context(p,args.protocol_sha),shapes)
    A.history(resumed_metadata,resumed_metadata['step']);A.read_best(prior,resumed_metadata,shapes)
    assert 25<=resumed_metadata['step']<100;del resumed
  if args.execute_training or args.child:''')
  train=replace(train,"    optimizer_runtime_previously_measured=False)",'''    optimizer_runtime_previously_measured=True,starting_total_updates=25,target_total_updates=100,
    additional_updates_requested=75,total_optimizer_updates=25,additional_committed_updates=0,
    predecessor=p['predecessor'],predecessor_cumulative_allocation_bytes=p['allocation_projection']['measured_first_run_cumulative_bytes'])''')
  train=replace(train,"    context=dict(protocol_sha256=args.protocol_sha,manifest_sha256=C.MANIFEST_SHA,hyperparameters=C.HYPER)",
    "    A=C.module('agent_distill_continue_transition',HERE/'agent-distill-continue-admit.py')\n    context=A.context(p,args.protocol_sha)")
  begin=train.index('    if args.resume_from:\n      arrays,metadata=S.load(')
  end=train.index("    ledger.phase('initial-optimizer-state-checkpoint')",begin)
  train=train[:begin]+'''    ledger.phase('predecessor-or-continuation-restoration')
    if args.resume_from:
      prior=args.resume_from.resolve();arrays,metadata=S.load(prior,context,shapes)
      A.history(metadata,metadata['step']);A.read_best(prior,metadata,shapes)
      assert 25<=metadata['step']<100
      transition=dict(kind='clean continuation recovery',from_root=str(prior),from_step=metadata['step'],
        new_context=context,latest_sha256=C.sha(prior/'latest.json'))
    else:
      old_state=C.module('agent_distill_continue_old_state_child',Path(p['predecessor']['state_source']))
      arrays,metadata,transition=A.predecessor(p,old_state);prior=Path(p['predecessor']['root'])
      transition.update(kind='explicit old-context to new-context transition',new_context=context)
    S.restore(block.weights,opt,arrays)
    restored=S.snapshot(block.weights,opt)
    for name in arrays:np.testing.assert_array_equal(restored[name].view(np.uint32),arrays[name].view(np.uint32))
    transition.update(restored_arrays_bit_exact=True,first_new_step=metadata['step']+1,
      first_command_index=metadata['step']%3)
    A.copy_best(prior,root,metadata,S);C.write(root/'transition.json',transition)
    del arrays,restored
'''+train[end:]
  train=replace(train,'initial_recovery=bool(args.resume_from)','initial_recovery=True')
  train=replace(train,"    start=metadata['step'];report['starting_checkpoint_step']=start",'''    start=metadata['step'];report.update(starting_checkpoint_step=start,total_optimizer_updates=start,additional_committed_updates=start-25)
    assert (start+1-1)%3==start%3 and (start!=25 or start%3==1)''')
  train=replace(train,'range(start+1,26)','range(start+1,101)')
  train=replace(train,"      metadata['step']=step","      metadata['step']=step;report.update(total_optimizer_updates=step,additional_committed_updates=step-25)")
  begin=train.index("          temporary=root/'best-next.npz'")
  end=train.index("      ledger.phase('checkpoint-'+str(step))",begin)
  train=train[:begin]+"          A.commit_best(root,block.weights,metadata,S)\n"+train[end:]
  train=replace(train,"    assert report['real_optimizer_updates']+start==25<=200",'''    assert report['real_optimizer_updates']+start==100<=200
    assert report['additional_committed_updates']==75 and report['total_optimizer_updates']==100
    A.history(metadata,100);A.read_best(root,metadata,shapes)
    report['lineage_cumulative_allocation_bytes']=report['predecessor_cumulative_allocation_bytes']+ledger.report['requested_bytes']''')
  train=replace(train,"    report['allocator_guards_retained_through_finalizers']=True",'''    report['allocator_guards_retained_through_finalizers']=True
    report['lineage_cumulative_allocation_bytes']=report['predecessor_cumulative_allocation_bytes']+ledger.report['requested_bytes']''')
  train=replace(train,"       blocked=['Fresh parent-reviewed KFD1 no-reset availability.'],full_length_qualification_passed=True)","       blocked=['Fresh parent-reviewed KFD1 no-reset availability.'],full_length_qualification_passed=True,predecessor_admitted=True)") if "       blocked=['Fresh parent-reviewed" in train else train
  for name,text in [('agent-distill-continue-common.py',common),('agent-distill-continue-train.py',train),('agent-distill-continue-state.py',state)]:
    ast.parse(text);(frozen/name).write_text(text)
  (frozen/'agent-distill-continue-ledger.py').write_bytes((old_frozen/'agent-distill-fit-ledger.py').read_bytes())
  (frozen/'agent-distill-continue-admit.py').write_bytes((HERE/'agent-distill-continue-admit.py').read_bytes())
  p=copy.deepcopy(p);p['hyperparameters']['updates']=100
  p.update(continuation=dict(starting_total_updates=25,additional_updates=75,target_total_updates=100,next_command_index=1),
    predecessor=dict(root=str(predecessor),protocol_path=str(OLD_PROTOCOL),protocol_sha256=OLD_SHA,
      committed_step=25,context=pointer['context'],metadata=pointer['metadata'],
      state_source=str(old_frozen/'agent-distill-fit-state.py'),
      files={name:sha(predecessor/name) for name in ('report.json','latest.json','journal.json','checkpoint-0025.npz','best.json','best.npz','wall.json')}),
    allocation_projection=projection(report),
    admission_status='One fixed 75-additional-update continuation, admitted from exact clean step25. Parent GPU execution only.',
    cumulative_bound_basis='Measured 25-update all-KFD allocation history; unchanged 256GiB per-process ceiling, never reset within process.')
  sources=dict(p['source_sha256'])
  sources.update({str(path):sha(path) for path in [OLD_PROTOCOL,*HERE.glob('agent-distill-continue-*.py'),*frozen.glob('*.py'),
    *(predecessor/name for name in p['predecessor']['files'])]})
  p['source_sha256']=sources;protocol=LIVE/'agent-distill-continue-protocol.json';write(protocol,p);digest=sha(protocol)
  base=['/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python',str(frozen/'agent-distill-continue-train.py'),'--protocol-sha',digest]
  write(LIVE/'agent-distill-continue-commands.json',dict(executed=False,protocol_sha256=digest,preflight=base,
    parent_snapshot=base+['--snapshot-availability','--output',str(ROOT/'agent-distill-continue-availability-001.json')],
    parent_pilot=base+['--availability-snapshot',str(ROOT/'agent-distill-continue-availability-001.json'),
      '--reviewed-availability-sha','<fresh reviewed snapshot SHA256>','--execute-training'],
    starting_total_updates=25,additional_updates=75,target_total_updates=100,timeout_seconds=900))
  print(json.dumps(dict(protocol=str(protocol),sha256=digest,gpu_executed=False,optimizer_updates=0,allocation_projection=p['allocation_projection']),indent=2))

if __name__=='__main__':main()

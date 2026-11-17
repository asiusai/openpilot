"""Expert pilot entrypoint with the exact qualified KFD alloc/free/owned-unmap ledger."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
QUALIFIED=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-distill-continue-artifacts/live-v1/frozen/agent-distill-continue-ledger.py')
QUALIFIED_SHA='f070ca77dedbc6f22027a2dcb063189b6c66647ae1436d022d0cbf7386b1663d'
TRAINER_SHA='45d2f3631bcd57f69936cdc32cbd4d21ff52987f23cb62cf29ab42eec58cc04f'


def load():
  path=HERE/'agent-distill-expert-train.py'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==TRAINER_SHA
  assert hashlib.sha256(QUALIFIED.read_bytes()).hexdigest()==QUALIFIED_SHA
  spec=importlib.util.spec_from_file_location('expert_ledger_unchanged_trainer',path)
  trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer;spec.loader.exec_module(trainer)
  return trainer


def guarded_child(T,args,records,manifest,admission):
  B,S=T.B,T.S
  K=B.module('expert_ledger_kfd_identity',HERE/'agent-distill-kfd1-common.py')
  K.no_gpu_fds();availability=K.passive_availability();K.require_no_observed_conflict(availability)
  assert availability['vram_free_bytes']>=B.BOUNDS['live_allocation_limit']+B.BOUNDS['free_reserve']
  root=args.output_dir.resolve();S.atomic_json(root/'availability-before-open.json',availability)
  def guard(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
      path=os.fsdecode(values[0])
      if path.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/')):
        assert path in {'/dev/kfd','/dev/dri/renderD129'}
  sys.addaudithook(guard)
  from tinygrad import Device
  from tinygrad.helpers import DEV,MAX_BUFFER_SIZE,PROFILE,USE_TC,ALLOW_TF32,NO_MEMORY_PLANNER,LRU
  from tinygrad.runtime.ops_amd import KFDIface,PMC,SQTT
  assert not Device._opened_devices and repr(DEV)==admission['target'] and DEV.target('AMD').indices=='1'
  assert not PMC.value and not SQTT.value and not PROFILE.value and MAX_BUFFER_SIZE.value==B.BOUNDS['max_single_buffer']
  assert USE_TC.value==ALLOW_TF32.value==0 and NO_MEMORY_PLANNER.value==1 and LRU.value==0
  L=B.module('expert_ledger_qualified_implementation',QUALIFIED)
  parameters=dict(B.BOUNDS,selected_gpu_id=27211,selected_pci='0000:0c:00.0')
  ledger=L.Ledger(parameters,K.passive_availability)
  original_alloc,original_free,original_unmap=KFDIface.alloc,KFDIface.free,KFDIface.unmap
  KFDIface.alloc=ledger.alloc(original_alloc)
  KFDIface.free=ledger.free(original_free)
  KFDIface.unmap=ledger.unmap(original_unmap)
  report=dict(complete=False,gpu_executed=False,allocation_budget=ledger.report,quality_qualified=False,
              manifest=manifest,hyperparameters=B.HYPER,bounds=B.BOUNDS,
              source_sha256={str(p):B.sha(p) for p in HERE.glob('agent-distill-expert-*.py')},
              qualified_ledger=dict(path=str(QUALIFIED),sha256=QUALIFIED_SHA),unchanged_trainer_sha256=TRAINER_SHA)
  ledger.phase('device-and-seed')
  try:
    dev=Device['AMD'];selected=K.selected_node(K.passive_topology())
    identity=dict(requested_kfd_index=1,logical_device_id=dev.device_id,filtered_kfd_nodes=list(KFDIface.gpus),
      interface=type(dev.iface).__name__,gpu_id=dev.iface.gpu_id,pci=Path(dev.iface.dev_sysfs_path).resolve().name,
      render_minor=dev.iface.props['drm_render_minor'],arch=dev.arch,renderer_target=repr(dev.renderer.target),
      is_usb=dev.is_usb,native_pci=dev.is_am(),can_recover=dev.can_recover,allocator=type(dev.allocator).__name__)
    K.validate_opened_identity(identity,selected);report.update(gpu_executed=True,actual_device=identity)
    weights=B.seed();seed_hash=hashlib.sha256()
    for name,value in sorted(weights.items()):seed_hash.update(name.encode());seed_hash.update(value.tobytes())
    block=B.Block(weights,'AMD')
    context=dict(manifest_sha256=manifest['manifest_sha256'],hyperparameters=B.HYPER,seed_sha256=seed_hash.hexdigest(),
                 block_sha256=B.sha(HERE/'agent-distill-expert-block.py'),synthetic_only=False)
    ledger.phase('unchanged-fit-and-checkpoints')
    report.update(T.fit(block,records,B.load,root,context,'AMD',resume=args.resume_from))
  except BaseException as error:
    report['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    # Same qualified pattern: do not restore guards before live native buffers finalize.
    report['allocator_guards_retained_through_finalizers']=True
    report['allocation_snapshot_before_native_finalizers']=True
    S.atomic_json(root/'report.json',report)


def main():
  trainer=load()
  trainer.child=lambda *args:guarded_child(trainer,*args)
  # The unchanged parent launcher must invoke this overlay again in its child process.
  trainer.__file__=str(Path(__file__).resolve())
  trainer.main()


if __name__=='__main__':main()

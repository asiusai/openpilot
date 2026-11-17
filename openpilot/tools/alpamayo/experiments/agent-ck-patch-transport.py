"""Bounded patch writes through real Buffer views and the pinned Buffer.copy_from API."""
import hashlib
import importlib
import time
from collections import Counter

ACTIVE=None


def finalize(report,plan):
  rows=[r for r in report['patch_writes'] if r['transport']=='Buffer.view.copy_from_staged_AMD_dword']
  assert len(rows)==len(report['copy_invocations'])==120 and report['copy_batches']
  assert set(report['copy_host_programs'])==set(plan['copy_host_programs'])
  counts=Counter((r['logical_buffer']['bytes'],r['logical_buffer']['offset_bytes'],r['offset'],r['bytes']) for r in rows)
  assert counts==Counter({tuple(x):2 for x in plan['copy_destinations']})
  for row in rows:
    assert row['complete'] and row['device_copy_bytes']==row['bytes']==4
    assert not row['physical_view']['host_mapped'] and row['staging']['host_mapped']
    assert row['physical_view']['gpu_address']==row['logical_buffer']['gpu_address']+row['offset']
    assert set(row['host_program_associations'])==set(plan['copy_host_programs'])
    assert not row['logical_buffer']['base_gpu_address']<=row['staging']['gpu_address']<row['logical_buffer']['base_gpu_address']+row['logical_buffer']['base_bytes']
  report['patch_transport_qualified']={'completed':True,'device_copy_calls':120,'staging_upload_bytes':480,
    'device_copy_bytes':480,'unique_dword_destinations':60,'each_mutated_and_restored':True,
    'payload_accounting':'All staging payloads are included in io.host_write_bytes/calls. Queue-control traffic follows the original excluded-overhead convention.'}


def preflight(device,plan,report):
  """Compile and authorize the actual hardware's copy submit path before numerical fixtures; do not link or execute it."""
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp,Ops,graph_rewrite
  from tinygrad import dtypes
  from tinygrad.engine import realize
  from tinygrad.runtime.support import hcq2
  identity=importlib.import_module('agent-ck-patch-identity')
  assert device.device=='AMD' and device.arch=='gfx1200' and tuple(device.target)==(12,0,0) and device.is_usb
  assert device.has_copy_queue
  buffers=[Buffer('AMD',4) for _ in range(2)]
  du,su=(UOp.from_buffer(b,dtypes.uint8) for b in buffers)
  started=time.monotonic()
  inputs=[]
  linear=graph_rewrite(UOp(Ops.LINEAR,src=(du.store_call(su),)),hcq2.pm_replace_buffers,ctx=(inputs,{}),walk=True)
  signature=identity.copy_call(linear.src[0]);cid=identity.key(signature)
  assert cid in plan['copy_calls'] and signature==plan['copy_calls'][cid]
  result=realize.compile_linear(linear,input_uops=inputs,cache=False)
  ids=[]
  for p in [u for u in result.toposort(enter_calls=True) if u.op is Ops.PROGRAM]:
    item=identity.host_program(p);pid=identity.key(item)
    assert pid in plan['copy_host_programs'] and item==plan['copy_host_programs'][pid],('Copy preflight host-program mismatch',pid,item)
    ids.append(pid)
  assert len(ids)==8 and set(ids)==set(plan['copy_host_programs'])
  assert all(not b.is_allocated() for b in buffers)
  report['patch_transport_preflight']={'completed':True,'copy_executed':False,'seconds':time.monotonic()-started,
    'copy_target':'USB+AMD:LLVM:gfx1200','host_program_associations':ids,'dummy_buffers_unallocated':True}


def info(buffer):
  storage=buffer.get_storage()
  return {'bytes':buffer.nbytes,'offset_bytes':buffer.offset,'base_bytes':buffer.base.nbytes,
    'gpu_address':int(storage.buf),'base_gpu_address':int(buffer.base.get_storage().buf),
    'host_mapped':storage.host is not None,'buffer_id':id(buffer),'base_buffer_id':id(buffer.base)}


def write(tensor,data,offset,device,report):
  global ACTIVE
  from tinygrad.device import Buffer,BufferSpec
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad import dtypes
  assert ACTIVE is None
  data=bytes(data);buffer=tensor.uop.buffer
  assert device.device==buffer.device=='AMD' and device.arch=='gfx1200' and tuple(device.target)==(12,0,0) and device.is_usb
  assert buffer.is_allocated() and len(data)>0 and offset>=0 and offset+len(data)<=buffer.nbytes
  assert offset%4==len(data)%4==0
  device.synchronize()
  target=buffer.view(len(data),offset).ensure_allocated()
  assert target.base is buffer.base and target.offset==buffer.offset+offset
  assert target.get_storage().buf==buffer.get_storage().buf+offset
  record={'phase':report['active_phase'],'offset':offset,'bytes':len(data),'payload_sha256':hashlib.sha256(data).hexdigest(),
    'logical_buffer':info(buffer),'physical_view':info(target),'complete':False}
  report.setdefault('patch_writes',[]).append(record)
  if (host:=target.get_storage().host) is not None:
    record['transport']='existing_host_mapping'
    host.view(size=len(data),fmt='B')[:]=data
    report['io']['host_write_bytes']+=len(data);report['io']['host_write_calls']+=1
  else:
    # Only the deliberately changed returned O/LSE dwords require this transport.
    assert len(data)==4 and buffer.nbytes in (8388608,131072)
    assert report['active_phase']=='validator_all_bit_negative_controls_A_normalized_candidate'
    assert device.has_copy_queue, 'The fixed patch transport requires the existing AMD SDMA queue'
    staging=Buffer(buffer.device,4,options=BufferSpec(cpu_access=True,nolru=True)).ensure_allocated()
    assert staging.get_storage().host is not None
    staging.host.view(size=4,fmt='B')[:]=data
    report['io']['host_write_bytes']+=4;report['io']['host_write_calls']+=1
    record.update(transport='Buffer.view.copy_from_staged_AMD_dword',staging=info(staging))
    du,su=UOp.from_buffer(target,dtypes.uint8),UOp.from_buffer(staging,dtypes.uint8)
    ACTIVE={'linear':UOp(Ops.LINEAR,src=(du.store_call(su),)),'target':target,'staging':staging,'record':record,'raw_checked':False}
    try:
      target.copy_from(staging)
      device.synchronize()
      assert ACTIVE['raw_checked'], 'Buffer.copy_from must pass the independently checked transport observer'
      record['device_copy_bytes']=4
    finally: ACTIVE=None
    # Staging and its exact view remain alive through the final fence above.
  device.synchronize()
  assert info(target)['gpu_address']==record['physical_view']['gpu_address']
  record['complete']=True

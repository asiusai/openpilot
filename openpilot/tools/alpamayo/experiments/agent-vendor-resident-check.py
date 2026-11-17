"""CPU-only real ELF / real HCQ link tests with virtual storage. No GPU opens."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import patch

FROZEN = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/vendor-source-v2/alpamayo/experiments')
METADATA = Path('/tmp/opencode/alpamayo-vendor-tuned-selected.json')
IMAGE = Path('/tmp/opencode/alpamayo-vendor-fp8.hsaco')
PINS = {IMAGE: '9dbb05bf273cd6a2a3b36c65fc2254ac95b5275804ae58940783e672f3f3ebc0',
        METADATA: 'ada4182c5187a405395edc6b138c742a13eb592db423d60c624b092bef8d1626',
        FROZEN/'vendor-gemm.py': 'aed3ad5bd725e6b7755c8f5860f1c5e2c61181d241445fad36512f76268e7450'}
SOLUTIONS = (27854, 28279)


def module(path):
  spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-', '_'), path)
  value = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(value)
  return value


def provenance():
  result = {}
  for path, expected in PINS.items():
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected, f'Pinned artifact changed: {path}'
    result[str(path)] = {'sha256': digest, 'bytes': path.stat().st_size}
  return result


@contextmanager
def vendor_scope():
  previous = os.environ.get('ALPAMAYO_VENDOR_SOLUTIONS')
  os.environ['ALPAMAYO_VENDOR_SOLUTIONS'] = str(METADATA)
  try: yield module(FROZEN/'vendor-gemm.py')
  finally:
    if previous is None: os.environ.pop('ALPAMAYO_VENDOR_SOLUTIONS', None)
    else: os.environ['ALPAMAYO_VENDOR_SOLUTIONS'] = previous


class HostView:
  """Validate every byte and every contiguous offset without storing duplicate images."""
  def __init__(self, record, expected, offset=0, size=None):
    self.record, self.expected, self.offset = record, expected, offset
    self.size = len(expected)-offset if size is None else size

  def view(self, offset=0, size=None, fmt='B'):
    assert fmt == 'B'
    return HostView(self.record, self.expected, self.offset+offset, self.size-offset if size is None else size)

  def __setitem__(self, index, value):
    assert isinstance(index, slice) and index.step in (None, 1)
    start, stop = self.offset+(index.start or 0), self.offset+(index.stop if index.stop is not None else self.size)
    assert start == self.record['bytes'] and stop <= len(self.expected)
    assert bytes(value) == self.expected[start:stop], 'Image bytes differ from the native relocated image'
    self.record['bytes'] = stop
    self.record['writes'] += 1


class MockAllocator:
  def __init__(self, image): self.image, self.records, self.frees = image, [], []

  def alloc(self, size, options):
    from tinygrad.device import BufferStorage
    assert size == len(self.image) and options.cpu_access
    record = {'address': 0x100000000+len(self.records)*0x10000000, 'bytes': 0, 'writes': 0}
    self.records.append(record)
    return BufferStorage(record['address'], host=HostView(record, self.image))

  def free(self, storage, size, options): self.frees.append(storage.buf)
  def _offset(self, address, size, offset): return address+offset


def device_for(image):
  result = SimpleNamespace(device='AMD', is_usb=True, arch='gfx1200', target=(12, 0, 0),
                           iface=SimpleNamespace(props={'lds_size_in_kb': 64}),
                           allocator=MockAllocator(image), compute_queue=object(), prof_enabled=False, cu_cnt=32, syncs=0)
  def sync(): result.syncs += 1
  result.synchronize = sync
  return result


@contextmanager
def virtual_link_storage(device):
  from tinygrad.device import Buffer
  from tinygrad.runtime.support import hcq2
  class MockBuffer(Buffer):
    def __init__(self, *args, **kwargs):
      kwargs['allocator'] = device.allocator
      super().__init__(*args, **kwargs)
  class Registry:
    def __getitem__(self, key):
      assert key == device.device
      return device
  with patch.object(hcq2, 'Device', Registry()), patch.object(hcq2, 'Buffer', MockBuffer): yield


def program(vendor, index=27854, shape=(2112, 9216, 2048)):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp
  m, n, k = shape
  specs = (((m,n),dtypes.bfloat16),((m,k),dtypes.fp8e4m3),((n,k),dtypes.fp8e4m3),
           ((1,),dtypes.float32),((1,),dtypes.float32),((n,),dtypes.float32),((n,),dtypes.float32),((m,n),dtypes.bfloat16))
  return vendor.program(*(UOp.placeholder(s,d,i) for i,(s,d) in enumerate(specs)), shape=shape,index=index)


def link_image(lib, count=1):
  from tinygrad.runtime.support.hcq2 import hcq_link
  from tinygrad.uop.ops import UOp, Ops
  # These are the same code-buffer GETADDR and BINARY STORE nodes as AMDQueue.exec.
  # GPU queue submission and command execution are deliberately absent.
  return hcq_link(UOp(Ops.LINEAR, src=tuple(lib.getaddr(('AMD',)) for _ in range(count))))


def fails(kind, function):
  try: function()
  except kind: return
  raise AssertionError(f'Expected {kind.__name__}')


def check():
  from tinygrad import Device
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import UOp, Ops
  assert not Device._opened_devices, 'Use a fresh CPU-only process'
  report = {'classification': 'CPU mocks; no USB, GPU execution, or performance evidence', 'pins': provenance(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             Path(__file__).parent.glob('agent-vendor-resident*.py')}}
  adapter = module(Path(__file__).with_name('agent-vendor-resident.py'))
  original_builder, original_loader = ops_amd.amd_build_program, ops_amd._amd_program_image
  original_abi, original_cache = ops_amd.AMDComputeQueue.kernargs, hcq2.hcq_compile_cache
  with vendor_scope() as vendor:
    fake = device_for(b'')
    native = {i: original_loader(fake, vendor.binary(i)) for i in SOLUTIONS}
    report['native_images'] = [{'solution': i, 'bytes': len(image), 'sha256': adapter.sha(image), 'descriptor': asdict(data)}
                               for i,(data,image) in native.items()]
    assert all(len(image) == 67880700 for _,image in native.values())
    assert native[27854][1] == native[28279][1], 'Only descriptor selection may differ'
    image = native[27854][1]
    mutable = bytearray(vendor.binary(27854))
    shoff = struct.unpack_from('<Q',mutable,40)[0]
    count = struct.unpack_from('<H',mutable,60)[0]
    section = next(shoff+64*i for i in range(count) if struct.unpack_from('<I',mutable,shoff+64*i+4)[0] == 1)
    flags = struct.unpack_from('<Q',mutable,section+8)[0]
    struct.pack_into('<Q',mutable,section+8,flags|1)
    fails(AssertionError,lambda:adapter.immutable_elf(bytes(mutable)))
    del mutable
    fake = device_for(image)
    # Only native parsing is memoized. The actual builder, HCQ link, Buffer,
    # address resolution, storage references and every image write stay real.
    by_blob = {vendor.binary(i): value for i,value in native.items()}
    def loaded(dev, blob):
      assert dev.target == (12,0,0)
      return by_blob[blob]
    with patch.object(ops_amd, '_amd_program_image', loaded), virtual_link_storage(fake), vendor.argument_adapter(fake):
      prg = program(vendor)
      fails(AssertionError,lambda:adapter.ResidentImages(fake,vendor,(27854,),max_bytes=1))
      assert not fake.allocator.records
      eager_links = []
      for _ in range(40):
        data, lib = original_builder(fake, prg, ('AMD',))
        eager_links.append(link_image(lib))
      _, lib = original_builder(fake, prg, ('AMD',))
      captured_link = link_image(lib, 40)
      assert len(fake.allocator.records) == 41
      assert all(r['bytes'] == len(image) and r['writes'] == 1 for r in fake.allocator.records)
      # Replays consume the linked graph, not hcq_link. Inspect its 40 addresses
      # repeatedly; no mock claims to execute a GPU graph.
      for _ in range(40): assert len(captured_link.src) == 40
      assert len(fake.allocator.records) == 41
      report['original_link_path'] = {'eager_links':40,'capture_links':1,'retained_replay_inspections':40,
                                     'image_uploads':41,'bytes_written':41*len(image),'each_write_exact':True}
      prior_allocations = len(fake.allocator.records)
      with adapter.install(fake, vendor, SOLUTIONS) as pool:
        assert hcq2.hcq_compile_cache is not original_cache
        hcq2.hcq_compile_cache['CPU-mock-owned-template'] = 'must not leak'
        resident_links = []
        for _ in range(40):
          data, lib = ops_amd.amd_build_program(fake, prg, ('AMD',))
          assert data == native[27854][0]
          assert lib.buffer.device == 'AMD'
          resident_links.append(link_image(lib))
        _, lib = ops_amd.amd_build_program(fake, prg, ('AMD',))
        captured_resident = link_image(lib,40)
        assert any(u.op is Ops.BUFFER and u.buffer is lib.buffer for u in captured_resident.toposort())
        for _ in range(40): assert len(captured_resident.src) == 40
        assert pool.stats['image_uploads'] == 1 and pool.stats['reuse_calls'] == 40
        report['resident_first_40_eager_plus_capture'] = {'image_uploads':1,'bytes_written':len(image),
                                                         'retained_replay_inspections':40,'each_write_exact':True}
        assert len(fake.allocator.records)-prior_allocations == 1
        assert fake.allocator.records[-1]['bytes'] == len(image)
        first = lib
        # New PROGRAM shape revalidates key and ABI metadata, reuses only exact ELF.
        p2 = program(vendor,shape=(2112,2048,9216))
        assert ops_amd.amd_build_program(fake,p2,('AMD',))[1] is first
        assert pool.stats['unique_programs'] == 2
        p3 = program(vendor,index=28279)
        d3, second = ops_amd.amd_build_program(fake,p3,('AMD',))
        assert second is not first and d3 == native[28279][0]
        assert pool.stats['image_uploads'] == 2
        changed = prg.replace(src=(*prg.src[:3],UOp(Ops.BINARY,arg=prg.src[3].arg+b'changed')))
        fails(AssertionError,lambda:ops_amd.amd_build_program(fake,changed,('AMD',)))
        assert pool.stats['image_uploads'] == 2
        report['resident_link_path'] = dict(pool.stats)
        # Native kernels still invoke the original builder and normal link path.
        from dataclasses import replace
        sink = prg.src[0].replace(arg=replace(prg.src[0].arg,name='native_delegation_control'))
        normal = prg.replace(src=(sink,*prg.src[1:]))
        _, normal_lib = ops_amd.amd_build_program(fake,normal,('AMD',))
        assert normal_lib.without_after.op is Ops.ALLOC
        link_image(normal_lib)
      assert pool.closed and pool.stats['builder_and_compile_cache_restored']
      fails(RuntimeError, pool.guard)
      # Linked graph owns image storage even after adapter bookkeeping closes.
      assert first.buffer.is_allocated() and second.buffer.is_allocated()
      assert ops_amd.amd_build_program is original_builder and hcq2.hcq_compile_cache is original_cache
      assert 'CPU-mock-owned-template' not in original_cache
      report['lifetime_and_program_checks'] = ['new_shape_validated_exact_elf_reused','second_elf_separate_allocation',
        'changed_binary_rejected','native_builder_delegated','linked_storage_alive_after_close','closed_scope_rejected']
      with adapter.install(fake,vendor,(27854,)) as pool:
        ops_amd.amd_build_program(fake,prg,('AMD',))
        old = fake.iface
        fake.iface = SimpleNamespace(props={'lds_size_in_kb':64})
        fails(RuntimeError,lambda:ops_amd.amd_build_program(fake,prg,('AMD',)))
        fake.iface = old
        assert pool.failed
      new_context = device_for(image)
      with adapter.install(new_context,vendor,(27854,)) as pool:
        _, fresh_lib = ops_amd.amd_build_program(new_context,prg,('AMD',))
        assert fresh_lib.buffer is not first.buffer
        assert len(new_context.allocator.records) == 1 and pool.stats['image_uploads'] == 1
      report['context_checks'] = ['changed_context_rejected','new_context_uploaded_fresh_image']
      # Deadline after one exact 1MiB write. Failed partial storage is never returned.
      tick = iter((0.,0.,31.,31.))
      with adapter.install(fake,vendor,(27854,),clock=lambda:next(tick)) as pool:
        fails(TimeoutError,lambda:ops_amd.amd_build_program(fake,prg,('AMD',)))
        assert pool.failed and not pool.entries and pool.stats['image_uploads'] == 0
        assert pool.stats['uploads'][0]['uploaded_bytes'] == 1<<20
        fails(RuntimeError,lambda:ops_amd.amd_build_program(fake,prg,('AMD',)))
        report['deadline_failure'] = dict(pool.stats)
    assert not vendor.ACTIVE and ops_amd.AMDComputeQueue.kernargs is original_abi
    # SIGINT's Python exception must unwind both independently scoped adapters.
    interrupted_pool = None
    def interrupted_allocation(size): raise KeyboardInterrupt('CPU-injected image allocation interruption')
    try:
      with vendor.argument_adapter(fake):
        with adapter.install(fake,vendor,(27854,),allocator=interrupted_allocation) as interrupted_pool:
          ops_amd.amd_build_program(fake,prg,('AMD',))
    except KeyboardInterrupt: pass
    assert interrupted_pool.failed and interrupted_pool.closed and interrupted_pool.stats['image_uploads'] == 0
    assert not vendor.ACTIVE and ops_amd.AMDComputeQueue.kernargs is original_abi
    assert ops_amd.amd_build_program is original_builder and hcq2.hcq_compile_cache is original_cache
    timed_pool = None
    tick = iter((0.,31.,31.))
    try:
      with vendor.argument_adapter(fake):
        with adapter.install(fake,vendor,(27854,),clock=lambda:next(tick)) as timed_pool:
          ops_amd.amd_build_program(fake,prg,('AMD',))
    except TimeoutError: pass
    else: raise AssertionError('Timeout must escape both scopes')
    assert timed_pool.failed and timed_pool.closed and timed_pool.stats['image_uploads'] == 0
    assert not vendor.ACTIVE and ops_amd.AMDComputeQueue.kernargs is original_abi
    assert ops_amd.amd_build_program is original_builder and hcq2.hcq_compile_cache is original_cache
    report['abi_finally_restoration'] = {'normal':True,'timeout':True,'keyboard_interrupt':True,'synchronizations':fake.syncs}
    report['safety_checks'] = ['mutable_loaded_section_rejected','byte_budget_checked_before_upload',
                              'private_compile_cache_discarded','allocation_interruption_never_cached']
  # Exercise the parent's FP8 staging and integer oracle without opening a runtime.
  runner = module(Path(__file__).with_name('agent-vendor-resident-run.py'))
  import numpy as np
  values = np.array([-2,-1,0,1,2],dtype=np.int32)
  raw = runner.small_integer_bits(values)
  exponent,mantissa = ((raw>>3)&15).astype(np.int32),(raw&7).astype(np.int32)
  decoded = np.where(exponent == 0,mantissa*2.**-9,(1+mantissa/8)*np.exp2(exponent-7))*np.where(raw&128,-1,1)
  np.testing.assert_array_equal(decoded,values)
  fails(AssertionError,lambda:runner.small_integer_bits(np.array([3],dtype=np.int32)))
  report['parent_smoke_staging'] = {'independent_fp8_decoder_exact':True,'out_of_range_integer_rejected':True,
                                   'gpu_helper_imported_without_opening_device':True}
  for key in ('resident_link_path','deadline_failure'):
    report[key]['timing_kind'] = 'CPU mocked write durations; no USB or GPU timings'
    for upload in report[key]['uploads']: upload['mock_wall_seconds'] = upload.pop('seconds')
  assert not Device._opened_devices
  report['opened_runtime_devices'] = []
  report['passed'] = True
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',required=True,type=Path)
  args = parser.parse_args()
  result = check()
  args.output.write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({'passed':result['passed'],'original':result['original_link_path'],
                    'resident_first_40_calls_uploads':1,'opened_runtime_devices':result['opened_runtime_devices']}))


if __name__ == '__main__': main()

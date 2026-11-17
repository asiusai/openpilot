"""Scoped resident vendor images for pinned tinygrad's USB AMD runtime.

No device is opened here. The caller owns the already-open device, vendor ABI
scope, synchronization, GPU serialization, and numerical acceptance.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import struct
import time


def sha(data): return hashlib.sha256(data).hexdigest()


def check_runtime_sources():
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  from tinygrad import device
  pins = ((ops_amd, '29e1914117c8ca8f17c6ed557a3dd1e684ef49d9315774d69dadb2c1b0233b9b'),
          (hcq2, 'dbab7aef4f8c226d43ca5d138381d5d757a9d8683dfca381ac32e6d826734480'),
          (device, '0d0ccf1c6553522987b4c307b0084cda983490383828d203978232a0b521ec9d'))
  for module, expected in pins:
    assert sha(Path(module.__file__).read_bytes()) == expected, f'Unreviewed tinygrad source: {module.__file__}'
  return {module.__name__: expected for module, expected in pins}


def immutable_elf(blob):
  """Reject mutable loaded sections; retain the unmodified original ELF."""
  assert blob[:6] == b'\x7fELF\x02\x01'
  offset = struct.unpack_from('<Q', blob, 40)[0]
  size, count, _ = struct.unpack_from('<HHH', blob, 58)
  assert size == 64 and offset+size*count <= len(blob)
  for i in range(count):
    section = struct.unpack_from('<IIQQQQIIQQ', blob, offset+i*size)
    # Pinned elf_loader loads SHT_PROGBITS, including its descriptor alias.
    if section[1] == 1: assert not section[2] & 1, 'Mutable PROGBITS cannot share resident code storage'


class ResidentImages:
  def __init__(self, device, vendor, solutions, *, seconds=30., chunk_bytes=1<<20, max_bytes=160<<20,
               allocator=None, clock=time.monotonic):
    from tinygrad.runtime import ops_amd
    from tinygrad.helpers import PROFILE
    sources = check_runtime_sources()
    assert vendor.ACTIVE and device.is_usb and device.arch == 'gfx1200'
    assert not PROFILE and not device.prof_enabled, 'This bounded adapter requires PROFILE=0'
    assert 1 <= seconds <= 60 and chunk_bytes % 4 == 0 and 4096 <= chunk_bytes <= 1<<20
    assert 1 <= len(solutions) <= 2 and len(set(solutions)) == len(solutions)
    assert 1 <= max_bytes <= 160<<20
    self.device, self.vendor, self.clock, self.seconds, self.chunk_bytes = device, vendor, clock, seconds, chunk_bytes
    self.allocate = allocator or self._allocate
    self.token = (device, device.iface, device.allocator, device.compute_queue, tuple(device.target), device.iface.props['lds_size_in_kb'])
    self.original = ops_amd.amd_build_program
    self.closed, self.failed = False, False
    self.entries, self.programs = {}, {}
    self.stats = {'image_uploads': 0, 'uploaded_bytes': 0, 'reuse_calls': 0, 'unique_programs': 0, 'uploads': [],
                  'tinygrad_source_sha256': sources,
                  'max_upload_seconds_per_image': seconds, 'max_resident_bytes': max_bytes,
                  'image_policy': 'One exact aliased ELF per device scope; no sharing across different ELF bytes',
                  'timing_kind': 'Startup upload durations only; no GEMM performance measurement'}
    self.approved = {}
    total = 0
    for index in solutions:
      blob = vendor.binary(index)
      immutable_elf(blob)
      data, image = ops_amd._amd_program_image(device, blob)
      assert isinstance(image, bytes) and len(image) % 4 == 0
      assert data.private_segment_size == 0 and data.kernargs_segment_size in (0, 176)
      assert data.wave32 and not data.enable_dispatch_ptr
      key = sha(blob)
      assert key not in {entry[0] for entry in self.approved.values()}
      self.approved[index] = (key, blob, data, image)
      total += len(image)
    assert total <= max_bytes, 'Resident image byte budget exceeded before any upload'
    self.stats['approved_images'] = [{'solution': i, 'elf_sha256': key, 'image_sha256': sha(image), 'bytes': len(image),
                                      'descriptor_offset': data.desc_offset, 'entry_point_offset': data.entry_point_offset}
                                     for i, (key, _, data, image) in self.approved.items()]

  def _allocate(self, size):
    from tinygrad.device import Buffer, BufferSpec
    return Buffer(self.device.device, size, options=BufferSpec(cpu_access=True, nolru=True),
                  allocator=self.device.allocator).ensure_allocated()

  def guard(self):
    if self.closed or self.failed: raise RuntimeError('Resident scope is closed or failed; start a fresh scope')
    dev = self.device
    now = (dev, dev.iface, dev.allocator, dev.compute_queue, tuple(dev.target), dev.iface.props['lds_size_in_kb'])
    if any(a is not b for a, b in zip(self.token[:4], now[:4])) or self.token[4:] != now[4:]:
      self.failed = True
      raise RuntimeError('Device context changed; resident images are invalid')
    assert self.vendor.ACTIVE, 'Vendor ABI scope ended before resident use'
    assert not dev.prof_enabled

  def build(self, device, program, devices):
    from tinygrad import dtypes
    from tinygrad.device import HCQ_RUNTIME_DEV
    from tinygrad.uop.ops import UOp, Ops
    self.guard()
    name = program.src[0].arg.name
    if name not in self.vendor.PROGRAMS:
      assert not name.startswith('alpamayo_vendor_'), 'Unregistered vendor program'
      return self.original(device, program, devices)
    assert device is self.device and devices == (self.device.device,), 'Different device context'
    index, shape = self.vendor.PROGRAMS[name]
    assert name == f'alpamayo_vendor_{index}_{shape[0]}_{shape[1]}_{shape[2]}'
    assert index in self.approved, 'Unapproved vendor solution'
    key, blob, data, image = self.approved[index]
    assert program.src[3].op is Ops.BINARY and program.src[3].arg == blob, 'Vendor program bytes changed; refuse stale residency'
    # A new PROGRAM is validated separately. Shape/grid changes can share this
    # exact immutable ELF, while ABI/grid construction remains in original code.
    if program.key not in self.programs:
      self.programs[program.key] = key
      self.stats['unique_programs'] += 1
    assert self.programs[program.key] == key
    if key not in self.entries:
      started, completed = self.clock(), 0
      upload = {'solution': index, 'elf_sha256': key, 'image_sha256': sha(image), 'bytes': len(image), 'complete': False}
      self.stats['uploads'].append(upload)
      try:
        buffer = self.allocate(len(image))
        # Keep partial storage owned until scope cleanup; a failed image is
        # never returned or installed in the usable entries map.
        self.partial = buffer
        host = buffer.get_storage().host
        assert host is not None, 'Resident upload requires the normal CPU-accessible program allocation'
        view = host.view(size=len(image), fmt='B')
        for offset in range(0, len(image), self.chunk_bytes):
          if self.clock()-started >= self.seconds: raise TimeoutError('Resident program upload exceeded its startup deadline')
          chunk = image[offset:offset+self.chunk_bytes]
          view[offset:offset+len(chunk)] = chunk
          completed += len(chunk)
        if self.clock()-started >= self.seconds: raise TimeoutError('Resident program upload exceeded its startup deadline')
        assert completed == len(image)
        value = UOp.from_buffer(buffer, dtypes.uint8, HCQ_RUNTIME_DEV.device)
        self.entries[key] = (buffer, value)
        self.partial = None
        upload.update(complete=True, uploaded_bytes=completed, seconds=self.clock()-started)
        self.stats['image_uploads'] += 1
        self.stats['uploaded_bytes'] += completed
      except BaseException as exc:
        self.failed = True
        upload.update(uploaded_bytes=completed, seconds=self.clock()-started, failure=f'{type(exc).__name__}: {exc}')
        raise
    else: self.stats['reuse_calls'] += 1
    # Return the existing storage. No ALLOC/BINARY STORE remains for hcq_link
    # to allocate or upload. Its GETADDR references keep the buffer alive.
    return data, self.entries[key][1]

  def close(self):
    self.closed = True
    self.entries.clear()
    self.programs.clear()
    self.partial = None


@contextmanager
def install(device, vendor, solutions, **kwargs):
  """Scope builder/cache to one unchanged context; restore on all exits.

  Keep every JIT invocation inside this scope. Check pool.guard() before calls.
  Device reset and concurrent use of other runtimes are unsupported.
  """
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  original, cache = ops_amd.amd_build_program, hcq2.hcq_compile_cache
  assert not getattr(original, '_agent_vendor_resident', False), 'Nested resident scope'
  resident = ResidentImages(device, vendor, solutions, **kwargs)
  def build(dev, program, devices): return resident.build(dev, program, devices)
  build._agent_vendor_resident = True
  ops_amd.amd_build_program = build
  hcq2.hcq_compile_cache = {}  # Templates in this scope may retain resident BUFFERs.
  try: yield resident
  finally:
    ops_amd.amd_build_program = original
    hcq2.hcq_compile_cache = cache
    resident.close()
    resident.stats['builder_and_compile_cache_restored'] = True

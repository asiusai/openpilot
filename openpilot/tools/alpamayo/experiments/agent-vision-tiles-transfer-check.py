"""Local input transfers and CPU checks. No GPU is opened by this program.

Exact byte shadows apply only to these exclusively owned, read-only kernel
inputs. External writes require invalidate() or force=True. No tensor version,
array identity, checksum, or prior fixture state substitutes for byte equality.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import weakref

import numpy as np

# Keep sources alive after an interrupted asynchronous copy. Normal completion
# releases sources after synchronize(). No automatic retry follows an error.
FAILED_STAGING = []


class InputTransfers:
  def __init__(self, device, *, mode='staging', skip_identical=True, stage_factory=None):
    assert mode in ('staging', 'direct')
    self.device, self.mode, self.skip_identical = device, mode, skip_identical
    self.stage_factory = stage_factory or self.cpu_stage
    self.entries, self.events, self.pending = {}, [], []

  @staticmethod
  def cpu_stage(payload):
    from tinygrad.device import Buffer
    return Buffer('CPU', len(payload), initial_value=payload)

  def register(self, tensors):
    for tensor in tensors:
      buffer = tensor.uop.buffer.ensure_allocated()
      assert buffer.base is buffer and buffer.offset == 0, 'Owned inputs must be whole allocations'
      assert buffer not in self.entries
      self.entries[buffer] = {'storage': buffer.get_storage(), 'base': buffer.base, 'offset': buffer.offset,
        'nbytes': buffer.nbytes, 'shape': tuple(tensor.shape), 'shadow': None}

  def upload(self, values, dtypes_):
    from tinygrad import Tensor
    from tinygrad.device import Buffer, BufferSpec
    from tinygrad.uop.ops import UOp
    tensors = []
    for value, dtype in zip(values, dtypes_, strict=True):
      assert value.dtype.itemsize == dtype.itemsize and value.nbytes > 0
      buffer = Buffer(self.device.device, value.nbytes, options=BufferSpec(cpu_access=True)).ensure_allocated()
      tensors.append(Tensor(UOp.from_buffer(buffer, dtype)).reshape(value.shape))
    tensors = tuple(tensors)
    self.register(tensors)
    self.replace(tensors, values)
    return tensors

  def invalidate(self, tensors=None):
    buffers = self.entries if tensors is None else [tensor.uop.buffer for tensor in tensors]
    for buffer in buffers: self.entries[buffer]['shadow'] = None

  def replace(self, tensors, values, *, force=False):
    self.device.synchronize()
    ready = []
    for tensor, value in zip(tensors, values, strict=True):
      buffer = tensor.uop.buffer
      assert buffer in self.entries, 'Input allocation changed after registration/capture'
      entry = self.entries[buffer]
      assert buffer.base is entry['base'] and buffer.offset == entry['offset'] and buffer.nbytes == entry['nbytes']
      assert buffer.get_storage() is entry['storage'], 'Input storage changed after registration/capture'
      assert tuple(tensor.shape) == entry['shape'] == tuple(value.shape)
      payload = np.ascontiguousarray(value).tobytes(order='C')
      assert len(payload) == buffer.nbytes
      skip = self.skip_identical and not force and entry['shadow'] is not None and payload == entry['shadow']
      ready.append((buffer, entry, payload, skip))
    # Check every argument before any copy. Mark all writes untrusted before
    # submitting, so partial errors and interrupted fences cannot bless a shadow.
    for _, entry, _, skip in ready:
      if not skip: entry['shadow'] = None
    try:
      for buffer, _, payload, skip in ready:
        if skip: continue
        if self.mode == 'staging':
          source = self.stage_factory(payload)
          self.pending.append(source)
          assert buffer.copy_from(source) is buffer
        else:
          # Explicit fallback only. This is the former slow upload operation.
          buffer.host.view(size=buffer.nbytes, fmt='B')[:] = payload
      self.device.synchronize()
    except BaseException:
      FAILED_STAGING.extend(self.pending)
      self.pending = []
      self.events.append({'completed': False, 'requested_write_bytes': sum(len(p) for _, _, p, skip in ready if not skip)})
      raise
    self.pending.clear()
    for _, entry, payload, skip in ready:
      if not skip and self.skip_identical: entry['shadow'] = payload
    result = {'completed': True, 'mode': self.mode, 'force': force, 'writes': [not skip for _, _, _, skip in ready],
      'written_bytes': sum(len(p) for _, _, p, skip in ready if not skip), 'skipped_bytes': sum(len(p) for _, _, p, skip in ready if skip),
      'same_allocations': True, 'completion_fence': True}
    self.events.append(result)
    return result

  def report(self):
    return {'mode': self.mode, 'exact_byte_shadows': self.skip_identical, 'gpu_transfer_speed_qualified': False,
      'events': self.events.copy(), 'written_bytes': sum(e.get('written_bytes', 0) for e in self.events),
      'skipped_bytes': sum(e.get('skipped_bytes', 0) for e in self.events)}


def cpu_checks():
  from types import SimpleNamespace
  class Stage:
    def __init__(self, payload): self.payload = payload
  class DeviceFake:
    device = 'FAKE'
    def __init__(self): self.queue, self.fences, self.fail_fence = [], 0, False
    def synchronize(self):
      self.fences += 1
      gc.collect()  # queued sources must stay alive through the completion fence
      while self.queue:
        destination, source = self.queue.pop(0)
        assert source() is not None, 'Staging source was freed before completion'
        destination.bytes[:] = source().payload
      if self.fail_fence is True or self.fail_fence == self.fences:
        self.fail_fence = False
        raise RuntimeError('fake interrupted completion')
  class BufferFake:
    def __init__(self, device, size):
      self.device, self.nbytes, self.base, self.offset = device, size, self, 0
      self.storage, self.bytes, self.calls, self.fail_copy = object(), bytearray([0xa5])*size, 0, False
    def ensure_allocated(self): return self
    def get_storage(self): return self.storage
    @property
    def host(self): raise AssertionError('Staging mode touched direct destination host writes')
    def copy_from(self, source):
      self.calls += 1
      self.device.queue.append((self, weakref.ref(source)))
      if self.fail_copy:
        self.fail_copy = False
        raise RuntimeError('fake submission interruption')
      return self
  def tensor(buffer, shape): return SimpleNamespace(uop=SimpleNamespace(buffer=buffer), shape=shape, version=0)
  device = DeviceFake()
  values = [np.arange(32, dtype=np.uint16).reshape(2, 4, 4), np.array([.125, .25], np.float32)]
  buffers = [BufferFake(device, a.nbytes) for a in values]
  tensors = tuple(tensor(b, a.shape) for b, a in zip(buffers, values, strict=True))
  tx = InputTransfers(device, stage_factory=Stage)
  tx.register(tensors)
  def exact():
    for buffer, value in zip(buffers, values, strict=True): assert bytes(buffer.bytes) == value.tobytes()
  initial = [v.copy() for v in values]
  assert tx.replace(tensors, values)['writes'] == [True, True]
  exact()
  calls = [b.calls for b in buffers]
  assert tx.replace(tensors, [v.copy() for v in values])['writes'] == [False, False]
  assert [b.calls for b in buffers] == calls
  # Mutate in place without changing the arrays, Tensor version, or buffer identities.
  unchanged_image = bytes(buffers[0].bytes[:values[0][0].nbytes])
  values[0][1, 0, 0] ^= 0x8001
  assert tx.replace(tensors, values)['writes'] == [True, False]
  exact()
  assert bytes(buffers[0].bytes[:values[0][0].nbytes]) == unchanged_image
  assert all(t.version == 0 for t in tensors)
  # A failure at the completion fence also keeps the shadow untrusted.
  values[0][1, 0, 2] += 1
  device.fail_fence = device.fences+2
  try: tx.replace(tensors, values)
  except RuntimeError: pass
  else: raise AssertionError('Fake completion interruption was lost')
  assert tx.entries[buffers[0]]['shadow'] is None
  assert tx.replace(tensors, values)['writes'] == [True, False]
  exact()
  # Validate all inputs before submitting even the first changed argument.
  values[0][1, 0, 3] += 1
  before_calls = [b.calls for b in buffers]
  try: tx.replace(tensors, [values[0], values[1].reshape(1, 2)])
  except AssertionError: pass
  else: raise AssertionError('Bad later argument bypassed validation')
  assert before_calls == [b.calls for b in buffers] and not device.queue
  assert tx.replace(tensors, values)['writes'] == [True, False]
  exact()
  # Logical C-order bytes are preserved for strided arrays too.
  values[0] = values[0][:, :, ::-1]
  assert not values[0].flags.c_contiguous
  assert tx.replace(tensors, values)['writes'] == [True, False]
  exact()
  values[1][1] = .5
  assert tx.replace(tensors, values)['writes'] == [False, True]
  exact()
  values = [v.copy() for v in initial]
  assert tx.replace(tensors, values)['writes'] == [True, True]  # changed then restored is still a write
  exact()
  # Equal numeric values with distinct bits must not bypass a write.
  values[1][0] = np.float32(0)
  tx.replace(tensors, values)
  values[1][0] = np.float32(-0.)
  assert tx.replace(tensors, values)['writes'] == [False, True]
  exact()
  # Invalidate after an explicitly external write; versions do not detect it.
  buffers[0].bytes[0] ^= 255
  tx.invalidate((tensors[0],))
  assert tx.replace(tensors, values)['writes'] == [True, False]
  exact()
  buffers[1].bytes[0] ^= 255
  assert tx.replace(tensors, values, force=True)['writes'] == [True, True]
  exact()
  # Allocation replacement cannot silently redirect a captured graph.
  previous = buffers[0].storage
  buffers[0].storage = object()
  try: tx.replace(tensors, values)
  except AssertionError: pass
  else: raise AssertionError('Changed storage passed identity gate')
  buffers[0].storage = previous
  substitute = tensor(BufferFake(device, buffers[0].nbytes), values[0].shape)
  try: tx.replace((substitute, tensors[1]), values)
  except AssertionError: pass
  else: raise AssertionError('Rebound tensor passed identity gate')
  # An interrupted submission never installs the intended byte shadow.
  values[0][1, 0, 1] += 1
  buffers[0].fail_copy = True
  try: tx.replace(tensors, values)
  except RuntimeError: pass
  else: raise AssertionError('Fake copy failure was lost')
  assert tx.entries[buffers[0]]['shadow'] is None
  device.synchronize()
  assert tx.replace(tensors, values)['writes'] == [True, False]
  exact()
  # The complete fixture update sequence must never skip any changed bytes.
  for state in range(20):
    before = [bytes(b.bytes) for b in buffers]
    if state % 3 == 0: values[0][state % 2, state % 4, state % 4] ^= 17
    if state % 5 == 0: values[1][state % 2] = np.float32(state*.03125)
    should_write = [old != v.tobytes() for old, v in zip(before, values, strict=True)]
    assert tx.replace(tensors, values)['writes'] == should_write
    exact()
  unshadowed = InputTransfers(device, stage_factory=Stage, skip_identical=False)
  unshadowed.register(tensors)
  for _ in range(2):
    assert unshadowed.replace(tensors, values)['writes'] == [True, True]
    exact()
  # Exercise the normal pinned Buffer.copy_from API and actual CPU JIT replay.
  from tinygrad import Device, TinyJit, dtypes
  real = InputTransfers(Device['CPU'])
  data = np.arange(16, dtype=np.float32).reshape(2, 8)
  inputs = real.upload([data], [dtypes.float32])
  identity = inputs[0].uop.buffer
  run = TinyJit(lambda x: (x*2+1).realize())
  for _ in range(3): result = run(*inputs)
  np.testing.assert_array_equal(result.numpy(), data*2+1)
  data[1] += 100
  real.replace(inputs, [data])
  assert inputs[0].uop.buffer is identity
  np.testing.assert_array_equal(run(*inputs).numpy(), data*2+1)
  restored = np.arange(16, dtype=np.float32).reshape(2, 8)
  real.replace(inputs, [restored])
  np.testing.assert_array_equal(run(*inputs).numpy(), restored*2+1)
  assert not any(name.startswith('AMD') for name in Device._opened_devices)
  return {'gpu_executed': False, 'gpu_speed_qualified': False, 'exact_bytes': True, 'one_image_isolation': True,
    'stale_version_bypassed': True, 'identical_bytes_untouched': True, 'changed_then_restored_written': True,
    'force_and_invalidation_checked': True, 'replaced_storage_rejected': True, 'interrupted_copy_shadow_rejected': True,
    'interrupted_completion_shadow_rejected': True, 'all_arguments_validated_before_writes': True,
    'noncontiguous_logical_bytes_exact': True, 'disabled_shadows_always_write': True,
    'all_twenty_mutation_steps_checked': True, 'normal_cpu_copy_from_and_captured_replay_exact': True,
    'fake_transfers': tx.report(), 'cpu_transfers': real.report()}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.environ.update(DEV='CPU', PROFILE='0', PYTHONDONTWRITEBYTECODE='1')
  report = cpu_checks()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'cpu_checks_passed': True, 'gpu_executed': False, 'output': str(args.output)}))


if __name__ == '__main__': main()

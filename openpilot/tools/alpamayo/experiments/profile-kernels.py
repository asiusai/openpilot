"""Stage-local GPU timestamps, including the USB backend's raw byte representation.

Profiling changes execution overhead. These runs cannot establish throughput.
"""
from collections import defaultdict
from contextlib import contextmanager
import struct
from types import MethodType
from tinygrad.helpers import PROFILE


class Recorder:
  def __init__(self, device, benchmark):
    assert PROFILE, 'Set PROFILE=1 before importing tinygrad'
    self.stage = 'other'
    self.reset()
    original_timer = benchmark.timed

    @contextmanager
    def timed(name, *args, **kwargs):
      previous, self.stage = self.stage, name
      try:
        with original_timer(name, *args, **kwargs):
          yield
      finally:
        self.stage = previous

    def collect(dev):
      buffers = {}
      for (buffer, _), entry in dev.prof_ents.items():
        if buffer not in buffers:
          values = buffer.host.view(fmt='Q')[:]
          if isinstance(values, (bytes, bytearray)) or (isinstance(values, memoryview) and values.itemsize == 1):
            values = struct.unpack('<'+'Q'*(len(values)//8), values)
          buffers[buffer] = values
        values = buffers[buffer]
        elapsed = (values[entry.en_id]-values[entry.st_id])/dev.timestamp_divider/1000
        if not 0 <= elapsed < 300000:
          # Retain invalid samples explicitly; never interpret them as durations.
          self.invalid.append({'stage': self.stage, 'name': str(entry.name),
                               'start': values[entry.st_id], 'end': values[entry.en_id], 'elapsed_ms': elapsed})
          continue
        row = self.stats[self.stage][str(entry.name)]
        row[0] += 1
        row[1] += elapsed
        row[2] = max(row[2], elapsed)
      dev.prof_ents.clear()

    device.collect_prof = MethodType(collect, device)
    # Tinygrad's generic final calibration also assumes typed USB reads.
    # This recorder uses device durations only and needs no host-clock mapping.
    device._at_profile_finalize = lambda: None
    benchmark.timed = timed

  def reset(self):
    self.stats = defaultdict(lambda: defaultdict(lambda: [0, 0., 0.]))
    self.invalid = []

  def report(self):
    return {'invalid_samples': self.invalid, 'stages': {stage: {'sum_kernel_ms': sum(v[1] for v in rows.values()),
                    'kernels': sorted([{'name': k, 'count': v[0], 'total_ms': v[1], 'max_ms': v[2]}
                                       for k, v in rows.items()], key=lambda x: -x['total_ms'])}
            for stage, rows in self.stats.items()}}

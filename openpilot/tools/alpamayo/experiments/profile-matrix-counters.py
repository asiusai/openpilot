"""Collect raw gfx1200 counters and device timestamps for existing FP8 kernels.

Counter collection changes scheduling. These measurements are diagnostic and
cannot establish model throughput or percentage of theoretical utilization.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import struct
from types import MethodType
import numpy as np

COUNTERS = ('SQ_BUSY_CYCLES', 'SQ_WAIT_ANY', 'SQ_INSTS_VALU', 'SQ_WAVES',
            'GRBM_GUI_ACTIVE', 'GL2C_HIT', 'GL2C_MISS')


def uint64_values(value):
  if isinstance(value, (bytes, bytearray)) or (isinstance(value, memoryview) and value.itemsize == 1):
    assert len(value) % 8 == 0
    return list(struct.unpack('<'+'Q'*(len(value)//8), value))
  return [int(x) for x in value]


class CounterRecorder:
  def __init__(self, device):
    self.label = 'setup'
    self.records = []
    self.timestamps = []
    self.invalid = []

    def collect(dev):
      if dev.prof_enabled:
        log = uint64_values(dev.prof_log.host.view(fmt='Q')[:])
        count = log[0]
        if count-dev.prof_read > dev.prof_slots:
          raise RuntimeError('Counter slots overwritten; diagnostic is invalid')
        for index in range(dev.prof_read, count):
          slot = index % dev.prof_slots
          blob = bytes(dev.pmc_buf.host.view(fmt='B')[slot*dev.pmc_size:(slot+1)*dev.pmc_size])
          counters = []
          for sample in dev.pmc_sched:
            values = uint64_values(blob[sample.off:sample.off+sample.size])
            counters.append({'schedule': asdict(sample), 'values': values, 'sum': sum(values)})
          self.records.append({'label': self.label, 'index': index, 'tag': log[1+slot], 'counters': counters})
        dev.prof_read = count
      buffers = {}
      for (buffer, _), entry in dev.prof_ents.items():
        if buffer not in buffers: buffers[buffer] = uint64_values(buffer.host.view(fmt='Q')[:])
        values = buffers[buffer]
        elapsed = (values[entry.en_id]-values[entry.st_id])/dev.timestamp_divider/1000
        row = {'label': self.label, 'name': str(entry.name), 'milliseconds': elapsed}
        if not 0 <= elapsed < 300000: self.invalid.append(row)
        else: self.timestamps.append(row)
      dev.prof_ents.clear()

    device.collect_prof = MethodType(collect, device)
    device._at_profile_finalize = lambda: None


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--metadata-only', action='store_true')
  parser.add_argument('--replays', type=int, default=5)
  args = parser.parse_args()
  from tinygrad.runtime.support.amd import import_pmc
  mapping = import_pmc((12, 0, 0))
  assert all(name in mapping for name in COUNTERS)
  report = {'schema': 1, 'counter_events': {name: mapping[name] for name in COUNTERS},
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limits': ['Raw counter values retain their instance dimensions.',
                       'No verified WMMA-specific instruction counter is used.',
                       'Counter and profiling overhead changes execution.',
                       'Packing, quantization, uploads and model execution are outside each matrix measurement.']}
  if args.metadata_only:
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    return
  assert os.environ.get('PROFILE') == '1' and os.environ.get('PMC') == '1'
  assert os.environ.get('PMC_COUNTERS') == ','.join(COUNTERS)
  from tinygrad import Device, Tensor, TinyJit, dtypes
  from tinygrad.device import Compiled, ProfileProgramEvent
  from amd_gemm import pack, _gemm
  device = Device[Device.DEFAULT]
  assert device.arch == 'gfx1200' and device.pmc_enabled
  recorder = CounterRecorder(device)
  rng = np.random.default_rng(9060)
  scale = Tensor([1.], dtype=dtypes.float32).reshape(()).realize()
  cases = []
  for label, m, n, k in (('integer_control', 17, 48, 32), ('prefill_fc1', 2112, 9216, 2048),
                         ('prefill_fc2', 2112, 2048, 9216), ('vision_fc1', 720, 4304, 1152),
                         ('expert_fc2', 64, 512, 3840)):
    recorder.label = 'setup_'+label
    a, b = (rng.integers(-3, 4, (rows, k), dtype=np.int32) for rows in (m, n))
    ap, bp = (pack(Tensor(x.astype(np.float32)).cast(dtypes.fp8e4m3)).realize() for x in (a, b))
    def multiply(x, weight):
      out = Tensor.empty(m, n, dtype=dtypes.bfloat16)
      return Tensor.custom_kernel(out, x, weight, scale, scale, fxn=partial(_gemm, packed_input=True))[0]
    fn = TinyJit(multiply)
    for _ in range(3): result = fn(ap, bp)
    device.synchronize()
    actual = result.bitcast(dtypes.uint16).numpy()
    if label == 'integer_control':
      expected = Tensor((a @ b.T).astype(np.float32), device='CPU').cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy()
      np.testing.assert_array_equal(actual, expected)
    for replay in range(args.replays):
      recorder.label = f'{label}/{replay}'
      fn(ap, bp)
      device.synchronize()
    cases.append({'label': label, 'shape': [m, n, k], 'useful_matmul_flops': 2*m*n*k})
  tags = {str(e.tag): e.name for e in Compiled.profile_events if isinstance(e, ProfileProgramEvent) and e.tag is not None}
  sums = defaultdict(lambda: defaultdict(int))
  for row in recorder.records:
    for counter in row['counters']: sums[row['label']][counter['schedule']['name']] += counter['sum']
  assert not recorder.invalid
  report.update(device=device.arch, cases=cases, records=recorder.records, timestamps=recorder.timestamps,
                invalid=recorder.invalid, program_tags=tags, sums=dict(sums))
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'output': str(args.output), 'matrix_cases': len(cases),
                    'counter_records': len(recorder.records), 'invalid': recorder.invalid}), flush=True)


if __name__ == '__main__': main()

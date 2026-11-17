"""CPU execution and gfx1200 compiler checks for isolated upstream backports.

Never opens a GPU. Compiler wall times do not measure inference or GPU speed.
Each compiler case runs in a fresh subprocess to avoid cached UOp properties.
"""
import argparse
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import textwrap
import time
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
PINNED = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
SOURCE = Path('/tmp/opencode/agent-upstream-source-20261116')
MODES = ('original', 'linearizer', 'recursive-properties')
SHAPES = ((2112, 9216, 2048), (2112, 2048, 9216), (2112, 4096, 2048),
          (16, 9216, 2048), (32, 9216, 2048), (48, 9216, 2048), (64, 9216, 2048))


def load_candidate():
  spec = importlib.util.spec_from_file_location('agent_upstream_candidates', HERE/'agent-upstream-candidates.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def digest(data): return hashlib.sha256(data).hexdigest()


def ast_body(source, *names):
  node = ast.parse(textwrap.dedent(source))
  for name in names: node = next(x for x in node.body if getattr(x, 'name', None) == name)
  return ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)


def check_provenance(candidate):
  pairs = (
    ('tinygrad/codegen/late/linearizer.py', ('linearize',), candidate.upstream_linearize),
    ('tinygrad/uop/ops.py', ('recursive_property', '__get__'), candidate.upstream_recursive_get),
    ('tinygrad/runtime/support/usb.py', ('USBMMIOInterface', '__getitem__'), candidate.upstream_usb_getitem),
  )
  records = []
  for rel, names, fn in pairs:
    source = (SOURCE/rel).read_text()
    assert ast_body(source, *names) == ast_body(inspect.getsource(fn), fn.__name__), fn.__name__
    records.append({'source': rel, 'source_sha256': digest(source.encode()), 'body_matches_upstream': True})
  return records


def check_usb_and_restore(candidate):
  import tinygrad.codegen as codegen
  import tinygrad.codegen.late.linearizer as linearizer_module
  from tinygrad.runtime.support.usb import USBMMIOInterface
  from tinygrad.uop.ops import recursive_property
  class FakeUSB:
    def __init__(self): self.data = bytearray(range(64)); self.calls = []
    def read(self, addr, size): self.calls.append(('read', addr, size)); return bytes(self.data[addr:addr+size])
    def pcie_mem_read(self, addr, size): self.calls.append(('pcie', addr, size)); return bytes(self.data[addr:addr+size])
  original = (codegen.linearize, linearizer_module.linearize, recursive_property.__get__, USBMMIOInterface.__getitem__)
  cache = dict(codegen.to_program_cache)
  fake = FakeUSB()
  raw = USBMMIOInterface(fake, 0, 64, 'B')
  assert isinstance(raw.view(fmt='Q')[:], bytes), 'Pinned regression no longer reproduced'
  cases = 0
  try:
    with candidate.scoped_backports(linearizer=True, recursive_properties=True, usb_typed_reads=True):
      assert codegen.linearize is linearizer_module.linearize is candidate.upstream_linearize
      for pcimem in (True, False):
        for fmt in ('B', 'H', 'I', 'Q', 'f', 'd'):
          view = USBMMIOInterface(fake, 0, 64, fmt, pcimem=pcimem)
          values = view[:]
          expected = bytes(fake.data) if fmt == 'B' else memoryview(fake.data).cast(fmt).tolist()
          assert values == expected, (fmt, values, expected)
          assert view[0:32//struct.calcsize(fmt)] == expected[:32//struct.calcsize(fmt)]
          assert view.view(offset=8, size=16, fmt='Q')[:] == list(struct.unpack('=QQ', fake.data[8:24]))
          if struct.calcsize(fmt) >= 4 or not pcimem:
            assert view[1] == int.from_bytes(fake.data[struct.calcsize(fmt):2*struct.calcsize(fmt)], 'little')
          cases += 1
      codegen.to_program_cache[('agent-upstream-restore-test',)] = None
      raise RuntimeError('agent-upstream-deliberate-rollback')
  except RuntimeError as err:
    assert str(err) == 'agent-upstream-deliberate-rollback', err
  assert original == (codegen.linearize, linearizer_module.linearize, recursive_property.__get__, USBMMIOInterface.__getitem__)
  assert codegen.to_program_cache == cache
  return {'mock_cases': cases, 'pinned_qword_slice_bug_reproduced': True, 'exception_restoration': True, 'hardware_access': False}


def resources(binary):
  from tinygrad.runtime.support.elf import elf_loader
  _, sections, _ = elf_loader(binary)
  note = next(x.content for x in sections if x.name == '.note')
  result = {}
  for key in ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count',
              'private_segment_fixed_size', 'group_segment_fixed_size', 'wavefront_size', 'workgroup_processor_mode'):
    field = ('.'+key).encode()
    pos = note.index(field)+len(field)
    tag = note[pos]
    if tag < 128: result[key] = tag
    elif tag in (204, 205, 206, 207):
      size = 1 << (tag-204)
      result[key] = int.from_bytes(note[pos+1:pos+1+size], 'big')
    else: raise ValueError((key, tag))
  return result


def mode_scope(candidate, mode):
  return nullcontext() if mode in ('original', 'upstream') else candidate.scoped_backports(
    linearizer=mode == 'linearizer', recursive_properties=mode == 'recursive-properties')


def compiler_worker(args, candidate):
  import tinygrad
  from tinygrad import Device, dtypes
  from tinygrad.helpers import Context, Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.uop.ops import Ops, UOp
  from tinygrad.codegen import to_program
  sys.path.insert(0, str(HERE.parent))
  import amd_gemm
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  class OfflineDevice:
    DEFAULT = 'offline'
    def __getitem__(self, key): return SimpleNamespace(renderer=renderer)
  # Clone the builder's globals instead of modifying its owning module.
  builder = FunctionType(amd_gemm._gemm.__code__, dict(amd_gemm._gemm.__globals__, Device=OfflineDevice()),
                         amd_gemm._gemm.__name__, amd_gemm._gemm.__defaults__, amd_gemm._gemm.__closure__)
  builder.__kwdefaults__ = amd_gemm._gemm.__kwdefaults__
  rows, cols, width = args.shape
  with Context(ALLOW_DEVICE_USAGE=0), mode_scope(candidate, args.mode):
    start = time.perf_counter_ns()
    out = UOp.placeholder((rows, cols), dtypes.bfloat16, 0)
    x = UOp.placeholder((((rows+15)//16)*16, width), dtypes.fp8e4m3, 1)
    w = UOp.placeholder((cols, width), dtypes.fp8e4m3, 2)
    xs, ws = (UOp.placeholder((1,), dtypes.float32, i) for i in (3, 4))
    kernel = builder(out, x, w, xs, ws)
    built = time.perf_counter_ns()
    program = to_program(kernel, renderer)
    compiled = time.perf_counter_ns()
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
  stem = f'agent-upstream-{rows}-{cols}-{width}-{args.mode}'
  (args.artifacts/(stem+'.ll')).write_text(source)
  (args.artifacts/(stem+'.hsaco')).write_bytes(binary)
  assert not Device._opened_devices, Device._opened_devices
  return {'mode': args.mode, 'shape_mnk': [rows, cols, width], 'cpu_build_ms': (built-start)/1e6,
          'cpu_lower_and_llvm_ms': (compiled-built)/1e6, 'source_sha256': digest(source.encode()),
          'binary_sha256': digest(binary), 'binary_bytes': len(binary), 'resources': resources(binary),
          'optimized_ast': kernel.arg.optimized, 'opened_devices': list(Device._opened_devices),
          'compiler': 'AMDLLVMRenderer / LLVMCompiler gfx1200 -cumode', 'builder_sha256': digest(Path(amd_gemm.__file__).read_bytes()),
          'tinygrad_source': str(Path(tinygrad.__file__).resolve())}


def execution_worker(args, candidate):
  import numpy as np
  from tinygrad import Device, Tensor, dtypes
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import AxisType, KernelInfo, UOp
  assert Device.DEFAULT == 'CPU', Device.DEFAULT
  with Context(ALLOW_DEVICE_USAGE=1), mode_scope(candidate, args.mode):
    r = UOp.range(0, 0, AxisType.LOOP)
    empty_result = Tensor.ones(1, device='CPU').contiguous().custom_kernel(Tensor.empty(1, device='CPU'),
      fxn=lambda p,q: q.after(p[0].store(8.), r)[0].store(64.).end(r).sink(arg=KernelInfo(opts_to_apply=())))[0].tolist()
    rng = np.random.default_rng(384813)
    cases = []
    for m, n, k in ((1, 19, 23), (17, 19, 23), (16, 32, 64)):
      av = rng.integers(-3, 4, size=(m, k)).astype(np.float32)
      bv = rng.integers(-3, 4, size=(k, n)).astype(np.float32)
      a, b = Tensor(av, device='CPU'), Tensor(bv, device='CPU')
      for iteration in range(3):
        if iteration == 2:
          av = -av+1
          a.assign(Tensor(av, device='CPU')).realize()
        actual = (a@b).cast(dtypes.float32).numpy()
        np.testing.assert_array_equal(actual, av.astype(np.float64)@bv.astype(np.float64))
        cases.append({'shape_mnk': [m, n, k], 'iteration': iteration, 'exact_integer_control': True})
  assert all(x.split(':')[0] in ('CPU', 'NPY') for x in Device._opened_devices), Device._opened_devices
  return {'mode': args.mode, 'empty_loop_result': empty_result, 'empty_loop_expected': [8.0],
          'empty_loop_pass': empty_result == [8.0], 'integer_controls': cases, 'opened_devices': list(Device._opened_devices)}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, default=HERE/'agent-upstream-cpu-report.json')
  parser.add_argument('--artifacts', type=Path, default=HERE/'agent-upstream-artifacts')
  parser.add_argument('--worker', choices=('compiler', 'execution'))
  parser.add_argument('--mode', choices=(*MODES, 'upstream'), default='original')
  parser.add_argument('--upstream-only', action='store_true', help='Extend an existing report using isolated upstream source')
  parser.add_argument('--shape', type=int, nargs=3, default=SHAPES[0])
  args = parser.parse_args()
  candidate = load_candidate()
  if args.worker:
    record = compiler_worker(args, candidate) if args.worker == 'compiler' else execution_worker(args, candidate)
    print(json.dumps(record))
    return
  args.artifacts.mkdir(exist_ok=True)
  assert all(p.name.startswith('agent-upstream-') for p in args.artifacts.iterdir()), args.artifacts
  report = json.loads(args.output.read_text()) if args.upstream_only else {
    'cpu_only': True, 'gpu_timing_or_quality_validation': False, 'pin': candidate.PIN, 'upstream': candidate.UPSTREAM,
    'commits': candidate.COMMITS, 'provenance': check_provenance(candidate), 'usb': check_usb_and_restore(candidate),
    'compiler': [], 'execution': []}
  if args.upstream_only: assert not any(r['mode'] == 'upstream' for r in report['compiler']), 'Already recorded upstream'
  env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(PINNED), DEV='CPU', ALLOW_DEVICE_USAGE='0',
             CACHELEVEL='0', CCACHE='0', BEAM='0', PROFILE='0', VIZ='0')
  for worker, cases in (('execution', [None]), ('compiler', SHAPES)):
    stop_worker = False
    if args.upstream_only and any(r['mode'] == 'upstream' for r in report[worker]): continue
    for shape in cases:
      for mode in (('upstream',) if args.upstream_only else MODES):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', worker, '--mode', mode, '--artifacts', str(args.artifacts)]
        if shape is not None: command += ['--shape', *map(str, shape)]
        worker_env = dict(env, ALLOW_DEVICE_USAGE='1' if worker == 'execution' else '0',
                          PYTHONPATH=str(SOURCE if mode == 'upstream' else PINNED))
        result = subprocess.run(command, env=worker_env, capture_output=True, text=True, timeout=90)
        if result.returncode:
          if mode != 'upstream': raise RuntimeError(f'{command}: exit {result.returncode}\n{result.stdout}\n{result.stderr}')
          error_path = args.artifacts/'agent-upstream-upstream-compile-error.txt'
          error_path.write_text(result.stdout+result.stderr)
          report[worker].append({'mode': mode, 'shape_mnk': list(shape), 'status': 'compile_error',
                                 'returncode': result.returncode, 'stderr_sha256': digest(result.stderr.encode()),
                                 'error': result.stderr.strip().splitlines()[-1], 'error_path': str(error_path),
                                 'command': command, 'remaining_upstream_shapes_skipped': True})
          args.output.write_text(json.dumps(report, indent=2)+'\n')
          stop_worker = True
          print(json.dumps(report[worker][-1]), flush=True)
          break
        record = json.loads(result.stdout)
        report[worker].append(record)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record), flush=True)
      if stop_worker: break
  for shape in SHAPES:
    records = [r for r in report['compiler'] if r['shape_mnk'] == list(shape) and 'binary_sha256' in r]
    for record in records:
      record['binary_matches_original'] = record['binary_sha256'] == records[0]['binary_sha256']
      record['source_matches_original'] = record['source_sha256'] == records[0]['source_sha256']
  report['all_controls_pass'] = all(case['exact_integer_control'] for r in report['execution'] for case in r['integer_controls'])
  report['candidate_empty_loop_regression_pass'] = all(r['empty_loop_pass'] for r in report['execution'] if r['mode'] == 'linearizer')
  args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()

"""Compile native bindings and prove complete 40-operation queues on the CPU."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', BEAM='0', DEBUG='0')
import importlib.util
spec = importlib.util.spec_from_file_location('agent_packed_binding_check', Path(__file__).with_name('agent-packed-epilogue-check.py'))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def main():
  from tinygrad import Device, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
  from tinygrad.runtime.support.hcq2 import _build_queues
  from tinygrad.uop.ops import UOp
  from amd_gemm import _gemm
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = LLVMCompiler('AMDGPU', 'gfx1200', '-cumode')
  records = []
  shapes = ((17, 48, 32), (65, 80, 64), (2112, 9216, 2048), (2027, 9216, 2048))
  for rows, columns, width in shapes:
    specs = [(((rows+15)//16*16, columns), dtypes.fp8e4m3), (((rows+15)//16*16, width), dtypes.fp8e4m3),
             ((columns, width), dtypes.fp8e4m3), ((1,), dtypes.float32), ((1,), dtypes.float32), ((1,), dtypes.float32),
             ((rows, columns), dtypes.bfloat16)]
    arguments = [UOp.placeholder(shape, dtype, i) for i, (shape, dtype) in enumerate(specs)]
    out, a, w, xs, ws, scale, scratch = arguments
    params = [UOp.param(slot, arg.dtype, arg.numel(), device='AMD') for slot, arg in zip((5, 0, 1, 2, 3, 4, 6), arguments)]
    matmul = to_program(H.E.bind_renderer(_gemm, renderer)(scratch, a, w, xs, ws), renderer)
    post = to_program(H.post_program(out, scratch, scale), renderer)
    fns = {'baseline': [matmul, post], 'old-fused': [to_program(H.E.bind_renderer(H.F.program, renderer)(*arguments[:6], rows=rows), renderer)]}
    for variant in H.P.TILES:
      fns[variant] = [to_program(H.P.program(*arguments[:6], rows=rows, variant=variant), renderer)]
    for label, programs in fns.items():
      calls = [program.call(*params) for _ in range(H.CALLS) for program in programs]
      batch = [(call, 'AMD', 'COMPUTE:0') for call in calls]
      ctx = SimpleNamespace(batch=batch, calls={('AMD', 'COMPUTE:0'): list(range(len(calls)))},
        peers={('AMD', 'COMPUTE:0'): set()}, queues={'AMD': ['COMPUTE:0']}, stamps=lambda *_: ())
      proof = H.queue_proof(batch, _build_queues(ctx), H.CALLS)
      records.append({'shape_mnk': [rows, columns, width], 'variant': label, 'offline_queue_proof': proof,
                      'global_sizes': [p.arg.global_size for p in programs], 'local_sizes': [p.arg.local_size for p in programs]})
  values = H.rounding_values()
  rows, columns = values.shape
  specs = [(((rows+15)//16*16, columns), dtypes.fp8e4m3), (values.shape, dtypes.float32), ((16, 16), dtypes.fp8e4m3),
           ((1,), dtypes.float32), ((1,), dtypes.float32), ((1,), dtypes.float32)]
  arguments = [UOp.placeholder(shape, dtype, i) for i, (shape, dtype) in enumerate(specs)]
  for variant in H.P.TILES:
    program = to_program(H.P.program(*arguments, rows=rows, variant=variant, epilogue_only=True), renderer)
    assert set(program.arg.globals) == set(range(6)) and tuple(program.arg.outs) == (0,)
    records.append({'shape': values.shape, 'variant': variant, 'independent_epilogue_probe_compiled': True,
                    'global_size': program.arg.global_size, 'local_size': program.arg.local_size})
  assert not any(device.startswith('AMD') for device in Device._opened_devices)
  H.E.write_json(args.output, {'complete': True, 'gpu_executed': False, 'records': records})
  print(f'Compiled {len(records)} binding/queue checks without GPU access')


if __name__ == '__main__': main()

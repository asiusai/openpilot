"""Reconstruct the production vision GELU/quantizer boundary without GPU execution."""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
from functools import partial
from types import FunctionType, SimpleNamespace

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
CACHE = Path('/tmp/opencode/alpamayo-asius-20261114/cache.db')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def production_sources():
  plan = json.loads((WORK/'current-profile-dynamic-2048/report.current-profile-plan.json').read_text())
  sources = {}
  for path in (HERE.parent/'model.py', HERE/'batched-vision.py', HERE/'quantize-reduction.py', HERE/'tiled-gemm.py', HERE.parent/'amd_gemm.py'):
    sources[str(path)] = sha(path.read_bytes())
    assert sources[str(path)] == plan['source_sha256'][str(path)], 'Production source changed since profile'
  return sources


def load(name):
  path = HERE/(name+'.py')
  spec = importlib.util.spec_from_file_location('agent_vision_gelu_'+name.replace('-', '_'), path)
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  return module


@contextmanager
def schedules():
  from tinygrad import Tensor
  old = Tensor.realize
  calls = []
  def realize(self, *others, **kwargs):
    calls.extend(self.schedule_linear(*others).src)
    return self
  Tensor.realize = realize
  try: yield calls
  finally: Tensor.realize = old


def fc1_output(renderer, images=2, rows=720, width=4304):
  from tinygrad import Tensor, dtypes
  original = load('tiled-gemm').kernel
  class Devices:
    DEFAULT = 'offline'
    def __getitem__(self, _): return SimpleNamespace(renderer=renderer)
  kernel = FunctionType(original.__code__, dict(original.__globals__, Device=Devices()), original.__name__, original.__defaults__)
  kernel.__kwdefaults__ = original.__kwdefaults__
  shapes = [((images*rows, width), dtypes.bfloat16), ((images*rows, 1152), dtypes.fp8e4m3),
    ((width, 1152), dtypes.fp8e4m3), ((images*rows,), dtypes.float32), ((1,), dtypes.float32), ((width,), dtypes.bfloat16)]
  buffers = [Tensor.empty(*shape, device='CPU', dtype=dt) for shape, dt in shapes]
  return Tensor.custom_kernel(*buffers, fxn=partial(kernel, tile_m=4, tile_n=4, row_scale=True))[0].reshape(images, rows, width)


def original_chain(images=2, rows=720, width=4304, renderer=None):
  from tinygrad import Tensor, dtypes
  from amd_gemm import pack
  x = (Tensor.empty(images*rows, width, device='CPU', dtype=dtypes.bfloat16).reshape(images, rows, width)
       if renderer is None else fc1_output(renderer, images, rows, width))
  # Invoke the original model method to obtain its actual post-fc1 expression.
  core = load('../model')
  boundary = []
  class Reached(Exception): pass
  def linear(value, name, *args, **kwargs):
    if name.endswith('.mlp.fc1'): return x
    if name.endswith('.mlp.fc2'): boundary.append(value); raise Reached()
    return Tensor.empty(images, rows, 1152, device='CPU', dtype=dtypes.bfloat16)
  model = SimpleNamespace(dtype=dtypes.bfloat16, weight_scales={}, norm=lambda value, *args: value,
    linear=linear, attention=lambda q, k, v: q)
  try: core.EdgeModel.vision_layer(model, Tensor.empty(images, rows, 1152, device='CPU', dtype=dtypes.bfloat16), 2)
  except Reached: pass
  assert len(boundary) == 1
  reduction = load('quantize-reduction')
  with schedules() as calls:
    quantized, scale = reduction.grouped_quantize(boundary[0], images)
    packed = pack(quantized).realize()
    scale.realize()
  return calls, (x, boundary[0], packed, scale)


def compile_calls(calls, renderer, directory, *, prefix='original'):
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import Ops
  compiler = load('agent-gemm-compile')
  records = []
  with sqlite3.connect('file:'+str(CACHE)+'?mode=ro', uri=True, timeout=2) as connection:
    for index, call in enumerate(calls):
      ast = call.body
      program = to_program(ast, renderer)
      source, binary = program.src[2].arg, program.src[3].arg
      name = program.src[0].arg.function_name
      record = dict(index=index, name=name, ast_key=ast.key.hex(), ast_ops=dict(Counter(u.op.name for u in ast.toposort())),
        source_sha256=sha(source.encode()), binary_sha256=sha(binary), global_size=program.arg.global_size,
        local_size=program.arg.local_size, resources=compiler.metadata(binary),
        source_counts={op:source.count(op) for op in ('load ', 'store ', 'fmul ', 'fadd ', 'fdiv ', 'exp2', '32767')})
      is_llvm = 'define amdgpu_kernel' in source
      if is_llvm:
        matches = connection.execute('SELECT COUNT(*) FROM "compile_llvm_gfx1200_-cumode_opt_24" WHERE key=?', (source,)).fetchone()[0]
        record['exact_source_found_in_existing_wgp_cache'] = matches == 1
      params = [u for u in ast.toposort() if u.op is Ops.PARAM]
      record['parameters'] = [dict(slot=u.arg.slot, dtype=str(u.dtype), elements=u.numel()) for u in sorted(params, key=lambda x:x.arg.slot)]
      stem = directory/f'agent-vision-gelu-{prefix}-{index}'
      stem.with_suffix('.cpp' if is_llvm else '.ll').unlink(missing_ok=True)
      stem.with_suffix('.ll' if is_llvm else '.cpp').write_text(source)
      stem.with_suffix('.s').write_text(compiler.disassemble(binary))
      records.append(record)
  return records


def cache_candidates():
  result = []
  with sqlite3.connect('file:'+str(CACHE)+'?mode=ro', uri=True, timeout=2) as connection:
    for mode in ('+', '-'):
      table = f'compile_llvm_gfx1200_{mode}cumode_opt_24'
      for (source,) in connection.execute('SELECT key FROM "'+table+'" WHERE instr(key,?)>0', ('@E_64560_32_3(',)):
        result.append(dict(table=table, source_sha256=sha(source.encode()),
          signature=next(line for line in source.splitlines() if 'define amdgpu_kernel' in line),
          exp2_count=source.count('exp2'), fdiv_count=source.count('fdiv '),
          per_image_scale_index='sdiv i32 %v0, 32280' in source))
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--artifacts', type=Path, default=HERE/'agent-vision-gelu-artifacts')
  args = parser.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', TC_OPT='2', BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0')
  from tinygrad import Device
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  renderer = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  renderer.compiler = load('wgp_compiler').WGPCompiler('gfx1200')
  args.artifacts.mkdir(parents=True, exist_ok=True)
  sources = production_sources()
  calls, _ = original_chain(renderer=renderer)
  report = dict(gpu_executed=False, runtime_binary_identity_not_captured=True, production_sources=sources,
    source_sha256=sha(Path(__file__).read_bytes()), cache_name_collisions=cache_candidates(),
    boundary=dict(images=2, rows_per_image=720, channels=4304, input='original rounded BF16 fc1 output'),
    original=compile_calls(calls, renderer, args.artifacts))
  assert not Device._opened_devices
  report['opened_devices'] = sorted(Device._opened_devices)
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(dict(gpu_executed=False, kernels=[dict(name=r['name'], source_sha256=r['source_sha256'],
    source_counts=r['source_counts']) for r in report['original']]), indent=2))


if __name__ == '__main__': main()

"""Scoped running-C repair for the demonstrated prefill-down 3x3 defect.

All original kernels and proof helpers come from the parent's frozen source.
No tinygrad globals, renderer, model dispatch, or other experiment files change.
"""
from functools import lru_cache, partial
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import FunctionType

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
FROZEN = WORK/'matrix-tiles-source/alpamayo/experiments'
ACTUAL = (2112, 2048, 9216)
SCOPED_CONTROLS = {(17, 48, 32), (49, 80, 64), (49, 80, 9216)}
_observations = None


def sha(raw): return hashlib.sha256(raw).hexdigest()


@lru_cache(None)
def frozen(name):
  path = FROZEN/(name+'.py')
  spec = importlib.util.spec_from_file_location('agent_matrix_accum_frozen_'+name.replace('-', '_'), path)
  result = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = result
  spec.loader.exec_module(result)
  return result


@lru_cache(None)
def helpers(): return frozen('agent-matrix-tiles-check')


def provenance():
  import amd_gemm
  import tinygrad
  root = Path(tinygrad.__file__).resolve().parent.parent
  prior_path = WORK/'agent-matrix-tiles-kernels/report.json'
  prior = json.loads(prior_path.read_text())
  for name, expected in prior['source_sha256'].items():
    assert sha(Path(name).read_bytes()) == expected, ('Frozen source changed', name)
  assert sha(Path(amd_gemm.__file__).read_bytes()) == prior['source_sha256'][str(FROZEN.parent/'amd_gemm.py')]
  files = [Path(__file__), HERE/'agent-matrix-accum-check.py', root/'tinygrad/codegen/__init__.py',
    root/'tinygrad/renderer/llvmir.py', root/'tinygrad/runtime/support/compiler_llvm.py',
    FROZEN/'agent-matrix-tiles-assembly.py']
  revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
  assert revision == prior['tinygrad_revision']
  return dict(sources={str(p): sha(p.read_bytes()) for p in files}, frozen_sources=prior['source_sha256'],
    imported_amd_gemm=str(Path(amd_gemm.__file__).resolve()), tinygrad_revision=revision,
    tinygrad_diff_sha256=sha(subprocess.check_output(['git', '-C', str(root), 'diff', '--binary', 'HEAD', '--', 'tinygrad'])),
    prior_gpu_report=str(prior_path), prior_gpu_report_sha256=sha(prior_path.read_bytes()),
    scoped_actual_shape_mnk=ACTUAL, scoped_tile=(3, 3), compiler='gfx1200 LLVM WGP -cumode, default O2')


def _absorb(wmma, reshape, add):
  from tinygrad import dtypes
  if wmma.dtype != dtypes.float32 or wmma.arg[1] != dtypes.fp8e4m3: return None
  # Equal shapes avoid inventing any new broadcasting behavior.
  assert add.shape == reshape.shape and math.prod(wmma.shape) == math.prod(reshape.shape)
  _observations.append(dict(wmma_shape=wmma.shape, reshape_shape=reshape.shape, accumulator_shape=add.shape))
  return (wmma+add.reshape(wmma.shape)).reshape(reshape.shape)


def private_lowering():
  import tinygrad.codegen as codegen
  from tinygrad.uop.ops import Ops, PatternMatcher, UPat
  observations = []
  absorb = FunctionType(_absorb.__code__, dict(_absorb.__globals__, _observations=observations), _absorb.__name__)
  rule = PatternMatcher([
    (UPat(Ops.RESHAPE, src=(UPat(Ops.WMMA, name='wmma'), UPat()), name='reshape')+UPat.var('add'), absorb),
  ])
  original = codegen.full_rewrite_to_sink
  result = FunctionType(original.__code__, dict(original.__globals__,
    pm_reduce_local=rule+codegen.pm_reduce_local,
    pm_expand_broadcast=rule+codegen.pm_expand_broadcast), original.__name__, original.__defaults__)
  result.__kwdefaults__ = original.__kwdefaults__
  return result, observations


def repaired_kernel(*args, raw_kernel, renderer):
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import Ops, ProgramInfo, UOp
  out, _, weight = args[:3]
  shape = (*out.shape, weight.shape[1])
  assert shape == ACTUAL or shape in SCOPED_CONTROLS, ('Unjustified repair scope', shape)
  assert (raw_kernel.keywords['tile_m'], raw_kernel.keywords['tile_n']) == (3, 3)
  ast = raw_kernel(*args)
  lower, _ = private_lowering()
  sink = lower(ast, renderer, optimize=not ast.arg.optimized)
  return to_program(UOp(Ops.PROGRAM, src=(sink,), arg=ProgramInfo.from_sink(sink, renderer.target)), renderer)


def functions(renderer, row_scale=False, *, repair=True, audit_tiles=((3, 3),)):
  result = helpers().functions(renderer, row_scale, audit_tiles)
  if repair:
    assert (3, 3) in audit_tiles
    fn = partial(repaired_kernel, raw_kernel=result['3x3'], renderer=renderer)
    fn.raw_kernel = result['3x3']
    result['3x3-accum'] = fn
  return result


def compile_program(shape, bias, row_scale, fn, renderer):
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import UOp, Ops, AxisType
  m, n, k = shape
  specs = [((m, n), dtypes.bfloat16), ((math.ceil(m/16)*16, k), dtypes.fp8e4m3),
    ((math.ceil(n/16)*16, k), dtypes.fp8e4m3), ((m if row_scale else 1,), dtypes.float32), ((1,), dtypes.float32)]
  if bias: specs.append(((n,), dtypes.bfloat16))
  args = [UOp.placeholder(s, dt, i) for i, (s, dt) in enumerate(specs)]
  ast = getattr(fn, 'raw_kernel', fn)(*args)
  reductions = [(u.axis_type.name, int(u.vmax+1)) for u in ast.toposort()
    if u.op is Ops.RANGE and u.axis_type in (AxisType.LOOP, AxisType.WEAK)]
  assert reductions == [('WEAK', k//16)], reductions
  program = fn(*args) if hasattr(fn, 'raw_kernel') else to_program(ast, renderer)
  return args, ast, program


def record(shape, bias, row_scale, fn, renderer, tile, artifact=None):
  from tinygrad.uop.ops import Ops, UOp
  args, _, program = compile_program(shape, bias, row_scale, fn, renderer)
  binary, source = program.src[3].arg, program.src[2].arg
  kernel = source[source.index('define amdgpu_kernel'):]
  loop = kernel[:kernel.index('loop_exit_2:')]
  wmma = [line.strip() for line in loop.splitlines() if 'call <8 x float> @llvm.amdgcn.wmma.' in line]
  zero = set(re.findall(r'(%v\d+) = .*<8 x float> .*float 0x0000000000000000', loop))
  c_args = [re.search(r'<8 x float> (%v\d+)\)', line).group(1) for line in wmma]
  adds = [line.strip() for line in loop.splitlines() if ' = fadd ' in line]
  assert len(wmma) == math.prod(tile)
  notes = subprocess.check_output(['/usr/lib/llvm-18/bin/llvm-readelf', '--notes', '-'], input=binary).decode()
  names = ('vgpr_count', 'sgpr_count', 'vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size',
    'group_segment_fixed_size', 'wavefront_size', 'workgroup_processor_mode')
  resources = {key: int(value) for key, value in re.findall(r'\.('+'|'.join(names)+r'):\s*(\d+)', notes)}
  assert resources['wavefront_size'] == 32 and resources['workgroup_processor_mode'] == 1
  assert all(resources[key] == 0 for key in ('vgpr_spill_count', 'sgpr_spill_count', 'private_segment_fixed_size', 'group_segment_fixed_size'))
  params = [UOp.param(slot, arg.dtype, arg.numel(), device='AMD') for slot, arg in zip((5, 0, 1, 2, 3, 4), args)]
  proof = helpers().E.prove_cpu_queue([program.call(*params)]*40, device='AMD')
  row = dict(status='compiled', tile=tile, global_size=program.arg.global_size, local_size=program.arg.local_size,
    resources=resources, binary_sha256=sha(binary), llvm_sha256=sha(source.encode()),
    k16_steps=shape[2]//16, llvm_wmma_per_step=len(wmma), llvm_zero_c_wmma=sum(arg in zero for arg in c_args),
    llvm_loop_fadd=len(adds), offline_serial_queue_proof=proof,
    epilogue_fmul_count=sum(' = fmul ' in line for line in kernel[kernel.index('loop_exit_2:'):].splitlines()),
    wmma_operand_pairs=sorted((u.src[0].key.hex(), u.src[1].key.hex()) for u in program.src[1].src if u.op is Ops.WMMA))
  if hasattr(fn, 'raw_kernel'):
    assert row['llvm_zero_c_wmma'] == row['llvm_loop_fadd'] == 0, row
  if artifact:
    artifact.with_suffix('.ll').write_text(source)
    artifact.with_suffix('.hsaco').write_bytes(binary)
    artifact.with_suffix('.notes').write_text(notes)
  return row, program

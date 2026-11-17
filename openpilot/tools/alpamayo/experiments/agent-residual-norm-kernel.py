"""Optional M2048/W2048 residual, frozen RMSNorm, and fixed packed-FP8 fusion.

Importing this module uses only the standard library. The parent owns GPU tests.
"""
import ast
from functools import lru_cache
import hashlib
from pathlib import Path

FROZEN = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/current-profile-dynamic-2048/report-sources')
NORM_SHA = '4271693fc1f3210f160eac0a4ddb69dad168612c712d4f77a755ef2eda178b7c'
ROWS, WIDTH = 2048, 2048


def sha(data): return hashlib.sha256(data).hexdigest()


def frozen_norm_source():
  path = FROZEN/'fused-norm.py'
  data = path.read_bytes()
  if sha(data) != NORM_SHA: raise ValueError('Frozen RMSNorm source changed')
  tree = ast.parse(data)
  function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compiled')
  function.decorator_list = []
  class SourceOnly:
    def __init__(self, arch): assert arch == 'gfx1200'
    def compile_cached(self, text): return b''
  namespace = {'HIPCompiler': SourceOnly}
  exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
  return namespace['compiled'](WIDTH, True, 1e-5)[0]


def replace_once(source, old, new):
  if source.count(old) != 1: raise ValueError('Unexpected frozen RMSNorm statement')
  return source.replace(old, new)


def source():
  text = frozen_norm_source()
  helpers = r'''
// Exact pinned tinygrad cast_float_to_bf16; the existing norm b() stays unchanged.
DEV u16 residual_b(float value) {
  unsigned bits=__builtin_bit_cast(unsigned,value);
  unsigned rounded=bits+((bits>>16)&1)+0x7fffu;
  unsigned special=bits|(((bits&0xffffu)!=0)<<16);
  unsigned mask=0u-(((-bits & 0x7f800000u)!=0));
  asm("" : "+v"(mask));
  return (u16)(((rounded&mask)|(special&~mask))>>16);
}
DEV unsigned char packed_f8(float value) {
  // Ordered selects match the original Tensor.clamp, including NaN propagation.
  value=value < -448.0f ? -448.0f : value;
  value=448.0f < value ? 448.0f : value;
  return (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32(value,value,0,false);
}
'''
  text = replace_once(text, 'extern "C" __attribute__((global,amdgpu_flat_work_group_size(256,256)))',
                      helpers+'\nextern "C" __attribute__((global,amdgpu_flat_work_group_size(256,256)))')
  text = replace_once(text, 'void alpamayo_norm(u16 *out,const u16 *x,const u16 *weight,const u16 *bias)',
    'void agent_residual_norm(u16 *out,unsigned char *packed,const u16 *a,const u16 *other,const u16 *weight,const float *scale)')
  text = replace_once(text, '  float mean=0;', '  u16 residual[WIDTH/256];\n  float mean=0;')
  # RMS is a compile-time constant. Preserve the original dead non-RMS branch text.
  text = text.replace('f(x[row*WIDTH+i])', 'f(residual[i/256])')
  text = replace_once(text,
    '  for(int i=t;i<WIDTH;i+=256) { float v=f(residual[i/256])-mean; squared+=v*v; }',
    '  for(int i=t;i<WIDTH;i+=256) {\n'
    '    u16 r=residual_b(f(a[row*WIDTH+i])+f(other[row*WIDTH+i]));\n'
    '    residual[i/256]=r; out[row*WIDTH+i]=r;\n'
    '    float v=f(r)-mean; squared+=v*v;\n  }')
  text = replace_once(text,
    '    out[row*WIDTH+i]=RMS?b(f(b(normalized))*f(weight[i])):b(normalized*f(weight[i])+f(bias[i]));',
    '    u16 weighted=b(f(b(normalized))*f(weight[i]));\n'
    '    float scaled=f(weighted)*__builtin_amdgcn_rcpf(scale[0]);\n'
    '    unsigned index=((row/16)*(WIDTH/16)+i/16)*256+(i%16/4)*64+(row%16)*4+i%4;\n'
    '    packed[index]=packed_f8(scaled);')
  # Dead branch references still require declarations. Its removal is source checked.
  begin, end = text.index('  if (!RMS) {'), text.index('  float squared=0;')
  assert text[begin:end].count('if (!RMS)') == 1
  text = text[:begin]+text[end:]
  return text


@lru_cache(None)
def compiled():
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  text = source()
  # compile(), not compile_cached(): no writes to another owner's compiler cache.
  return text, HIPCompiler('gfx1200').compile(text)


def program(out, packed, a, other, weight, scale):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  dependency = a.flatten()[0].load().float()+other.flatten()[0].load().float()+weight[0].load().float()+scale[0].load()
  sink = UOp.sink(out.flatten()[0].store(dependency.cast(dtypes.bfloat16)),
                  packed.flatten()[0].store(dependency.cast(dtypes.fp8e4m3)),
                  UOp.special(ROWS, 'gidx0'), UOp.special(256, 'lidx0'),
                  arg=KernelInfo(name='agent_residual_norm'))
  text, binary = compiled()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())),
                             UOp(Ops.SOURCE, arg=text), UOp(Ops.BINARY, arg=binary)))


def fused(a, other, weight, scale):
  from tinygrad import Tensor, dtypes
  assert a.shape == other.shape == (1, ROWS, WIDTH)
  assert weight.shape == (WIDTH,) and scale.numel() == 1
  assert a.dtype == other.dtype == weight.dtype == dtypes.bfloat16 and scale.dtype == dtypes.float32
  assert a.device == other.device == weight.device == scale.device
  out = Tensor.empty(a.shape, dtype=dtypes.bfloat16, device=a.device)
  packed = Tensor.empty(ROWS, WIDTH, dtype=dtypes.fp8e4m3, device=a.device)
  values = Tensor.custom_kernel(out, packed, a.contiguous(), other.contiguous(), weight.contiguous(), scale.reshape(1), fxn=program)
  values[0].realize(values[1])
  return values[0], values[1]

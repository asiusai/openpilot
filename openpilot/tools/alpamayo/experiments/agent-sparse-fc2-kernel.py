"""One 64x64 / four-by-four fragment candidate, with exact overflow fallback.

Arithmetic equivalence is UNVERIFIED. SWMMAC K32 must pass the separate gate.
No model hook, quantization, pruning, fitting, or device access at import.
"""
from functools import lru_cache, partial
import importlib.util
from pathlib import Path
from types import FunctionType, SimpleNamespace

EXP = Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
BASELINE = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/activation-sparsity-devicefix-source/alpamayo')


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def source(m, n, k, mode='hybrid', raw=False, initial=False):
  if m <= 0 or n <= 0 or m%64 or n%64 or k <= 0 or k%32 or mode not in ('hybrid', 'dense', 'compress'):
    raise ValueError('This single scheme requires full 64x64 output tiles and K32')
  if mode == 'compress':
    import hashlib
    if hashlib.sha256((EXP/'agent-sparse-kernel.py').read_bytes()).hexdigest()!='8aaf1879a078d3410bdc9841fe9c333ba897e318783a3a296658b11b9374e4d6':
      raise ValueError('The GPU-verified compression source changed')
    old = load(EXP/'agent-sparse-kernel.py', 'fc2_verified_compressor')
    return old.source(m,n,k,'compress').replace('void agent_sparse(', 'void agent_sparse_fc2_compress(')
  defines = dict(M=m,N=n,K=k,HYBRID=int(mode=='hybrid'),RAW=int(raw),INITIAL=int(initial))
  return ''.join(f'#define {key} {value}\n' for key,value in defines.items())+r'''
#pragma clang fp contract(off)
#define DEV __attribute__((device,always_inline)) inline
typedef int i2 __attribute__((ext_vector_type(2)));
typedef int i4 __attribute__((ext_vector_type(4)));
typedef float f8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
DEV unsigned addr(unsigned r,unsigned k) {
  return (r/16*(K/16)+k/16)*256+(k%16/4)*64+r%16*4+k%4;
}
DEV int word(const unsigned char* p,unsigned r,unsigned k) {return *(const int*)(p+addr(r,k));}
DEV int peer(int value,unsigned lane) {return __builtin_amdgcn_ds_bpermute((lane^16)*4,value);}
DEV unsigned short bf(float value) {
  unsigned b=__builtin_bit_cast(unsigned,value);
  unsigned regular=b+((b>>16)&1)+0x7fffu;
  unsigned special=b|(((b&0xffffu)!=0)<<16);
  unsigned mask=0u-(((-b&0x7f800000u)!=0));
  return (unsigned short)(((regular&mask)|(special&~mask))>>16);
}
#if RAW
typedef float output_t;
#else
typedef unsigned short output_t;
#endif
extern "C" __attribute__((global,amdgpu_flat_work_group_size(32,32)))
void agent_sparse_fc2_gemm(output_t* out,const unsigned char* x,const unsigned char* w,
 const float* xs,const float* ws,const i2* values,const unsigned* metadata,const unsigned* flags,const float* seed) {
  unsigned lane=__ockl_get_local_id(0),block=__ockl_get_group_id(0);
  unsigned mr=(block/(N/64))*64,nr=(block%(N/64))*64;
  f8 c[4][4];
  #pragma unroll
  for(int i=0;i<4;i++) {
    #pragma unroll
    for(int j=0;j<4;j++) {
      c[i][j]=(f8){0};
#if INITIAL
      #pragma unroll
      for(int q=0;q<8;q++)c[i][j][q]=seed[(mr+i*16+(lane/16)*8+q)*N+nr+j*16+lane%16];
#endif
    }
  }
  #pragma clang loop unroll(disable)
  for(unsigned kt=0;kt<K/32;kt++) {
    unsigned tile[4],ok[4],any_dense=0;
    #pragma unroll
    for(int i=0;i<4;i++) {
      tile[i]=((mr/16)+i)*(K/32)+kt;
#if HYBRID
      ok[i]=flags[tile[i]];
#else
      ok[i]=0;
#endif
      any_dense|=!ok[i];
    }
    i4 b32[4];i2 b0[4],b1[4];
    #pragma unroll
    for(int j=0;j<4;j++) {
      unsigned col=nr+j*16+lane%16,base=kt*32+(lane/16)*16;
      b32[j]=(i4){word(w,col,base),word(w,col,base+4),word(w,col,base+8),word(w,col,base+12)};
      if(any_dense) {
        // Peer lanes have the SAME output column and opposite K16 half.
        // Derive both original K16 fragments without extra global B loads.
        int p0=peer(b32[j][0],lane),p1=peer(b32[j][1],lane);
        int p2=peer(b32[j][2],lane),p3=peer(b32[j][3],lane);
        b0[j]=lane<16?(i2){b32[j][0],b32[j][1]}:(i2){p2,p3};
        b1[j]=lane<16?(i2){p0,p1}:(i2){b32[j][2],b32[j][3]};
      }
    }
    #pragma unroll
    for(int i=0;i<4;i++) {
      if(ok[i]) {
        i2 a=values[tile[i]*32+lane];int meta=(int)metadata[tile[i]*32+lane];
        #pragma unroll
        for(int j=0;j<4;j++)c[i][j]=__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a,b32[j],c[i][j],meta);
      } else {
        unsigned row=mr+i*16+lane%16,base=kt*32+(lane/16)*8;
        i2 a0=(i2){word(x,row,base),word(x,row,base+4)};
        i2 a1=(i2){word(x,row,base+16),word(x,row,base+20)};
        #pragma unroll
        for(int j=0;j<4;j++) {
          c[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0,b0[j],c[i][j]);
          c[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1,b1[j],c[i][j]);
        }
      }
    }
  }
  float scale=xs[0]*ws[0];
  #pragma unroll
  for(int i=0;i<4;i++) {
    #pragma unroll
    for(int j=0;j<4;j++) {
      #pragma unroll
      for(int q=0;q<8;q++) {
        unsigned row=mr+i*16+(lane/16)*8+q,col=nr+j*16+lane%16;
#if RAW
        out[row*N+col]=c[i][j][q];
#else
        out[row*N+col]=bf(c[i][j][q]*scale);
#endif
      }
    }
  }
}
'''


@lru_cache(None)
def compiled(m,n,k,mode='hybrid',raw=False,initial=False):
  import hashlib,json
  root=Path(__file__).resolve().parent
  text=source(m,n,k,mode,raw,initial)
  report_path=root/'agent-sparse-fc2-cpu.json'
  stem=f'{m}-{n}-{k}-{mode}-raw{int(raw)}-C{int(initial)}'
  artifacts=root/'agent-sparse-fc2-cpu-compiled'
  if report_path.exists():
    report=json.loads(report_path.read_text())
    record=next((r for r in report.get('compiler',[])if r['name']==stem),None)
    qualified=report.get('cpu_only') and report.get('complete') and report.get('all_pass')
    qualified=qualified and report.get('source_hashes',{}).get(Path(__file__).name)==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if qualified and record is not None:
      binary=(artifacts/(stem+'.hsaco')).read_bytes()
      if (artifacts/(stem+'.cpp')).read_text()!=text or hashlib.sha256(binary).hexdigest()!=record['binary_sha256']:
        raise ValueError('Frozen precompiled sparse fc2 binary/source mismatch')
      return text,binary
  from tinygrad.device import Compiler
  from tinygrad.runtime.support import compiler_amd
  original=compiler_amd.compile_hip
  def options(action,value): return compiler_amd.set_options(action,value.replace(b'-mcumode',b'-mno-cumode'))
  function=FunctionType(original.__code__,dict(original.__globals__,set_options=options),argdefs=original.__defaults__)
  class WGP(Compiler):
    def compile(self,text):return function(text,'gfx1200')
  return text,WGP('agent_sparse_fc2_wgp_gfx1200_v1').compile_cached(text)


def program(*params,m,n,k,mode,raw=False,initial=False):
  from tinygrad import dtypes
  from tinygrad.helpers import Target
  from tinygrad.uop.ops import UOp,Ops,KernelInfo,ProgramInfo
  outputs=(0,1,2) if mode=='compress' else (0,)
  reads=tuple(i for i in range(len(params)) if i not in outputs)
  blocks=m//16*(k//32) if mode=='compress' else m//64*(n//64)
  buffers=tuple(p.buf_uop for p in params)
  dependencies=[params[i].flatten().index(0).load().cast(dtypes.uint32) for i in reads]
  dummy=sum(dependencies[1:],dependencies[0])
  name='agent_sparse_fc2_compress' if mode=='compress' else 'agent_sparse_fc2_gemm'
  sink=UOp.sink(*[params[i].flatten().index(0).store(dummy.cast(params[i].dtype)) for i in outputs],
    UOp.special(blocks,'gidx0'),UOp.special(32,'lidx0'),arg=KernelInfo(name=name))
  info=ProgramInfo(global_size=(blocks,1,1),local_size=(32,1,1),params=buffers,
    globals=tuple(range(len(params))),outs=outputs,ins=reads,target=Target('AMD','LLVM','gfx1200'))
  text,binary=compiled(m,n,k,mode,raw,initial)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)),arg=info)


def allocate(m,n,k,device):
  from tinygrad import Tensor,dtypes
  t=m//16*(k//32)
  return (Tensor.empty(t,32,2,device=device,dtype=dtypes.uint32).realize(),
    Tensor.empty(t,32,device=device,dtype=dtypes.uint32).realize(),
    Tensor.empty(t,device=device,dtype=dtypes.uint32).realize())


def compress(x,outputs):
  from tinygrad import Tensor
  m,k=x.shape
  return tuple(Tensor.custom_kernel(*outputs,x,fxn=partial(program,m=m,n=64,k=k,mode='compress'))[:3])


def gemm(x,w,xs,ws,compressed,out,seed,mode='hybrid',initial=False):
  from tinygrad import Tensor,dtypes
  m,k=x.shape;n=w.shape[0]
  if x.dtype!=dtypes.uint8 or w.dtype!=dtypes.uint8:raise ValueError('Require original packed byte storage')
  return Tensor.custom_kernel(out,x,w,xs,ws,*compressed,seed,
    fxn=partial(program,m=m,n=n,k=k,mode=mode,raw=out.dtype==dtypes.float32,initial=initial))[0]


def original_program(*params,renderer=None):
  from tinygrad import Device
  module=load(BASELINE/'amd_gemm.py','sparse_fc2_original_gemm')
  selected=Device[Device.DEFAULT].renderer if renderer is None else renderer
  class Devices:
    DEFAULT='offline-original'
    def __getitem__(self,key):return SimpleNamespace(renderer=selected)
  fn=FunctionType(module._gemm.__code__,dict(module._gemm.__globals__,Device=Devices()),argdefs=module._gemm.__defaults__)
  return fn(*params,packed_input=True)


def original(x,w,xs,ws,out):
  from tinygrad import Tensor,dtypes
  return Tensor.custom_kernel(out,x.bitcast(dtypes.fp8e4m3),w.bitcast(dtypes.fp8e4m3),xs,ws,fxn=original_program)[0]

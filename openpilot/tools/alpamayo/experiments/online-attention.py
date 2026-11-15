"""Experimental RDNA4 BF16 online softmax, without a full score matrix."""
from functools import lru_cache,partial
import importlib.util
import os
from pathlib import Path
from tinygrad import Tensor,dtypes
from tinygrad.uop.ops import UOp,Ops,KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler


@lru_cache(None)
def compiled(heads,kv_heads,rows,keys,causal,waves=4,width=128,block_keys=16,stage_kv=False,masked=False,wave_sync=False,wgp=False,native_exp=False,register_prob=False,packed_k=False,key_padding=0,prob_padding=0,local_fences=False,dpp=False,kv_load=0,value_layout=False,probability_mma=False):
  text=r'''
#define DEV __attribute__((device)) __attribute__((always_inline)) inline
typedef unsigned short u16;
typedef short s8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
DEV float frombf(u16 x) { return __builtin_bit_cast(float, (unsigned)x << 16); }
DEV short tobf(float x) { unsigned b=__builtin_bit_cast(unsigned,x); return (b+0x7fff+((b>>16)&1))>>16; }
template<int offset> DEV float shuffle(float x) {
  return __builtin_bit_cast(float,__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int,x),offset));
}
DEV float maximum(float x) {
  x=__builtin_fmaxf(x,shuffle<0x201f>(x)); x=__builtin_fmaxf(x,shuffle<0x101f>(x));
  x=__builtin_fmaxf(x,shuffle<0x081f>(x)); x=__builtin_fmaxf(x,shuffle<0x041f>(x)); return x;
}
DEV float total(float x) {
  x+=shuffle<0x201f>(x); x+=shuffle<0x101f>(x); x+=shuffle<0x081f>(x); x+=shuffle<0x041f>(x); return x;
}
DEV f8 mma(s8 a,s8 b,f8 c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a,b,c); }
DEV int key_index(int row,int dim) {
  #if PACKEDK
  return ((row/16)*((WIDTH+15)/16)+dim/16)*256+(dim%16/4)*64+(row%16)*4+dim%4;
  #else
  return row*(((WIDTH+15)/16)*16+KEYPAD)+dim;
  #endif
}
DEV float lane_read(float x, int source) {
  return __builtin_bit_cast(float,__builtin_amdgcn_ds_bpermute(source*4,__builtin_bit_cast(int,x)));
}
DEV f8 transpose_probability(f8 x, int lane) {
  // Exchange the three register row bits with the three low lane column bits.
  // Each pair needs one exchange; the other value stays in its source lane.
  for(int bit=1;bit<8;bit*=2) for(int j=0;j<8;j++) if(!(j&bit)) {
    float a=x[j], b=x[j|bit];
    float exchanged=lane_read((lane&bit)?a:b,lane^bit);
    x[j]=(lane&bit)?exchanged:a;
    x[j|bit]=(lane&bit)?b:exchanged;
  }
  // The high row and column bits both live in the lane index.
  int source=(lane&7)|((lane&8)<<1)|((lane&16)>>1);
  for(int j=0;j<8;j++) x[j]=lane_read(x[j],source);
  return x;
}
extern "C" __attribute__((global,amdgpu_flat_work_group_size(1,THREADS)))
void alpamayo_online(u16 *out,const u16 *q,const u16 *k,const u16 *v
#if MASKED
,const float *mask
#endif
) {
  constexpr int H=HEADS, KH=KVHEADS, R=ROWS, K=KEYS, W=WIDTH, D=(W+15)/16;
  int thread=__ockl_get_local_id(0), lane=thread%32, wave=thread/32, block=__ockl_get_group_id(0);
  int group=block%((R+16*WAVES-1)/(16*WAVES));
  int tile=group*WAVES+wave, head=block/((R+16*WAVES-1)/(16*WAVES)), kh=(head/H)*KH+(head%H)/(H/KH);
  int qr=tile*16+lane%16;
  __attribute__((shared,aligned(16))) short probabilities[WAVES*16*(16+PROBPAD)];
  #if STAGEKV
  __attribute__((shared,aligned(16))) short shared_k[BLOCKKEYS*(D*16+KEYPAD)],shared_v[BLOCKKEYS*D*16];
  #endif
  short *prob=probabilities+wave*16*(16+PROBPAD);
  s8 query[D];
  for(int d=0;d<D;d++) for(int i=0;i<8;i++) {
    int dim=d*16+(lane/16)*8+i;
    query[d][i]=qr<R && dim<W?q[(head*R+qr)*W+dim]:0;
  }
  f8 accum[D]={}; f8 denom={}, maxima;
  for(int i=0;i<8;i++) maxima[i]=-__builtin_inff();
  int limit=(group+1)*WAVES*16;
  int end=CAUSAL?(K<limit?K:limit):K;
  for(int start=0;start<end;start+=BLOCKKEYS) {
    #if STAGEKV
    for(int index=thread;index<BLOCKKEYS*D*16;index+=THREADS) {
      int row=index/(D*16),dim=index%(D*16);
      shared_k[key_index(row,dim)]=start+row<K && dim<W?k[(kh*K+start+row)*W+dim]:0;
      shared_v[index]=start+row<K && dim<W?v[(kh*K+start+row)*W+dim]:0;
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
    #endif
    f8 scores[BLOCKKEYS/16]={};
    for(int sub=0;sub<BLOCKKEYS/16;sub++) for(int d=0;d<D;d++) {
      s8 key;
      for(int i=0;i<8;i++) { int dim=d*16+(lane/16)*8+i;
        int col=start+sub*16+lane%16;
        #if STAGEKV
        key[i]=shared_k[key_index(sub*16+lane%16,dim)];
        #else
        key[i]=(col<K && dim<W)?k[(kh*K+col)*W+dim]:0;
        #endif
      }
      scores[sub]=mma(query[d],key,scores[sub]);
    }
    for(int i=0;i<8;i++) {
      int row=tile*16+(lane/16)*8+i;
      float localmax=-__builtin_inff();
      for(int sub=0;sub<BLOCKKEYS/16;sub++) {
        int col=start+sub*16+lane%16;
        scores[sub][i]=(col<K && (!CAUSAL || col<=row))?scores[sub][i]*SCALE:-__builtin_inff();
        #if MASKED
        if(col<K) scores[sub][i]+=mask[col];
        #endif
        localmax=__builtin_fmaxf(localmax,scores[sub][i]);
      }
      float next=__builtin_fmaxf(maxima[i],maximum(localmax));
      float scale=next==-__builtin_inff()?1.0f:__builtin_exp2f((maxima[i]-next)*1.4426950408889634f);
      float sum=0;
      for(int sub=0;sub<BLOCKKEYS/16;sub++) {
        scores[sub][i]=next==-__builtin_inff()?0.0f:__builtin_exp2f((scores[sub][i]-next)*1.4426950408889634f);
        sum+=scores[sub][i];
      }
      for(int d=0;d<D;d++) accum[d][i]*=scale;
      denom[i]=denom[i]*scale+total(sum); maxima[i]=next;
    }
    for(int sub=0;sub<BLOCKKEYS/16;sub++) {
    #if PROBMMA
    // Reinterpreting a WMMA accumulator as an input transposes its tile.
    // Multiplication by identity changes the register layout for the next WMMA.
    s8 stored, identity={};
    for(int i=0;i<8;i++) {
      stored[i]=tobf(scores[sub][i]);
      identity[i]=lane%16==(lane/16)*8+i?tobf(1.0f):0;
    }
    f8 transposed=mma(stored,identity,(f8){});
    s8 probability;
    for(int i=0;i<8;i++) probability[i]=tobf(transposed[i]);
    #elif REGISTERPROB
    f8 transposed=transpose_probability(scores[sub],lane);
    s8 probability;
    for(int i=0;i<8;i++) probability[i]=tobf(transposed[i]);
    #else
    for(int i=0;i<8;i++) prob[((lane/16)*8+i)*(16+PROBPAD)+lane%16]=tobf(scores[sub][i]);
    #if WAVESYNC
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"wavefront"); __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"wavefront");
    #else
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
    #endif
    s8 probability;
    for(int i=0;i<8;i++) probability[i]=prob[(lane%16)*(16+PROBPAD)+(lane/16)*8+i];
    #endif
    for(int d=0;d<D;d++) {
      s8 value;
      for(int i=0;i<8;i++) { int key=start+sub*16+(lane/16)*8+i;
        #if STAGEKV
        value[i]=shared_v[(sub*16+(lane/16)*8+i)*D*16+d*16+lane%16];
        #else
        value[i]=key<K && d*16+lane%16<W?v[(kh*K+key)*W+d*16+lane%16]:0;
        #endif
      }
      accum[d]=mma(probability,value,accum[d]);
    }
    #if !REGISTERPROB && !PROBMMA
    #if WAVESYNC
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"wavefront"); __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"wavefront");
    #else
    __builtin_amdgcn_s_barrier();
    #endif
    #endif
    }
    #if (WAVESYNC || REGISTERPROB || PROBMMA) && STAGEKV
    // All waves must finish reading shared K/V before the next block replaces it.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
    #endif
  }
  for(int d=0;d<D;d++) for(int i=0;i<8;i++) {
    int row=tile*16+(lane/16)*8+i;
    if(row<R && d*16+lane%16<W) out[(head*R+row)*W+d*16+lane%16]=tobf(accum[d][i]/denom[i]);
  }
}
'''
  if kv_load:
    assert kv_load in (1,2) and stage_kv and width%8==0
    text=text.replace('typedef unsigned short u16;', '''typedef unsigned short u16;
typedef unsigned short u8 __attribute__((ext_vector_type(8)));
DEV u8 load_eight(const u16 *source,int head,int row,int dim) {
  if(row>=KEYS || dim>=WIDTH) return (u8){};
  return *(const volatile u8 *)(source+(head*KEYS+row)*WIDTH+dim);
}''')
    old='''    for(int index=thread;index<BLOCKKEYS*D*16;index+=THREADS) {
      int row=index/(D*16),dim=index%(D*16);
      shared_k[key_index(row,dim)]=start+row<K && dim<W?k[(kh*K+start+row)*W+dim]:0;
      shared_v[index]=start+row<K && dim<W?v[(kh*K+start+row)*W+dim]:0;
    }'''
    assert text.count(old)==1
    if kv_load==1:
      replacement='''    for(int index=thread*8;index<BLOCKKEYS*D*16;index+=THREADS*8) {
      int row=index/(D*16),dim=index%(D*16);
      u8 loaded_k=load_eight(k,kh,start+row,dim), loaded_v=load_eight(v,kh,start+row,dim);
      for(int i=0;i<8;i++) { shared_k[key_index(row,dim+i)]=loaded_k[i]; shared_v[index+i]=loaded_v[i]; }
    }'''
    else:
      loop='  for(int start=0;start<end;start+=BLOCKKEYS) {'
      assert text.count(loop)==1
      text=text.replace(loop,'''  constexpr int LOADS=(BLOCKKEYS*D*16+THREADS*8-1)/(THREADS*8);
  u8 pending_k[LOADS], pending_v[LOADS];
  for(int slot=0;slot<LOADS;slot++) {
    int index=(thread+slot*THREADS)*8, row=index/(D*16), dim=index%(D*16);
    pending_k[slot]=row<BLOCKKEYS?load_eight(k,kh,row,dim):(u8){};
    pending_v[slot]=row<BLOCKKEYS?load_eight(v,kh,row,dim):(u8){};
  }
'''+loop)
      replacement='''    for(int slot=0;slot<LOADS;slot++) {
      int index=(thread+slot*THREADS)*8, row=index/(D*16), dim=index%(D*16);
      if(index<BLOCKKEYS*D*16) for(int i=0;i<8;i++) {
        shared_k[key_index(row,dim+i)]=pending_k[slot][i]; shared_v[index+i]=pending_v[slot][i];
      }
    }'''
      after='''    #endif
    f8 scores[BLOCKKEYS/16]={};'''
      assert text.count(after)==1
      text=text.replace(after,'''    #endif
    // Fetch the next immutable K/V block while the current LDS tile is consumed.
    if(start+BLOCKKEYS<end) for(int slot=0;slot<LOADS;slot++) {
      int index=(thread+slot*THREADS)*8, row=index/(D*16), dim=index%(D*16);
      pending_k[slot]=row<BLOCKKEYS?load_eight(k,kh,start+BLOCKKEYS+row,dim):(u8){};
      pending_v[slot]=row<BLOCKKEYS?load_eight(v,kh,start+BLOCKKEYS+row,dim):(u8){};
    }
    f8 scores[BLOCKKEYS/16]={};''')
    text=text.replace(old,replacement)
  if value_layout:
    assert kv_load==1 and stage_kv
    old='''      u8 loaded_k=load_eight(k,kh,start+row,dim), loaded_v=load_eight(v,kh,start+row,dim);'''
    assert text.count(old)==1
    text=text.replace(old,'''      int vd=index/(BLOCKKEYS*16), within=index%(BLOCKKEYS*16);
      int vk=within/256, fragment=within%256;
      int voffset=((kh*D+vd)*((K+15)/16)+start/16+vk)*256+fragment;
      u8 loaded_k=load_eight(k,kh,start+row,dim);
      u8 loaded_v=start+vk*16<K?*(const volatile u8*)(v+voffset):(u8){};''')
    old='''      for(int i=0;i<8;i++) { int key=start+sub*16+(lane/16)*8+i;
        #if STAGEKV
        value[i]=shared_v[(sub*16+(lane/16)*8+i)*D*16+d*16+lane%16];
        #else
        value[i]=key<K && d*16+lane%16<W?v[(kh*K+key)*W+d*16+lane%16]:0;
        #endif
      }'''
    assert text.count(old)==1
    text=text.replace(old,'''      value=*(const s8*)(shared_v+(d*(BLOCKKEYS/16)+sub)*256+lane*8);''')
  text=text.replace('BLOCKKEYS',str(block_keys)).replace('STAGEKV',str(int(stage_kv))).replace('MASKED',str(int(masked))).replace('WAVESYNC',str(int(wave_sync)))
  text=text.replace('REGISTERPROB',str(int(register_prob)))
  text=text.replace('PROBMMA',str(int(probability_mma)))
  text=text.replace('PACKEDK',str(int(packed_k)))
  text=text.replace('KEYPAD',str(key_padding)).replace('PROBPAD',str(prob_padding))
  if native_exp:
    text=text.replace('__builtin_exp2f', '__builtin_amdgcn_exp2f')
  if dpp:
    # RDNA row_xmask permutes within each 16-lane row. Keep XOR 8/4/2/1 order.
    old='__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int,x),offset)'
    assert text.count(old)==1
    text=text.replace(old,'__builtin_amdgcn_mov_dpp(__builtin_bit_cast(int,x),0x160|(offset>>10),0xf,0xf,true)')
  if local_fences:
    # Inter-thread communication uses LDS only. Global Q/K/V are read-only;
    # each thread writes its own final output after the last barrier.
    # Match ROCm __work_group_barrier(__CLK_LOCAL_MEM_FENCE).
    text=text.replace('"workgroup")', '"workgroup","local")')
    text=text.replace('"wavefront")', '"wavefront","local")')
  for key,value in {'KVHEADS':kv_heads,'HEADS':heads,'ROWS':rows,'KEYS':keys,'CAUSAL':int(causal),'THREADS':waves*32,'WAVES':waves,'WIDTH':width,'SCALE':repr(width**-.5)+'f'}.items():text=text.replace(key,str(value))
  compiler = HIPCompiler('gfx1200')
  if wgp:
    spec = importlib.util.spec_from_file_location('attention_hip_compiler', Path(__file__).with_name('hip-compiler.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compiler = module.WGPCompiler('gfx1200')
  return text,compiler.compile_cached(text)


def program(out,q,k,v,*masks,shape,causal):
  batch,heads,rows,width=shape;kv_heads=k.shape[1];keys=k.shape[2]
  assert width in (72,128)
  waves=int(os.environ.get('ALPAMAYO_ONLINE_WAVES','4'))
  if masks:
    waves=int(os.environ.get('ALPAMAYO_EXPERT_WAVES','1'))
  block_keys=int(os.environ.get('ALPAMAYO_ONLINE_BLOCK_KEYS','16'))
  stage_kv=bool(int(os.environ.get('ALPAMAYO_ONLINE_STAGE_KV','0')))
  wave_sync=bool(int(os.environ.get('ALPAMAYO_ONLINE_WAVE_SYNC','0')))
  wgp=bool(int(os.environ.get('ALPAMAYO_HIP_WGP','0')))
  native_exp=bool(int(os.environ.get('ALPAMAYO_ONLINE_NATIVE_EXP','0')))
  register_prob=bool(int(os.environ.get('ALPAMAYO_ONLINE_REGISTER_PROB','0')))
  packed_k=bool(int(os.environ.get('ALPAMAYO_ONLINE_PACKED_K','0')))
  key_padding=int(os.environ.get('ALPAMAYO_ONLINE_KEY_PADDING','0'))
  prob_padding=int(os.environ.get('ALPAMAYO_ONLINE_PROB_PADDING','0'))
  local_fences=bool(int(os.environ.get('ALPAMAYO_ONLINE_LOCAL_FENCES','0')))
  dpp=bool(int(os.environ.get('ALPAMAYO_ONLINE_DPP','0')))
  kv_load=int(os.environ.get('ALPAMAYO_ONLINE_KV_LOAD','0'))
  value_layout=bool(int(os.environ.get('ALPAMAYO_ONLINE_VALUE_LAYOUT','0'))) and rows>=int(os.environ.get('ALPAMAYO_ONLINE_VALUE_MIN_ROWS','0'))
  probability_mma=bool(int(os.environ.get('ALPAMAYO_ONLINE_PROBABILITY_MMA','0')))
  assert key_padding in (0,2,4) and prob_padding in (0,2,4)
  assert not (packed_k and key_padding)
  assert waves in (1,2,4,8,16)
  assert block_keys in (16,32,64)
  sink=UOp.sink(out.flatten()[0].store(q.flatten()[0].load()+k.flatten()[0].load()+v.flatten()[0].load()+
                                     (masks[0].flatten()[0].load() if masks else 0)),
    UOp.special(batch*heads*((rows+16*waves-1)//(16*waves)),'gidx0'),UOp.special(32*waves,'lidx0'),arg=KernelInfo(name='alpamayo_online'))
  text,binary=compiled(heads,kv_heads,rows,keys,causal,waves,width,block_keys,stage_kv,bool(masks),wave_sync,wgp,native_exp,register_prob,packed_k,key_padding,prob_padding,local_fences,dpp,kv_load,value_layout,probability_mma)
  return UOp(Ops.PROGRAM,src=(sink,UOp(Ops.LINEAR,src=tuple(sink.toposort())),UOp(Ops.SOURCE,arg=text),UOp(Ops.BINARY,arg=binary)))


def pack_values(v):
  batch,heads,keys,width=v.shape
  # Preserve every BF16 bit while arranging each WMMA operand as eight adjacent values.
  return (v.bitcast(dtypes.uint16).pad(((0,0),(0,0),(0,(-keys)%16),(0,(-width)%16)))
          .reshape(batch,heads,(keys+15)//16,2,8,(width+15)//16,16)
          .permute(0,1,5,2,3,6,4).contiguous().bitcast(dtypes.bfloat16))


def attention(q,k,v,causal=True,mask=None):
  assert q.dtype==k.dtype==v.dtype==dtypes.bfloat16
  assert q.shape[-1]==k.shape[-1]==v.shape[-1] and q.shape[-1] in (72,128)
  out=Tensor.empty(q.shape,device=q.device,dtype=q.dtype)
  masks=() if mask is None else (mask.float().contiguous(),)
  if int(os.environ.get('ALPAMAYO_ONLINE_VALUE_LAYOUT','0')) and q.shape[2]>=int(os.environ.get('ALPAMAYO_ONLINE_VALUE_MIN_ROWS','0')):
    v=pack_values(v)
  return Tensor.custom_kernel(out,q.contiguous(),k.contiguous(),v.contiguous(),*masks,fxn=partial(program,shape=q.shape,causal=causal))[0]


def install(model,expert=False):
  from types import MethodType
  original=model.attention
  def attend(self,q,k,v,mask=None,causal=False):
    if mask is None and q.shape[2]>=128 and q.shape[-1] in (72,128):
      return attention(q,k,v,causal)
    if expert and q.shape[2:]==(64,128) and mask is not None and mask.numel()==k.shape[2] and not causal:
      return attention(q,k,v,causal,mask)
    return original(q,k,v,mask,causal)
  model.attention=MethodType(attend,model)

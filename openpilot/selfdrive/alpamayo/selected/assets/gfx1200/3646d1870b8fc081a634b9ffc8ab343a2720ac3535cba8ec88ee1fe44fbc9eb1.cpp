
#define DEV __attribute__((device)) __attribute__((always_inline)) inline
typedef unsigned short u16;
typedef unsigned short u8 __attribute__((ext_vector_type(8)));
DEV u8 load_eight(const u16 *source,int head,int row,int dim) {
  if(row>=2048 || dim>=128) return (u8){};
  return *(const volatile u8 *)(source+(head*2048+row)*128+dim);
}
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
  #if 0
  return ((row/16)*((128+15)/16)+dim/16)*256+(dim%16/4)*64+(row%16)*4+dim%4;
  #else
  return row*(((128+15)/16)*16+4)+dim;
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
extern "C" __attribute__((global,amdgpu_flat_work_group_size(1,256)))
void alpamayo_online(u16 *out,const u16 *q,const u16 *k,const u16 *v
#if 0
,const float *mask
#endif
) {
  constexpr int H=16, KH=8, R=2048, K=2048, W=128, D=(W+15)/16;
  int thread=__ockl_get_local_id(0), lane=thread%32, wave=thread/32, block=__ockl_get_group_id(0);
  int group=block%((R+16*8-1)/(16*8));
  int tile=group*8+wave, head=block/((R+16*8-1)/(16*8)), kh=(head/H)*KH+(head%H)/(H/KH);
  int qr=tile*16+lane%16;
  __attribute__((shared,aligned(16))) short probabilities[8*16*(16+0)];
  #if 1
  __attribute__((shared,aligned(16))) short shared_k[32*(D*16+4)],shared_v[32*D*16];
  #endif
  short *prob=probabilities+wave*16*(16+0);
  s8 query[D];
  for(int d=0;d<D;d++) for(int i=0;i<8;i++) {
    int dim=d*16+(lane/16)*8+i;
    query[d][i]=qr<R && dim<W?q[(head*R+qr)*W+dim]:0;
  }
  f8 accum[D]={}; f8 denom={}, maxima;
  for(int i=0;i<8;i++) maxima[i]=-__builtin_inff();
  int limit=(group+1)*8*16;
  int end=1?(K<limit?K:limit):K;
  for(int start=0;start<end;start+=32) {
    #if 1
    for(int index=thread*8;index<32*D*16;index+=256*8) {
      int row=index/(D*16),dim=index%(D*16);
      int vd=index/(32*16), within=index%(32*16);
      int vk=within/256, fragment=within%256;
      int voffset=((kh*D+vd)*((K+15)/16)+start/16+vk)*256+fragment;
      u8 loaded_k=load_eight(k,kh,start+row,dim);
      u8 loaded_v=start+vk*16<K?*(const volatile u8*)(v+voffset):(u8){};
      for(int i=0;i<8;i++) { shared_k[key_index(row,dim+i)]=loaded_k[i]; shared_v[index+i]=loaded_v[i]; }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
    #endif
    f8 scores[32/16]={};
    for(int sub=0;sub<32/16;sub++) for(int d=0;d<D;d++) {
      s8 key;
      for(int i=0;i<8;i++) { int dim=d*16+(lane/16)*8+i;
        int col=start+sub*16+lane%16;
        #if 1
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
      for(int sub=0;sub<32/16;sub++) {
        int col=start+sub*16+lane%16;
        scores[sub][i]=(col<K && (!1 || col<=row))?scores[sub][i]*0.08838834764831845f:-__builtin_inff();
        #if 0
        if(col<K) scores[sub][i]+=mask[col];
        #endif
        localmax=__builtin_fmaxf(localmax,scores[sub][i]);
      }
      float next=__builtin_fmaxf(maxima[i],maximum(localmax));
      float scale=next==-__builtin_inff()?1.0f:__builtin_exp2f((maxima[i]-next)*1.4426950408889634f);
      float sum=0;
      for(int sub=0;sub<32/16;sub++) {
        scores[sub][i]=next==-__builtin_inff()?0.0f:__builtin_exp2f((scores[sub][i]-next)*1.4426950408889634f);
        sum+=scores[sub][i];
      }
      for(int d=0;d<D;d++) accum[d][i]*=scale;
      denom[i]=denom[i]*scale+total(sum); maxima[i]=next;
    }
    for(int sub=0;sub<32/16;sub++) {
    #if 1
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
    #elif 0
    f8 transposed=transpose_probability(scores[sub],lane);
    s8 probability;
    for(int i=0;i<8;i++) probability[i]=tobf(transposed[i]);
    #else
    for(int i=0;i<8;i++) prob[((lane/16)*8+i)*(16+0)+lane%16]=tobf(scores[sub][i]);
    #if 0
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"wavefront"); __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"wavefront");
    #else
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"workgroup"); __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"workgroup");
    #endif
    s8 probability;
    for(int i=0;i<8;i++) probability[i]=prob[(lane%16)*(16+0)+(lane/16)*8+i];
    #endif
    for(int d=0;d<D;d++) {
      s8 value;
      value=*(const s8*)(shared_v+(d*(32/16)+sub)*256+lane*8);
      accum[d]=mma(probability,value,accum[d]);
    }
    #if !0 && !1
    #if 0
    __builtin_amdgcn_fence(__ATOMIC_RELEASE,"wavefront"); __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"wavefront");
    #else
    __builtin_amdgcn_s_barrier();
    #endif
    #endif
    }
    #if (0 || 0 || 1) && 1
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

#define M 2048
#define N 9216
#define K 2048
#define TM 4
#define TN 4
#define PADM 2048
#define MTILES 32
#define NTILES 144
#define EPILOGUE_ONLY 0

#pragma clang fp contract(off)
#define DEV __attribute__((device,always_inline)) inline
typedef int i2 __attribute__((ext_vector_type(2)));
typedef float f8 __attribute__((ext_vector_type(8)));
extern "C" __attribute__((device)) unsigned long __ockl_get_local_id(unsigned);
extern "C" __attribute__((device)) unsigned long __ockl_get_group_id(unsigned);
// Exact pinned tinygrad BF16 cast, including subnormal and special-value rules.
DEV unsigned short bf(float value) {
  unsigned b=__builtin_bit_cast(unsigned,value);
  unsigned regular=b+((b>>16)&1)+0x7fffu;
  unsigned special=b|(((b&0xffffu)!=0)<<16);
  unsigned mask=0u-(((-b & 0x7f800000u)!=0));
  asm("" : "+v"(mask));
  return (unsigned short)(((regular&mask)|(special&~mask))>>16);
}
DEV float frombf(unsigned short value) { return __builtin_bit_cast(float,((unsigned)value)<<16); }
DEV float activate(float sum,float scale,float inverse_output_scale) {
  float rounded=frombf(bf(sum*scale));
  float positive=__builtin_fmaxf(rounded,0.0f);
  float squared=frombf(bf(positive*positive));
  // Match actual AMD LLVM lowering of fdiv contract afn: one hardware reciprocal
  // followed by a rounded FP32 multiply. Full HIP division emits different arithmetic.
  float scaled=squared*inverse_output_scale;
  return __builtin_fmaxf(-448.0f,__builtin_fminf(448.0f,scaled));
}
DEV unsigned pack_four(float a,float b,float c,float d) {
  unsigned low=__builtin_amdgcn_cvt_pk_fp8_f32(a,b,0,false);
  return __builtin_amdgcn_cvt_pk_fp8_f32(c,d,low,true);
}
DEV i2 fragment(const unsigned *ptr,unsigned offset) { return (i2){(int)ptr[offset],(int)ptr[offset+32]}; }
extern "C" __attribute__((global,amdgpu_flat_work_group_size(32,32)))
void agent_packed_epilogue(unsigned *__restrict out,const unsigned *__restrict x,
                          const unsigned *__restrict weight,const float *__restrict xs,
                          const float *__restrict ws,const float *__restrict output_scale) {
  unsigned lane=__ockl_get_local_id(0),block=__ockl_get_group_id(0);
  unsigned mt=block/NTILES,nt=block%NTILES;
  f8 accum[TM][TN];
  #pragma unroll
  for(unsigned i=0;i<TM;i++) {
    #pragma unroll
    for(unsigned j=0;j<TN;j++) accum[i][j]=(f8){0};
  }
#if EPILOGUE_ONLY
  // Independent float input probe covers rounding boundaries without a matrix reduction.
  #pragma unroll
  for(unsigned i=0;i<TM;i++) {
    #pragma unroll
    for(unsigned j=0;j<TN;j++) {
      unsigned row=mt*(16*TM)+i*16+lane%16,col=nt*(16*TN)+j*16+(lane/16)*8;
      #pragma unroll
      for(unsigned v=0;v<8;v++) accum[i][j][v]=(row<M && col+v<N)?((const float *)x)[row*N+col+v]:0.0f;
    }
  }
#else
  unsigned ao[TM],bo[TN];
  #pragma unroll
  for(unsigned i=0;i<TM;i++) {
    unsigned ar=mt*TM+i;
    ar=ar<(M+15)/16?ar:(M+15)/16-1;
    ao[i]=ar*(K/16)*64+lane;
  }
  #pragma unroll
  for(unsigned j=0;j<TN;j++) {
    unsigned br=nt*TN+j;
    br=br<N/16?br:N/16-1;
    bo[j]=br*(K/16)*64+lane;
  }
  i2 a[TM],b[TN];
  #pragma unroll
  for(unsigned i=0;i<TM;i++) a[i]=fragment(x,ao[i]);
  #pragma unroll
  for(unsigned j=0;j<TN;j++) b[j]=fragment(weight,bo[j]);
  #pragma clang loop unroll(disable)
  for(unsigned k=0;k<K/16;k++) {
    #pragma unroll
    for(unsigned i=0;i<TM;i++) {
      #pragma unroll
      for(unsigned j=0;j<TN;j++)
        accum[i][j]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b[j],a[i],accum[i][j]);
    }
    if(k+1<K/16) {
      #pragma unroll
      for(unsigned i=0;i<TM;i++) a[i]=fragment(x,ao[i]+(k+1)*64);
      #pragma unroll
      for(unsigned j=0;j<TN;j++) b[j]=fragment(weight,bo[j]+(k+1)*64);
    }
  }
#endif
  float scale=xs[0]*ws[0],next_scale=__builtin_amdgcn_rcpf(output_scale[0]);
  #pragma unroll
  for(unsigned j=0;j<TN;j++) {
    #pragma unroll
    for(unsigned i=0;i<TM;i++) {
      unsigned row=mt*(16*TM)+i*16+lane%16,col=nt*(16*TN)+j*16+(lane/16)*8;
      if(row<PADM && col<N) {
        // Packed byte address: (row/16*(N/16)+col/16)*256+(col%16/4)*64+(row%16)*4+col%4.
        // col%4 is zero. Each lane writes two four-byte words separated by 64 bytes.
        unsigned location=(row/16*(N/16)+col/16)*64+(col%16/4)*16+row%16;
        unsigned words[2];
        #pragma unroll
        for(unsigned q=0;q<2;q++) {
          float a=activate(accum[i][j][q*4],scale,next_scale);
          float b=activate(accum[i][j][q*4+1],scale,next_scale);
          float c=activate(accum[i][j][q*4+2],scale,next_scale);
          float d=activate(accum[i][j][q*4+3],scale,next_scale);
          words[q]=row<M?pack_four(a,b,c,d):0u;
        }
        out[location]=words[0];
        out[location+16]=words[1];
      }
    }
  }
}

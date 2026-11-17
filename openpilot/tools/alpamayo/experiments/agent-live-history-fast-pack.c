/* Exact uint8 permutation: CHW -> merge-block/patch/HWC, no image arithmetic. */
#include <stdint.h>
#include <tmmintrin.h>

#define M(i,c) ((i)%3==(c) ? (i)/3 : -128)
#define S(j,c) _mm_setr_epi8(M(16*(j)+0,c),M(16*(j)+1,c),M(16*(j)+2,c),M(16*(j)+3,c), \
  M(16*(j)+4,c),M(16*(j)+5,c),M(16*(j)+6,c),M(16*(j)+7,c), \
  M(16*(j)+8,c),M(16*(j)+9,c),M(16*(j)+10,c),M(16*(j)+11,c), \
  M(16*(j)+12,c),M(16*(j)+13,c),M(16*(j)+14,c),M(16*(j)+15,c))
#define BLOCK(j) _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r,S(j,0)),_mm_shuffle_epi8(g,S(j,1))),_mm_shuffle_epi8(b,S(j,2)))

void pack_320_576(uint8_t *restrict dst, const uint8_t *restrict src) {
  const int plane=320*576;
  for (int block_y=0;block_y<10;block_y++)
    for (int block_x=0;block_x<18;block_x++)
      for (int merge_y=0;merge_y<2;merge_y++)
        for (int merge_x=0;merge_x<2;merge_x++)
          for (int y=0;y<16;y++) {
            const int offset=((block_y*2+merge_y)*16+y)*576+(block_x*2+merge_x)*16;
            const __m128i r=_mm_loadu_si128((const __m128i *)(src+offset));
            const __m128i g=_mm_loadu_si128((const __m128i *)(src+plane+offset));
            const __m128i b=_mm_loadu_si128((const __m128i *)(src+2*plane+offset));
            _mm_storeu_si128((__m128i *)(dst+0),BLOCK(0));
            _mm_storeu_si128((__m128i *)(dst+16),BLOCK(1));
            _mm_storeu_si128((__m128i *)(dst+32),BLOCK(2));
            dst+=48;
          }
}

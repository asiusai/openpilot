#include <stddef.h>
#include <stdint.h>

/* Factors contain the first weight, 1-fraction. SciPy derives the second
 * weight as 1-first_weight. Preserve that rounding and each term's order. */
static inline uint8_t sample(const uint8_t *src, uint32_t a, uint32_t b,
                             uint32_t c, uint32_t d, double dx, double dy) {
  double value = 0.;
  value += (src[a] * dy) * dx;
  value += (src[b] * dy) * (1.-dx);
  value += (src[c] * (1.-dy)) * dx;
  value += (src[d] * (1.-dy)) * (1.-dx);
  /* Valid weights bound value to [0,255], within double rounding error.
   * Adding .5 then truncating remains in [0,255]; redundant clamps inhibit SIMD. */
  return (uint8_t)(int32_t)(value+.5);
}

/* Four contiguous index planes and two contiguous fraction planes. */
void warp_soa(const uint8_t *restrict src, uint8_t *restrict dst,
              const uint32_t *restrict indices, const double *restrict fractions,
              size_t pixels, size_t source_pixels, size_t begin, size_t end) {
  const uint32_t *a = indices, *b = indices+pixels, *c = indices+2*pixels, *d = indices+3*pixels;
  const double *dx = fractions, *dy = fractions+pixels;
  for (size_t channel = 0; channel < 3; ++channel) {
    const uint8_t *in = src+channel*source_pixels;
    uint8_t *out = dst+channel*pixels;
    for (size_t i = begin; i < end; ++i)
      out[i] = sample(in, a[i], b[i], c[i], d[i], dx[i], dy[i]);
  }
}

/* Reuse each map element across all channels. */
void warp_channels(const uint8_t *restrict src, uint8_t *restrict dst,
                   const uint32_t *restrict indices, const double *restrict fractions,
                   size_t pixels, size_t source_pixels, size_t begin, size_t end) {
  const uint32_t *a = indices, *b = indices+pixels, *c = indices+2*pixels, *d = indices+3*pixels;
  const double *dx = fractions, *dy = fractions+pixels;
  for (size_t i = begin; i < end; ++i) {
    const uint32_t ai = a[i], bi = b[i], ci = c[i], di = d[i];
    const double x = dx[i], y = dy[i];
    dst[i] = sample(src, ai, bi, ci, di, x, y);
    dst[pixels+i] = sample(src+source_pixels, ai, bi, ci, di, x, y);
    dst[2*pixels+i] = sample(src+2*source_pixels, ai, bi, ci, di, x, y);
  }
}

/* Tiling keeps map data hot between channel passes. */
void warp_tiled(const uint8_t *restrict src, uint8_t *restrict dst,
                const uint32_t *restrict indices, const double *restrict fractions,
                size_t pixels, size_t source_pixels, size_t begin, size_t end) {
  const uint32_t *a = indices, *b = indices+pixels, *c = indices+2*pixels, *d = indices+3*pixels;
  const double *dx = fractions, *dy = fractions+pixels;
  for (size_t tile = begin; tile < end; tile += 256) {
    size_t stop = tile+256 < end ? tile+256 : end;
    for (size_t channel = 0; channel < 3; ++channel) {
      const uint8_t *in = src+channel*source_pixels;
      uint8_t *out = dst+channel*pixels;
      for (size_t i = tile; i < stop; ++i)
        out[i] = sample(in, a[i], b[i], c[i], d[i], dx[i], dy[i]);
    }
  }
}

/* Explicit gather staging exposes independent arithmetic to portable SIMD. */
void warp_staged(const uint8_t *restrict src, uint8_t *restrict dst,
                 const uint32_t *restrict indices, const double *restrict fractions,
                 size_t pixels, size_t source_pixels, size_t begin, size_t end) {
  const uint32_t *a = indices, *b = indices+pixels, *c = indices+2*pixels, *d = indices+3*pixels;
  const double *dx = fractions, *dy = fractions+pixels;
  for (size_t tile = begin; tile < end; tile += 64) {
    size_t length = tile+64 < end ? 64 : end-tile;
    for (size_t channel = 0; channel < 3; ++channel) {
      const uint8_t *in = src+channel*source_pixels;
      uint8_t *out = dst+channel*pixels;
      double aa[64], bb[64], cc[64], dd[64];
      for (size_t j = 0; j < length; ++j) {
        size_t i = tile+j;
        aa[j] = in[a[i]]; bb[j] = in[b[i]];
        cc[j] = in[c[i]]; dd[j] = in[d[i]];
      }
      for (size_t j = 0; j < length; ++j) {
        size_t i = tile+j;
        double value = 0.;
        value += (aa[j] * dy[i]) * dx[i];
        value += (bb[j] * dy[i]) * (1.-dx[i]);
        value += (cc[j] * (1.-dy[i])) * dx[i];
        value += (dd[j] * (1.-dy[i])) * (1.-dx[i]);
        out[i] = (uint8_t)(int32_t)(value+.5);
      }
    }
  }
}

/* Interleaved indices and fractions retain spatial locality. */
void warp_aos(const uint8_t *restrict src, uint8_t *restrict dst,
              const uint32_t *restrict indices, const double *restrict fractions,
              size_t pixels, size_t source_pixels, size_t begin, size_t end) {
  for (size_t i = begin; i < end; ++i) {
    const uint32_t a = indices[4*i], b = indices[4*i+1], c = indices[4*i+2], d = indices[4*i+3];
    const double dx = fractions[2*i], dy = fractions[2*i+1];
    dst[i] = sample(src, a, b, c, d, dx, dy);
    dst[pixels+i] = sample(src+source_pixels, a, b, c, d, dx, dy);
    dst[2*pixels+i] = sample(src+2*source_pixels, a, b, c, d, dx, dy);
  }
}

/* OpenMP SIMD has no runtime dependency and permits independent pixel lanes. */
void warp_simd(const uint8_t *restrict src, uint8_t *restrict dst,
               const uint32_t *restrict indices, const double *restrict fractions,
               size_t pixels, size_t source_pixels, size_t begin, size_t end) {
  const uint32_t *a = indices, *b = indices+pixels, *c = indices+2*pixels, *d = indices+3*pixels;
  const double *dx = fractions, *dy = fractions+pixels;
  for (size_t channel = 0; channel < 3; ++channel) {
    const uint8_t *in = src+channel*source_pixels;
    uint8_t *out = dst+channel*pixels;
    #pragma omp simd
    for (size_t i = begin; i < end; ++i)
      out[i] = sample(in, a[i], b[i], c[i], d[i], dx[i], dy[i]);
  }
}

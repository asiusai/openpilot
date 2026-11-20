#include <stddef.h>
#include <stdint.h>

/* Match scipy.ndimage order=1, mode=nearest, including uint8 rounding.
 * Keep double multiplication order and disable floating-point contraction. */
void warp_u8(const uint8_t *restrict src, uint8_t *restrict dst, const int32_t *restrict offsets,
             const double *restrict fractions, size_t pixels, size_t source_pixels,
             size_t width, size_t height) {
  for (size_t i = 0; i < pixels; ++i) {
    size_t x = offsets[2*i], y = offsets[2*i+1];
    size_t p = y*width+x, q = p+(x+1 < width);
    size_t r = p+(y+1 < height ? width : 0), s = r+(x+1 < width);
    double dx = fractions[2*i], dy = fractions[2*i+1];
    for (size_t channel = 0; channel < 3; ++channel) {
      const uint8_t *in = src + channel * source_pixels;
      double value = 0.;
      value += (in[p] * (1.-dy)) * (1.-dx);
      value += (in[q] * (1.-dy)) * dx;
      value += (in[r] * dy) * (1.-dx);
      value += (in[s] * dy) * dx;
      dst[channel*pixels+i] = value > 255. ? 255 : value < 0. ? 0 : (uint8_t)(value+.5);
    }
  }
}

/* The same samples from interleaved HWC RGB, as the video decoder returns it. A planar copy of the full camera frame
 * cost about as much as the colour conversion itself. The arithmetic and the output are identical to warp_u8. */
void warp_hwc_u8(const uint8_t *restrict src, uint8_t *restrict dst, const int32_t *restrict offsets,
                 const double *restrict fractions, size_t pixels, size_t width, size_t height) {
  for (size_t i = 0; i < pixels; ++i) {
    size_t x = offsets[2*i], y = offsets[2*i+1];
    size_t p = y*width+x, q = p+(x+1 < width);
    size_t r = p+(y+1 < height ? width : 0), s = r+(x+1 < width);
    double dx = fractions[2*i], dy = fractions[2*i+1];
    for (size_t channel = 0; channel < 3; ++channel) {
      double value = 0.;
      value += (src[3*p+channel] * (1.-dy)) * (1.-dx);
      value += (src[3*q+channel] * (1.-dy)) * dx;
      value += (src[3*r+channel] * dy) * (1.-dx);
      value += (src[3*s+channel] * dy) * dx;
      dst[channel*pixels+i] = value > 255. ? 255 : value < 0. ? 0 : (uint8_t)(value+.5);
    }
  }
}

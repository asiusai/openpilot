#include <stddef.h>
#include <stdint.h>

/* Match scipy.ndimage order=1, mode=nearest, including uint8 rounding.
 * Keep double multiplication order and disable floating-point contraction. */
void warp_u8(const uint8_t *src, uint8_t *dst, const int32_t *offsets,
             const double *fractions, size_t pixels, size_t source_pixels,
             size_t width, size_t height) {
  for (size_t channel = 0; channel < 3; ++channel) {
    const uint8_t *in = src + channel * source_pixels;
    uint8_t *out = dst + channel * pixels;
    for (size_t i = 0; i < pixels; ++i) {
      size_t x = offsets[2*i], y = offsets[2*i+1];
      size_t nx = x+1 < width ? x+1 : x;
      size_t ny = y+1 < height ? y+1 : y;
      double dx = fractions[2*i], dy = fractions[2*i+1];
      double value = 0.;
      value += (in[y*width+x] * (1.-dy)) * (1.-dx);
      value += (in[y*width+nx] * (1.-dy)) * dx;
      value += (in[ny*width+x] * dy) * (1.-dx);
      value += (in[ny*width+nx] * dy) * dx;
      out[i] = value > 255. ? 255 : value < 0. ? 0 : (uint8_t)(value+.5);
    }
  }
}

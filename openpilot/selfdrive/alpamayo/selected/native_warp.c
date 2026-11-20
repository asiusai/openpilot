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
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static inline uint8_t round_u8(double value) {
  return value > 255. ? 255 : value < 0. ? 0 : (uint8_t)(value+.5);
}

void warp_hwc_u8(const uint8_t *restrict src, uint8_t *restrict dst, const int32_t *restrict offsets,
                 const double *restrict fractions, size_t pixels, size_t width, size_t height) {
  size_t i = 0;
#if defined(__aarch64__)
  /* Two pixels per iteration in float64x2 lanes: the same IEEE operations in the same order as the scalar loop. */
  const float64x2_t one = vdupq_n_f64(1.);
  for (; i+1 < pixels; i += 2) {
    double dxs[2] = {fractions[2*i], fractions[2*i+2]}, dys[2] = {fractions[2*i+1], fractions[2*i+3]};
    float64x2_t dx = vld1q_f64(dxs), dy = vld1q_f64(dys);
    float64x2_t ndx = vsubq_f64(one, dx), ndy = vsubq_f64(one, dy);
    size_t p[2], q[2], r[2], t[2];
    for (int k = 0; k < 2; ++k) {
      size_t x = offsets[2*(i+k)], y = offsets[2*(i+k)+1];
      p[k] = y*width+x; q[k] = p[k]+(x+1 < width);
      r[k] = p[k]+(y+1 < height ? width : 0); t[k] = r[k]+(x+1 < width);
    }
    for (size_t channel = 0; channel < 3; ++channel) {
      double a[2] = {src[3*p[0]+channel], src[3*p[1]+channel]}, b[2] = {src[3*q[0]+channel], src[3*q[1]+channel]};
      double c[2] = {src[3*r[0]+channel], src[3*r[1]+channel]}, d[2] = {src[3*t[0]+channel], src[3*t[1]+channel]};
      float64x2_t value = vmulq_f64(vmulq_f64(vld1q_f64(a), ndy), ndx);
      value = vaddq_f64(value, vmulq_f64(vmulq_f64(vld1q_f64(b), ndy), dx));
      value = vaddq_f64(value, vmulq_f64(vmulq_f64(vld1q_f64(c), dy), ndx));
      value = vaddq_f64(value, vmulq_f64(vmulq_f64(vld1q_f64(d), dy), dx));
      dst[channel*pixels+i] = round_u8(vgetq_lane_f64(value, 0));
      dst[channel*pixels+i+1] = round_u8(vgetq_lane_f64(value, 1));
    }
  }
#endif
  for (; i < pixels; ++i) {
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
      dst[channel*pixels+i] = round_u8(value);
    }
  }
}

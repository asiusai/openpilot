/* Check concurrent submissions, uneven partitions, mutation, and pool lifetime. */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*warp_fn)(const uint8_t *, uint8_t *, const uint32_t *, const double *,
                        size_t, size_t, size_t, size_t);
void warp_soa(const uint8_t *, uint8_t *, const uint32_t *, const double *, size_t, size_t, size_t, size_t);
void warp_simd(const uint8_t *, uint8_t *, const uint32_t *, const double *, size_t, size_t, size_t, size_t);
void *warp_pool_create(size_t);
void warp_pool_run(void *, warp_fn, const uint8_t *, uint8_t *, const uint32_t *, const double *, size_t, size_t);
void warp_pool_destroy(void *);
typedef struct { void *pool; size_t pixels, seed; int failed; } task;

static void *check(void *argument) {
  task *t = argument;
  const size_t source_pixels = 97;
  uint8_t *src = malloc(3*source_pixels), *expected = malloc(3*t->pixels), *actual = malloc(3*t->pixels);
  uint32_t *indices = malloc(4*t->pixels*sizeof(*indices));
  double *factors = malloc(2*t->pixels*sizeof(*factors));
  if (!src || !expected || !actual || !indices || !factors) abort();
  for (size_t i = 0; i < 4*t->pixels; ++i) indices[i] = (i*17+t->seed)%source_pixels;
  for (size_t i = 0; i < 2*t->pixels; ++i) factors[i] = (double)((i*11+t->seed)%65)/64.;
  for (size_t round = 0; round < 100; ++round) {
    for (size_t i = 0; i < 3*source_pixels; ++i) src[i] = (uint8_t)(i*13+round*7+t->seed);
    warp_soa(src, expected, indices, factors, t->pixels, source_pixels, 0, t->pixels);
    warp_pool_run(t->pool, warp_simd, src, actual, indices, factors, t->pixels, source_pixels);
    if (memcmp(expected, actual, 3*t->pixels)) { t->failed = 1; break; }
  }
  free(factors); free(indices); free(actual); free(expected); free(src);
  return NULL;
}

int main(void) {
  const size_t sizes[] = {1, 17, 513};
  for (size_t workers = 2; workers <= 3; ++workers) {
    for (size_t size = 0; size < 3; ++size) {
      void *pool = warp_pool_create(workers);
      if (!pool) return 2;
      pthread_t producer;
      task one = {pool, sizes[size], 3, 0}, two = {pool, sizes[size], 19, 0};
      if (pthread_create(&producer, NULL, check, &two)) return 3;
      check(&one);
      pthread_join(producer, NULL);
      warp_pool_destroy(pool);
      if (one.failed || two.failed) return 4;
    }
  }
  puts("1200 concurrent submissions matched; six pool lifetimes passed");
  return 0;
}

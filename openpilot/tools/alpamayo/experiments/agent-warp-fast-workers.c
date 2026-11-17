#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef void (*warp_fn)(const uint8_t *, uint8_t *, const uint32_t *, const double *,
                        size_t, size_t, size_t, size_t);

typedef struct warp_pool warp_pool;
typedef struct { warp_pool *pool; size_t lane; } worker_arg;
struct warp_pool {
  pthread_mutex_t submit, mutex;
  pthread_cond_t ready, done;
  pthread_t threads[2];
  worker_arg workers[2];
  size_t count, generation, remaining;
  int stopping;
  warp_fn fn;
  const uint8_t *src;
  uint8_t *dst;
  const uint32_t *indices;
  const double *factors;
  size_t pixels, source_pixels;
};

static void *worker(void *argument) {
  worker_arg *arg = argument;
  warp_pool *pool = arg->pool;
  size_t generation = 0;
  pthread_mutex_lock(&pool->mutex);
  for (;;) {
    while (!pool->stopping && generation == pool->generation)
      pthread_cond_wait(&pool->ready, &pool->mutex);
    if (pool->stopping) break;
    generation = pool->generation;
    const size_t begin = pool->pixels*arg->lane/pool->count;
    const size_t end = pool->pixels*(arg->lane+1)/pool->count;
    pthread_mutex_unlock(&pool->mutex);
    pool->fn(pool->src, pool->dst, pool->indices, pool->factors,
             pool->pixels, pool->source_pixels, begin, end);
    pthread_mutex_lock(&pool->mutex);
    if (--pool->remaining == 0) pthread_cond_signal(&pool->done);
  }
  pthread_mutex_unlock(&pool->mutex);
  return NULL;
}

void *warp_pool_create(size_t count) {
  if (count < 2 || count > 3) return NULL;
  warp_pool *pool = calloc(1, sizeof(*pool));
  if (!pool) return NULL;
  if (pthread_mutex_init(&pool->submit, NULL)) goto fail0;
  if (pthread_mutex_init(&pool->mutex, NULL)) goto fail1;
  if (pthread_cond_init(&pool->ready, NULL)) goto fail2;
  if (pthread_cond_init(&pool->done, NULL)) goto fail3;
  pool->count = count;
  for (size_t i = 0; i < count-1; ++i) {
    pool->workers[i] = (worker_arg){pool, i+1};
    if (pthread_create(&pool->threads[i], NULL, worker, &pool->workers[i])) {
      pthread_mutex_lock(&pool->mutex);
      pool->stopping = 1;
      pthread_cond_broadcast(&pool->ready);
      pthread_mutex_unlock(&pool->mutex);
      for (size_t j = 0; j < i; ++j) pthread_join(pool->threads[j], NULL);
      goto fail4;
    }
  }
  return pool;
fail4: pthread_cond_destroy(&pool->done);
fail3: pthread_cond_destroy(&pool->ready);
fail2: pthread_mutex_destroy(&pool->mutex);
fail1: pthread_mutex_destroy(&pool->submit);
fail0: free(pool);
  return NULL;
}

void warp_pool_run(warp_pool *pool, warp_fn fn, const uint8_t *src, uint8_t *dst,
                   const uint32_t *indices, const double *factors, size_t pixels, size_t source_pixels) {
  pthread_mutex_lock(&pool->submit);
  pthread_mutex_lock(&pool->mutex);
  pool->fn = fn; pool->src = src; pool->dst = dst;
  pool->indices = indices; pool->factors = factors;
  pool->pixels = pixels; pool->source_pixels = source_pixels;
  pool->remaining = pool->count-1;
  ++pool->generation;
  pthread_cond_broadcast(&pool->ready);
  pthread_mutex_unlock(&pool->mutex);
  fn(src, dst, indices, factors, pixels, source_pixels, 0, pixels/pool->count);
  pthread_mutex_lock(&pool->mutex);
  while (pool->remaining) pthread_cond_wait(&pool->done, &pool->mutex);
  pthread_mutex_unlock(&pool->mutex);
  pthread_mutex_unlock(&pool->submit);
}

void warp_pool_destroy(warp_pool *pool) {
  if (!pool) return;
  pthread_mutex_lock(&pool->submit);
  pthread_mutex_lock(&pool->mutex);
  pool->stopping = 1;
  pthread_cond_broadcast(&pool->ready);
  pthread_mutex_unlock(&pool->mutex);
  for (size_t i = 0; i < pool->count-1; ++i) pthread_join(pool->threads[i], NULL);
  pthread_mutex_unlock(&pool->submit);
  pthread_cond_destroy(&pool->done);
  pthread_cond_destroy(&pool->ready);
  pthread_mutex_destroy(&pool->mutex);
  pthread_mutex_destroy(&pool->submit);
  free(pool);
}

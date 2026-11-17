"""Isolated exact CPU sampler with bounded geometry maps and optional worker threads."""
from collections import OrderedDict
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import threading

import numpy as np

from calibration_trials import calibrated_coordinates

DIRECTORY = Path(__file__).resolve().parent
ARTIFACTS = DIRECTORY/'agent-warp-fast-artifacts'
KINDS = ('soa', 'channels', 'tiled', 'staged', 'aos', 'simd')
FLAGS = ('-O3', '-shared', '-fPIC', '-std=c11', '-fno-fast-math', '-ffp-contract=off', '-fopenmp-simd', '-pthread')


def build(compiler='clang', native=False, baseline=False):
  compiler = shutil.which(compiler)
  if compiler is None:
    raise RuntimeError('Requested C compiler is unavailable')
  source = DIRECTORY/('native-warp.c' if baseline else 'agent-warp-fast-kernel.c')
  sources = [source] if baseline else [source, DIRECTORY/'agent-warp-fast-workers.c']
  flags = list(FLAGS)+(['-march=native'] if native else [])
  version = subprocess.check_output([compiler, '--version'], text=True)
  digest = hashlib.sha256(b''.join(p.read_bytes() for p in sources)+version.encode()+platform.machine().encode()+repr(flags).encode()).hexdigest()
  ARTIFACTS.mkdir(exist_ok=True)
  output = ARTIFACTS/f'{digest}.so'
  command = [compiler, *flags, *(str(p) for p in sources), '-o', str(output)]
  if not output.exists():
    subprocess.run(command, check=True)
  lib = ctypes.CDLL(str(output))
  for name in (('u8',) if baseline else KINDS):
    fn = getattr(lib, 'warp_'+name)
    fn.argtypes = [ctypes.c_void_p]*4+[ctypes.c_size_t]*4
    fn.restype = None
  if not baseline:
    lib.warp_pool_create.argtypes, lib.warp_pool_create.restype = [ctypes.c_size_t], ctypes.c_void_p
    lib.warp_pool_run.argtypes, lib.warp_pool_run.restype = [ctypes.c_void_p]*6+[ctypes.c_size_t]*2, None
    lib.warp_pool_destroy.argtypes, lib.warp_pool_destroy.restype = [ctypes.c_void_p], None
  lib.provenance = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        all_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                        library=str(output), library_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        compiler_version=version, command=command, native=native, architecture=platform.machine())
  return lib


def pack_maps(shape, sy, sx, kind='channels'):
  h, w = shape
  if h < 1 or w < 1 or h*w > np.iinfo(np.uint32).max:
    raise ValueError('Source dimensions exceed the uint32 index contract')
  sx, sy = np.asarray(sx, dtype=np.float64), np.asarray(sy, dtype=np.float64)
  if sx.shape != sy.shape or sx.ndim != 2 or not np.isfinite(sx).all() or not np.isfinite(sy).all():
    raise ValueError('Require equally shaped finite two-dimensional coordinate maps')
  fx, fy = np.floor(sx), np.floor(sy)
  x, y = np.clip(fx, 0, w-1).astype(np.uint32), np.clip(fy, 0, h-1).astype(np.uint32)
  nx, ny = np.clip(fx+1, 0, w-1).astype(np.uint32), np.clip(fy+1, 0, h-1).astype(np.uint32)
  axis = -1 if kind == 'aos' else 0
  indices = np.ascontiguousarray(np.stack([y*w+x, y*w+nx, ny*w+x, ny*w+nx], axis=axis))
  # SciPy clamps neighbor indices, retains outside fractions, and complements twice.
  fractions = np.ascontiguousarray(np.stack([1.-(sx-fx), 1.-(sy-fy)], axis=axis))
  indices.flags.writeable = fractions.flags.writeable = False
  return indices, fractions, sx.shape


class Sampler:
  def __init__(self, kind='channels', workers=1, compiler='clang', native=False, max_maps=8, max_bytes=96*1024**2):
    if kind not in KINDS or not 1 <= workers <= 3 or not 1 <= max_maps <= 8 or max_bytes <= 0:
      raise ValueError('Invalid bounded sampler configuration')
    self.kind, self.workers = kind, workers
    self.lib = build(compiler, native)
    self.fn = getattr(self.lib, 'warp_'+kind)
    self.cache, self.max_maps, self.max_bytes = OrderedDict(), max_maps, max_bytes
    self.cache_bytes = self.hits = self.misses = self.evictions = 0
    self.lock = threading.Lock()
    self.pool = self.lib.warp_pool_create(workers) if workers > 1 else None
    if workers > 1 and not self.pool:
      raise RuntimeError('Native CPU worker creation failed')
    self.fn_pointer = ctypes.cast(self.fn, ctypes.c_void_p)

  def maps(self, shape, camera, source, target, calibration):
    rotation = tuple(calibration['rpyCalib'])
    wide_rotation = tuple(calibration['wideFromDeviceEuler'])
    key = (tuple(shape), camera, json.dumps(source, sort_keys=True), json.dumps(target, sort_keys=True), rotation, wide_rotation)
    with self.lock:
      if key in self.cache:
        self.hits += 1
        self.cache.move_to_end(key)
        return self.cache[key]
      self.misses += 1
      sy, sx = calibrated_coordinates(shape, camera, source, target, calibration, False, False)
      packed = pack_maps(shape, sy, sx, self.kind)
      size = packed[0].nbytes+packed[1].nbytes
      if size > self.max_bytes:
        raise ValueError('One coordinate map exceeds the configured cache byte limit')
      while self.cache and (len(self.cache) >= self.max_maps or self.cache_bytes+size > self.max_bytes):
        _, old = self.cache.popitem(last=False)
        self.cache_bytes -= old[0].nbytes+old[1].nbytes
        self.evictions += 1
      self.cache[key] = packed
      self.cache_bytes += size
      return packed

  def sample(self, array, packed, out=None):
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[0] != 3 or not array.flags.c_contiguous:
      raise ValueError('Require contiguous uint8 CHW RGB')
    indices, fractions, shape = packed
    if out is None:
      out = np.empty((3, *shape), dtype=np.uint8)
    elif out.dtype != np.uint8 or out.shape != (3, *shape) or not out.flags.c_contiguous or np.shares_memory(array, out):
      raise ValueError('Invalid destination array')
    pixels = shape[0]*shape[1]
    args = (array.ctypes.data, out.ctypes.data, indices.ctypes.data, fractions.ctypes.data, pixels, array.shape[1]*array.shape[2])
    if self.pool is None:
      self.fn(*args, 0, pixels)
    else:
      self.lib.warp_pool_run(self.pool, self.fn_pointer, *args)
    return out

  def calibrated_image(self, array, camera, source, target, calibration):
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[0] != 3 or not array.flags.c_contiguous:
      raise ValueError('Require contiguous uint8 CHW RGB')
    return self.sample(array, self.maps(tuple(array.shape[-2:]), camera, source, target, calibration))

  def state(self):
    return dict(kind=self.kind, workers=self.workers, maps=len(self.cache), map_bytes=self.cache_bytes,
                max_maps=self.max_maps, max_bytes=self.max_bytes, hits=self.hits, misses=self.misses,
                evictions=self.evictions, provenance=self.lib.provenance)

  def close(self):
    if self.pool is not None:
      self.lib.warp_pool_destroy(self.pool)
      self.pool = None

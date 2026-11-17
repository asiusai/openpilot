"""Bounded ray-map cache and exact uint8 bilinear camera sampling."""
import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np

from calibration_trials import cached_calibrated_coordinates


@lru_cache(maxsize=1)
def library():
  source = Path(__file__).with_name('native-warp.c')
  compiler = shutil.which('clang') or shutil.which('cc')
  if compiler is None:
    raise RuntimeError('A C compiler is required to build the native camera sampler')
  flags = ['-O3', '-shared', '-fPIC', '-ffp-contract=off', '-std=c11']
  version = subprocess.check_output([compiler, '--version'])
  digest = hashlib.sha256(source.read_bytes()+version+platform.machine().encode()+repr(flags).encode()).hexdigest()
  cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'alpamayo-native-warp'
  cache.mkdir(parents=True, exist_ok=True)
  output = cache/(digest+'.so')
  if not output.exists():
    with tempfile.TemporaryDirectory(dir=cache) as directory:
      temporary = Path(directory)/'warp.so'
      subprocess.run([compiler, *flags, str(source), '-o', str(temporary)], check=True)
      os.replace(temporary, output)
  lib = ctypes.CDLL(str(output))
  lib.warp_u8.argtypes = [ctypes.c_void_p]*4+[ctypes.c_size_t]*4
  lib.warp_u8.restype = None
  return lib


@lru_cache(maxsize=8)
def maps(shape, camera, source_json, target_json, rotation, wide_rotation):
  sy, sx = cached_calibrated_coordinates(shape, camera, source_json, target_json, rotation, wide_rotation, False, False)
  h, w = shape
  sx, sy = np.clip(sx, 0, w-1), np.clip(sy, 0, h-1)
  xy = np.stack([sx, sy], axis=-1)
  offsets = np.floor(xy).astype(np.int32)
  fractions = xy-offsets
  return offsets, fractions


def calibrated_image(array, camera, source, target, calibration):
  if array.dtype != np.uint8 or array.ndim != 3 or array.shape[0] != 3 or not array.flags.c_contiguous:
    raise ValueError('Native sampler requires contiguous uint8 CHW RGB')
  offsets, fractions = maps(tuple(array.shape[-2:]), camera, json.dumps(source, sort_keys=True),
      json.dumps(target, sort_keys=True), tuple(calibration['rpyCalib']), tuple(calibration['wideFromDeviceEuler']))
  h, w = array.shape[-2:]
  out = np.empty((3, *offsets.shape[:2]), dtype=np.uint8)
  library().warp_u8(array.ctypes.data, out.ctypes.data, offsets.ctypes.data, fractions.ctypes.data,
                    offsets.size//2, h*w, w, h)
  return out

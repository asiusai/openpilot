"""Replay calibrated camera preprocessing inside the measured inference boundary."""
from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np

from calibration_trials import asof
from native_warp import calibrated_image, library


class NativeFrames:
  def __init__(self, manifest, calibrations, raw_cache):
    manifest = json.loads(Path(manifest).read_text())
    if manifest['camera_input_profile'] != 'calibrated_vehicle_history' or manifest['camera_subset'] != [1,6]:
      raise ValueError('Native preprocessing requires the calibrated two-camera fixture')
    self.rows = {Path(r['sample']).name:r for r in manifest['schedule']}
    if len(self.rows) != len(manifest['schedule']):
      raise ValueError('Duplicate fixture timestamps')
    self.source, self.target = manifest['source_camera_geometry'], manifest['target_camera_geometry']
    records = json.loads(Path(calibrations).read_text())
    self.calibrations = (np.array([r['ns'] for r in records]), records)
    self.raw_cache = Path(raw_cache)
    self.images = OrderedDict()
    library()

  def prepare(self, sample):
    start = time.perf_counter()
    row = self.rows[Path(sample).name]
    images, hits, misses = [], 0, 0
    for frame in row['frames']:
      camera, index = frame['view'], frame['video_index']
      calibration, stamp = asof(self.calibrations, frame['capture_ns'], 1_000_000_000)
      if stamp != frame['image_calibration_ns'] or frame['capture_ns'] > row['query_ns']:
        raise ValueError('Noncausal or mismatched camera calibration')
      key = (camera, index, stamp)
      if key not in self.images:
        raw = np.load(self.raw_cache/str(camera)/f'{index}.npy', allow_pickle=False, mmap_mode='r')
        self.images[key] = calibrated_image(raw, camera, self.source, self.target, calibration)
        misses += 1
      else:
        hits += 1
      images.append(self.images[key][None])
      self.images.move_to_end(key)
      while len(self.images) > 144:
        self.images.popitem(last=False)
    pixels = np.stack(images)
    return pixels, dict(native_calibration_ms=(time.perf_counter()-start)*1000,
                        calibrated_frame_hits=hits, calibrated_frame_misses=misses)

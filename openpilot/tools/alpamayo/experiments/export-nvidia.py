"""Freeze real six-camera SDK inputs for identical native and NVIDIA replay."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from physical_ai_av import PhysicalAIAVDatasetInterface
from alpamayo2_edge.load_physical_aiavdataset import load_physical_aiavdataset


class CameraCadence:
  """Expose every Nth real frame, selecting the latest allowed capture causally."""
  def __init__(self, source, hz):
    self.source = source
    rate = 1e6/float(np.median(np.diff(source.timestamps)))
    self.stride = round(rate/hz)
    if self.stride < 1 or abs(rate/self.stride-hz) > 0.05:
      raise ValueError(f'Cannot obtain {hz} Hz by integer decimation of {rate} Hz')
    self.timestamps = source.timestamps[::self.stride]

  def decode_images_from_timestamps(self, requested):
    indices = np.searchsorted(self.timestamps,requested,side='right')-1
    if np.any(indices < 0):
      raise ValueError('Requested history precedes the retained camera frames')
    selected = self.timestamps[indices]
    images, actual = self.source.decode_images_from_timestamps(selected)
    np.testing.assert_array_equal(actual,selected)
    assert np.all(actual <= requested)
    return images, actual


class CachedDataset:
  def __init__(self, revision, camera_hz=None):
    self.source = PhysicalAIAVDatasetInterface(revision=revision)
    self.features, self.revision = self.source.features, revision
    self.cache, self.camera_rates = {}, {}
    self.camera_hz = camera_hz

  def get_clip_feature(self, clip, feature, **kwargs):
    key = (clip, feature)
    if key not in self.cache:
      print(f'Loading {feature}', flush=True)
      value = self.source.get_clip_feature(clip, feature, **kwargs)
      if str(feature).startswith('camera'):
        original_rate = 1e6/float(np.median(np.diff(value.timestamps)))
        if self.camera_hz:
          value = CameraCadence(value,self.camera_hz)
        stamps = np.asarray(value.timestamps)
        intervals = np.diff(stamps)
        self.camera_rates[feature] = dict(frames=len(stamps), first_us=int(stamps[0]), last_us=int(stamps[-1]),
                                         median_interval_us=float(np.median(intervals)),
                                         median_fps=float(1e6/np.median(intervals)), source_fps=original_rate)
      self.cache[key] = value
    return self.cache[key]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--clip', default='ea7bbd31-b7a5-4972-8dbd-7089e6b53de4')
  parser.add_argument('--revision', default='14ed7a77897edafb113dc86fde8b8173ae7fb028')
  parser.add_argument('--times-us', nargs='+', type=int, default=[4000000])
  parser.add_argument('--nav-text', default='Turn right in 30m')
  parser.add_argument('--camera-hz', type=float, help='Causal integer decimation of the real camera stream')
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(4)
  avdi = CachedDataset(args.revision,args.camera_hz)
  for timestamp in args.times_us:
    started = time.perf_counter()
    data = load_physical_aiavdataset(args.clip, timestamp, nav_text=args.nav_text, avdi=avdi, maybe_stream=True)
    output = args.output/f'sample-{timestamp}'
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for key, value in data.items():
      if isinstance(value, torch.Tensor):
        path = output/f'{key}.npy'
        np.save(path, value.numpy(), allow_pickle=False)
        hashes[path.name] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    np.save(output/'nav_text.npy', np.array(args.nav_text), allow_pickle=False)
    ego = avdi.get_clip_feature(args.clip, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
    future_times = timestamp + np.arange(1,65,dtype=np.int64)*100000
    if future_times[-1] <= ego.time_range[1]:
      current, future = ego(np.array([timestamp], dtype=np.int64)), ego(future_times)
      xyz = current.pose.rotation[0].inv().apply(future.pose.translation-current.pose.translation[0])
      np.save(output/'reference_xyz.npy', xyz.astype(np.float32), allow_pickle=False)
    metadata = {key:value for key,value in data.items() if not isinstance(value, torch.Tensor)}
    metadata.update(camera_rates=avdi.camera_rates, sha256=hashes,
                    shape=list(data['image_frames'].shape), export_seconds=time.perf_counter()-started,
                    sampling='NVIDIA loader with causal camera decimation' if args.camera_hz else 'Unmodified NVIDIA loader and native camera capture cadence',
                    requested_camera_hz=args.camera_hz)
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(sample=str(output), **metadata)), flush=True)


if __name__ == '__main__':
  main()

"""Check exact bicubic pixels and CPU cost for current six-camera cache misses."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torchvision.transforms.functional import resize, InterpolationMode


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sample', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(4)
  frames = torch.from_numpy(np.load(args.sample/'image_frames.npy', mmap_mode='c')).squeeze(1)
  cameras = np.load(args.sample/'camera_indices.npy')
  groups = [([int(i) for camera in (1, 6) for i in np.flatnonzero(cameras == camera)[-2:]], (320, 576)),
            ([int(i) for camera in (0, 2, 3, 5) for i in np.flatnonzero(cameras == camera)[-2:]], (224, 416))]
  results = []
  for indices, size in groups:
    source = [frames[i] for i in indices]
    reference = None
    for batched in (False, True):
      times = []
      for iteration in range(7):
        begin = time.perf_counter()
        if batched:
          output = resize(torch.stack(source), list(size), interpolation=InterpolationMode.BICUBIC)
        else:
          output = torch.stack([resize(frame, list(size), interpolation=InterpolationMode.BICUBIC) for frame in source])
        elapsed = (time.perf_counter()-begin)*1000
        if iteration >= 2:
          times.append(elapsed)
      if reference is None:
        reference = output.clone()
      else:
        assert torch.equal(output, reference), 'Batched resize changed source pixels'
      record = dict(images=len(indices), size=size, batched=batched, median_ms=float(np.median(times)), exact=True)
      results.append(record)
      args.output.write_text(json.dumps(results, indent=2)+'\n')
      print(json.dumps(record), flush=True)


if __name__ == '__main__':
  main()

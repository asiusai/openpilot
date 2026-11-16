"""Prepare and draw a recorded GPS map inside the exported route video."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageOps

SIZE = (320, 250)


def gps_points(logs, start_ns, duration):
  from openpilot.cereal import log
  import zstandard
  sources = defaultdict(dict)
  for path in sorted(logs.glob('*/rlog.zst')):
    with path.open('rb') as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoder:
      raw = decoder.read()
    for event in log.Event.read_multiple_bytes(raw):
      kind = event.which()
      if not event.valid or kind not in ('gpsLocationExternal', 'gpsLocation'):
        continue
      fix = getattr(event, kind)
      values = [fix.latitude, fix.longitude, fix.horizontalAccuracy]
      if (not fix.hasFix or not all(map(math.isfinite, values)) or not -90 < fix.latitude < 90
          or not -180 <= fix.longitude <= 180 or fix.horizontalAccuracy <= 0 or fix.unixTimestampMillis <= 0):
        continue
      sources[kind].setdefault(int(fix.unixTimestampMillis), {
        'time_s': (int(event.logMonoTime)-start_ns)/1e9, 'latitude': fix.latitude, 'longitude': fix.longitude,
        'accuracy_m': fix.horizontalAccuracy, 'utc_ms': int(fix.unixTimestampMillis), 'source': str(fix.source)})
  source = next((k for k in ('gpsLocationExternal', 'gpsLocation') if sources[k]), None)
  if source is None:
    raise ValueError('No valid recorded GPS fixes')
  selected = [p for p in sorted(sources[source].values(), key=lambda p: p['time_s']) if -1 <= p['time_s'] <= duration+1]
  if len(selected) < 2:
    raise ValueError('No GPS coverage for the video interval')
  return {'message': source, 'points': selected,
          'timing': 'GPS publication logMonoTime aligned to camera capture time; measurement latency is not corrected',
          'interpolation': 'Consecutive recorded fixes at most one second apart; stale marker hidden',
          'map': 'OpenStreetMap, https://www.openstreetmap.org/copyright'}


def world(latitude, longitude):
  lat = np.deg2rad(np.clip(latitude, -85.05112878, 85.05112878))
  return np.stack(((np.asarray(longitude)+180)/360, (1-np.log(np.tan(lat)+1/np.cos(lat))/np.pi)/2), axis=-1)


def prepare(args):
  from camera_profiles import replay_module
  replay = replay_module()
  replay.check_mount(args.output, args.recording_uuid)
  manifest = replay.read_json(args.output/'manifest.json')
  data = gps_points(args.logs, manifest['capture_start_ns'], manifest['duration_s'])
  points = data['points']
  coordinates = world([p['latitude'] for p in points], [p['longitude'] for p in points])
  low, high = coordinates.min(axis=0), coordinates.max(axis=0)
  center = (low+high)/2
  zoom = min(18., max(3., float(np.log2(np.min((np.array(SIZE)-[56, 72])/(256*np.maximum(high-low, 1e-8)))))))
  tile_zoom = min(19, math.ceil(zoom)+1)
  scale = 256*2**tile_zoom
  extent = np.array(SIZE)/(256*2**zoom)
  top, bottom = (center-extent/2)*scale, (center+extent/2)*scale
  first, last = np.floor(top/256).astype(int), np.floor(bottom/256).astype(int)
  tile_count = int(np.prod(last-first+1))
  if tile_count > 36:
    raise ValueError('Unexpected map tile coverage')
  canvas = Image.new('RGB', tuple((last-first+1)*256))
  args.tile_cache.mkdir(parents=True, exist_ok=True)
  tiles = []
  for y in range(first[1], last[1]+1):
    for x in range(first[0], last[0]+1):
      cache = args.tile_cache/f'{tile_zoom}-{x}-{y}.png'
      if not cache.exists():
        request = urllib.request.Request(f'https://tile.openstreetmap.org/{tile_zoom}/{x}/{y}.png',
                                         headers={'User-Agent': 'AsiusRouteVideo/1.0 (https://asius.ai)'})
        with urllib.request.urlopen(request, timeout=30) as response:
          payload = response.read()
        with Image.open(io.BytesIO(payload)) as image:
          image.load()
          if image.size != (256, 256):
            raise ValueError('Invalid map tile')
        cache.write_bytes(payload)
      with Image.open(cache) as tile:
        canvas.paste(tile.convert('RGB'), ((x-first[0])*256, (y-first[1])*256))
      tiles.append({'z': tile_zoom, 'x': int(x), 'y': int(y), 'sha256': hashlib.sha256(cache.read_bytes()).hexdigest()})
  box = tuple(np.concatenate((top-first*256, bottom-first*256)))
  background = canvas.transform((SIZE[0]*2, SIZE[1]*2), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC)
  background = ImageOps.colorize(ImageOps.invert(ImageOps.grayscale(background)), '#121c27', '#a3b0bd')
  replay.check_mount(args.output, args.recording_uuid)
  background.save(args.output/'map-background.png')
  replay.write_json(args.output/'gps-route.json', data)
  replay.write_json(args.output/'map-layout.json', {'size': SIZE, 'center': center.tolist(), 'zoom': zoom,
    'tiles': tiles, 'attribution': '© OpenStreetMap contributors', 'copyright_url': 'https://www.openstreetmap.org/copyright',
    'attribution_location': 'Accompanying publication description, provided in post-caption.txt'})
  print(f'Prepared in-video map: {len(points)} GPS fixes, {tile_count} map tiles', flush=True)


class RouteMap:
  def __init__(self, output):
    self.data = json.loads((output/'gps-route.json').read_text())
    layout = json.loads((output/'map-layout.json').read_text())
    self.times = np.array([p['time_s'] for p in self.data['points']])
    if np.any(np.diff(self.times) <= 0):
      raise ValueError('GPS timestamps must increase')
    p = self.data['points']
    self.xy = (world([v['latitude'] for v in p], [v['longitude'] for v in p])-layout['center'])*256*2**layout['zoom']+np.array(SIZE)/2
    self.background = Image.open(output/'map-background.png').convert('RGB')
    self.missing = 0
    self.mask = Image.new('L', self.background.size)
    ImageDraw.Draw(self.mask).rounded_rectangle((0, 0, self.mask.width-1, self.mask.height-1), radius=32, fill=255)

  def position(self, seconds):
    index = int(np.searchsorted(self.times, seconds, side='right'))-1
    if index < 0 or seconds-self.times[index] > 1:
      return None
    if index+1 < len(self.times) and self.times[index+1]-self.times[index] <= 1:
      fraction = np.clip((seconds-self.times[index])/(self.times[index+1]-self.times[index]), 0, 1)
      return self.xy[index]+fraction*(self.xy[index+1]-self.xy[index]), index
    return self.xy[index], index

  def frame(self, seconds):
    image = self.background.copy()
    draw = ImageDraw.Draw(image)
    position = self.position(seconds)
    for i in range(1, len(self.xy)):
      if self.times[i]-self.times[i-1] > 1:
        continue
      segment = [tuple(self.xy[i-1]*2), tuple(self.xy[i]*2)]
      draw.line(segment, fill='#08101a', width=12)
      draw.line(segment, fill='#ed655c' if self.times[i] <= seconds else '#8093a9', width=6)
    if position is not None:
      point, index = position
      if index+1 < len(self.times) and self.times[index+1]-self.times[index] <= 1:
        draw.line([tuple(self.xy[index]*2), tuple(point*2)], fill='#ed655c', width=6)
      x, y = point*2
      draw.ellipse((x-16, y-16, x+16, y+16), fill='#ffffff', outline='#111a25', width=3)
      draw.ellipse((x-8, y-8, x+8, y+8), fill='#ed655c')
    else:
      self.missing += 1
    image.putalpha(self.mask)
    return image.resize(SIZE, Image.Resampling.LANCZOS)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--logs', type=Path, required=True)
  parser.add_argument('--tile-cache', type=Path, required=True)
  parser.add_argument('--recording-uuid')
  prepare(parser.parse_args())

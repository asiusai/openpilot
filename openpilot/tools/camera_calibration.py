#!/usr/bin/env python3
"""Camera calibration bench for Asius v0: live view and lens alignment readouts in a browser.

Run on the device, then open the printed address on any computer or phone in the same network:

  cd /data/openpilot && python3 openpilot/tools/camera_calibration.py

The page shows the road, wide and driver cameras and measures, live:
  - focus: sharpness in a 5 x 3 grid, with the best value seen, and the left/right and top/bottom balance (lens tilt)
  - optical centre: where the lens axis meets the sensor, against the frame centre
      wide and driver (fisheye): the centre of the image circle, from the dark corners. Needs a bright scene.
      road: the centre of the brightness fall-off. Needs an evenly lit, plain surface that fills the view.
  - road to wide angle: yaw, pitch and roll between the two views. Needs a scene more than 10 m away.
  - frame sync: start-of-frame offsets between the cameras
While parked, the tool asks the manager for the camera preview (IsDriverViewEnabled) and removes the request on exit.
Without a manager it starts camerad itself. It needs only numpy and Pillow from the openpilot environment.
"""
import argparse
import atexit
import io
import json
import math
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

W, H = 1344, 760
F_ROAD = 1141.5   # px, pinhole (openpilot _os_config)
F_WIDE = 425.25   # px, equidistant fisheye (openpilot _os_fisheye); the driver camera has the same lens
PIXEL_MM = 0.004  # one output pixel on the sensor (2 um pixels, binned 2 x 2)
GX, GY = 5, 3     # sharpness grid
# Acceptance limits: optical centre within 0.08 mm (road) and 0.04 mm (wide), road to wide angle within 1 degree,
# sharpness balance at least 0.8. Edge width in pixels: 2.0 is a perfect edge.
LIMITS = {'road_centre_px': 20, 'wide_centre_px': 10, 'pair_deg': 1.0, 'balance': 0.8, 'sharp_px': 2.6, 'soft_px': 3.2}
BASEDIR = Path(__file__).resolve().parents[2]


def box_mean(a, k):
  """Mean over k x k blocks (the image is cut to a multiple of k)."""
  h, w = a.shape[0]//k*k, a.shape[1]//k*k
  return a[:h, :w].reshape(h//k, k, w//k, k).mean(axis=(1, 3))


def blur3(a, n=1):
  """n passes of a 3 x 3 box blur with edge replication."""
  for _ in range(n):
    p = np.pad(a, 1, mode='edge')
    a = (p[:-2, :-2]+p[:-2, 1:-1]+p[:-2, 2:]+p[1:-1, :-2]+p[1:-1, 1:-1]+p[1:-1, 2:]+p[2:, :-2]+p[2:, 1:-1]+p[2:, 2:])/9.
  return a


def detail(a):
  """Fine detail of 1 to 2 px: each pixel against its four neighbours, squared."""
  out = np.zeros_like(a)
  out[1:-1, 1:-1] = (a[1:-1, 1:-1]-.25*(a[:-2, 1:-1]+a[2:, 1:-1]+a[1:-1, :-2]+a[1:-1, 2:]))**2
  return out


def sharpness(y, previous=None):
  """Fine detail against mid-size detail in each grid cell. The ratio depends little on the scene content.

  Sensor noise looks like fine detail. With the previous picture of a still scene, the noise part is measured from the
  difference of the two pictures and removed. Returns the grid (None where a cell has too little texture), the
  fine-detail level of the centre, and the picture as float for the next call.
  """
  a = y.astype(np.float32)
  fine = detail(a)
  noise = detail(a-previous)/2. if previous is not None and previous.shape == a.shape else None
  q = box_mean(a, 4)
  mid = (blur3(q, 1)-blur3(q, 4))**2           # detail of about 10 to 30 px
  def net(rows, cols):
    f = float(fine[rows, cols].mean())
    return max(f-float(noise[rows, cols].mean()), 0.) if noise is not None else f
  ch, cw = H//GY, W//GX
  grid = []
  for gy in range(GY):
    row = []
    for gx in range(GX):
      f = net(slice(gy*ch, (gy+1)*ch), slice(gx*cw, (gx+1)*cw))
      m = float(mid[gy*ch//4:(gy+1)*ch//4, gx*cw//4:(gx+1)*cw//4].mean())
      row.append(round(100*f/m, 1) if m > 1.5 else None)
    grid.append(row)
  centre = 10*math.sqrt(net(slice(H//2-100, H//2+100), slice(W//2-160, W//2+160)))
  return grid, centre, a


def edge_widths(a):
  """Width in pixels of the strongest edges in each grid cell: edge contrast over its steepest slope.

  The value depends little on the scene: a perfect edge gives 2.0, a sharp lens 2.0 to 2.6, a soft one more than 3.2.
  Returns the grid (None where a cell has no clear edge) and the value of the centre region.
  """
  s = (a[:, :-2]+2*a[:, 1:-1]+a[:, 2:])/4.
  s = (s[:-2]+2*s[1:-1]+s[2:])/4.            # light smoothing against sensor noise
  dx, dy = np.zeros_like(s), np.zeros_like(s)
  dx[:, 1:-1], dy[1:-1] = (s[:, 2:]-s[:, :-2])/2., (s[2:]-s[:-2])/2.
  g = np.hypot(dx, dy)
  h, w = s.shape
  t = np.arange(-5, 6, dtype=np.float32)
  def cell(y0, y1, x0, x1, count=48):
    gg = g[y0:y1, x0:x1]
    flat = gg.ravel()
    idx = np.argpartition(flat, -count*6)[-count*6:][::6]   # spread over the strongest points
    idx = idx[flat[idx] > 3.]
    if len(idx) < 8:
      return None
    py, px = np.unravel_index(idx, gg.shape)
    py, px = py+y0, px+x0
    ux, uy = dx[py, px]/g[py, px], dy[py, px]/g[py, px]
    xx = np.clip(np.rint(px[:, None]+t[None, :]*ux[:, None]).astype(np.int32), 0, w-1)
    yy = np.clip(np.rint(py[:, None]+t[None, :]*uy[:, None]).astype(np.int32), 0, h-1)
    prof = s[yy, xx]
    contrast = prof.max(1)-prof.min(1)
    slope = np.abs(np.diff(prof, axis=1)).max(1)
    good = contrast > 20.
    return round(float(np.median(contrast[good]/slope[good])), 2) if good.sum() >= 8 else None
  ch, cw = h//GY, w//GX
  grid = [[cell(gy*ch+4, (gy+1)*ch-4, gx*cw+4, (gx+1)*cw-4) for gx in range(GX)] for gy in range(GY)]
  return grid, cell(h//2-140, h//2+140, w//2-220, w//2+220, count=96)


def balance(grid):
  """Left/right and top/bottom ratios of the edge widths (sharper side over softer side) and which side is softer."""
  def ratio(a, b, names):
    a, b = [v for v in a if v is not None], [v for v in b if v is not None]
    if not a or not b:
      return None
    a, b = float(np.mean(a)), float(np.mean(b))
    return {'ratio': round(min(a, b)/max(a, b), 2), 'soft': names[0] if a > b else names[1]}
  left = [r[c] for r in grid for c in (0, 1)]
  right = [r[c] for r in grid for c in (GX-2, GX-1)]
  return {'lr': ratio(left, right, ('left', 'right')), 'tb': ratio(grid[0], grid[-1], ('top', 'bottom'))}


def fit_circle(points):
  p = np.array(points, float)
  a = np.c_[2*p[:, 0], 2*p[:, 1], np.ones(len(p))]
  sol, *_ = np.linalg.lstsq(a, (p**2).sum(1), rcond=None)
  cx, cy = sol[:2]
  r = math.sqrt(max(sol[2]+cx*cx+cy*cy, 0.))
  return cx, cy, r, np.hypot(p[:, 0]-cx, p[:, 1]-cy)-r


def image_circle(y):
  """Centre of a fisheye lens's image circle from the edges of the dark corners (top and bottom rows)."""
  k = 4
  s = blur3(y[1::k, 1::k].astype(np.float32), 2)   # every fourth pixel; small picture x is full picture x = 4 x + 1
  h, w = s.shape
  side = w//5  # the dark corners are inside the outer fifth of the row
  points = []
  for row in [*range(1, h//3), *range(h-h//3, h-1)]:
    line = s[row]
    for left in (True, False):
      seg = line[:side+15] if left else line[::-1][:side+15]
      ref = float(np.median(seg[side:side+15]))
      if ref < 40:
        continue
      below = np.flatnonzero(seg[:side] < .5*ref)
      if not len(below) or below[-1] >= side-2 or below[-1] < 1:
        continue
      x = below[-1]
      x = x+(.5*ref-seg[x])/max(seg[x+1]-seg[x], 1e-3)
      x = x if left else (w-1)-x
      points.append((x*k+1., row*k+1.))
  # The fit needs all four corners: one side alone does not fix the centre.
  corners = [sum(1 for x, y in points if (x < W/2) == left and (y < H/2) == top) for left in (True, False) for top in (True, False)]
  if min(corners) < 4:
    # A one-sided estimate is not used: dark objects and bright lights near the edge move it by 50 px and more.
    return {'ok': False, 'msg': 'Dark corners not found in all four corners. Point the camera at a bright, plain scene.',
            'points': [[round(a), round(b)] for a, b in points[::3]]}
  cx, cy, r, res = fit_circle(points)
  keep = np.abs(res) < max(2.5*res.std(), 2.)
  if keep.sum() >= 12:
    points = [p for p, good in zip(points, keep, strict=True) if good]
    cx, cy, r, res = fit_circle(points)
  rms = float(res.std())
  out = centre_result(cx, cy, F_WIDE, fisheye=True)
  out.update(ok=rms < 6., circle=[round(float(cx), 1), round(float(cy), 1), round(r, 1)], rms=round(rms, 1),
             points=[[round(a), round(b)] for a, b in points[::3]],
             msg='' if rms < 6. else 'The corner edges are uneven. Use a bright, plain scene.')
  return out


def falloff_centre(y):
  """Centre of the brightness fall-off of a rectilinear lens on an evenly lit, plain surface."""
  k = 16
  s = box_mean(y.astype(np.float32), k)
  h, w = s.shape
  yy, xx = np.mgrid[:h, :w].astype(np.float64)
  x, z = ((xx+.5)*k-.5-(W-1)/2)/W, ((yy+.5)*k-.5-(H-1)/2)/W
  good = (s > 30) & (s < 245)
  if good.mean() < .8:
    return {'ok': False, 'msg': 'Too dark or overexposed for the fall-off fit.'}
  a = np.c_[np.ones(good.sum()), x[good], z[good], x[good]**2, z[good]**2, x[good]*z[good]]
  sol, *_ = np.linalg.lstsq(a, s[good], rcond=None)
  resid = float((a@sol-s[good]).std()/sol[0])
  hess = np.array([[2*sol[3], sol[5]], [sol[5], 2*sol[4]]])
  drop = float(-(sol[3]*.25+sol[4]*(H/W/2)**2)/sol[0])  # fall-off at the frame edges against the centre
  if not (hess[0, 0] < 0 and np.linalg.det(hess) > 0):
    return {'ok': False, 'msg': 'No brightness fall-off found. Fill the view with an evenly lit, plain surface.', 'flatness': round(100*resid, 1)}
  px, pz = np.linalg.solve(hess, -sol[1:3])*W
  out = centre_result(px+(W-1)/2, pz+(H-1)/2, F_ROAD, fisheye=False)
  ok = resid < .03 and drop > .02 and abs(px) < W/3 and abs(pz) < H/3
  out.update(ok=bool(ok), flatness=round(100*resid, 1), falloff=round(100*drop, 1),
             msg='' if ok else 'The scene is not plain enough. Fill the view with an evenly lit white surface (wall, paper, diffuser).')
  return out


def centre_result(cx, cy, focal, fisheye):
  dx, dy = cx-(W-1)/2, cy-(H-1)/2
  angle = (lambda d: math.degrees(d/focal)) if fisheye else (lambda d: math.degrees(math.atan(d/focal)))
  cx, cy, dx, dy = float(cx), float(cy), float(dx), float(dy)
  return {'cx': round(cx, 1), 'cy': round(cy, 1), 'dx': round(dx, 1), 'dy': round(dy, 1), 'offset_px': round(math.hypot(dx, dy), 1),
          'mm': [round(dx*PIXEL_MM, 3), round(dy*PIXEL_MM, 3)], 'deg': [round(angle(dx), 2), round(angle(dy), 2)]}


class PairFit:
  """Angle between the road and wide views: both are drawn as one small pinhole view and compared by phase correlation."""
  W0, H0 = 448, 256

  def __init__(self):
    yy, xx = np.mgrid[:self.H0, :self.W0].astype(np.float64)
    dx, dy = (xx-(self.W0-1)/2)/F_WIDE, (yy-(self.H0-1)/2)/F_WIDE
    self.road_map = self.bilinear(F_ROAD/2*dx+(W/2-1)/2, F_ROAD/2*dy+(H/2-1)/2, W//2, H//2)   # from the half-size road image
    rad = np.hypot(dx, dy)
    k = np.divide(np.arctan(rad), rad, out=np.ones_like(rad), where=rad > 1e-12)
    self.wide_map = self.bilinear(F_WIDE*dx*k+(W-1)/2, F_WIDE*dy*k+(H-1)/2, W, H)
    wy, wx = np.hanning(self.H0), np.hanning(self.W0)
    self.window = wy[:, None]*wx[None, :]

  @staticmethod
  def bilinear(x, y, w, h):
    x, y = np.clip(x, 0, w-1.001), np.clip(y, 0, h-1.001)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = (x-x0).astype(np.float32), (y-y0).astype(np.float32)
    return y0, x0, fx, fy

  @staticmethod
  def sample(img, m):
    y0, x0, fx, fy = m
    return img[y0, x0]*(1-fx)*(1-fy)+img[y0, x0+1]*fx*(1-fy)+img[y0+1, x0]*(1-fx)*fy+img[y0+1, x0+1]*fx*fy

  @staticmethod
  def normalise(a):
    hp = blur3(a, 1)-blur3(a, 6)
    return hp/np.sqrt(blur3(hp*hp, 6)+4.)

  def shift(self, a, b):
    """Sub-pixel (dx, dy) with b(x+dx, y+dy) ~ a(x, y), and the correlation peak height."""
    fa, fb = np.fft.rfft2(a), np.fft.rfft2(b)
    x = np.conj(fa)*fb
    c = np.fft.irfft2(x/(np.abs(x)+1e-6), s=a.shape)
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    def sub(v0, v1, v2):
      den = v0-2*v1+v2
      return .5*(v0-v2)/den if den < 0 else 0.
    h, w = c.shape
    sx = ix+sub(c[iy, ix-1], c[iy, ix], c[iy, (ix+1) % w])
    sy = iy+sub(c[iy-1, ix], c[iy, ix], c[(iy+1) % h, ix])
    return (sx-w if sx > w/2 else sx), (sy-h if sy > h/2 else sy), float(c[iy, ix])

  def measure(self, road, wide):
    r = self.normalise(self.sample(box_mean(road.astype(np.float32), 2), self.road_map))
    w = self.normalise(self.sample(wide.astype(np.float32), self.wide_map))
    dx, dy, peak = self.shift(r*self.window, w*self.window)
    if peak < .04 or abs(dx) > self.W0/4 or abs(dy) > self.H0/4:
      return {'ok': False, 'msg': 'The two views do not match. Use a detailed scene more than 10 m away.', 'peak': round(peak, 3)}
    # Roll from the vertical shift of the left and right halves, after the common shift is removed.
    half = self.W0//2
    wl = np.hanning(half)[None, :]*np.hanning(self.H0)[:, None]
    ws = np.roll(w, (-int(round(dy)), -int(round(dx))), axis=(0, 1))
    _, dyl, pl = self.shift(r[:, :half]*wl, ws[:, :half]*wl)
    _, dyr, pr = self.shift(r[:, half:]*wl, ws[:, half:]*wl)
    roll = math.degrees(math.atan((dyr-dyl)/half)) if min(pl, pr) > .03 and max(abs(dyl), abs(dyr)) < 12 else None
    # A scene point that is further right (down) in the wide view means that the wide camera looks further left (up).
    yaw, pitch = -math.degrees(math.atan(dx/F_WIDE)), math.degrees(math.atan(dy/F_WIDE))
    return {'ok': True, 'msg': '', 'yaw': round(yaw, 2), 'pitch': round(pitch, 2), 'roll': None if roll is None else round(roll, 2),
            'peak': round(peak, 3), 'shift_px': [round(float(dx), 1), round(float(dy), 1)]}


class Camera(threading.Thread):
  """Holds the newest frame of one camera and its measurements."""

  def __init__(self, key, title, stream, fisheye, stop):
    super().__init__(daemon=True, name=f'cam-{key}')
    self.key, self.title, self.stream, self.fisheye, self.stop = key, title, stream, fisheye, stop
    self.lock = threading.Lock()
    self.y = self.uv = None
    self.frame_id = 0
    self.frame_time = 0.
    self.times = []
    self.measure = {}
    self.peak = 0.
    self.selected = key == 'road'
    self.previous = None
    self.best_width = None
    self.preview_cache = (None, None)

  def run(self):
    from msgq.visionipc import VisionIpcClient
    client, last_measure = None, 0.
    while not self.stop.is_set():
      try:
        if client is None:
          client = VisionIpcClient('camerad', self.stream, True)
        if not client.is_connected() and not client.connect(False):
          time.sleep(.5)
          continue
        buf = client.recv(500)
        if buf is None:
          continue
        data = np.asarray(buf.data, dtype=np.uint8)
        y = data[:buf.uv_offset].reshape(-1, buf.stride)[:buf.height, :buf.width].copy()
        uv = data[buf.uv_offset:buf.uv_offset+buf.stride*(buf.height//2)].reshape(-1, buf.stride)[:, :buf.width].copy()
        now = time.monotonic()
        with self.lock:
          self.y, self.uv, self.frame_id, self.frame_time = y, uv, client.frame_id, now
          self.times = [t for t in self.times if now-t < 2.]+[now]
        if now-last_measure > (.25 if self.selected else 1.) and y.shape == (H, W):
          last_measure = now
          self.update_measure(y)
        time.sleep(.07 if self.selected else .25)  # the selected camera is copied about 10 times per second
      except Exception as error:
        self.measure = {'error': f'{type(error).__name__}: {error}'}
        client = None
        time.sleep(1.)

  def update_measure(self, y):
    _, level, self.previous = sharpness(y, self.previous)
    grid, width = edge_widths(self.previous)
    self.peak = max(self.peak, level)
    if width is not None:
      self.best_width = width if self.best_width is None else min(self.best_width, width)
    m = {'grid': grid, 'focus': round(level, 2), 'peak': round(self.peak, 2), 'balance': balance(grid), 'width': width, 'best_width': self.best_width,
         'mean': round(float(y[::8, ::8].mean()), 1), 'clipped': round(float((y[::8, ::8] > 250).mean())*100, 1)}
    try:
      m['centre'] = image_circle(y) if self.fisheye else falloff_centre(y)
    except Exception as error:
      m['centre'] = {'ok': False, 'msg': f'{type(error).__name__}: {error}'}
    self.measure = m

  def live(self):
    return self.y is not None and time.monotonic()-self.frame_time < 1.5

  def state(self):
    with self.lock:
      fps = (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times) > 2 else 0.
    return {'title': self.title, 'live': self.live(), 'fps': round(fps, 1), 'frame_id': self.frame_id, 'fisheye': self.fisheye, **self.measure}

  def jpeg(self, zoom=None, quality=78):
    """Half-size colour view, or a full-resolution grey crop around `zoom` (x, y, factor) for focusing."""
    with self.lock:
      y, uv, frame_id = self.y, self.uv, self.frame_id
    if y is None:
      return None
    key = (frame_id, zoom)
    if self.preview_cache[0] == key:
      return self.preview_cache[1]
    out = io.BytesIO()
    if zoom is not None:
      factor = int(zoom[2]) if len(zoom) > 2 and int(zoom[2]) in (2, 4, 8) else 2
      cw, ch = y.shape[1]//factor, y.shape[0]//factor
      x0 = int(min(max(zoom[0]-cw//2, 0), y.shape[1]-cw))
      y0 = int(min(max(zoom[1]-ch//2, 0), y.shape[0]-ch))
      crop = np.ascontiguousarray(y[y0:y0+ch, x0:x0+cw])
      if factor > 2:  # enlarge on the device without smoothing, so single sensor pixels stay visible
        crop = np.repeat(np.repeat(crop, factor//2, axis=0), factor//2, axis=1)
      Image.fromarray(crop).save(out, 'JPEG', quality=92)
    else:
      yy = y[::2, ::2].astype(np.float32)
      u, v = uv[:yy.shape[0], 0::2].astype(np.float32)-128., uv[:yy.shape[0], 1::2].astype(np.float32)-128.
      rgb = np.stack([yy+1.402*v, yy-.344*u-.714*v, yy+1.772*u], -1)
      Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(out, 'JPEG', quality=quality)
    self.preview_cache = (key, out.getvalue())
    return self.preview_cache[1]


class Bench:
  def __init__(self, args):
    from openpilot.cereal.visionipc import VisionStreamType
    self.args = args
    self.stop = threading.Event()
    self.cams = {
      'road': Camera('road', 'Road', VisionStreamType.VISION_STREAM_NARROW_ROAD, False, self.stop),
      'wide': Camera('wide', 'Wide', VisionStreamType.VISION_STREAM_WIDE_ROAD, True, self.stop),
      'driver': Camera('driver', 'Driver', VisionStreamType.VISION_STREAM_CABIN, True, self.stop),
    }
    self.pair = {'ok': False, 'msg': 'Waiting for the road and wide cameras.'}
    self.sync = {}
    self.exposure = {}
    self.camera_source = 'waiting'
    self.camerad = None
    self.preview_requested = False
    self.started = time.monotonic()

  def start(self):
    for cam in self.cams.values():
      cam.start()
    for target in (self.pair_loop, self.state_loop, self.camera_loop):
      threading.Thread(target=target, daemon=True).start()

  # ----- camera source: a running camerad, the manager's preview, or a camerad started here

  def camera_loop(self):
    from openpilot.common.params import Params
    params = Params()
    while not self.stop.is_set():
      live = any(cam.live() for cam in self.cams.values())
      waited = time.monotonic()-self.started
      if live:
        self.camera_source = 'own camerad' if self.camerad else ('preview requested by this tool' if self.preview_requested else 'running camerad')
      elif self.camerad is None and not self.preview_requested and waited > 2. and not self.args.no_start:
        params.put_bool('IsDriverViewEnabled', True)
        self.preview_requested = True
        self.camera_source = 'asked the manager for the camera preview'
      elif self.camerad is None and waited > 14. and not self.args.no_start and not self.process_runs('camerad'):
        # No manager answered the preview request: run camerad here.
        self.camera_source = 'starting camerad'
        self.camerad = subprocess.Popen(['./camerad'], cwd=BASEDIR/'openpilot/system/camerad', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      self.stop.wait(1.)

  @staticmethod
  def process_runs(name):
    for path in Path('/proc').glob('[0-9]*/comm'):
      try:
        if path.read_text().strip() == name:
          return True
      except OSError:
        pass
    return False

  def close(self):
    self.stop.set()
    if self.preview_requested:
      try:
        from openpilot.common.params import Params
        Params().remove('IsDriverViewEnabled')
      except Exception:
        pass
    if self.camerad is not None:
      self.camerad.terminate()

  # ----- measurements across cameras

  def pair_loop(self):
    fit = PairFit()
    road, wide = self.cams['road'], self.cams['wide']
    history = []
    while not self.stop.is_set():
      self.stop.wait(.7)
      if not (road.live() and wide.live()):
        self.pair, history = {'ok': False, 'msg': 'Waiting for the road and wide cameras.'}, []
        continue
      try:
        with road.lock:
          a = road.y
        with wide.lock:
          b = wide.y
        result = fit.measure(a, b)
        if result['ok']:
          history = (history+[(result['yaw'], result['pitch'])])[-6:]
          spread = float(np.ptp([h[0] for h in history])+np.ptp([h[1] for h in history])) if len(history) > 2 else None
          result['steady'] = spread is not None and spread < .3
        else:
          history = []
        self.pair = result
      except Exception as error:
        self.pair = {'ok': False, 'msg': f'{type(error).__name__}: {error}'}

  def state_loop(self):
    """Frame timing and exposure from the camera state messages."""
    import openpilot.cereal.messaging as messaging
    names = {'road': 'narrowRoadCameraState', 'wide': 'wideRoadCameraState', 'driver': 'cabinCameraState'}
    sm = messaging.SubMaster(list(names.values()))
    sof = {k: {} for k in names}
    while not self.stop.is_set():
      sm.update(200)
      for key, name in names.items():
        if sm.updated[name]:
          msg = sm[name]
          sof[key][msg.frameId] = msg.timestampSof
          for old in [i for i in sof[key] if i < msg.frameId-40]:
            del sof[key][old]
          self.exposure[key] = {'lines': int(msg.integLines), 'gain': round(float(msg.gain), 2)}
      def offset(a, b):
        ids = sorted(set(sof[a]) & set(sof[b]))[-20:]
        return round(float(np.median([sof[a][i]-sof[b][i] for i in ids]))/1e6, 3) if len(ids) >= 5 else None
      self.sync = {'road_wide_ms': offset('road', 'wide'), 'driver_road_ms': offset('driver', 'road')}

  def state(self):
    return {'time': time.strftime('%H:%M:%S'), 'serial': serial(), 'camera_source': self.camera_source, 'limits': LIMITS, 'size': [W, H],
            'cams': {k: {**cam.state(), 'exposure': self.exposure.get(k)} for k, cam in self.cams.items()},
            'pair': self.pair, 'sync': self.sync}

  def save(self):
    """Write the readouts and one full-resolution picture per camera."""
    folder = Path(self.args.output)/serial()/time.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, exist_ok=True)
    state = self.state()
    for cam in state['cams'].values():
      if isinstance(cam.get('centre'), dict):
        cam['centre'].pop('points', None)
    (folder/'report.json').write_text(json.dumps(state, indent=1)+'\n')
    for key, cam in self.cams.items():
      with cam.lock:
        y = cam.y
      if y is not None:
        Image.fromarray(y).save(folder/f'{key}.jpg', 'JPEG', quality=92)
    return str(folder)


def serial():
  try:
    return Path('/sys/devices/soc0/serial_number').read_text().strip()
  except OSError:
    return socket.gethostname()


def addresses(port):
  found = []
  try:
    out = subprocess.check_output(['ip', '-4', '-o', 'addr', 'show', 'scope', 'global'], text=True)
    found = [line.split()[3].split('/')[0] for line in out.splitlines()]
  except (OSError, subprocess.CalledProcessError):
    pass
  return [f'http://{a}:{port}/' for a in found] or [f'http://{socket.gethostname()}:{port}/']


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera calibration</title>
<style>
:root { --bg:#14171a; --panel:#1f2428; --line:#343b42; --text:#e8ecef; --dim:#9aa5ae; --good:#4cc47a; --bad:#ef6a5b; --warn:#e8b64c; --accent:#5aa9e6; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif; }
header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:16px; margin:0 8px 0 0; }
header .info { color:var(--dim); font-size:13px; }
main { display:grid; grid-template-columns:minmax(0, 1fr) 360px; gap:16px; padding:16px; }
@media (max-width: 900px) { main { grid-template-columns:1fr; } }
.tabs { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
.tab { position:relative; border:2px solid var(--line); border-radius:8px; overflow:hidden; cursor:pointer; background:#000; width:176px; }
.tab.sel { border-color:var(--accent); }
.tab img { display:block; width:100%; aspect-ratio:1344/760; object-fit:cover; background:#000; }
.tab span { position:absolute; left:6px; top:4px; font-weight:600; text-shadow:0 0 4px #000; }
.tab i { position:absolute; right:6px; top:6px; width:10px; height:10px; border-radius:50%; background:var(--bad); }
.tab.live i { background:var(--good); }
.view { position:relative; background:#000; border-radius:8px; overflow:hidden; }
.view img { display:block; width:100%; aspect-ratio:1344/760; image-rendering:pixelated; }
.view canvas { position:absolute; inset:0; width:100%; height:100%; }
.bar { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0; align-items:center; }
button { background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:6px; padding:7px 12px; font:inherit; cursor:pointer; }
button.on { border-color:var(--accent); color:var(--accent); }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin-bottom:12px; }
.card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--dim); margin:0 0 8px; }
.row { display:flex; justify-content:space-between; gap:10px; padding:3px 0; }
.row b { font-variant-numeric:tabular-nums; font-weight:600; text-align:right; }
.good { color:var(--good); } .bad { color:var(--bad); } .warn { color:var(--warn); } .dim { color:var(--dim); }
.note { color:var(--dim); font-size:13px; margin-top:6px; }
.meter { height:10px; background:#0d0f11; border-radius:5px; overflow:hidden; margin:6px 0 2px; }
.meter div { height:100%; background:var(--accent); width:0; }
ol { margin:0; padding-left:18px; color:var(--dim); font-size:13px; }
</style></head><body>
<header><h1>Camera calibration</h1><span class="info" id="info"></span></header>
<main>
<section>
  <div class="tabs" id="tabs"></div>
  <div class="view"><img id="main" alt="camera view"><canvas id="overlay" width="1344" height="760"></canvas></div>
  <div class="bar">
    <button id="zoom">Zoom: off</button>
    <button id="grid" class="on">Sharpness grid</button>
    <button id="reset">Reset best focus</button>
    <button id="save">Save report</button>
    <span class="dim" id="saved"></span>
  </div>
  <div class="card"><h2>How to use</h2><ol>
    <li><b>Focus:</b> aim at a target with clear edges at the distance you need (road camera: more than 10 m).
      Turn the lens until the edge width is smallest, then lock it. Check that left/right and top/bottom are balanced on a flat target.</li>
    <li><b>Optical centre, wide and driver:</b> aim at a bright, plain wall. The circle must be found in all four corners.</li>
    <li><b>Optical centre, road:</b> fill the view with an evenly lit white surface.
      Move the lens holder until the offset is inside the limit, then tighten both screws in small alternate steps.</li>
    <li><b>Road to wide angle:</b> aim both cameras at a detailed scene more than 10 m away.
      At 2 m the cameras' distance from each other adds about 1 degree.</li>
  </ol></div>
</section>
<aside>
  <div class="card"><h2>Focus <span id="camname"></span></h2>
    <div class="row"><span>Edge width now (smaller is sharper)</span><b id="width">-</b></div>
    <div class="row"><span>Smallest seen</span><b id="bestwidth">-</b></div>
    <div class="row"><span>Detail level (for rough focus)</span><b id="focus">-</b></div>
    <div class="meter"><div id="focusbar"></div></div>
    <div class="row"><span>Highest detail level seen</span><b id="peak">-</b></div>
    <div class="row"><span>Left / right balance</span><b id="lr">-</b></div>
    <div class="row"><span>Top / bottom balance</span><b id="tb">-</b></div>
    <div class="note" id="focusnote"></div>
  </div>
  <div class="card"><h2>Optical centre</h2>
    <div class="row"><span>Offset from the frame centre</span><b id="coff">-</b></div>
    <div class="row"><span>Right (+) / left (-)</span><b id="cdx">-</b></div>
    <div class="row"><span>Down (+) / up (-)</span><b id="cdy">-</b></div>
    <div class="row"><span>As a view angle</span><b id="cdeg">-</b></div>
    <div class="row"><span>Limit</span><b id="clim">-</b></div>
    <div class="note" id="cnote"></div>
  </div>
  <div class="card"><h2>Road to wide angle</h2>
    <div class="row"><span>Yaw (wide view right of road: +)</span><b id="yaw">-</b></div>
    <div class="row"><span>Pitch (wide view above road: +)</span><b id="pitch">-</b></div>
    <div class="row"><span>Roll (wide picture turned clockwise: +)</span><b id="roll">-</b></div>
    <div class="note" id="pnote"></div>
  </div>
  <div class="card"><h2>Picture and timing</h2>
    <div class="row"><span>Pictures shown per second</span><b id="fps">-</b></div>
    <div class="row"><span>Brightness / overexposed</span><b id="bright">-</b></div>
    <div class="row"><span>Exposure lines / gain</span><b id="expo">-</b></div>
    <div class="row"><span>Road minus wide start</span><b id="s1">-</b></div>
    <div class="row"><span>Driver minus road start</span><b id="s2">-</b></div>
  </div>
</aside>
</main>
<script>
const $ = id => document.getElementById(id);
let sel = 'road', zoom = null, zf = 0, showGrid = true, state = null, lost = false;
const half = () => [672 / zf, 380 / zf];   // half of the zoomed window in sensor pixels
const clampZoom = (x, y) => [Math.min(Math.max(x, half()[0]), 1344 - half()[0]), Math.min(Math.max(y, half()[1]), 760 - half()[1])];
function zoomLabel() { $('zoom').textContent = zf ? 'Zoom: ' + zf + 'x (click the picture to move)' : 'Zoom: off'; $('zoom').classList.toggle('on', !!zf); }
const names = ['road', 'wide', 'driver'];
const tabs = $('tabs');
for (const k of names) {
  const d = document.createElement('div');
  d.className = 'tab'; d.id = 'tab-' + k;
  d.innerHTML = '<img alt=""><span></span><i></i>';
  d.onclick = () => { sel = k; zoom = null; zf = 0; zoomLabel(); setStream(); draw(); };
  tabs.appendChild(d);
}
function setStream() {
  const q = zoom ? '?zoom=' + Math.round(zoom[0]) + ',' + Math.round(zoom[1]) + ',' + zf : '';
  $('main').src = '/stream/' + sel + q;
  for (const k of names) $('tab-' + k).classList.toggle('sel', k === sel);
}
$('zoom').onclick = () => {
  zf = {0: 2, 2: 4, 4: 8, 8: 0}[zf];
  zoom = zf ? clampZoom(...(zoom || [671.5, 379.5])) : null;
  zoomLabel(); setStream(); draw();
};
$('grid').onclick = () => { showGrid = !showGrid; $('grid').classList.toggle('on', showGrid); draw(); };
$('reset').onclick = () => fetch('/api/reset', {method: 'POST'});
$('save').onclick = async () => {
  const r = await (await fetch('/api/save', {method: 'POST'})).json();
  $('saved').textContent = 'Saved on the device: ' + r.folder;
};
$('overlay').onclick = e => {
  if (!zoom) return;
  const r = e.target.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  zoom = clampZoom(zoom[0] + (x - .5) * 2 * half()[0], zoom[1] + (y - .5) * 2 * half()[1]);
  setStream(); draw();
};
function cls(el, c) { el.className = c || ''; }
function show(id, text, c) { const el = $(id); el.textContent = text; cls(el, c); }
function draw() {
  const c = $('overlay'), g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  if (!state) return;
  const cam = state.cams[sel];
  // picture coordinates: the whole frame, or the zoomed half-size window
  const map = zoom ? (x, y) => [(x - (zoom[0] - half()[0])) * zf, (y - (zoom[1] - half()[1])) * zf] : (x, y) => [x, y];
  const sc = zoom ? zf : 1;
  g.lineWidth = 2; g.font = '600 26px system-ui'; g.textAlign = 'center'; g.textBaseline = 'middle';
  // frame centre
  let [fx, fy] = map(671.5, 379.5);
  g.strokeStyle = 'rgba(255,255,255,.75)';
  g.beginPath(); g.moveTo(fx - 60, fy); g.lineTo(fx + 60, fy); g.moveTo(fx, fy - 60); g.lineTo(fx, fy + 60); g.stroke();
  if (showGrid && cam.grid && !zoom) {
    cam.grid.forEach((row, gy) => row.forEach((v, gx) => {
      const x = (gx + .5) * 1344 / 5, y = (gy + .5) * 760 / 3 + 70;
      g.fillStyle = v === null ? 'rgba(255,255,255,.4)' : (v <= state.limits.sharp_px ? '#4cc47a' : (v <= state.limits.soft_px ? '#e8b64c' : '#ef6a5b'));
      g.strokeStyle = 'rgba(0,0,0,.8)'; g.lineWidth = 5;
      const t = v === null ? '–' : v.toFixed(1);
      g.strokeText(t, x, y); g.fillText(t, x, y);
    }));
    g.strokeStyle = 'rgba(255,255,255,.18)'; g.lineWidth = 1;
    for (let i = 1; i < 5; i++) { g.beginPath(); g.moveTo(i * 1344 / 5, 0); g.lineTo(i * 1344 / 5, 760); g.stroke(); }
    for (let i = 1; i < 3; i++) { g.beginPath(); g.moveTo(0, i * 760 / 3); g.lineTo(1344, i * 760 / 3); g.stroke(); }
  }
  const ce = cam.centre;
  if (ce && ce.cx !== undefined) {
    const col = ce.ok ? '#5aa9e6' : 'rgba(232,182,76,.9)';
    const [x, y] = map(ce.cx, ce.cy);
    g.strokeStyle = col; g.lineWidth = 3;
    g.beginPath(); g.arc(x, y, 16, 0, 7); g.moveTo(x - 30, y); g.lineTo(x + 30, y); g.moveTo(x, y - 30); g.lineTo(x, y + 30); g.stroke();
    g.beginPath(); g.moveTo(fx, fy); g.lineTo(x, y); g.stroke();
    if (ce.circle) { g.lineWidth = 2; g.beginPath(); const [cx, cy] = map(ce.circle[0], ce.circle[1]); g.arc(cx, cy, ce.circle[2] * sc, 0, 7); g.stroke(); }
  }
  if (ce && ce.points) { g.fillStyle = '#e8b64c'; for (const p of ce.points) { const [x, y] = map(p[0], p[1]); g.fillRect(x - 3, y - 3, 6, 6); } }
}
function bal(id, b) {
  if (!b) return show(id, '-', 'dim');
  const ok = b.ratio >= state.limits.balance;
  show(id, b.ratio.toFixed(2) + (ok ? '' : ' (' + b.soft + ' is softer)'), ok ? 'good' : 'bad');
}
function render() {
  const s = state, cam = s.cams[sel];
  $('info').textContent = 'Device ' + s.serial + ' · ' + s.time + ' · cameras: ' + s.camera_source;
  for (const k of names) { const t = $('tab-' + k); t.classList.toggle('live', s.cams[k].live); t.querySelector('span').textContent = s.cams[k].title; }
  $('camname').textContent = '· ' + cam.title;
  if (cam.focus !== undefined) {
    const pct = cam.peak > 0 ? cam.focus / cam.peak : 0;
    show('focus', cam.focus.toFixed(1) + '  (' + Math.round(100 * pct) + '% of highest)');
    const wc = v => v <= s.limits.sharp_px ? 'good' : (v <= s.limits.soft_px ? 'warn' : 'bad');
    const word = v => v <= s.limits.sharp_px ? '  sharp' : (v <= s.limits.soft_px ? '  nearly' : '  soft');
    if (cam.width !== null && cam.width !== undefined) show('width', cam.width.toFixed(2) + ' px' + word(cam.width), wc(cam.width));
    else show('width', 'no clear edge', 'dim');
    show('bestwidth', cam.best_width === null || cam.best_width === undefined ? '-' : cam.best_width.toFixed(2) + ' px');
    $('focusbar').style.width = Math.round(100 * Math.min(pct, 1)) + '%';
    show('peak', cam.peak.toFixed(1));
    bal('lr', cam.balance && cam.balance.lr); bal('tb', cam.balance && cam.balance.tb);
    $('focusnote').textContent = 'Grid numbers: edge width in pixels. Up to ' + s.limits.sharp_px + ' is sharp, above ' + s.limits.soft_px +
      ' is soft; a dash means no clear edge in the cell. Objects at other distances than the focus target read softer.';
  } else {
    for (const id of ['focus', 'peak', 'lr', 'tb', 'width', 'bestwidth']) show(id, '-', 'dim');
    $('focusnote').textContent = cam.error || (cam.live ? '' : 'No picture from this camera.');
  }
  const ce = cam.centre, lim = sel === 'road' ? s.limits.road_centre_px : s.limits.wide_centre_px;
  show('clim', lim + ' px (' + (lim * 0.004).toFixed(2) + ' mm)');
  if (ce && ce.dx !== undefined) {
    const c = !ce.ok ? 'warn' : (ce.offset_px <= lim ? 'good' : 'bad');
    show('coff', ce.offset_px.toFixed(0) + ' px' + (ce.ok ? (ce.offset_px <= lim ? '  OK' : '  too far') : '  (uncertain)'), c);
    show('cdx', ce.dx.toFixed(0) + ' px  ' + ce.mm[0].toFixed(2) + ' mm', c);
    show('cdy', ce.dy.toFixed(0) + ' px  ' + ce.mm[1].toFixed(2) + ' mm', c);
    show('cdeg', ce.deg[0].toFixed(1) + '° / ' + ce.deg[1].toFixed(1) + '°');
  } else for (const id of ['coff', 'cdx', 'cdy', 'cdeg']) show(id, '-', 'dim');
  const fitNote = c => cam.fisheye ? 'Image circle fit, edge scatter ' + c.rms + ' px.'
    : 'Fall-off fit: ' + c.falloff + '% darker at the edges, unevenness ' + c.flatness + '%.';
  $('cnote').textContent = ce ? (ce.msg || fitNote(ce)) : '';
  const p = s.pair, pl = s.limits.pair_deg;
  if (p.ok) {
    show('yaw', p.yaw.toFixed(2) + '°', Math.abs(p.yaw) <= pl ? 'good' : 'bad');
    show('pitch', p.pitch.toFixed(2) + '°', Math.abs(p.pitch) <= pl ? 'good' : 'bad');
    show('roll', p.roll === null ? '-' : p.roll.toFixed(2) + '°', p.roll === null ? 'dim' : (Math.abs(p.roll) <= pl ? 'good' : 'bad'));
    $('pnote').textContent = (p.steady ? 'Steady. ' : 'Settling. ') + 'Limit ' + pl + '° in yaw and pitch. Valid for a scene more than 10 m away.';
  } else { for (const id of ['yaw', 'pitch', 'roll']) show(id, '-', 'dim'); $('pnote').textContent = p.msg || ''; }
  show('fps', cam.fps ? cam.fps.toFixed(1) : '-');
  show('bright', cam.mean !== undefined ? cam.mean.toFixed(0) + ' / ' + cam.clipped.toFixed(1) + '%' : '-', cam.clipped > 5 ? 'warn' : '');
  show('expo', cam.exposure ? cam.exposure.lines + ' / ' + cam.exposure.gain : '-');
  const ms = v => v === null || v === undefined ? '-' : v.toFixed(2) + ' ms';
  show('s1', ms(s.sync.road_wide_ms)); show('s2', ms(s.sync.driver_road_ms));
  draw();
}
async function poll() {
  try { state = await (await fetch('/api/state?cam=' + sel)).json(); if (lost) { lost = false; setStream(); } render(); }
  catch (e) { lost = true; $('info').textContent = 'No connection to the device.'; }
  setTimeout(poll, 400);
}
setInterval(() => { for (const k of names) $('tab-' + k).querySelector('img').src = '/frame/' + k + '.jpg?t=' + Date.now(); }, 1500);
setStream(); poll();
</script></body></html>
"""


def make_handler(bench):
  class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
      pass

    def send(self, body, kind, code=200):
      self.send_response(code)
      self.send_header('Content-Type', kind)
      self.send_header('Content-Length', str(len(body)))
      self.send_header('Cache-Control', 'no-store')
      self.end_headers()
      self.wfile.write(body)

    def do_GET(self):
      url = urlparse(self.path)
      parts = url.path.strip('/').split('/')
      try:
        if url.path == '/':
          self.send(PAGE.encode(), 'text/html; charset=utf-8')
        elif url.path == '/api/state':
          selected = parse_qs(url.query).get('cam', [''])[0]
          if selected in bench.cams:
            for key, cam in bench.cams.items():
              cam.selected = key == selected
          self.send(json.dumps(bench.state()).encode(), 'application/json')
        elif parts[0] == 'frame' and parts[1].removesuffix('.jpg') in bench.cams:
          jpeg = bench.cams[parts[1].removesuffix('.jpg')].jpeg(quality=60)
          self.send(jpeg or b'', 'image/jpeg', 200 if jpeg else 503)
        elif parts[0] == 'stream' and parts[1] in bench.cams:
          zoom = parse_qs(url.query).get('zoom')
          self.stream(bench.cams[parts[1]], tuple(float(v) for v in zoom[0].split(',')) if zoom else None)
        else:
          self.send(b'not found', 'text/plain', 404)
      except (BrokenPipeError, ConnectionResetError):
        pass
      except (IndexError, ValueError):
        self.send(b'not found', 'text/plain', 404)

    def do_POST(self):
      if self.path == '/api/reset':
        for cam in bench.cams.values():
          cam.peak, cam.best_width = 0., None
        self.send(b'{}', 'application/json')
      elif self.path == '/api/save':
        self.send(json.dumps({'folder': bench.save()}).encode(), 'application/json')
      else:
        self.send(b'not found', 'text/plain', 404)

    def stream(self, cam, zoom):
      """Motion JPEG: the browser shows each part as it arrives."""
      self.send_response(200)
      self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
      self.send_header('Cache-Control', 'no-store')
      self.send_header('Connection', 'close')
      self.end_headers()
      self.close_connection = True
      last = None
      while not bench.stop.is_set():
        if cam.frame_id != last and cam.live():
          last = cam.frame_id
          jpeg = cam.jpeg(zoom)
          self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n')
        time.sleep(.1)
  return Handler


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--host', default='0.0.0.0', help='address to listen on (default: every interface)')
  parser.add_argument('--port', type=int, default=8088)
  parser.add_argument('--output', default='/data/camera-calibration', help='folder for saved reports')
  parser.add_argument('--no-start', action='store_true', help='only use cameras that already run; never start them')
  args = parser.parse_args()
  sys.path.insert(0, str(BASEDIR))
  bench = Bench(args)
  atexit.register(bench.close)
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
  bench.start()
  server = ThreadingHTTPServer((args.host, args.port), make_handler(bench))
  server.daemon_threads = True
  print('Camera calibration is running. Open one of these addresses:', flush=True)
  for address in addresses(args.port):
    print(' ', address, flush=True)
  print('Anyone in the same network can see the cameras while this runs. Stop with Ctrl+C.', flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  main()

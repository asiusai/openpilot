"""Online lateral calibration of Alpamayo plans, in the manner of torqued.

Alpamayo plans turn less than the car did: the 1 s heading change was 0.6-0.9 of the measured one, and the ratio differs
between drives and speeds. Fixed gains helped one drive and hurt another. This calibrator learns a gain per speed bucket from
driver-steered samples only: each plan's predicted 1 s heading change against the heading change that `deviceMotion` then
measured. Engaged samples are excluded because the car follows the corrected plan there.

Only plans that turn the same way as the car teach it, so a turn the driver chose without route guidance does not.
Each bucket keeps decayed weighted least-squares sums with a prior of gain 1 and Huber weights, and stores them atomically.
"""
import json
import os
from collections import deque
from pathlib import Path

import numpy as np

VERSION = 1
EDGES = (0., 5., 8., 12., 20., 60.)  # m/s
HORIZON_NS = 1_000_000_000
PRIOR, STRENGTH, DECAY = 1.0, 0.02, 0.999
GAIN_LIMITS = (0.8, 2.0)
MIN_PREDICTED = 0.02  # rad of heading change in 1 s
MAX_ACTUAL = 1.0
MIN_SPEED = 3.  # m/s
SAVE_INTERVAL_NS = 60_000_000_000


class LateralCalibration:
  def __init__(self, path):
    self.path = Path(path)
    self.sxx, self.sxy, self.samples = np.zeros(len(EDGES)-1), np.zeros(len(EDGES)-1), 0
    try:
      saved = json.loads(self.path.read_text())
      if saved.get('version') == VERSION and saved.get('edges') == list(EDGES):
        self.sxx, self.sxy = np.array(saved['sxx'], float), np.array(saved['sxy'], float)
        self.samples = int(saved.get('samples', 0))
    except (OSError, ValueError, KeyError, TypeError):
      pass  # no or incompatible calibration: start from the prior
    self.pending = deque()  # (query_ns, speed, predicted heading change, steered by the controller)
    self.yaw = deque()  # (ns, left-positive yaw, unwrapped)
    self.saved_ns = 0

  @staticmethod
  def bucket(v):
    return int(np.clip(np.searchsorted(EDGES, v, side='right')-1, 0, len(EDGES)-2))

  def gain(self, v):
    b = self.bucket(v)
    return float(np.clip((self.sxy[b]+STRENGTH*PRIOR)/(self.sxx[b]+STRENGTH), *GAIN_LIMITS))

  def learn(self, v, predicted, actual):
    if v < MIN_SPEED or abs(predicted) < MIN_PREDICTED or np.sign(predicted) != np.sign(actual) or abs(actual) > MAX_ACTUAL:
      return False
    b = self.bucket(v)
    residual = actual-self.gain(v)*predicted
    weight = 1. if abs(residual) < .05 else .05/abs(residual)
    self.sxx *= DECAY
    self.sxy *= DECAY
    self.sxx[b] += weight*predicted*predicted
    self.sxy[b] += weight*predicted*actual
    self.samples += 1
    return True

  def plan(self, query_ns, speed, predicted):
    """A received plan: its predicted heading change (left-positive) 1 s after its query."""
    if np.isfinite(predicted) and np.isfinite(speed):
      self.pending.append([int(query_ns), float(speed), float(predicted), False])

  def motion(self, ns, ned_yaw, steered, now_ns=None):
    """A deviceMotion sample. `steered`: the controller steered without a driver steering input."""
    left = -float(ned_yaw)
    if self.yaw:
      last = self.yaw[-1][1]
      left = last+(left-last+np.pi) % (2*np.pi)-np.pi  # unwrap
    self.yaw.append((int(ns), left))
    while self.yaw and self.yaw[0][0] < ns-3*HORIZON_NS:
      self.yaw.popleft()
    for entry in self.pending:
      if entry[0] <= ns <= entry[0]+HORIZON_NS:
        entry[3] |= steered
    while self.pending and self.pending[0][0]+HORIZON_NS <= ns:
      query, speed, predicted, controlled = self.pending.popleft()
      if controlled or self.yaw[0][0] > query:
        continue
      times = np.array([t for t, _ in self.yaw]); values = np.array([y for _, y in self.yaw])
      actual = float(np.interp(query+HORIZON_NS, times, values)-np.interp(query, times, values))
      self.learn(speed, predicted, actual)
    if now_ns is not None and now_ns-self.saved_ns > SAVE_INTERVAL_NS:
      self.save()
      self.saved_ns = now_ns

  def save(self):
    data = {'version': VERSION, 'edges': list(EDGES), 'sxx': self.sxx.tolist(), 'sxy': self.sxy.tolist(), 'samples': self.samples,
            'gains': [self.gain((a+b)/2) for a, b in zip(EDGES[:-1], EDGES[1:], strict=True)]}
    temporary = self.path.with_suffix('.tmp')
    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      temporary.write_text(json.dumps(data))
      os.replace(temporary, self.path)
    except OSError:
      pass  # calibration persistence must never stop planning

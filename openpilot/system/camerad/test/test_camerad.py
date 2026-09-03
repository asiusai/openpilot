#!/usr/bin/env python3

import time
import unittest
import numpy as np

from openpilot.common.parameterized import parameterized
from openpilot.common.hardware import ASIUS_HARDWARE
from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal.services import SERVICE_LIST
from openpilot.tools.lib.log_time_series import msgs_to_time_series
from openpilot.system.camerad.snapshot import get_snapshots
from openpilot.selfdrive.pandad.pandad import panda_ready, flash_panda
from openpilot.selfdrive.test.helpers import collect_logs, log_collector, processes_context

TEST_TIMESPAN = 10
CAMERAS = ('narrowRoadCameraState', 'cabinCameraState', 'wideRoadCameraState')
EXPOSURE_STABLE_FRAMES = 20
EXPOSURE_TOLERANCE = 0.2
MAX_TEST_TIME = 25
STARTUP_FRAME_IGNORE = 10
MAX_ISOLATED_FRAME_DROPS = 2


def _numpy_rgb2gray(im):
  return np.clip(im[:,:,2] * 0.114 + im[:,:,1] * 0.587 + im[:,:,0] * 0.299, 0, 255).astype(np.uint8)

def _image_stats(im):
  h, w = im.shape[:2]
  gray = _numpy_rgb2gray(im[h//10:9*h//10, w//10:9*w//10]) / 255.
  return float(np.mean(gray)), float(np.std(gray))

def _converged(ts, cam):
  measured, target = ts[cam]['measuredGreyFraction'], ts[cam]['targetGreyFraction']
  # in a scene too dark (or bright) to reach the target, auto exposure settles at its limit instead
  ev = ts[cam]['exposureValPercent']
  saturated = ((ev > 99.9) & (measured < target)) | ((ev < 0.1) & (measured > target))
  return (np.abs(measured - target) < EXPOSURE_TOLERANCE * target) | saturated

def run_and_log(procs, services, duration):
  with processes_context(procs):
    return collect_logs(services, duration)

def _camera_session():
  """Single camerad session that collects logs and exposure data.
     Runs until exposure stabilizes (min TEST_TIMESPAN seconds for enough log data)."""
  with processes_context(["camerad"]), log_collector(CAMERAS) as (raw_logs, lock):
    images = {cam: [] for cam in CAMERAS}
    start = time.monotonic()
    while time.monotonic() - start < MAX_TEST_TIME:
      rpic, dpic = get_snapshots(frame="narrowRoadCameraState", front_frame="cabinCameraState")
      wpic, _ = get_snapshots(frame="wideRoadCameraState")
      for cam, img in zip(CAMERAS, [rpic, dpic, wpic], strict=True):
        images[cam].append(_image_stats(img))

      if time.monotonic() - start >= TEST_TIMESPAN:
        with lock:
          ts = msgs_to_time_series(raw_logs)
        if all(len(ts[cam]['t']) >= EXPOSURE_STABLE_FRAMES and
               _converged(ts, cam)[-EXPOSURE_STABLE_FRAMES:].all() for cam in CAMERAS):
          break

  with lock:
    ts = msgs_to_time_series(raw_logs)

  for cam in CAMERAS:
    # measured from the first frame, camerad takes a few seconds to start
    expected_frames = SERVICE_LIST[cam].frequency * (ts[cam]['t'][-1] - ts[cam]['t'][0]) + 1
    cnt = len(ts[cam]['t'])
    assert expected_frames*0.8 < cnt < expected_frames*1.2, f"unexpected frame count {cam}: {expected_frames=}, got {cnt}"

    timestamps = ts[cam]['timestampSof'][STARTUP_FRAME_IGNORE:] / 1e6
    dts = np.abs(np.diff(timestamps) - 1000/SERVICE_LIST[cam].frequency)
    bad_intervals = np.count_nonzero(dts >= 1.0)
    assert bad_intervals <= MAX_ISOLATED_FRAME_DROPS, \
      f"{cam} dts(ms) out of spec: {bad_intervals} bad intervals, " \
      f"max diff {dts.max()}, 99 percentile {np.percentile(dts, 99)}"
    assert dts.max() < 1000/SERVICE_LIST[cam].frequency + 1.0, f"{cam} dropped consecutive frames"

  return ts, images

class TestCamerad(OpenpilotTestCase):
  COMMA_HARDWARE_TEST = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Panda generates camera FSIN, including the cabin camera's 25ms stagger.
    if not panda_ready():
      flash_panda()
    assert panda_ready(), "panda not ready for camera synchronization"
    cls.logs, cls.images = _camera_session()

  @parameterized.expand(CAMERAS, names=("cam",))
  def test_camera_exposure(self, cam):
    # auto exposure converges to its target
    converged = _converged(self.logs, cam)
    assert len(converged) >= EXPOSURE_STABLE_FRAMES and converged[-EXPOSURE_STABLE_FRAMES:].all(), \
      f"{cam}: exposure didn't converge, {converged.sum()}/{len(converged)} frames converged"

  @parameterized.expand(CAMERAS, names=("cam",))
  def test_camera_image(self, cam):
    # the image isn't black, saturated, or flat
    mean, std = self.images[cam][-1]
    assert 0.02 < mean < 0.9, f"{cam}: bad image brightness {mean=:.3f}"
    # a dead sensor or pipeline is flat to within noise, the rack scenes have little contrast, e.g. std=0.016
    assert std > 0.005, f"{cam}: flat image {std=:.3f}"

  def test_frame_skips(self):
    for c in CAMERAS:
      frame_diffs = np.diff(self.logs[c]['frameId'][STARTUP_FRAME_IGNORE:] if ASIUS_HARDWARE else self.logs[c]['frameId'])
      if ASIUS_HARDWARE:
        assert frame_diffs.min() == 1, f"{c} has duplicate or decreasing frame IDs"
        assert frame_diffs.max() <= 2, f"{c} dropped consecutive frames"
        dropped_frames = np.count_nonzero(frame_diffs != 1)
        assert dropped_frames <= MAX_ISOLATED_FRAME_DROPS, f"{c} dropped too many frames: {dropped_frames}"
      else:
        assert set(frame_diffs) == {1, }, f"{c} has frame skips"

  def test_frame_sync(self):
    if ASIUS_HARDWARE:
      self.skipTest("Asius cameras are not hardware-synchronized")

    SYNCED_CAMS = ('narrowRoadCameraState', 'wideRoadCameraState')
    n = range(len(self.logs['narrowRoadCameraState']['t'][:-10]))

    frame_ids = {i: [self.logs[cam]['frameId'][i] for cam in CAMERAS] for i in n}
    assert all(len(set(v)) == 1 for v in frame_ids.values()), "frame IDs not aligned"

    # road and wide cameras should be synced within 1.1ms
    synced_times = {i: [self.logs[cam]['timestampSof'][i] for cam in SYNCED_CAMS] for i in n}
    diffs = {i: (max(ts) - min(ts))/1e6 for i, ts in synced_times.items()}
    laggy_frames = {k: v for k, v in diffs.items() if v > 1.1}
    assert len(laggy_frames) == 0, f"Frames not synced properly: {laggy_frames=}"

    # cabin camera should be staggered ~25ms from road camera
    for i in n:
      offset_ms = abs(self.logs['cabinCameraState']['timestampSof'][i] - self.logs['narrowRoadCameraState']['timestampSof'][i]) / 1e6
      assert 20 < offset_ms < 30, f"cabin camera stagger out of range at frame {i}: {offset_ms:.1f}ms (expected ~25ms)"

  def test_frame_sync(self):
    if ASIUS:
      self.skipTest("Asius v1 cameras do not share a hardware frame-sync clock")

    SYNCED_CAMS = ('narrowRoadCameraState', 'wideRoadCameraState')
    n = range(len(self.logs['narrowRoadCameraState']['t'][:-10]))

    frame_ids = {i: [self.logs[cam]['frameId'][i] for cam in CAMERAS] for i in n}
    assert all(len(set(v)) == 1 for v in frame_ids.values()), "frame IDs not aligned"

    # road and wide cameras should be synced within 1.1ms
    synced_times = {i: [self.logs[cam]['timestampSof'][i] for cam in SYNCED_CAMS] for i in n}
    diffs = {i: (max(ts) - min(ts))/1e6 for i, ts in synced_times.items()}
    laggy_frames = {k: v for k, v in diffs.items() if v > 1.1}
    assert len(laggy_frames) == 0, f"Frames not synced properly: {laggy_frames=}"

    # cabin camera should be staggered ~25ms from road camera
    for i in n:
      offset_ms = abs(self.logs['cabinCameraState']['timestampSof'][i] - self.logs['narrowRoadCameraState']['timestampSof'][i]) / 1e6
      assert 20 < offset_ms < 30, f"cabin camera stagger out of range at frame {i}: {offset_ms:.1f}ms (expected ~25ms)"
  def test_sanity_checks(self):
    self._sanity_checks(self.logs)

  def _sanity_checks(self, ts):
    for c in CAMERAS:
      assert c in ts
      assert len(ts[c]['t']) > 20

      # should monotonically increase
      assert np.all(np.diff(ts[c]['frameId']) >= 1)
      assert 0 not in ts[c]['requestId']
      assert np.all(np.diff(ts[c]['requestId']) >= 1)

      # EOF > SOF
      assert np.all((ts[c]['timestampEof'] - ts[c]['timestampSof']) > 0)

      # logMonoTime > SOF
      assert np.all((ts[c]['t'] - ts[c]['timestampSof']/1e9) > 1e-7)

      # EOF timestamps can be reconstructed from SOF or supplied directly by the camera driver.
      assert np.all((ts[c]['t'] - ts[c]['timestampEof']/1e9) > -0.10)        # when EOF > logMonoTime, it should never be more than two frames

if __name__ == "__main__":
  unittest.main()

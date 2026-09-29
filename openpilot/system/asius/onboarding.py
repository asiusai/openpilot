"""Device-owned app setup state and the driver-monitoring training check."""
import math
import threading
import time

from openpilot.common.params import Params
from openpilot.common.version import terms_version, training_version


def setup_status(params: Params) -> dict:
  accepted = params.get("HasAcceptedTerms") == terms_version
  trained = params.get("CompletedTrainingVersion") == training_version
  return {"termsVersion": terms_version, "trainingVersion": training_version,
          "termsAccepted": accepted, "trainingCompleted": trained, "complete": accepted and trained}


class Onboarding:
  def __init__(self, params=None, sm=None, clock=time.monotonic):
    self.params = params if params is not None else Params()
    self.sm = sm
    self.clock = clock
    self.lock = threading.RLock()
    self.timer = None
    self.progress = 0.
    self.last_sample = None
    self.checked_version = None

  def parked(self):
    if not self.params.get_bool("IsOffroad"):
      raise PermissionError("Park before completing setup")

  def accept(self, version: str):
    with self.lock:
      self.parked()
      if version != terms_version:
        raise ValueError("Terms changed; reload setup")
      self.checked_version = None
      self.params.put("HasAcceptedTerms", version, block=True)
      return setup_status(self.params)

  def stop_check(self):
    with self.lock:
      if self.timer:
        self.timer.cancel()
        self.timer = None
        self.params.put_bool("IsDriverViewEnabled", False, block=True)
      self.last_sample = None
      self.progress = 0.

  def check_driver(self):
    with self.lock:
      self.parked()
      if not setup_status(self.params)["termsAccepted"]:
        raise PermissionError("Accept the terms first")
      if self.sm is None:
        import openpilot.cereal.messaging as messaging
        self.sm = messaging.SubMaster(["driverMonitoringState", "driverStateV2"])
      if self.timer:
        self.timer.cancel()
      self.params.put_bool("IsDriverViewEnabled", True, block=True)
      self.timer = threading.Timer(5., self.stop_check)
      self.timer.daemon = True
      self.timer.start()
      self.sm.update(0)
      now = self.clock()
      fresh = all(self.sm.valid[s] and self.sm.seen[s] and 0 <= now - self.sm.logMonoTime[s] / 1e9 < 1.
                  for s in ("driverMonitoringState", "driverStateV2"))
      looking = False
      if fresh:
        dm = self.sm["driverMonitoringState"]
        driver = self.sm["driverStateV2"].rightDriverData if dm.isRHD else self.sm["driverStateV2"].leftDriverData
        orientation = driver.faceOrientation
        looking = dm.visionPolicyState.faceDetected and len(orientation) == 3 and all(abs(math.degrees(a)) < 30. for a in orientation[:2])
      elapsed = now - self.last_sample if self.last_sample is not None else 0.
      self.progress = min(4., self.progress + elapsed) if looking and 0 <= elapsed <= 2. else 0.
      self.last_sample = now
      if self.progress >= 4.:
        self.checked_version = training_version
      complete = self.checked_version == training_version
      return {"faceDetected": bool(looking), "progress": self.progress / 4., "complete": complete}

  def complete(self, version: str, record_front: bool, share_data: bool):
    with self.lock:
      self.parked()
      if version != training_version or not setup_status(self.params)["termsAccepted"]:
        raise ValueError("Setup changed; reload terms and training")
      if self.checked_version != training_version:
        raise PermissionError("Complete the driver monitoring check first")
      if type(record_front) is not bool or type(share_data) is not bool:
        raise ValueError("Choose whether to record and share cabin data")
      self.params.put_bool("RecordFront", record_front, block=True)
      self.params.put_bool("ShareDrivingData", share_data, block=True)
      self.params.put("CompletedTrainingVersion", version, block=True)
      self.stop_check()
      return setup_status(self.params)

  def reset(self):
    with self.lock:
      self.parked()
      self.stop_check()
      self.checked_version = None
      self.params.put("HasAcceptedTerms", "0", block=True)
      self.params.put("CompletedTrainingVersion", "0", block=True)
      return setup_status(self.params)

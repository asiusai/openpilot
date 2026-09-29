"""Device-owned app terms and training state."""
import threading

from openpilot.common.params import Params
from openpilot.common.version import terms_version, training_version


def setup_status(params: Params) -> dict:
  accepted = params.get("HasAcceptedTerms") == terms_version
  trained = params.get("CompletedTrainingVersion") == training_version
  return {"termsVersion": terms_version, "trainingVersion": training_version,
          "termsAccepted": accepted, "trainingCompleted": trained, "complete": accepted and trained}


class Onboarding:
  def __init__(self, params=None):
    self.params = params if params is not None else Params()
    self.lock = threading.RLock()

  def parked(self):
    if not self.params.get_bool("IsOffroad"):
      raise PermissionError("Park before completing setup")

  def accept(self, version: str):
    with self.lock:
      self.parked()
      if version != terms_version:
        raise ValueError("Terms changed; reload setup")
      self.params.put("HasAcceptedTerms", version, block=True)
      return setup_status(self.params)

  def complete(self, version: str, record_front: bool, share_data: bool):
    with self.lock:
      self.parked()
      if version != training_version or not setup_status(self.params)["termsAccepted"]:
        raise ValueError("Setup changed; reload terms and training")
      if type(record_front) is not bool or type(share_data) is not bool:
        raise ValueError("Choose whether to record and share cabin data")
      self.params.put_bool("RecordFront", record_front, block=True)
      self.params.put_bool("ShareDrivingData", share_data, block=True)
      self.params.put("CompletedTrainingVersion", version, block=True)
      return setup_status(self.params)

  def reset(self):
    with self.lock:
      self.parked()
      self.params.put("HasAcceptedTerms", "0", block=True)
      self.params.put("CompletedTrainingVersion", "0", block=True)
      return setup_status(self.params)

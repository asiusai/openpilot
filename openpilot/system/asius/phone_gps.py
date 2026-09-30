"""An authorized Bluetooth phone supplies the standard 1 Hz GPS service.

Fresh device-issued sessions and increasing sequence numbers bound replay and
latency using monotonic time. The phone's UTC timestamp is GPS metadata only.
"""
import math
import secrets
import time
from collections.abc import Callable

from openpilot.cereal import messaging

PHONE_GPS_METHODS = {"startPhoneGps", "updatePhoneGps", "stopPhoneGps"}
MAX_FIX_AGE_MS = 3000
SESSION_TIMEOUT = 5.0
MAX_ACCURACY = 100.0


def number(value, name: str, low: float, high: float) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
    raise ValueError(f"Invalid phone GPS {name}")
  return float(value)


def gps_message(fix: dict | None):
  msg = messaging.new_message('gpsLocation', valid=fix is not None)
  gps = msg.gpsLocation
  gps.source = 'external'
  gps.vNED = [0., 0., 0.]
  gps.verticalAccuracy = 10000.
  gps.speedAccuracy = 1000.
  gps.bearingAccuracyDeg = 180.
  if fix is None:
    gps.hasFix = False
    return msg
  if not isinstance(fix, dict):
    raise ValueError("Invalid phone GPS fix")
  number(fix.get('ageMs'), 'age', 0, MAX_FIX_AGE_MS)
  gps.latitude = number(fix.get('latitude'), 'latitude', -90, 90)
  gps.longitude = number(fix.get('longitude'), 'longitude', -180, 180)
  gps.horizontalAccuracy = number(fix.get('accuracy'), 'accuracy', 0, 100000)
  gps.unixTimestampMillis = int(number(fix.get('timestamp'), 'timestamp', 1, 253402300799000))
  source = fix.get('source')
  if source not in ('android', 'iOS', 'external'):
    raise ValueError("Invalid phone GPS source")
  gps.source = source
  if fix.get('altitude') is not None:
    gps.altitude = number(fix['altitude'], 'altitude', -1500, 100000)
    if fix.get('altitudeAccuracy') is not None:
      gps.verticalAccuracy = number(fix['altitudeAccuracy'], 'altitude accuracy', 0, 100000)
  if fix.get('speed') is not None:
    gps.speed = number(fix['speed'], 'speed', 0, 200)
    if fix.get('speedAccuracy') is not None:
      gps.speedAccuracy = number(fix['speedAccuracy'], 'speed accuracy', 0, 1000)
  if fix.get('heading') is not None:
    gps.bearingDeg = number(fix['heading'], 'heading', 0, 360)
    if fix.get('headingAccuracy') is not None:
      gps.bearingAccuracyDeg = number(fix['headingAccuracy'], 'heading accuracy', 0, 360)
    if fix.get('speed') is not None:
      bearing = math.radians(gps.bearingDeg)
      gps.vNED = [gps.speed * math.cos(bearing), gps.speed * math.sin(bearing), 0.]
  gps.hasFix = gps.horizontalAccuracy <= MAX_ACCURACY
  return msg


class PhoneGps:
  def __init__(self, publish: Callable, monotonic: Callable = time.monotonic):
    self.publish = publish
    self.monotonic = monotonic
    self.owner: str | None = None
    self.session: str | None = None
    self.sequence = -1
    self.timestamp = 0
    self.deadline = 0.
    self.fix_deadline = 0.
    self.has_fix = False

  def clear(self) -> None:
    if self.owner is not None:
      self.publish(gps_message(None))
    self.owner = self.session = None
    self.has_fix = False

  def tick(self, authorized_peers=None) -> None:
    now = self.monotonic()
    if self.owner is not None and (now >= self.deadline or (authorized_peers is not None and self.owner not in authorized_peers)):
      self.clear()
    elif self.has_fix and now >= self.fix_deadline:
      self.publish(gps_message(None))
      self.has_fix = False

  def start(self, peer: str) -> dict:
    self.tick()
    if self.owner is not None and self.owner != peer:
      raise RuntimeError("Another phone is supplying GPS. Turn off GPS sharing on that phone first.")
    self.clear()
    self.owner, self.session = peer, secrets.token_hex(24)
    self.sequence = -1
    self.timestamp = 0
    self.deadline = self.monotonic() + SESSION_TIMEOUT
    return {"session": self.session, "timeoutMs": int(SESSION_TIMEOUT * 1000)}

  def update(self, peer: str, session: str, sequence: int, fix: dict | None) -> dict:
    self.tick()
    if self.owner != peer or self.session is None or session != self.session:
      raise ValueError("Phone GPS session expired. Reconnect GPS sharing.")
    if type(sequence) is not int or not self.sequence < sequence <= 2**53 - 1:
      raise ValueError("Phone GPS sequence must increase")
    msg = gps_message(fix)
    if fix is not None and msg.gpsLocation.unixTimestampMillis <= self.timestamp:
      raise ValueError("Phone GPS fix must be newer than the previous fix")
    now = self.monotonic()
    self.sequence = sequence
    if fix is not None:
      self.timestamp = msg.gpsLocation.unixTimestampMillis
    self.deadline = now + SESSION_TIMEOUT
    self.has_fix = msg.gpsLocation.hasFix
    self.fix_deadline = now + (MAX_FIX_AGE_MS - fix['ageMs']) / 1000 if fix is not None else now
    self.publish(msg)
    return {"accepted": True, "hasFix": self.has_fix}

  def stop(self, peer: str, session: str) -> dict:
    stopped = self.owner == peer and self.session == session
    if stopped:
      self.clear()
    return {"stopped": stopped}

  def dispatcher(self, peer: str) -> dict:
    return {
      "startPhoneGps": lambda: self.start(peer),
      "updatePhoneGps": lambda session, sequence, fix: self.update(peer, session, sequence, fix),
      "stopPhoneGps": lambda session: self.stop(peer, session),
    }

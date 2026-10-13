"""Optional offline speech for the alerts normally shown on the device screen."""
import json
import threading
import wave
from pathlib import Path

import numpy as np

from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog

VOICE_DIR = Path(BASEDIR) / "openpilot/selfdrive/assets/sounds/voice"
MESSAGES = json.loads((VOICE_DIR / "messages.json").read_text())
EVENT_VOICES = {event: key for key, item in MESSAGES.items() for event in item["events"]}
NO_ENTRY_VOICES = {event: key for key, item in MESSAGES.items() for event in item["no_entry"]}
OFFROAD_VOICES = {param: key for key, item in MESSAGES.items() for param in item["offroad"]}
PROCESS_VOICES = {"modeld": "model_error", "camerad": "camera_error"}
SAMPLE_RATE = 48000
RETRIGGER_SECONDS = 30.
SPEECH_GAP_SECONDS = 2.


def alert_voice(state):
  # Empty engagement/disengagement alerts must not speak their event's no-entry reason.
  if state is None or state.alertSize == 0:
    return None
  event, _, alert_type = state.alertType.partition("/")
  # selfdrived selects noEntry only when an engagement attempt is refused.
  if alert_type == "noEntry" and event in NO_ENTRY_VOICES:
    return NO_ENTRY_VOICES[event]
  if event == "processNotRunning":
    for process in state.alertText2.split(", "):
      if process in PROCESS_VOICES:
        return PROCESS_VOICES[process]
  return EVENT_VOICES.get(event)


class VoiceAlerts:
  def __init__(self, params):
    self.params = params
    self.enabled = False
    self.next_poll = 0.
    self.offroad = []
    self.loaded = {}
    self.load_attempted = False
    self.announced = set()
    self.last_played = {}
    self.next_speech = 0.
    self.lock = threading.Lock()
    self.current = None
    self.frame = 0

  @property
  def playing(self):
    return self.current is not None

  def load(self):
    self.load_attempted = True
    # A missing speech asset must not prevent soundd from playing standard alerts.
    for key in MESSAGES:
      try:
        with wave.open(str(VOICE_DIR / f"{key}.wav")) as wav:
          if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError("Expected mono 48 kHz signed 16-bit PCM")
          samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
          if not len(samples) or len(samples) > 8 * SAMPLE_RATE or np.max(np.abs(samples)) > .851:
            raise ValueError("Invalid speech length or peak level")
          self.loaded[key] = samples
      except (OSError, EOFError, wave.Error, ValueError):
        cloudlog.exception("Cannot load voice alert %s", key)

  def update(self, state, onroad, critical, now):
    if now >= self.next_poll:
      self.enabled = self.params.get_bool("VoiceAlertsEnabled")
      self.offroad = [key for param, key in OFFROAD_VOICES.items() if self.params.get(param)] if self.enabled and not onroad else []
      self.next_poll = now + 1.
      if self.enabled and not self.load_attempted:
        self.load()

    key = alert_voice(state) if onroad else None
    active = [key] if key else ([] if onroad else self.offroad)
    self.announced.intersection_update(active)
    with self.lock:
      if not self.enabled or critical:
        self.current = None
        self.frame = 0
        return
      if self.playing or now < self.next_speech:
        return
      for key in active:
        if key in self.loaded and key not in self.announced and now - self.last_played.get(key, -RETRIGGER_SECONDS) >= RETRIGGER_SECONDS:
          self.current = self.loaded[key]
          self.frame = 0
          self.announced.add(key)
          self.last_played[key] = now
          self.next_speech = now + len(self.current) / SAMPLE_RATE + SPEECH_GAP_SECONDS
          break

  def mix(self, alert, volume, critical):
    with self.lock:
      if critical or not self.enabled:
        self.current = None
        self.frame = 0
        return alert
      if not self.playing:
        return alert
      count = min(len(alert), len(self.current) - self.frame)
      # Speech peaks are <= 0.85. Reserve the remaining headroom for existing chimes.
      result = alert.copy()
      result[:count] = .149 * result[:count] + volume * self.current[self.frame:self.frame + count]
      self.frame += count
      if self.frame == len(self.current):
        self.current = None
      return result

"""Speak the local Bluetooth Numeric Comparison value without network access."""
import threading
import wave
from pathlib import Path

import numpy as np

from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware import ASIUS_HARDWARE
from openpilot.common.swaglog import cloudlog
from openpilot.system.asius.pairing_confirmation import ANNOUNCEMENTS, read_confirmation, read_announcement

SAMPLE_RATE = 48000


class PairingSpeech:
  def __init__(self, root: Path = Path(BASEDIR) / 'openpilot/selfdrive/assets/sounds/voice', button_confirmation: bool = ASIUS_HARDWARE):
    self.digits = {}
    self.outcomes = {}
    self.instructions = {}
    self.confirm_instruction = 'confirm_button' if button_confirmation else 'confirm_screen'
    try:
      for digit in [*'0123456789', *ANNOUNCEMENTS, 'compare', 'confirm_button', 'confirm_screen']:
        with wave.open(str(root / f'pairing_{digit}.wav')) as wav:
          if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError('Invalid pairing speech format')
          target = self.digits if digit.isdigit() else self.outcomes if digit in ANNOUNCEMENTS else self.instructions
          target[digit] = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.
    except (OSError, ValueError, wave.Error):
      self.digits.clear()
      self.outcomes.clear()
      self.instructions.clear()
      cloudlog.exception('asius.pairing.speech_unavailable')
    self.request_id = None
    self.announcement_id = None
    self.clip_kind = None
    self.clip = None
    self.offset = 0
    self.lock = threading.Lock()

  @property
  def playing(self) -> bool:
    return self.clip is not None

  def update(self, onroad: bool, critical: bool, now: float) -> None:
    status = None if onroad or critical else read_confirmation(now=now)
    announcement = None if critical else read_announcement(now=now)
    # The refusal is useful with ignition on. All actual pairing speech stays offroad.
    if onroad and announcement and announcement['outcome'] != 'ignition_on':
      announcement = None
    with self.lock:
      if not status:
        if self.clip_kind == 'digits' or not announcement:
          self.clip = None
        if announcement and announcement['id'] != self.announcement_id:
          self.announcement_id = announcement['id']
          self.clip = self.outcomes.get(announcement['outcome'])
          self.clip_kind = 'outcome'
          self.offset = 0
        return
      # Do not play an older start message after the comparison has finished.
      if announcement:
        self.announcement_id = announcement['id']
      if status['id'] == self.request_id:
        return
      self.request_id = status['id']
      digits = status.get('digits', '')
      if (len(self.digits) != 10 or len(self.instructions) != 3 or
          len(digits) != 6 or any(digit not in self.digits for digit in digits)):
        self.clip = None
        return
      # Include leading zeros. Clips already have natural word spacing.
      self.clip = np.concatenate([self.instructions['compare'], *[self.digits[digit] for digit in digits],
                                  self.instructions[self.confirm_instruction]])
      self.clip_kind = 'digits'
      self.offset = 0

  def mix(self, audio: np.ndarray, volume: float, critical: bool) -> np.ndarray:
    with self.lock:
      if critical:
        self.clip = None
      if self.clip is None:
        return audio
      count = min(len(audio), len(self.clip) - self.offset)
      output = audio * .15
      output[:count] += self.clip[self.offset:self.offset + count] * volume
      self.offset += count
      if self.offset >= len(self.clip):
        self.clip = None
      return np.clip(output, -1., 1.)

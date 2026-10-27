"""USB M10 receiver transport for Asius. Receiver configuration is RAM-only."""
import struct
import time
from pathlib import Path

from openpilot.cereal import messaging
from openpilot.common.serial import Serial
from openpilot.common.swaglog import cloudlog
from openpilot.system.ubloxd.ubloxd import UbloxMsgParser, UbxFramer
from openpilot.system.ubloxd.ubx import Ubx


# CP2102N bridges are candidates, not receiver identities. Verify MON-VER before configuring.
PORT_PATTERN = 'usb-Silicon_Labs_CP2102N_USB_to_UART_Bridge_Controller_*-if00-port0'
CONFIG = {
  0x30210001: (2, 100),  # CFG-RATE-MEAS: 10 Hz
  0x30210002: (2, 1),    # CFG-RATE-NAV
  0x20110021: (1, 4),    # CFG-NAVSPG-DYNMODEL: automotive
  0x10740001: (1, 1),    # CFG-UART1OUTPROT-UBX
  0x10740002: (1, 0),    # CFG-UART1OUTPROT-NMEA
  0x20910007: (1, 1),    # CFG-MSGOUT-UBX_NAV_PVT_UART1: every epoch
  0x20910016: (1, 10),   # CFG-MSGOUT-UBX_NAV_SAT_UART1: 1 Hz
}
STALE_TIMEOUT = 1.5  # Leave headroom for reads and scheduling before the two-second consumer expiry.


def packet(cls: int, mid: int, payload: bytes = b'') -> bytes:
  body = bytes((cls, mid)) + struct.pack('<H', len(payload)) + payload
  a = b = 0
  for value in body:
    a = (a + value) & 255
    b = (b + a) & 255
  return b'\xb5\x62' + body + bytes((a, b))


class M10:
  def __init__(self, tty: Serial):
    self.tty = tty
    self.framer = UbxFramer()

  def receive(self) -> list[bytes]:
    return self.framer.add_data(time.monotonic(), self.tty.read(4096))

  def request(self, cls: int, mid: int, payload: bytes = b'') -> bytes:
    self.tty.write(packet(cls, mid, payload))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
      for frame in self.receive():
        if frame[2:4] == bytes((cls, mid)):
          return frame[6:-2]
        if frame[2:4] in (b'\x05\x00', b'\x05\x01') and frame[6:-2] == bytes((cls, mid)):
          if frame[3] == 0:
            raise ValueError('M10 rejected configuration')
          return b''
    raise TimeoutError('M10 response timeout')

  def configure(self) -> None:
    version = self.request(0x0a, 0x04)
    if len(version) < 40 or version[30:40].rstrip(b'\0') != b'000A0000' or b'PROTVER=34.10' not in version:
      raise ValueError('Unsupported USB GNSS receiver')
    payload = b'\x00\x01\x00\x00'  # RAM layer only
    for key, (size, value) in CONFIG.items():
      payload += struct.pack('<I', key) + value.to_bytes(size, 'little')
    self.request(0x06, 0x8a, payload)
    for key, (size, value) in CONFIG.items():
      reply = self.request(0x06, 0x8b, b'\x00\x00\x00\x00' + struct.pack('<I', key))
      if len(reply) != 8 + size or reply[4:8] != struct.pack('<I', key) or int.from_bytes(reply[8:], 'little') != value:
        raise ValueError('M10 configuration readback failed')
    self.tty.reset_input_buffer()
    self.framer.reset()


class Publisher:
  def __init__(self, pm):
    self.pm = pm
    self.parser = UbloxMsgParser()
    self.last_epoch: int | None = None
    self.last_sample = time.monotonic()

  def invalidate(self) -> None:
    msg = messaging.new_message('gpsLocationExternal', valid=False)
    msg.gpsLocationExternal.source = 'ublox'
    self.pm.send('gpsLocationExternal', msg)

  def publish(self, frame: bytes) -> None:
    kind = frame[2:4]
    if kind == b'\x01\x07':
      if len(frame) != 100:
        return
      pvt = Ubx.NavPvt.from_bytes(frame[6:-2])
      # iTOW wraps weekly. Duplicated or out-of-order epochs must not refresh a fix.
      if self.last_epoch is not None and not 0 < (pvt.i_tow - self.last_epoch) % 604800000 < 302400000:
        return
      self.last_epoch = pvt.i_tow
      self.last_sample = time.monotonic()
      _, msg = self.parser.parse_frame(frame)
      gps = msg.gpsLocationExternal
      gps.hasFix = bool(pvt.flags & 1) and pvt.fix_type == 3 and not pvt.flags3 & 1
      msg.valid = pvt.valid & 7 == 7 and gps.unixTimestampMillis > 0
      gps.hasFix = gps.hasFix and msg.valid
      if not msg.valid:
        gps.unixTimestampMillis = 0
      self.pm.send('gpsLocationExternal', msg)
    elif kind == b'\x01\x35' and len(frame) >= 16 and len(frame) == 16 + frame[11] * 12:
      result = self.parser.parse_frame(frame)
      if result is not None:
        self.pm.send(*result)


def run_receiver(tty: Serial, pm) -> None:
  receiver = M10(tty)
  receiver.configure()
  cloudlog.info('M10 USB GNSS configured at 10 Hz')
  publisher = Publisher(pm)
  try:
    while True:
      frames = receiver.receive()
      if frames:
        raw = messaging.new_message(None, valid=True)
        raw.ubloxRaw = b''.join(frames)
        pm.send('ubloxRaw', raw)
      for frame in frames:
        publisher.publish(frame)
      if time.monotonic() - publisher.last_sample > STALE_TIMEOUT:
        raise TimeoutError('M10 navigation output stalled')
  finally:
    publisher.invalidate()


def main() -> None:
  pm = messaging.PubMaster(['gpsLocationExternal', 'ubloxGnss', 'ubloxRaw'])
  while True:
    for port in sorted(Path('/dev/serial/by-id').glob(PORT_PATTERN)):
      try:
        with Serial(str(port), baudrate=38400, timeout=0.1, exclusive=True) as tty:
          run_receiver(tty, pm)
      except (OSError, ValueError, TimeoutError):
        cloudlog.exception('M10 USB GNSS disconnected or unavailable')
    time.sleep(1)


if __name__ == '__main__':
  main()

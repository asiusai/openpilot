import struct
from unittest.mock import Mock

import pytest

from openpilot.system.ubloxd.m10d import M10, CONFIG, Publisher, packet, run_receiver
from openpilot.system.ubloxd.ubloxd import UbxFramer


def pvt(epoch=100, valid=7, fix=3, flags=1, flags3=0):
  data = bytearray(92)
  struct.pack_into('<IHBBBBBBIiBBBBiiiiIIiiiiiIIH', data, 0,
                   epoch, 2026, 10, 27, 14, 0, 0, valid, 25, -100000000,
                   fix, flags, 0, 12, 267420000, 583560000, 68000, 49000,
                   2000, 3000, 1000, -2000, 3000, 2236, 30000000, 200, 500000, 100)
  data[78] = flags3
  return packet(1, 7, data)


def test_framer_noise_split_checksum():
  frame = pvt()
  parser = UbxFramer()
  assert parser.add_data(0, b'$GNGGA,noise\r\n\xb5') == []
  assert parser.add_data(0, frame[1:40]) == []
  assert parser.add_data(0, frame[40:]) == [frame]
  damaged = frame[:-1] + bytes((frame[-1] ^ 255,))
  assert parser.add_data(0, damaged + frame) == [frame]


def test_pvt_units_and_timestamp():
  pm = Mock()
  publisher = Publisher(pm)
  publisher.publish(pvt())
  service, msg = pm.send.call_args.args
  gps = msg.gpsLocationExternal
  assert service == 'gpsLocationExternal' and msg.valid and gps.hasFix
  assert gps.latitude == pytest.approx(58.356)
  assert gps.longitude == pytest.approx(26.742)
  assert gps.altitude == 68  # ellipsoid, not 49 m MSL
  assert list(gps.vNED) == pytest.approx([1, -2, 3])
  assert gps.horizontalAccuracy == 2 and gps.verticalAccuracy == 3
  assert gps.unixTimestampMillis == 1793109599900
  assert gps.satelliteCount == 12 and gps.source == 'ublox'


@pytest.mark.parametrize('kwargs', [{'valid': 0}, {'valid': 3}, {'fix': 2}, {'flags': 0}, {'flags3': 1}])
def test_invalid_fix(kwargs):
  pm = Mock()
  Publisher(pm).publish(pvt(**kwargs))
  assert not pm.send.call_args.args[1].gpsLocationExternal.hasFix


def test_duplicate_out_of_order_and_week_wrap():
  pm = Mock()
  publisher = Publisher(pm)
  publisher.publish(pvt(604799900))
  publisher.publish(pvt(0))
  received = publisher.last_sample
  publisher.publish(pvt(0))
  publisher.publish(pvt(604799900))
  assert pm.send.call_count == 2
  assert publisher.last_sample == received
  publisher.invalidate()
  assert not pm.send.call_args.args[1].valid


def test_configuration_ram_only_and_readback():
  tty = Mock()
  receiver = M10(tty)
  version = b'ROM SPG 5.10'.ljust(30, b'\0') + b'000A0000\0\0' + b'PROTVER=34.10'
  replies = [version, b''] + [b'\x01\x00\x00\x00' + struct.pack('<I', k) + v.to_bytes(n, 'little')
                              for k, (n, v) in CONFIG.items()]
  receiver.request = Mock(side_effect=replies)
  receiver.configure()
  assert receiver.request.call_args_list[1].args[:2] == (6, 0x8a)
  assert receiver.request.call_args_list[1].args[2][:4] == b'\x00\x01\x00\x00'
  receiver.request = Mock(return_value=b'not a receiver')
  with pytest.raises(ValueError, match='Unsupported'):
    receiver.configure()
  assert receiver.request.call_count == 1
  receiver.request = Mock(side_effect=[version, b'', b'wrong readback'])
  with pytest.raises(ValueError, match='readback'):
    receiver.configure()


def test_nak_and_unplug():
  receiver = M10(Mock())
  receiver.receive = Mock(return_value=[packet(5, 0, b'\x06\x8a')])
  with pytest.raises(ValueError, match='rejected'):
    receiver.request(6, 0x8a)


def test_stall_invalidates(monkeypatch):
  monkeypatch.setattr(M10, 'configure', lambda self: None)
  monkeypatch.setattr(M10, 'receive', lambda self: [])
  now = iter([0., 3.])
  monkeypatch.setattr('openpilot.system.ubloxd.m10d.time.monotonic', lambda: next(now, 3.))
  pm = Mock()
  with pytest.raises(TimeoutError, match='stalled'):
    run_receiver(Mock(), pm)
  assert not pm.send.call_args.args[1].valid


def test_stream_and_disconnect(monkeypatch):
  monkeypatch.setattr(M10, 'configure', lambda self: None)
  monkeypatch.setattr(M10, 'receive', Mock(side_effect=[[pvt()], OSError('USB removed')]))
  pm = Mock()
  with pytest.raises(OSError, match='USB removed'):
    run_receiver(Mock(), pm)
  raw, fix, lost = [call.args for call in pm.send.call_args_list]
  assert raw[0] == 'ubloxRaw' and bytes(raw[1].ubloxRaw) == pvt()
  assert fix[1].gpsLocationExternal.hasFix
  assert not lost[1].valid and not lost[1].gpsLocationExternal.hasFix

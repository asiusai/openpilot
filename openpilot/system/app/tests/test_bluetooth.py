import unittest
from unittest.mock import AsyncMock, patch

from openpilot.system.app.bluetoothd import BlePeerEngine


class TestBluetoothAuthorization(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.engine = BlePeerEngine.__new__(BlePeerEngine)
    self.engine.dongle_id = "device"
    self.engine.active_peers = {}
    self.engine.send_body = AsyncMock()

  async def test_pairing_known_peer_closes_pairing_mode(self):
    self.engine.unpack_body = lambda _: ("app", {"type": "ble-pair-request", "requestId": "pair"}, False)
    with patch('openpilot.system.app.bluetoothd.load_authorized_peers', return_value={"app": {}}), \
         patch('openpilot.system.app.bluetoothd.disable_pairing_mode') as close, \
         patch('openpilot.system.app.bluetoothd.get_device_name', return_value="Asius v0"):
      await self.engine.handle_encrypted(b'')
    close.assert_called_once()
    self.assertEqual(self.engine.send_body.call_args.args[1]['type'], 'pair-response')

  async def test_session_confirms_the_request_for_a_known_peer(self):
    self.engine.unpack_body = lambda _: ("app", {"type": "ble-session", "requestId": "fresh"}, False)
    with patch('openpilot.system.app.bluetoothd.load_authorized_peers', return_value={"app": {}}):
      await self.engine.handle_encrypted(b'')
    self.engine.send_body.assert_awaited_once_with("app", {"type": "ble-session", "ready": True, "requestId": "fresh"})

  async def test_first_new_peer_closes_window_and_second_is_rejected(self):
    pairing = True
    peers = {}

    def close():
      nonlocal pairing
      pairing = False

    def authorize(peer, label):
      peers[peer] = {"label": label}

    with patch('openpilot.system.app.bluetoothd.load_authorized_peers', side_effect=lambda: peers), \
         patch('openpilot.system.app.bluetoothd.pairing_mode_active', side_effect=lambda: pairing), \
         patch('openpilot.system.app.bluetoothd.disable_pairing_mode', side_effect=close), \
         patch('openpilot.system.app.bluetoothd.authorize_peer', side_effect=authorize), \
         patch('openpilot.system.app.bluetoothd.is_dongle_id', return_value=True), \
         patch('openpilot.system.app.bluetoothd.get_device_name', return_value="Asius v0"):
      self.engine.unpack_body = lambda _: ("first", {"type": "ble-pair-request", "publicKey": "first", "requestId": "one"}, False)
      await self.engine.handle_encrypted(b'')
      self.engine.unpack_body = lambda _: ("second", {"type": "ble-pair-request", "publicKey": "second", "requestId": "two"}, False)
      with self.assertRaisesRegex(PermissionError, 'pairing mode is not active'):
        await self.engine.handle_encrypted(b'')
    self.assertEqual(list(peers), ['first'])
    self.assertFalse(pairing)

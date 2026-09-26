import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from dbus_fast import Variant
from dbus_fast.errors import DBusError

from openpilot.system.app.bluetoothd import Advertisement, BLE_SERVICE_UUID, BlePeerEngine, PairingAgent, StatusCharacteristic, keep_advertising


class TestBluetoothDiscovery(unittest.IsolatedAsyncioTestCase):
  def test_status_is_public_read_only_and_reflects_pairing_mode(self):
    status = StatusCharacteristic('public-device-key')
    self.assertEqual(status.Flags, ['read'])
    read = StatusCharacteristic.ReadValue.__wrapped__
    for active in (True, False):
      with patch('openpilot.system.app.bluetoothd.pairing_mode_active', return_value=active):
        value = read(status, {})
        self.assertEqual(json.loads(value), {"v": 1, "publicKey": "public-device-key", "pairingMode": active})
        self.assertEqual(read(status, {"offset": Variant('q', 10)}), value[10:])
        with self.assertRaises(DBusError):
          read(status, {"offset": Variant('q', len(value) + 1)})

  def test_discoverable_without_allowing_new_bonds_outside_pairing_mode(self):
    with patch('openpilot.system.app.bluetoothd.pairing_mode_active', return_value=False), \
         patch('openpilot.system.app.bluetoothd.get_device_name', return_value="Asius v0"):
      advertisement = Advertisement()
      self.assertEqual(advertisement.ServiceUUIDs, [BLE_SERVICE_UUID])
      self.assertEqual(advertisement.LocalName, 'Asius v0')
      self.assertTrue(advertisement.Discoverable)
      with self.assertRaises(DBusError):
        PairingAgent.require_pairing_mode()

  async def test_closing_pairing_does_not_restart_advertising(self):
    stop = MagicMock()
    stop.is_set.side_effect = [False, False, True]
    stop.wait = AsyncMock()
    bus = MagicMock()
    with patch('openpilot.system.app.bluetoothd.connected_device_count', new=AsyncMock(return_value=0)), \
         patch('openpilot.system.app.bluetoothd.pairing_mode_active', side_effect=[True, False]), \
         patch('openpilot.system.app.bluetoothd.get_device_name', return_value='Asius v0'), \
         patch('openpilot.system.app.bluetoothd.set_adapter_property', new=AsyncMock()) as set_property, \
         patch('openpilot.system.app.bluetoothd.refresh_advertisement', new=AsyncMock()) as refresh:
      await keep_advertising(bus, '/adapter', stop)
    refresh.assert_awaited_once_with(bus, '/adapter')
    self.assertEqual([call.args[2] for call in set_property.call_args_list], ['Pairable', 'Pairable'])
    self.assertEqual([call.args[3].value for call in set_property.call_args_list], [True, False])


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

  async def test_retired_phone_gps_commands_are_not_exposed_over_bluetooth(self):
    for method in ('startPhoneGps', 'updatePhoneGps', 'stopPhoneGps'):
      await self.engine.handle_rpc('app', {'jsonrpc': '2.0', 'id': method, 'method': method})
      response = self.engine.send_body.call_args.args[1]
      self.assertEqual(response['error']['code'], -32601)
    await self.engine.handle_rpc('app', {'jsonrpc': '2.0', 'id': 'echo', 'method': 'echo', 'params': {'s': 'still connected'}})
    self.assertEqual(self.engine.send_body.call_args.args[1]['result'], 'still connected')

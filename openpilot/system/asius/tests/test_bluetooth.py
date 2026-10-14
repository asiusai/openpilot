import pytest
import json
import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from cryptography.hazmat.primitives.asymmetric import ed25519, ec
from openpilot.system.asius.session import SecureSession, wrap_peer, unwrap_peer
from openpilot.system.asius.recording_keys import public_text

from dbus_fast import Message, Variant
from dbus_fast.errors import DBusError

from openpilot.system.asius.bluetoothd import (
  Advertisement, BLE_SERVICE_UUID, BlePeerEngine, PairingAgent, StatusCharacteristic, keep_advertising, bluez_registration_lost,
)


class TestBluetoothDiscovery(unittest.IsolatedAsyncioTestCase):
  def test_adapter_reset_invalidates_service_registration(self):
    def removed(path, interfaces):
      return Message.new_signal('/', 'org.freedesktop.DBus.ObjectManager', 'InterfacesRemoved', 'oas', [path, interfaces])
    self.assertTrue(bluez_registration_lost(removed('/org/bluez/hci0', ['org.bluez.Adapter1']), '/org/bluez/hci0'))
    self.assertTrue(bluez_registration_lost(removed('/org/bluez/hci0', ['org.bluez.GattManager1']), '/org/bluez/hci0'))
    self.assertFalse(bluez_registration_lost(removed('/org/bluez/hci1', ['org.bluez.Adapter1']), '/org/bluez/hci0'))
    self.assertFalse(bluez_registration_lost(removed('/org/bluez/hci0/dev_123', ['org.bluez.Device1']), '/org/bluez/hci0'))

  def test_bluez_restart_invalidates_service_registration(self):
    def changed(name, old, new):
      return Message.new_signal('/org/freedesktop/DBus', 'org.freedesktop.DBus', 'NameOwnerChanged', 'sss', [name, old, new])
    self.assertTrue(bluez_registration_lost(changed('org.bluez', ':1.2', ''), '/org/bluez/hci0'))
    self.assertTrue(bluez_registration_lost(changed('org.bluez', ':1.2', ':1.3'), '/org/bluez/hci0'))
    self.assertFalse(bluez_registration_lost(changed('org.bluez', '', ':1.3'), '/org/bluez/hci0'))
    self.assertFalse(bluez_registration_lost(changed('other.service', ':1.2', ''), '/org/bluez/hci0'))

  async def test_slow_rpc_does_not_block_phone_gps(self):
    engine = BlePeerEngine.__new__(BlePeerEngine)
    engine.rpc_active = 0
    engine.incoming = asyncio.Queue()
    engine.send_authenticated_error = AsyncMock()
    slow_started, release, gps = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def handle(payload, device):
      assert device == "/phone"
      if json.loads(payload)["method"] == "searchNavigation":
        slow_started.set()
        await release.wait()
      else:
        gps.set()
    engine.handle_encrypted = handle
    stop = asyncio.Event()
    task = asyncio.create_task(engine.process_messages(stop))
    try:
      await engine.incoming.put(('/phone', b'{"method":"searchNavigation"}'))
      await asyncio.wait_for(slow_started.wait(), 1)
      await engine.incoming.put(('/phone', b'{"method":"updatePhoneGps"}'))
      await asyncio.wait_for(gps.wait(), 0.2)
    finally:
      release.set()
      stop.set()
      await asyncio.wait_for(task, 1)

  def test_status_is_public_read_only_and_reflects_pairing_mode(self):
    status = StatusCharacteristic('public-device-key')
    self.assertEqual(status.Flags, ['read'])
    read = StatusCharacteristic.ReadValue.__wrapped__
    for active in (True, False):
      with patch('openpilot.system.asius.bluetoothd.pairing_mode_active', return_value=active):
        value = read(status, {})
        self.assertEqual(json.loads(value), {"v": 1, "publicKey": "public-device-key", "pairingMode": active})
        self.assertEqual(read(status, {"offset": Variant('q', 10)}), value[10:])
        with self.assertRaises(DBusError):
          read(status, {"offset": Variant('q', len(value) + 1)})

  def test_name_and_discovery_are_hidden_outside_pairing_mode(self):
    with patch('openpilot.system.asius.bluetoothd.pairing_mode_active', return_value=False), \
         patch('openpilot.system.asius.bluetoothd.methods.getDeviceName', return_value="Asius v0"):
      advertisement = Advertisement()
      self.assertEqual(advertisement.ServiceUUIDs, [BLE_SERVICE_UUID])
      self.assertEqual(advertisement.LocalName, '')
      self.assertFalse(advertisement.Discoverable)
      with self.assertRaises(DBusError):
        PairingAgent.require_pairing_mode()

  def test_pairing_mode_announces_name_and_service(self):
    with patch('openpilot.system.asius.bluetoothd.pairing_mode_active', return_value=True), \
         patch('openpilot.system.asius.bluetoothd.methods.getDeviceName', return_value="Asius v0"):
      advertisement = Advertisement()
      self.assertEqual(advertisement.LocalName, 'Asius v0')
      self.assertTrue(advertisement.Discoverable)
      self.assertEqual(advertisement.ServiceUUIDs, [BLE_SERVICE_UUID])

  async def test_closing_pairing_refreshes_advertisement(self):
    stop = MagicMock()
    stop.is_set.side_effect = [False, False, True]
    stop.wait = AsyncMock()
    bus = MagicMock()
    with patch('openpilot.system.asius.bluetoothd.connected_device_count', new=AsyncMock(return_value=0)), \
         patch('openpilot.system.asius.bluetoothd.pairing_mode_active', side_effect=[True, False]), \
         patch('openpilot.system.asius.bluetoothd.methods.getDeviceName', return_value='Asius v0'), \
         patch('openpilot.system.asius.bluetoothd.set_adapter_property', new=AsyncMock()) as set_property, \
         patch('openpilot.system.asius.bluetoothd.refresh_advertisement', new=AsyncMock()) as refresh:
      await keep_advertising(bus, '/adapter', stop)
    self.assertEqual(refresh.await_count, 2)
    self.assertEqual([call.args[2] for call in set_property.call_args_list], ['Pairable', 'Alias', 'Pairable'])
    self.assertEqual([call.args[3].value for call in set_property.call_args_list], [True, 'Asius v0', False])

  async def test_advertising_recovers_after_connected_refresh_fails(self):
    stop = MagicMock()
    stop.is_set.side_effect = [False, False, False, True]
    stop.wait = AsyncMock()
    bus = MagicMock()
    with patch('openpilot.system.asius.bluetoothd.connected_device_count', new=AsyncMock(side_effect=[0, 1, 0])), \
         patch('openpilot.system.asius.bluetoothd.pairing_mode_active', return_value=False), \
         patch('openpilot.system.asius.bluetoothd.methods.getDeviceName', return_value='comma 3X'), \
         patch('openpilot.system.asius.bluetoothd.set_adapter_property', new=AsyncMock()), \
         patch('openpilot.system.asius.bluetoothd.cloudlog.exception'), \
         patch('openpilot.system.asius.bluetoothd.refresh_advertisement', new=AsyncMock(side_effect=[None, RuntimeError('busy'), None])) as refresh:
      await keep_advertising(bus, '/adapter', stop)
    self.assertEqual(refresh.await_count, 3)


class TestBluetoothAuthorization(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.device_key = ed25519.Ed25519PrivateKey.generate()
    self.client = SecureSession(ed25519.Ed25519PrivateKey.generate(), 'bluetooth')
    self.engine = BlePeerEngine.__new__(BlePeerEngine)
    self.engine.rpc_active = 0
    self.engine.dongle_id = SecureSession(self.device_key, 'bluetooth').public_key
    self.engine.active_peers = {}
    self.engine.sessions = {}
    self.engine.peer_devices = {}
    self.engine.tx = MagicMock(send_text=AsyncMock(return_value=True))
    self.engine.approvals = MagicMock()
    self.storage_key = public_text(ec.generate_private_key(ec.SECP256R1()))

  async def establish(self):
    with patch('openpilot.system.asius.bluetoothd.get_device_private_key', return_value=self.device_key):
      await self.engine.handle_encrypted(wrap_peer(self.client.begin(self.engine.dongle_id)).encode(), '/phone')
    reply = self.engine.tx.send_text.call_args.args[0]
    self.client.receive(unwrap_peer(reply, self.client.public_key))

  async def deliver(self, body):
    message = wrap_peer(self.client.encrypt(self.engine.dongle_id, json.dumps(body)))
    await self.engine.handle_encrypted(message.encode(), '/phone')
    return message

  def response(self):
    wire = unwrap_peer(self.engine.tx.send_text.call_args.args[0], self.client.public_key)
    return json.loads(self.client.receive(wire)['text'])

  async def test_bond_alone_never_authorizes_an_app(self):
    await self.establish()
    self.engine.approvals.consume.return_value = False
    with patch('openpilot.system.asius.bluetoothd.pairing_mode_active', return_value=True), \
         patch('openpilot.system.asius.bluetoothd.load_authorized_peers', return_value={}), \
         patch('openpilot.system.asius.bluetoothd.authorize_peer') as authorize:
      await self.deliver({'type': 'ble-pair-request', 'publicKey': self.client.public_key, 'storagePublicKey': self.storage_key, 'requestId': 'pair'})
    authorize.assert_not_called()
    self.assertEqual(self.response()['type'], 'ble-error')
    self.engine.approvals.consume.assert_called_once_with('/phone')

  async def test_approved_peer_enrollment_binds_recording_key_and_closes_mode(self):
    await self.establish()
    self.engine.approvals.consume.side_effect = lambda device: device == '/phone'
    with patch('openpilot.system.asius.bluetoothd.pairing_mode_active', return_value=True), \
         patch('openpilot.system.asius.bluetoothd.load_authorized_peers', return_value={}), \
         patch('openpilot.system.asius.bluetoothd.authorize_peer') as authorize, \
         patch('openpilot.system.asius.bluetoothd.disable_pairing_mode') as close, \
         patch('openpilot.system.asius.bluetoothd.methods.getDeviceName', return_value='Asius v0'):
      await self.deliver({'type': 'ble-pair-request', 'publicKey': self.client.public_key, 'storagePublicKey': self.storage_key, 'requestId': 'pair'})
    authorize.assert_called_once_with(self.client.public_key, self.storage_key, None)
    close.assert_called_once()
    self.assertEqual(self.response()['type'], 'pair-response')

  async def test_known_app_reconnect_requires_a_fresh_authenticated_session(self):
    await self.establish()
    with patch('openpilot.system.asius.bluetoothd.load_authorized_peers', return_value={self.client.public_key: {}}):
      message = await self.deliver({'type': 'ble-session', 'requestId': 'fresh'})
      self.assertEqual(self.response(), {'type': 'ble-session', 'ready': True, 'requestId': 'fresh'})
      with self.assertRaises(ValueError):
        await self.engine.handle_encrypted(message.encode(), '/phone')
    self.engine.approvals.consume.assert_not_called()

  async def test_unknown_app_cannot_send_commands(self):
    await self.establish()
    self.engine.handle_rpc = AsyncMock()
    with patch('openpilot.system.asius.bluetoothd.load_authorized_peers', return_value={}):
      await self.deliver({'method': 'reboot', 'id': 1})
      self.assertEqual(self.response()['type'], 'ble-error')
    self.engine.handle_rpc.assert_not_awaited()

  async def test_second_client_and_disconnect_preserve_first_session(self):
    await self.establish()
    second = SecureSession(ed25519.Ed25519PrivateKey.generate(), 'bluetooth')
    self.engine.assembler = MagicMock(partial={})
    with patch('openpilot.system.asius.bluetoothd.get_device_private_key', return_value=self.device_key):
      await self.engine.handle_encrypted(wrap_peer(second.begin(self.engine.dongle_id)).encode(), '/second')
    second.receive(unwrap_peer(self.engine.tx.send_text.call_args.args[0], second.public_key))
    with patch('openpilot.system.asius.bluetoothd.load_authorized_peers', return_value={self.client.public_key: {}, second.public_key: {}}):
      wire = wrap_peer(second.encrypt(self.engine.dongle_id, json.dumps({'type': 'ble-session', 'requestId': 'second'})))
      await self.engine.handle_encrypted(wire.encode(), '/second')
      reply = unwrap_peer(self.engine.tx.send_text.call_args.args[0], second.public_key)
      self.assertTrue(json.loads(second.receive(reply)['text'])['ready'])
      await self.deliver({'type': 'ble-session', 'requestId': 'first'})
      self.assertEqual(self.response()['requestId'], 'first')
      self.engine.disconnect('/second')
      self.assertNotIn('/second', self.engine.sessions)
      await self.deliver({'type': 'ble-session', 'requestId': 'still-first'})
      self.assertEqual(self.response()['requestId'], 'still-first')

  async def test_agent_rejects_just_works(self):
    agent = PairingAgent(MagicMock())
    with self.assertRaises(DBusError):
      PairingAgent.RequestAuthorization.__wrapped__(agent, '/phone')



@pytest.fixture(autouse=True)
def ignition_off_for_bluetooth_tests(monkeypatch):
  monkeypatch.setattr('openpilot.system.asius.access_policy.ignition_state', lambda: False)

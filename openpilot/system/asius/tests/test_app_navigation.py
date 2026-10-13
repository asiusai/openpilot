import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from openpilot.common.params import Params, ParamKeyFlag
from openpilot.selfdrive.navigation.navd import RouteEngine
from openpilot.selfdrive.navigation.renderer import MAJOR, STREET, road_style
from openpilot.selfdrive.navigation.tests.test_navigation import field, varint
from openpilot.selfdrive.navigation.vector_tile import decode_roads
from openpilot.system.asius import methods
from openpilot.system.asius import navigation
from openpilot.system.asius.mapbox import MapboxTiles


def test_navigation_rpc_uses_upstream_params(tmp_path, monkeypatch):
  params = Params(str(tmp_path))
  monkeypatch.setattr(methods.upstream_athena, 'Params', lambda: params)
  monkeypatch.setattr(navigation, 'Params', lambda: params)
  place = {'latitude': 59.4, 'longitude': 24.7, 'name': 'Home', 'address': 'Tallinn, Estonia'}
  for transport in ('network', 'bluetooth'):
    dispatcher = methods.dispatcher_for_peer('test')
    result = json.loads(methods.handle({'id': 1, 'method': 'setNavDestination', 'params': place}, dispatcher))
    assert result['result']['success'] == 1
    assert {key: params.get('NavDestination')[key] for key in place} == place
    assert params.get_bool('NavigationEnabled')
    result = json.loads(methods.handle({'id': 2, 'method': 'setNavDestination', 'params': {**place, 'latitude': 100}}, dispatcher))
    assert result['result']['success'] == 0
    assert params.get('NavDestination')['latitude'] == 59.4
    methods.handle({'id': 3, 'method': 'clearNavDestination'}, dispatcher)
    assert params.get('NavDestination') is None
    if transport == 'bluetooth':
      from openpilot.system.asius.bluetoothd import BlePeerEngine
      engine = BlePeerEngine.__new__(BlePeerEngine)
      engine.rpc_active = 0
      engine.send_body = AsyncMock()
      asyncio.run(engine.handle_rpc('test', {'id': 4, 'method': 'setNavDestination', 'params': place}))
      assert engine.send_body.call_args.args[1]['result']['success'] == 1
      asyncio.run(engine.handle_rpc('test', {'id': 5, 'method': 'getRecentNavDestinations'}))
      assert engine.send_body.call_args.args[1]['result'] == [place]
    else:
      result = json.loads(methods.handle({'id': 5, 'method': 'getRecentNavDestinations'}, dispatcher))
      assert result['result'] == [place]


def test_recent_destinations_persist_and_merge(tmp_path, monkeypatch):
  monkeypatch.setattr(navigation, 'Params', lambda: Params(str(tmp_path)))
  params = Params(str(tmp_path))
  for index in range(25):
    assert navigation.setNavDestination(59, 24 + index / 100, f'Place {index}', 'Address')['success'] == 1
  recent = navigation.getRecentNavDestinations()
  assert len(recent) == 20
  assert recent[0]['name'] == 'Place 24'
  assert recent[-1]['name'] == 'Place 5'
  assert navigation.setNavDestination(59, 24.1, ' Renamed place ', ' Updated address ')['success'] == 1
  recent = navigation.getRecentNavDestinations()
  assert len(recent) == 20
  assert recent[0] == {'latitude': 59, 'longitude': 24.1, 'name': 'Renamed place', 'address': 'Updated address'}
  params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START | ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION)
  assert params.get('NavDestination') is None
  assert Params(str(tmp_path)).get(navigation.RECENT_DESTINATIONS) == recent


@pytest.mark.parametrize('field,value', [('latitude', float('nan')), ('longitude', float('inf')), ('latitude', True), ('name', ''), ('name', 'x' * 257), ('address', 'x' * 1025)])
def test_invalid_destination_cannot_change_route_or_history(tmp_path, monkeypatch, field, value):
  monkeypatch.setattr(navigation, 'Params', lambda: Params(str(tmp_path)))
  place = {'latitude': 59, 'longitude': 24, 'name': 'Home', 'address': 'Tallinn'}
  navigation.setNavDestination(**place)
  assert navigation.setNavDestination(**{**place, field: value})['success'] == 0
  assert navigation.getRecentNavDestinations() == [place]
  assert {key: Params(str(tmp_path)).get('NavDestination')[key] for key in place} == place


def test_parallel_transport_updates_do_not_drop_destinations(tmp_path, monkeypatch):
  from concurrent.futures import ThreadPoolExecutor
  monkeypatch.setattr(navigation, 'Params', lambda: Params(str(tmp_path)))
  # Independent Params handles and lock descriptors, like separate transport processes.
  with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(lambda index: navigation.setNavDestination(59, 24 + index / 100, f'Place {index}', ''), range(20)))
  assert all(result['success'] == 1 for result in results)
  assert len(navigation.getRecentNavDestinations()) == 20


def test_malformed_history_does_not_block_navigation(tmp_path, monkeypatch):
  monkeypatch.setattr(navigation, 'Params', lambda: Params(str(tmp_path)))
  Params(str(tmp_path)).put(navigation.RECENT_DESTINATIONS, [None, {}, {'latitude': True}], block=True)
  assert navigation.getRecentNavDestinations() == []
  assert navigation.setNavDestination(59, 24, 'Home', '') == {'success': 1}
  assert len(navigation.getRecentNavDestinations()) == 1


def test_route_engine_uses_asius_proxy_without_mapbox_token(monkeypatch):
  monkeypatch.delenv('MAPBOX_TOKEN', raising=False)
  values = {}
  params = MagicMock()
  params.get.side_effect = values.get
  with patch('openpilot.selfdrive.navigation.navd.Api'):
    engine = RouteEngine(MagicMock(), MagicMock(), params, store=MagicMock())
    engine.api.get_token.return_value = 'signed-device-proof'
  engine.last_position = (24.7, 59.4)
  with patch('requests.get') as get:
    get.return_value.json.return_value = {'routes': []}
    engine.calculate_route((24.8, 59.5))
  assert get.call_args.args[0].startswith('https://api.asius.ai/v1/nav/directions/')
  assert 'access_token' not in get.call_args.kwargs['params']
  assert get.call_args.kwargs['headers']['Authorization'] == 'Bearer signed-device-proof'


def mapbox_road_tile(road_class, structure='none'):
  geometry = b''.join(varint(v) for v in [9, 4096, 0, 10, 0, 8192])
  feature = field(2, b'\x00\x00\x01\x01') + field(3, 2) + field(4, geometry)
  layer = field(15, 2) + field(1, b'road') + field(2, feature)
  layer += field(3, b'class') + field(3, b'structure')
  layer += field(4, field(1, road_class.encode())) + field(4, field(1, structure.encode())) + field(5, 4096)
  return field(3, layer)


@pytest.mark.parametrize('kind,style', [('primary', MAJOR), ('street', STREET)])
def test_mapbox_roads_use_upstream_styles(kind, style):
  roads = decode_roads(mapbox_road_tile(kind, 'bridge'))
  assert len(roads) == 1
  assert road_style(roads[0]) == style
  assert roads[0].bridge and not roads[0].tunnel
  assert roads[0].points.tolist() == [[0.5, 0], [0.5, 1]]
  assert decode_roads(mapbox_road_tile('path')) == []
  assert decode_roads(mapbox_road_tile('ferry')) == []


def test_mapbox_cache_and_redacted_errors():
  tiles = MapboxTiles()
  tiles.api = MagicMock()
  try:
    with patch('requests.get') as get:
      get.return_value.content = b'tile'
      assert tiles.get(15, 12, 34) == tiles.get(15, 12, 34) == b'tile'
      assert get.call_count == 1
      get.side_effect = requests.HTTPError('private token-bearing URL')
      with pytest.raises(OSError, match='^Unable to fetch Mapbox tiles$'):
        tiles.get(15, 12, 35)
  finally:
    tiles.close()


def test_device_navigation_proof_uses_base58_identity(monkeypatch):
  from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
  from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
  from openpilot.system.asius import mapbox
  from openpilot.system.asius.identity import bytes_to_identity
  key = Ed25519PrivateKey.generate().public_key()
  monkeypatch.setattr(mapbox, 'get_device_public_key', lambda: key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode())
  monkeypatch.setattr(mapbox, 'sign_jwt', lambda claims, expiry_seconds: (claims, expiry_seconds))
  claims, expiry = mapbox.NavigationApi().get_token()
  assert claims == {'type': 'navigation', 'aud': 'asius-navigation', 'identity': bytes_to_identity(key.public_bytes(Encoding.Raw, PublicFormat.Raw))}
  assert expiry == 60

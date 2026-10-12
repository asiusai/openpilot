import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from openpilot.common.params import Params
from openpilot.selfdrive.navigation.navd import RouteEngine
from openpilot.selfdrive.navigation.renderer import MAJOR, STREET, road_style
from openpilot.selfdrive.navigation.tests.test_navigation import field, varint
from openpilot.selfdrive.navigation.vector_tile import decode_roads
from openpilot.system.asius import methods
from openpilot.system.asius.mapbox import MapboxTiles


def test_navigation_rpc_uses_upstream_params(tmp_path, monkeypatch):
  params = Params(str(tmp_path))
  monkeypatch.setattr(methods.upstream_athena, 'Params', lambda: params)
  for transport in ('network', 'bluetooth'):
    dispatcher = methods.dispatcher_for_peer('test')
    result = json.loads(methods.handle({'id': 1, 'method': 'setNavDestination', 'params': {'latitude': 59.4, 'longitude': 24.7}}, dispatcher))
    assert result['result']['success'] == 1
    assert params.get('NavDestination') == {'latitude': 59.4, 'longitude': 24.7}
    result = json.loads(methods.handle({'id': 2, 'method': 'setNavDestination', 'params': {'latitude': 100, 'longitude': 0}}, dispatcher))
    assert result['result']['success'] == 0
    assert params.get('NavDestination')['latitude'] == 59.4
    methods.handle({'id': 3, 'method': 'clearNavDestination'}, dispatcher)
    assert params.get('NavDestination') is None
    if transport == 'bluetooth':
      from openpilot.system.asius.bluetoothd import BlePeerEngine
      engine = BlePeerEngine.__new__(BlePeerEngine)
      engine.send_body = AsyncMock()
      asyncio.run(engine.handle_rpc('test', {'id': 4, 'method': 'setNavDestination', 'params': {'latitude': 59, 'longitude': 24}}))
      assert engine.send_body.call_args.args[1]['result']['success'] == 1


def test_route_engine_uses_asius_proxy_without_mapbox_token(monkeypatch):
  monkeypatch.delenv('MAPBOX_TOKEN', raising=False)
  values = {}
  params = MagicMock()
  params.get.side_effect = values.get
  with patch('openpilot.selfdrive.navigation.navd.Api'):
    engine = RouteEngine(MagicMock(), MagicMock(), params)
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

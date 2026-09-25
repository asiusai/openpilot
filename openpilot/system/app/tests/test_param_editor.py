import base64
import datetime
from types import SimpleNamespace

import pytest

from openpilot.common.params import Params
from openpilot.system.app.param_editor import CHUNK_BYTES, ParameterEditor, decode_value, encode_value


class ForkParams:
  def __init__(self):
    self.values = {'IsOffroad': True, 'ForkOnly': 2**60, 'Secret': 'private', 'Long': '🙂' * 6000}
    self.defaults = {'Unset': 1.25}
    self.types = {'IsOffroad': 'bool', 'ForkOnly': 'int', 'Secret': 'string', 'Long': 'string', 'Unset': 'float'}

  def all_keys(self):
    return [key.encode() for key in self.types]

  def check_key(self, key):
    if key not in self.types:
      raise ValueError('Unknown parameter')

  def get_type(self, key):
    return SimpleNamespace(name=self.types[key].upper())

  def get(self, key):
    return self.values.get(key)

  def get_default_value(self, key):
    return self.defaults.get(key)

  def get_bool(self, key):
    return self.values.get(key, False)

  def put(self, key, value, block=False):
    assert block
    self.values[key] = value

  def remove(self, key):
    self.values.pop(key, None)


@pytest.fixture
def editor():
  params = ForkParams()
  return ParameterEditor({'Secret'}, lambda: params), params


def entry(editor, key):
  after = ''
  while True:
    page = editor.catalog(after)
    for value in page['params']:
      if value['name'] == key:
        return value
    assert page['next']
    after = page['next']


def write(editor, key, text, **kwargs):
  return editor.write('peer', key, entry(editor, key)['revision'], 'test', 0, base64.b64encode(text.encode()).decode(), True, **kwargs)


def test_catalog_follows_fork_and_does_not_expose_protected_or_large_values(editor):
  api, _ = editor
  first = api.catalog(limit=2)
  second = api.catalog(first['next'], limit=2)
  assert first['total'] == 5 and first['canEdit']
  assert {p['name'] for p in first['params']}.isdisjoint(p['name'] for p in second['params'])
  assert entry(api, 'ForkOnly')['value'] == str(2**60)
  assert entry(api, 'Unset')['defaultValue'] == '1.25'
  assert entry(api, 'Unset')['isSet'] is False
  assert entry(api, 'Secret')['value'] is None
  assert entry(api, 'Long')['value'] is None
  assert entry(api, 'Long')['size'] == 24000


def test_atomic_chunks_conflicts_and_peer_isolation(editor):
  api, params = editor
  revision = entry(api, 'Long')['revision']
  output, offset = bytearray(), 0
  while True:
    part = api.read('Long', revision, offset)
    output.extend(base64.b64decode(part['chunk']))
    offset = part['nextOffset']
    if part['done']:
      break
  assert output.decode() == params.values['Long']
  replacement = ('ä🙂' * 2000).encode()
  part = base64.b64encode(replacement[:CHUNK_BYTES]).decode()
  api.write('peer', 'Long', revision, 'long', 0, part, False)
  assert params.values['Long'] == '🙂' * 6000
  with pytest.raises(ValueError):
    api.write('other', 'Long', revision, 'long', CHUNK_BYTES, '', True)
  with pytest.raises(ValueError):
    api.write('peer', 'Long', revision, 'long', 1, '', True)
  api.write('peer', 'Long', revision, 'long', CHUNK_BYTES, base64.b64encode(replacement[CHUNK_BYTES:]).decode(), True)
  assert params.values['Long'].encode() == replacement
  with pytest.raises(ValueError, match='changed'):
    api.read('Long', revision)
  write(api, 'Long', '', clear=True)
  assert 'Long' not in params.values


def test_protected_and_onroad_writes_rejected_even_mid_upload(editor):
  api, params = editor
  with pytest.raises(ValueError, match='Managed'):
    write(api, 'Secret', 'new')
  revision = entry(api, 'ForkOnly')['revision']
  api.write('peer', 'ForkOnly', revision, 'pending', 0, 'MTIz', False)
  params.values['IsOffroad'] = False
  with pytest.raises(ValueError, match='Park'):
    api.write('peer', 'ForkOnly', revision, 'pending', 3, '', True)
  assert params.values['ForkOnly'] == 2**60
  params.values['IsOffroad'] = True
  params.values['ForkOnly'] = 5
  with pytest.raises(ValueError, match='changed'):
    api.write('peer', 'ForkOnly', revision, 'pending', 3, '', True)


@pytest.mark.parametrize('kind,text', [('bool', 'yes'), ('int', '1.2'), ('float', 'NaN'),
                                       ('json', '{"x":1e999}'), ('json', 'null'), ('bytes', '!!!'), ('time', 'no')])
def test_invalid_types(kind, text):
  with pytest.raises(ValueError):
    decode_value(text, kind)


@pytest.mark.parametrize('kind,value', [('bool', True), ('int', 2**60), ('float', 1.25),
                                        ('json', {'x': [1, False]}), ('string', 'ä🙂'), ('bytes', bytes(range(256))),
                                        ('time', datetime.datetime(2026, 9, 25, tzinfo=datetime.UTC))])
def test_typed_round_trip(kind, value):
  assert decode_value(encode_value(value, kind), kind) == value


def test_actual_device_registry(tmp_path):
  params = Params(str(tmp_path))
  params.put_bool('IsOffroad', True)
  api = ParameterEditor({'DongleId'}, lambda: params)
  names = []
  after = ''
  while True:
    page = api.catalog(after)
    names.extend(value['name'] for value in page['params'])
    if not page['next']:
      break
    after = page['next']
  assert 'OpenpilotEnabledToggle' in names
  assert len(names) == len(params.all_keys())
  write(api, 'ExperimentalMode', 'true')
  assert params.get_bool('ExperimentalMode')

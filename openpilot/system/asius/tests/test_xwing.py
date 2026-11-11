import json
from pathlib import Path

import pytest

from openpilot.system.asius.xwing import PrivateKey, PublicKey


@pytest.mark.parametrize('vector', json.loads(Path(__file__).with_name('xwing.json').read_text()))
def test_upstream_vectors(vector):
  key = PrivateKey(bytes.fromhex(vector['seed']))
  public, ciphertext, secret = (bytes.fromhex(vector[field]) for field in ('pk', 'ct', 'ss'))
  assert key.public_key().public_bytes_raw() == public
  assert key.decapsulate(ciphertext) == secret
  generated_secret, generated_ciphertext = PublicKey(public).encapsulate()
  assert key.decapsulate(generated_ciphertext) == generated_secret
  for offset in (0, 1088):
    changed = bytearray(ciphertext)
    changed[offset] ^= 1
    assert key.decapsulate(bytes(changed)) != secret
  with pytest.raises(ValueError):
    key.decapsulate(ciphertext[:-1])
  with pytest.raises(ValueError):
    PublicKey(public[:-1])
  with pytest.raises(ValueError):
    key.decapsulate(ciphertext[:1088] + bytes(32))
  with pytest.raises(ValueError):
    PublicKey(public[:1184] + bytes(32)).encapsulate()
  with pytest.raises(ValueError):
    PublicKey(bytes([255]) * 1152 + public[1152:]).encapsulate()

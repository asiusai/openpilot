from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.common.version import terms_version, training_version

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openpilot.system.asius.service_registration import registration_proof, set_service_hosts, TERMS_VERSION


def test_registration_proof_binds_device_challenge_and_terms(tmp_path):
  params = Params(str(tmp_path))
  params.put("HasAcceptedTerms", terms_version, block=True)
  key = Ed25519PrivateKey.generate()
  with patch('openpilot.system.asius.identity.get_device_identity', return_value='device-key'), \
       patch('openpilot.system.asius.relayd.get_device_private_key', return_value=key), \
       patch('openpilot.system.asius.relayd.wall_time', return_value=1_800_000_000):
    token = registration_proof('ab' * 32, TERMS_VERSION, params)
  proof = jwt.decode(token, key.public_key(), algorithms=['EdDSA'], audience='asius-services',
                     options={'verify_exp': False, 'verify_iat': False, 'verify_nbf': False})
  assert proof['type'] == 'service-registration'
  assert proof['device'] == 'device-key'
  assert proof['challenge'] == 'ab' * 32
  assert proof['terms'] == TERMS_VERSION
  assert proof['openpilotTerms'] == terms_version
  assert proof['exp'] - proof['iat'] == 300


@pytest.mark.parametrize('challenge,terms', [('short', TERMS_VERSION), ('AB' * 32, TERMS_VERSION), ('ab' * 32, 'old'), (None, TERMS_VERSION)])
def test_registration_rejects_invalid_requests(challenge, terms):
  with pytest.raises(ValueError):
    registration_proof(challenge, terms)


def test_registration_requires_current_openpilot_terms(tmp_path):
  params = Params(str(tmp_path))
  for value in ['0', 'old']:
    params.put('HasAcceptedTerms', value, block=True)
    with pytest.raises(PermissionError):
      registration_proof('ab' * 32, TERMS_VERSION, params)


def test_service_choice_preserves_training_and_supports_bluetooth_only(tmp_path):
  params = Params(str(tmp_path))
  assert params.get('RelayEnabled', return_default=True) is True
  assert params.get('CloudUploadEnabled', return_default=True) is True
  params.put_bool('IsOffroad', True, block=True)
  params.put('HasAcceptedTerms', terms_version, block=True)
  params.put('CompletedTrainingVersion', training_version, block=True)
  assert set_service_hosts('', '', params) == {'relay': '', 'cloud': ''}
  assert not params.get_bool('RelayEnabled')
  assert not params.get_bool('CloudUploadEnabled')
  set_service_hosts('wss://relay.example/', 'https://cloud.example:8443/', params)
  assert params.get('RelayHost') == 'wss://relay.example'
  assert params.get_bool('RelayEnabled')
  assert params.get_bool('CloudUploadEnabled')
  assert params.get('HasAcceptedTerms') == terms_version
  assert params.get('CompletedTrainingVersion') == training_version
  with pytest.raises(ValueError):
    set_service_hosts('wss://valid.example', 'http://invalid.example', params)
  assert params.get('RelayHost') == 'wss://relay.example'
  params.put_bool('IsOffroad', False, block=True)
  with pytest.raises(PermissionError):
    set_service_hosts('', '', params)


@pytest.mark.parametrize('relay,cloud', [
  ('ws://relay.example', ''), ('wss://relay.example/path', ''), ('wss://@relay.example', ''),
  ('', 'https://user@cloud.example'), ('', 'https://cloud.example?query'), ('', 'https://cloud.example#fragment'),
  ('', 'https://cloud.example:65536'), ('', 'https://cloud.example/../'),
])
def test_invalid_services_do_not_change_device_settings(tmp_path, relay, cloud):
  params = Params(str(tmp_path))
  params.put_bool('IsOffroad', True, block=True)
  with pytest.raises(ValueError):
    set_service_hosts(relay, cloud, params)
  assert params.get('RelayHost') is None
  assert params.get('CloudHost') is None
  assert params.get('RelayEnabled', return_default=True) is True
  assert params.get('CloudUploadEnabled', return_default=True) is True

import pytest

from openpilot.common.params import Params
from openpilot.common.version import terms_version, training_version
from openpilot.system.asius.onboarding import Onboarding, setup_status


@pytest.fixture
def setup(tmp_path):
  params = Params(str(tmp_path))
  params.put_bool('IsOffroad', True, block=True)
  return Onboarding(params), params


def test_existing_acceptance_is_preserved(setup):
  _, params = setup
  params.put('HasAcceptedTerms', terms_version, block=True)
  params.put('CompletedTrainingVersion', training_version, block=True)
  assert setup_status(params)['complete']


def test_fresh_device_requires_terms_and_training(setup):
  flow, params = setup
  assert params.get('HasAcceptedTerms', return_default=True) == '0'
  assert params.get('CompletedTrainingVersion', return_default=True) == '0'
  assert not setup_status(params)['complete']
  with pytest.raises(ValueError):
    flow.complete(training_version, False, False)


def test_terms_versions_and_training_without_camera(setup):
  flow, params = setup
  with pytest.raises(ValueError):
    flow.accept('old')
  flow.accept(terms_version)
  with pytest.raises(ValueError):
    flow.complete('old', False, False)
  assert flow.complete(training_version, False, False)['complete']
  assert not params.get_bool('RecordFront')
  assert not params.get_bool('ShareDrivingData')
  assert not params.get_bool('IsDriverViewEnabled')
  assert setup_status(Params(params.get_param_path().rsplit('/', 1)[0]))['complete']


def test_reset_requires_parked(setup):
  flow, params = setup
  flow.accept(terms_version)
  flow.complete(training_version, True, True)
  params.put_bool('IsOffroad', False, block=True)
  with pytest.raises(PermissionError):
    flow.reset()
  assert setup_status(params)['complete']
  params.put_bool('IsOffroad', True, block=True)
  assert not flow.reset()['complete']
  assert params.get('HasAcceptedTerms') == '0'
  assert params.get('CompletedTrainingVersion') == '0'

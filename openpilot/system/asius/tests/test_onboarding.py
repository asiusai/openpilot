from types import SimpleNamespace

import pytest

from openpilot.common.params import Params
from openpilot.common.version import terms_version, training_version
from openpilot.system.asius.onboarding import Onboarding, setup_status


class DriverState:
  def __init__(self):
    self.now = 100.
    self.fresh = True
    self.face = True
    self.orientation = [0., 0., 0.]
    self.seen = dict.fromkeys(['driverMonitoringState', 'driverStateV2'], True)
    self.valid = self.seen.copy()
    self.logMonoTime = {}

  def update(self, _):
    self.logMonoTime = dict.fromkeys(self.seen, (self.now if self.fresh else self.now - 10.) * 1e9)

  def __getitem__(self, key):
    if key == 'driverMonitoringState':
      return SimpleNamespace(isRHD=False, visionPolicyState=SimpleNamespace(faceDetected=self.face))
    return SimpleNamespace(leftDriverData=SimpleNamespace(faceOrientation=self.orientation))


@pytest.fixture
def setup(tmp_path):
  params = Params(str(tmp_path))
  params.put_bool('IsOffroad', True, block=True)
  sm = DriverState()
  flow = Onboarding(params, sm, lambda: sm.now)
  yield flow, params, sm
  flow.stop_check()


def detected(flow, sm):
  for _ in range(6):
    result = flow.check_driver()
    sm.now += 1.
  return result


def test_existing_acceptance_is_preserved(setup):
  _, params, _ = setup
  params.put('HasAcceptedTerms', terms_version, block=True)
  params.put('CompletedTrainingVersion', training_version, block=True)
  assert setup_status(params)['complete']


def test_fresh_device_requires_terms_and_training(setup):
  flow, params, _ = setup
  assert params.get('HasAcceptedTerms', return_default=True) == '0'
  assert params.get('CompletedTrainingVersion', return_default=True) == '0'
  assert not setup_status(params)['complete']
  with pytest.raises(PermissionError):
    flow.check_driver()


def test_terms_versions_and_training_order(setup):
  flow, params, sm = setup
  with pytest.raises(ValueError):
    flow.accept('old')
  flow.accept(terms_version)
  with pytest.raises(PermissionError):
    flow.complete(training_version, False, False)
  assert detected(flow, sm)['complete']
  assert flow.complete(training_version, False, False)['complete']
  assert not params.get_bool('RecordFront')
  assert not params.get_bool('ShareDrivingData')
  assert not params.get_bool('IsDriverViewEnabled')
  assert setup_status(Params(params.get_param_path().rsplit('/', 1)[0]))['complete']


@pytest.mark.parametrize('failure', ['stale', 'missing', 'turned', 'gap'])
def test_driver_check_rejects_bad_or_stale_samples(setup, failure):
  flow, _, sm = setup
  flow.accept(terms_version)
  if failure == 'stale':
    sm.fresh = False
  if failure == 'missing':
    sm.face = False
  if failure == 'turned':
    sm.orientation = [0., 1., 0.]
  for _ in range(6):
    result = flow.check_driver()
    sm.now += 3. if failure == 'gap' else 1.
  assert not result['complete']


def test_reset_requires_parked_and_invalidates_old_check(setup):
  flow, params, sm = setup
  flow.accept(terms_version)
  detected(flow, sm)
  flow.complete(training_version, True, True)
  params.put_bool('IsOffroad', False, block=True)
  with pytest.raises(PermissionError):
    flow.reset()
  assert setup_status(params)['complete']
  params.put_bool('IsOffroad', True, block=True)
  assert not flow.reset()['complete']
  flow.accept(terms_version)
  with pytest.raises(PermissionError):
    flow.complete(training_version, False, False)


def test_stopping_driver_check_releases_camera(setup):
  flow, params, _ = setup
  flow.accept(terms_version)
  flow.check_driver()
  assert params.get_bool('IsDriverViewEnabled')
  flow.stop_check()
  assert not params.get_bool('IsDriverViewEnabled')

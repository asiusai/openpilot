import time
from unittest.mock import Mock

import pytest

from openpilot.cereal import messaging
from openpilot.system.hardware.chestnut import app_status


@pytest.fixture
def gpu(monkeypatch):
  monkeypatch.setattr(app_status, 'chestnut_compiled', lambda: True)
  sm = messaging.SubMaster(['deviceState', 'modelV2'])
  values = {}
  params = Mock()
  params.get.side_effect = values.get

  def publish(name, data, age=0, valid=True):
    msg = messaging.new_message(name)
    msg.valid = valid
    msg.logMonoTime = time.monotonic_ns() - int(age * 1e9)
    setattr(msg, name, data)
    sm.update_msgs(time.monotonic(), [msg])

  return lambda: app_status.gpu_status(sm, params), publish, values


def test_presence_does_not_claim_the_big_model_is_running(gpu, monkeypatch):
  status, publish, values = gpu
  assert status() == 'unknown'
  publish('deviceState', {'started': False, 'chestnutPresent': True})
  assert status() == 'ready'
  publish('deviceState', {'started': True, 'chestnutPresent': True})
  assert status() == 'loading'
  values['ChestnutLoading'] = True
  publish('modelV2', {'big': False})
  assert status() == 'loading'
  values['ChestnutLoading'] = False
  assert status() == 'failed'
  monkeypatch.setattr(app_status, 'chestnut_compiled', lambda: False)
  assert status() == 'uncompiled'


def test_active_requires_fresh_valid_big_model_and_successful_initialization(gpu):
  status, publish, values = gpu
  publish('deviceState', {'started': True, 'chestnutPresent': True})
  publish('modelV2', {'big': True})
  assert status() != 'active'
  values['ChestnutActive'] = True
  assert status() == 'active'
  publish('modelV2', {'big': True}, age=1.1)
  assert status() == 'failed'
  publish('modelV2', {'big': True}, valid=False)
  assert status() == 'failed'
  publish('modelV2', {'big': False})
  assert status() == 'failed'
  publish('modelV2', {'big': True})
  assert status() == 'active'
  values['ChestnutActive'] = False
  assert status() == 'failed'


def test_disconnection_is_only_a_fault_after_gpu_use(gpu):
  status, publish, values = gpu
  publish('deviceState', {'started': True, 'chestnutPresent': False})
  publish('modelV2', {'big': False})
  assert status() == 'disconnected'
  values['ChestnutActive'] = True
  assert status() == 'failed'
  publish('deviceState', {'started': False, 'chestnutPresent': False})
  assert status() == 'disconnected'


@pytest.mark.parametrize('alert', app_status.CHESTNUT_ALERTS)
def test_hardware_fault_overrides_apparent_gpu_readiness(gpu, alert):
  status, publish, values = gpu
  publish('deviceState', {'started': True, 'chestnutPresent': True})
  publish('modelV2', {'big': True})
  values['ChestnutActive'] = True
  values[alert] = {'text': 'Fault'}
  assert status() == 'failed'
  del values[alert]
  assert status() == 'active'


@pytest.mark.parametrize('age,valid', [(5.1, True), (0, False)])
def test_unavailable_device_state_cannot_claim_any_gpu_state(gpu, age, valid):
  status, publish, values = gpu
  publish('deviceState', {'started': True, 'chestnutPresent': True}, age=age, valid=valid)
  publish('modelV2', {'big': True})
  values['ChestnutActive'] = True
  assert status() == 'unknown'

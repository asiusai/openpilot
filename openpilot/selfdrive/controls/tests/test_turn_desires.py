from types import SimpleNamespace

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper, LANE_CHANGE_SPEED_MIN, TURN_SPEED_MAX, LaneChangeState


def car(**values):
  return SimpleNamespace(**{
    'vEgo': 5.0, 'leftBlinker': False, 'rightBlinker': False,
    'leftBlindspot': False, 'rightBlindspot': False,
    'steeringPressed': False, 'steeringTorque': 0.0, **values,
  })


@pytest.mark.parametrize('left,right,blind_left,blind_right,expected', [
  (True, False, False, False, log.Desire.turnLeft),
  (False, True, False, False, log.Desire.turnRight),
  (True, False, True, False, log.Desire.none),
  (False, True, False, True, log.Desire.none),
  (True, False, False, True, log.Desire.turnLeft),
  (False, True, True, False, log.Desire.turnRight),
  (False, False, False, False, log.Desire.none),
  (True, True, False, False, log.Desire.none),
])
def test_signal_and_blindspot_conditions(left, right, blind_left, blind_right, expected):
  helper = DesireHelper()
  helper.update(car(leftBlinker=left, rightBlinker=right, leftBlindspot=blind_left, rightBlindspot=blind_right), True, 1.0)
  assert helper.desire == expected
  assert helper.lane_change_state == LaneChangeState.off


@pytest.mark.parametrize('speed,expected', [
  (0.0, log.Desire.turnLeft),
  (TURN_SPEED_MAX - 0.001, log.Desire.turnLeft),
  (TURN_SPEED_MAX, log.Desire.none),
  (TURN_SPEED_MAX + 0.001, log.Desire.none),
  (LANE_CHANGE_SPEED_MIN - 0.001, log.Desire.none),
  (LANE_CHANGE_SPEED_MIN, log.Desire.none),
  (LANE_CHANGE_SPEED_MIN + 0.001, log.Desire.none),
  (-1.0, log.Desire.none),
  (float('nan'), log.Desire.none),
])
def test_turn_speed_boundary(speed, expected):
  helper = DesireHelper()
  helper.update(car(vEgo=speed, leftBlinker=True), True, 1.0)
  assert helper.desire == expected


def test_turn_clears_on_disengagement_signal_cancellation_hazards_and_blindspot():
  helper = DesireHelper()
  for stopped, active in [(car(leftBlinker=True), False), (car(), True),
                          (car(leftBlinker=True, rightBlinker=True), True),
                          (car(leftBlinker=True, leftBlindspot=True), True)]:
    helper.update(car(leftBlinker=True), True, 1.0)
    assert helper.desire == log.Desire.turnLeft
    helper.update(stopped, active, 1.0)
    assert helper.desire == log.Desire.none
    assert helper.lane_change_state == LaneChangeState.off


def test_signal_direction_changes_and_held_signal_are_consistent():
  helper = DesireHelper()
  for _ in range(100):
    helper.update(car(leftBlinker=True), True, 1.0)
    assert helper.desire == log.Desire.turnLeft
  helper.update(car(rightBlinker=True), True, 1.0)
  assert helper.desire == log.Desire.turnRight


@pytest.mark.parametrize('left', [True, False])
def test_held_signal_crosses_turn_and_lane_change_thresholds(left):
  helper = DesireHelper()
  state = car(vEgo=TURN_SPEED_MAX, leftBlinker=left, rightBlinker=not left)
  for speed in [TURN_SPEED_MAX, LANE_CHANGE_SPEED_MIN - 0.001, TURN_SPEED_MAX]:
    state.vEgo = speed
    helper.update(state, True, 1.0)
    assert helper.desire == log.Desire.none
    assert helper.lane_change_state == LaneChangeState.off

  state.vEgo = TURN_SPEED_MAX - 0.001
  helper.update(state, True, 1.0)
  assert helper.desire == (log.Desire.turnLeft if left else log.Desire.turnRight)

  for speed in [TURN_SPEED_MAX, LANE_CHANGE_SPEED_MIN]:
    state.vEgo = speed
    helper.update(state, True, 1.0)
    assert helper.desire == log.Desire.none
    assert helper.lane_change_state == LaneChangeState.off


@pytest.mark.parametrize('left,torque,desire', [(True, 1.0, log.Desire.laneChangeLeft), (False, -1.0, log.Desire.laneChangeRight)])
def test_above_threshold_retains_lane_change_confirmation(left, torque, desire):
  helper = DesireHelper()
  state = car(vEgo=LANE_CHANGE_SPEED_MIN, leftBlinker=left, rightBlinker=not left)
  helper.update(state, True, 1.0)
  assert helper.lane_change_state == LaneChangeState.preLaneChange
  helper.update(state, True, 1.0)
  assert helper.desire == log.Desire.none
  state.steeringPressed, state.steeringTorque = True, -torque
  helper.update(state, True, 1.0)
  assert helper.desire == log.Desire.none
  state.steeringTorque = torque
  state.leftBlindspot, state.rightBlindspot = left, not left
  helper.update(state, True, 1.0)
  assert helper.desire == log.Desire.none
  state.leftBlindspot, state.rightBlindspot = False, False
  helper.update(state, True, 1.0)
  assert helper.desire == desire
  helper.update(state, False, 1.0)
  assert helper.desire == log.Desire.none


def test_crossing_speed_threshold_does_not_start_unconfirmed_lane_change():
  helper = DesireHelper()
  state = car(leftBlinker=True, steeringPressed=True, steeringTorque=1)
  helper.update(state, True, 1.0)
  assert helper.desire == log.Desire.turnLeft
  state.vEgo = LANE_CHANGE_SPEED_MIN + 1
  helper.update(state, True, 1.0)
  assert helper.desire == log.Desire.none
  assert helper.lane_change_state == LaneChangeState.off
  state.leftBlinker = False
  helper.update(state, True, 1.0)
  state.leftBlinker = True
  helper.update(state, True, 1.0)
  helper.update(state, True, 1.0)
  assert helper.desire == log.Desire.laneChangeLeft
  state.vEgo = 5
  state.leftBlindspot = True
  helper.update(state, True, 1.0)
  assert helper.desire == log.Desire.none
  assert helper.lane_change_state == LaneChangeState.off

import math
import struct
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.system.sensord.sensors.asius_imu import transform_asius_imu
from openpilot.system.sensord.sensors import lsm6ds3_accel, lsm6ds3_gyro


# Independent physical bases from the saved v5 assembly. Columns are axes in
# the case frame. U6 is at 0 degrees on B.Cu; the compute assembly is flipped.
# The road lens points along camera-module -Y, after its -28 degree case tilt.
CASE_FROM_IMU = np.diag([-1.0, -1.0, 1.0])
CASE_FROM_DEVICE = np.array([
  [0.0, -1.0, 0.0],
  [-0.8829475928589292, 0.0, -0.4694715627858867],
  [0.4694715627858867, 0.0, -0.8829475928589292],
])


def device_from_event(v):
  # locationd's conversion from published sensor coordinates.
  return np.array([-v[2], -v[1], -v[0]])


def test_asius_imu_device_frame_rotation():
  # locationd converts the published Android sensor convention into the
  # camera-aligned device frame with this matrix.
  device_from_sensor = np.array([
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
  ])
  sensor_transform = np.column_stack([
    transform_asius_imu([1.0, 0.0, 0.0]),
    transform_asius_imu([0.0, 1.0, 0.0]),
    transform_asius_imu([0.0, 0.0, 1.0]),
  ])

  device_transform = device_from_sensor @ sensor_transform @ device_from_sensor.T
  expected = rot_from_euler([0.0, np.radians(-118.0), 0.0])

  np.testing.assert_allclose(device_transform, expected, atol=1e-12)
  np.testing.assert_allclose(sensor_transform.T @ sensor_transform, np.eye(3), atol=1e-12)
  np.testing.assert_allclose(np.linalg.det(sensor_transform), 1.0, atol=1e-12)


@pytest.mark.parametrize('device_vector', [
  [0.0, 0.0, -9.81],  # gravity with the road camera level
  [1.0, 0.0, 0.0],    # forward acceleration / roll rate
  [0.0, 1.0, 0.0],    # rightward acceleration / pitch rate
  [0.0, 0.0, 1.0],    # downward acceleration / yaw rate
])
def test_asius_imu_matches_physical_axes(device_vector):
  raw_x, raw_y, raw_z = CASE_FROM_IMU.T @ CASE_FROM_DEVICE @ device_vector
  actual = device_from_event(transform_asius_imu([raw_y, -raw_x, raw_z]))
  np.testing.assert_allclose(actual, device_vector, atol=1e-12)


@pytest.mark.parametrize('asius', [False, True])
@pytest.mark.parametrize('module,sensor_class,measurement,scale', [
  (lsm6ds3_accel, lsm6ds3_accel.LSM6DS3_Accel, 'acceleration', 9.81 * 2.0 / (1 << 15)),
  (lsm6ds3_gyro, lsm6ds3_gyro.LSM6DS3_Gyro, 'gyroUncalibrated', math.radians(8.75 / 1000.0)),
])
def test_sensor_event_axis_mapping(asius, module, sensor_class, measurement, scale):
  counts = np.array([1234, -4567, 8910])
  timestamp = 123456789
  with patch('openpilot.system.sensord.sensors.i2c_sensor.SMBus'), patch.object(module, 'ASIUS_HARDWARE', asius):
    sensor = sensor_class(1)
    sensor.source = log.SensorEventData.SensorSource.lsm6ds3trc
    with patch.object(sensor, 'read', side_effect=[b'\x03', struct.pack('<hhh', *counts)]):
      event = sensor.get_event(timestamp)

  actual = np.array(getattr(event, measurement).v)
  raw = counts * scale
  if asius:
    expected = CASE_FROM_DEVICE.T @ CASE_FROM_IMU @ raw
    np.testing.assert_allclose(device_from_event(actual), expected, rtol=1e-6, atol=1e-7)
  else:
    np.testing.assert_allclose(actual, [raw[1], -raw[0], raw[2]], rtol=1e-6, atol=1e-7)
  assert event.timestamp == timestamp
  assert event.source == log.SensorEventData.SensorSource.lsm6ds3trc

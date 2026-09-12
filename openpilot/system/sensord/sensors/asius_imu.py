import math

_SENSOR_PITCH = math.radians(90.0 + 28.0)
_SIN_PITCH = math.sin(_SENSOR_PITCH)
_COS_PITCH = math.cos(_SENSOR_PITCH)


def transform_asius_imu(v: list[float]) -> list[float]:
  """Align stock LSM6DS3 event coordinates with Asius v0's road camera.

  Panda v5's IMU is on B.Cu at 0 degrees. The case holds the road-camera
  optical axis 28 degrees from the Panda plane (62 from its normal).
  After the driver's raw [y, -x, z] mapping, rotate +118 degrees about sensor
  Y. locationd's [-z, -y, -x] conversion makes this a -118 degree device-frame
  pitch correction. Apply the same rotation to acceleration and angular rate.
  """
  x, y, z = v
  return [
    _COS_PITCH * x + _SIN_PITCH * z,
    y,
    -_SIN_PITCH * x + _COS_PITCH * z,
  ]

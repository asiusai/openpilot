"""Independent geometry, causal history and longitudinal metric checks."""
import copy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_trials import calibrated_image, vehicle_history
from route_longitudinal import TIMES, kinematics, path_shape, slowing


def history_checks():
  query = 10_000_000_000
  times = query + np.arange(-50, 3, dtype=np.int64)*50_000_000
  cp = dict(rotationalInertiaPerMass=2., wheelbase=2.7, centerToFront=1.2,
            tireStiffnessFrontPerMass=60., tireStiffnessRearPerMass=70., steerRatio=15.)
  records = {
    'carParams': (np.array([0]), [cp]),
    'carState': (times, [dict(vEgo=10., steeringAngleDeg=0.) for _ in times]),
    'vehicleParameters': (times, [dict(stiffnessFactor=1., steerRatio=15., angleOffsetDeg=0., roll=0.) for _ in times]),
  }
  row = dict(query_ns=query, ego_history_xyz=np.zeros((1, 20, 3)).tolist(),
             ego_history_rot=np.broadcast_to(np.eye(3), (1, 20, 3, 3)).tolist())
  xyz, rotations, _ = vehicle_history(row, records)
  np.testing.assert_allclose(xyz[0, :, 0], np.arange(-19, 1), atol=1e-5)
  np.testing.assert_array_equal(xyz[0, :, 1:], 0)
  np.testing.assert_array_equal(rotations, row['ego_history_rot'])
  future = copy.deepcopy(records)
  for kind in ('carState', 'vehicleParameters'):
    for stamp, value in zip(*future[kind], strict=True):
      if stamp > query:
        value.update({key: 100. for key in value})
  changed = vehicle_history(row, future)
  np.testing.assert_array_equal(changed[0], xyz)
  np.testing.assert_array_equal(changed[1], rotations)
  turning = copy.deepcopy(records)
  for value in turning['carState'][1]:
    value['steeringAngleDeg'] = 10.
  xyz, rotations, audit = vehicle_history(row, turning)
  assert audit['left_yaw_rate_rad_s'] > 0 and xyz[0, 0, 1] > 0
  assert Rotation.from_matrix(rotations[0, 0]).as_euler('xyz')[2] < 0
  np.testing.assert_array_equal(xyz[0, -1], 0)
  np.testing.assert_allclose(rotations[0, -1], np.eye(3), atol=1e-7)


def projection_checks():
  # A pinhole camera can be checked independently with an ordinary homography.
  width, height, focal = 768, 432, 650.
  cx, cy = (width-1)/2, (height-1)/2
  source = {'cameras': {'6': dict(prepared_size=[width, height], content_box=[0, 0, width, height],
                                 focal_pixels=[focal, focal], principal_point=[cx, cy], projection='pinhole')}}
  yy, xx = np.mgrid[:height, :width]
  image = np.stack([xx/4, yy/2, xx/8+yy/4]).astype(np.float32)
  fraction = 2*focal*math.tan(math.radians(15))/width
  intrinsic = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.]])
  target = np.array([[focal/fraction, 0, cx], [0, focal/fraction, cy], [0, 0, 1.]])
  for angle in (0., .04):
    # Device yaw becomes negative camera-image yaw around the vertical image axis.
    r = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                  [-math.sin(angle), 0, math.cos(angle)]])
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ (intrinsic @ r @ np.linalg.inv(target)).T
    sx, sy = pixels[..., 0]/pixels[..., 2], pixels[..., 1]/pixels[..., 2]
    expected = np.stack([sx/4, sy/2, sx/8+sy/4]).astype(np.uint8)
    actual = calibrated_image(image, 6, source, {}, {'rpyCalib': [0, 0, angle], 'wideFromDeviceEuler': [0, 0, 0]})
    assert np.abs(actual.astype(int)-expected).max() <= 1


def longitudinal_checks():
  straight = lambda speed: np.column_stack([TIMES*speed, np.zeros((64, 2))])
  np.testing.assert_allclose(kinematics(straight(10)), 10, atol=1e-12)
  assert slowing(kinematics(straight(10)))['onset_s'] is None
  slowing_path = np.column_stack([12*TIMES-.5*TIMES**2, np.zeros((64, 2))])
  result = slowing(kinematics(slowing_path))
  assert result['onset_s'] < .5
  np.testing.assert_allclose(result['minimum_smoothed_acceleration_m_s2'], -1, atol=1e-10)
  shape = path_shape(straight(15), straight(10))
  assert shape['mean_error_m'] < 1e-10 and shape['mean_absolute_lateral_error_m'] < 1e-10
  assert shape['shared_distance_m'] == 64
  assert path_shape(straight(0), straight(10))['mean_error_m'] is None


if __name__ == '__main__':
  history_checks()
  projection_checks()
  longitudinal_checks()
  print('PASS: causal learned history, steering sign, calibration homography, speed, slowing and equal-distance path metrics')

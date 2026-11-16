"""Check trajectory axes, camera height, calibration and ego-motion compensation."""
import numpy as np
from scipy.spatial.transform import Rotation
from route_video import Geometry, LoggedModels, AXES

g = object.__new__(Geometry)
g.ns = np.array([1_000_000_000, 1_100_000_000])
g.p = np.array([[0., 0., 0.], [1., 0., 0.]])
g.r = np.stack([np.eye(3)]*2)
g.v = np.array([[10., 0., 0.]]*2)
g.intrinsics = np.array([[400., 0., 640.], [0., 400., 360.], [0., 0., 1.]])
g.cal_ns = np.array([900_000_000])
g.calibrations = [{'ns': 900_000_000, 'rpyCalib': [0., 0., 0.], 'wideFromDeviceEuler': [0., 0., 0.], 'height': [1.2]}]
row = {'query_ns': 1_000_000_000, 'calibration_rpy': [0., 0., 0.]}
path = np.tile([20., 2., 0.], (64, 1))
pixels, valid = g.project(path, row, row['query_ns'])
np.testing.assert_allclose(pixels, np.tile([600., 384.], (64, 1)))
assert valid.all()  # Forward-left-up maps left of the principal point.
pixels, valid = g.project(path, row, 1_100_000_000)
np.testing.assert_allclose(pixels, np.tile([640.-800./19, 360.+480./19], (64, 1)))
assert valid.all()  # Spatial points still ahead remain visible as predictions age.
pixels, _ = g.project(path, row, 1_125_000_000)
np.testing.assert_allclose(pixels, np.tile([640.-800./18.75, 360.+480./18.75], (64, 1)))

row['calibration_rpy'] = [.01, -.06, -.08]
g.calibrations[0].update(rpyCalib=row['calibration_rpy'], wideFromDeviceEuler=[-.01, .02, .03])
pixels, _ = g.project(path, row, row['query_ns'])
device_from_calib = Rotation.from_euler('xyz', row['calibration_rpy']).as_matrix()
wide_from_device = Rotation.from_euler('xyz', g.calibrations[0]['wideFromDeviceEuler']).as_matrix()
view_from_device = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
ground_path = path @ AXES + [0.,0.,1.2]
homogeneous = (g.intrinsics @ view_from_device @ wide_from_device @ device_from_calib @ ground_path.T).T
np.testing.assert_allclose(pixels, homogeneous[:,:2]/homogeneous[:,2:], atol=1e-10)
_, valid = g.project(-path, row, row['query_ns'])
assert not valid.any()
crossing, valid = g.project(np.array([[0., 0., 0.], [20., 0., 0.]]), row, row['query_ns'])
assert len(crossing) == 3 and valid.tolist() == [False, True, True]
print('Route projection axes, calibration, ground height and pose compensation passed')

# Logged openpilot frames advance independently inside one Alpamayo second.
models = object.__new__(LoggedModels)
models.ns = np.arange(1_000_000_000, 2_000_000_000, 50_000_000, dtype=np.int64)
models.rpy = np.zeros((20, 3))
models.xyz = np.zeros((20, 64, 3))
models.xyz[:, :, 0] = np.arange(20)[:, None]
models.selected, models.missing = set(), 0
for i, ns in enumerate(models.ns):
  selected, origin = models.at(int(ns)+20_000_000)
  assert origin['query_ns'] == int(ns) and np.all(selected[:, 0] == i)
assert len(models.selected) == 20
assert models.at(999_999_999) is None and models.at(2_100_000_001) is None

# Recorded future motion is evaluated at the video time, not the 1 Hz tick.
g.ns = np.arange(1_000_000_000, 9_000_000_000, 100_000_000, dtype=np.int64)
g.p = np.column_stack(((g.ns-g.ns[0])/1e9*10, np.zeros((len(g.ns), 2))))
g.r = np.tile(np.eye(3), (len(g.ns), 1, 1))
g.v = np.tile([10., 0., 0.], (len(g.ns), 1))
g.calibrations[0].update(rpyCalib=[0., 0., 0.], wideFromDeviceEuler=[0., 0., 0.])
actual, origin = g.actual(1_025_000_000)
np.testing.assert_allclose(actual[:, 0], np.arange(1, 65), atol=1e-10)
np.testing.assert_allclose(actual[:, 1:], 0, atol=1e-10)
assert origin['query_ns'] == 1_025_000_000
print('Independent 20 Hz openpilot updates, stale-frame rejection and gray motion path passed')

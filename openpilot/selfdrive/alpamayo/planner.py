"""Select a time-aligned GPU trajectory at the regular driving-model cadence."""
import numpy as np

PERIOD = 500_000_000
T = np.arange(65) * .1
CHECK_TIMES = np.arange(1, 6)*.5
ACTION_TIMES = np.arange(22)*.1


def packet(result):
  """Only completed, in-slot predictions can claim a slot. A rejected slot invalidates the previous plan."""
  if result['startup'] or result['period_ms'] != 500 or not result['slot_deadline_met']:
    return None
  xyz = np.asarray(result['prediction']['xyz'])[0]
  rot = np.asarray(result['prediction']['rot'])[0]
  if result['trajectory_axes'] != 'forward/left/up' or not np.allclose(result['trajectory_times_s'], T[1:]):
    raise ValueError('Unsupported trajectory convention')
  return dict(queryMonoTime=result['input']['query_ns'], readyMonoTime=result['output_ready_ns'], slotDeadline=result['slot_deadline_ns'],
              x=xyz[:, 0].tolist(), y=xyz[:, 1].tolist(), yaw=np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).tolist())


class PlanSelector:
  def __init__(self):
    self.query = self.deadline = self.ready = self.streak = 0
    self.curve = None

  def reject(self):
    self.curve, self.streak = None, 0

  def receive(self, plan, now):
    q, r, d = int(plan.queryMonoTime), int(plan.readyMonoTime), int(plan.slotDeadline)
    if not (self.query < q <= r <= now < d+PERIOD and d-PERIOD <= r <= d and r-q <= PERIOD+250_000_000 and d > self.deadline):
      self.reject()
      return
    if any(len(col) != 64 for col in (plan.x, plan.y, plan.yaw)):
      self.reject()
      return
    x, y, yaw = np.array([[0., *col] for col in (plan.x, plan.y, plan.yaw)])
    if not np.isfinite([x, y, yaw]).all():
      self.reject()
      return
    yaw = np.unwrap(yaw)
    speed = np.hypot(np.diff(x), np.diff(y)) / .1
    speed = np.interp(T, T[1:]-.05, speed)
    self.curve = x, y, yaw, speed, np.gradient(speed, .1), np.gradient(yaw, .1)
    self.streak = self.streak+1 if d-self.deadline == PERIOD else 1
    self.query, self.ready, self.deadline = q, r, d

  def action(self, now, v_ego, lat_delay, long_delay, local_path, local_action):
    from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan, should_stop
    if self.curve is None or self.streak < 2 or not self.ready <= now < min(self.deadline+PERIOD, self.query+1_000_000_000):
      return None
    if (not np.isfinite([v_ego, local_action.desiredCurvature, local_action.desiredAcceleration]).all() or
        not (1. <= v_ego and 0 < lat_delay <= 2. and 0 < long_delay <= 2.) or local_action.shouldStop):
      return None
    age = (now-self.query)/1e9
    x, y, yaw, speed, accel, yaw_rate = self.curve
    # Compare in the current vehicle frame. A turn starts at the current heading, not the exposure heading.
    theta = np.interp(age, T, yaw)
    future = CHECK_TIMES
    dx = np.interp(age+future, T, x)-np.interp(age, T, x)
    dy = np.interp(age+future, T, y)-np.interp(age, T, y)
    gx, gy = np.cos(theta)*dx+np.sin(theta)*dy, np.sin(theta)*dx-np.cos(theta)*dy
    ts, sx, sy = (np.asarray(list(col)) for col in local_path)
    if (len(ts) < 2 or len(sx) != len(ts) or len(sy) != len(ts) or not np.isfinite([sx, sy, ts]).all()
        or np.any(np.diff(ts) <= 0) or ts[0] > future[0] or ts[-1] < future[-1] or
        np.max(abs(gy-np.interp(future, ts, sy))) > 1.25 or np.max(abs(gx-np.interp(future, ts, sx))) > 3.):
      return None
    at = ACTION_TIMES
    def sample(v):
      return np.interp(age+at, T, v)
    v = sample(speed)
    curvature = get_curvature_from_plan(theta-sample(yaw), -sample(yaw_rate), at, v_ego, lat_delay)
    a = get_accel_from_plan(v, sample(accel), at, long_delay)
    if (abs(v[0]-v_ego) > 2. or abs(curvature-local_action.desiredCurvature)*v_ego**2 > 1. or
        abs(a-local_action.desiredAcceleration) > 1. or (local_action.desiredAcceleration < -.5 and a > local_action.desiredAcceleration)):
      return None
    return dict(desiredCurvature=float(curvature), desiredAcceleration=float(a), shouldStop=should_stop(v_ego, a))

  def apply(self, model, now, speed, lat_delay, long_delay):
    """Called on a fresh modelV2 builder; preserve the local action when unavailable."""
    candidate = self.action(now, speed, lat_delay, long_delay, (model.position.t, model.position.x, model.position.y), model.action)
    if candidate is not None:
      model.action = candidate
      model.alpamayoQueryMonoTime = self.query

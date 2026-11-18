"""Select a time-aligned GPU trajectory at the regular driving-model cadence."""
import numpy as np

PERIOD = 500_000_000
HANDOFF_GRACE = 50_000_000  # One frame of the local 20 Hz driving model.
MAX_DELIVERY_AGE = PERIOD+250_000_000
T = np.arange(65) * .1
ACTION_TIMES = np.arange(22)*.1


def deliverable(query, ready, deadline, now):
  return (0 < query <= deadline-PERIOD <= ready <= now < deadline+HANDOFF_GRACE
          and now-query <= MAX_DELIVERY_AGE)


def packet(result):
  """Keep the strict benchmark deadline separate from eligibility for live handoff."""
  q,r,d = result['input']['query_ns'], result['output_ready_ns'], result['slot_deadline_ns']
  if result['startup'] or result['period_ms'] != 500 or not deliverable(q,r,d,r):
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
    self.last_on_time = False

  def reject(self):
    self.curve, self.streak = None, 0
    self.last_on_time = False

  def receive(self, plan, now, published=None):
    q, r, d = int(plan.queryMonoTime), int(plan.readyMonoTime), int(plan.slotDeadline)
    if q <= self.query and d <= self.deadline:
      return  # An older or repeated packet cannot replace a newer result.
    published = r if published is None else published
    if not (q > self.query and d > self.deadline and r <= published <= now and deliverable(q,r,d,published)
            and now-published <= HANDOFF_GRACE and now-q <= MAX_DELIVERY_AGE):
      self.reject()
      return
    on_time = published < d
    consecutive = d-self.deadline == PERIOD
    if not on_time and not (consecutive and self.streak >= 2 and self.last_on_time):
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
    self.curve = yaw, speed, np.gradient(speed, .1), np.gradient(yaw, .1)
    self.streak = self.streak+1 if consecutive else 1
    self.last_on_time = on_time
    self.query, self.ready, self.deadline = q, r, d

  def candidate_action(self, now, v_ego, lat_delay, long_delay):
    """Time-align the GPU plan without consulting any other model's path or action."""
    from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
    from openpilot.selfdrive.controls.lib.drive_helpers import MAX_CURVATURE, MAX_LATERAL_ACCEL_NO_ROLL, get_accel_from_plan, get_curvature_from_plan, should_stop
    if self.curve is None or self.streak < 2 or not self.ready <= now < min(self.deadline+PERIOD, self.query+1_000_000_000):
      return None
    if not np.isfinite(v_ego) or not (1. <= v_ego and 0 < lat_delay <= 2. and 0 < long_delay <= 2.):
      return None
    age = (now-self.query)/1e9
    yaw, speed, accel, yaw_rate = self.curve
    def sample(v):
      return np.interp(age+ACTION_TIMES, T, v)
    v = sample(speed)
    if abs(v[0]-v_ego) > 2.:
      return None
    theta = np.interp(age, T, yaw)
    curvature = get_curvature_from_plan(theta-sample(yaw), -sample(yaw_rate), ACTION_TIMES, v_ego, lat_delay)
    a = get_accel_from_plan(v, sample(accel), ACTION_TIMES, long_delay)
    if (not np.isfinite([curvature,a]).all() or not ACCEL_MIN <= a <= ACCEL_MAX or
        abs(curvature) > MAX_CURVATURE or abs(curvature)*v_ego**2 > MAX_LATERAL_ACCEL_NO_ROLL):
      return None
    return dict(desiredCurvature=float(curvature), desiredAcceleration=float(a), shouldStop=should_stop(v_ego, a))

  def apply(self, model, now, speed, lat_delay, long_delay):
    """Called on a fresh modelV2 builder; preserve the local action when unavailable."""
    candidate = self.candidate_action(now, speed, lat_delay, long_delay)
    if candidate is not None:
      model.action = candidate
      model.alpamayoQueryMonoTime = self.query

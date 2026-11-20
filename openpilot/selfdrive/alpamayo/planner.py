"""Select a time-aligned Alpamayo trajectory; the driving action comes only from Alpamayo plans.

A plan is usable when it was ready within MAX_LATENCY of its camera query and while that query is younger than
MAX_PLAN_AGE, whatever the scheduling mode. Bad or late packets are ignored; the current plan simply ages out.
"""
import numpy as np

PERIOD = 500_000_000
FREE = 0  # Free-running: each prediction starts when the previous one finishes.
PERIODS = (FREE, 250_000_000, 500_000_000)
FREE_WINDOW = 400_000_000  # Free-running supervisor: slot bookkeeping deadline after selection.
MAX_LATENCY = 500_000_000  # Camera query to plan ready.
MAX_PLAN_AGE = 1_500_000_000  # A plan drives while its camera query is at most this old; its horizon is 6.4 s.
# In the car a plan is ready 350-550 ms after its query, and one superseded 250 ms slot leaves a 500 ms query gap.
CONSECUTIVE_GAP = 1_000_000_000  # Consecutive plans have queries at most this far apart (two superseded slots).
HANDOFF_GRACE = 100_000_000  # modeld reads plans once per 20 Hz camera frame, after its own model run; allow two frames.
T = np.arange(65) * .1
ACTION_TIMES = np.arange(22)*.1


def should_stop_now(v_ego, accel):
  from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
  return should_stop(v_ego, accel)


def configured_period():
  """Slot period from ALPAMAYO_CONFIG for the supervisor's scheduling."""
  import json, os
  from pathlib import Path
  path = os.environ.get('ALPAMAYO_CONFIG')
  period = json.loads(Path(path).read_text()).get('period_ms', 500)*1_000_000 if path and os.path.isfile(path) else PERIOD
  if period not in PERIODS:
    raise ValueError('Alpamayo planner requires a free-running, 250 or 500 ms period')
  return period


def deliverable(query, ready, now):
  return 0 < query < ready <= now and ready-query <= MAX_LATENCY and now-query < MAX_PLAN_AGE


def packet(result):
  """Plan message fields, or None for startup results and plans too late to drive."""
  q,r,d = result['input']['query_ns'], result['output_ready_ns'], result['slot_deadline_ns']
  if result['startup'] or not deliverable(q,r,r):
    return None
  xyz = np.asarray(result['prediction']['xyz'])[0]
  rot = np.asarray(result['prediction']['rot'])[0]
  if result['trajectory_axes'] != 'forward/left/up' or not np.allclose(result['trajectory_times_s'], T[1:]):
    raise ValueError('Unsupported trajectory convention')
  return dict(queryMonoTime=result['input']['query_ns'], readyMonoTime=result['output_ready_ns'], slotDeadline=d,
              x=xyz[:, 0].tolist(), y=xyz[:, 1].tolist(), yaw=np.arctan2(rot[:, 1, 0], rot[:, 0, 0]).tolist())


class PlanSelector:
  def __init__(self):
    self.query = self.ready = self.streak = 0
    self.curve = None
    self.held = None

  def receive(self, plan, now, published=None):
    q, r = int(plan.queryMonoTime), int(plan.readyMonoTime)
    published = r if published is None else published
    if q <= self.query or not (r <= published <= now and deliverable(q,r,published) and now-published <= HANDOFF_GRACE):
      return  # An older, repeated, late or stale packet never replaces the current plan.
    if any(len(col) != 64 for col in (plan.x, plan.y, plan.yaw)):
      return
    x, y, yaw = np.array([[0., *col] for col in (plan.x, plan.y, plan.yaw)])
    if not np.isfinite([x, y, yaw]).all():
      return
    yaw = np.unwrap(yaw)
    speed = np.hypot(np.diff(x), np.diff(y)) / .1
    speed = np.interp(T, T[1:]-.05, speed)
    self.curve = yaw, speed, np.gradient(speed, .1), np.gradient(yaw, .1)
    self.streak = self.streak+1 if 0 < q-self.query <= CONSECUTIVE_GAP else 1
    self.query, self.ready = q, r

  def candidate_action(self, now, v_ego, lat_delay, long_delay):
    """Time-align the GPU plan without consulting any other model's path or action."""
    from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
    from openpilot.selfdrive.controls.lib.drive_helpers import MAX_CURVATURE, MAX_LATERAL_ACCEL_NO_ROLL, get_accel_from_plan, get_curvature_from_plan, should_stop
    if self.curve is None or self.streak < 2 or not self.ready <= now < self.query+MAX_PLAN_AGE:
      return None
    # Low speeds are valid: the curvature conversion clamps speed, and stops and starts need the plan's acceleration.
    if not np.isfinite(v_ego) or not (0. <= v_ego and 0 < lat_delay <= 2. and 0 < long_delay <= 2.):
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
    """Called on a fresh modelV2 builder. The action always comes from Alpamayo, never from the local model.

    Without a current plan, the last Alpamayo curvature is held with non-positive acceleration and
    alpamayoQueryMonoTime stays zero, so selfdrived blocks engagement or soft-disables.
    """
    candidate = self.candidate_action(now, speed, lat_delay, long_delay)
    if candidate is not None:
      model.action = candidate
      model.alpamayoQueryMonoTime = self.query
      self.held = candidate
    else:
      held = self.held or dict(desiredCurvature=0., desiredAcceleration=0., shouldStop=False)
      accel = min(held['desiredAcceleration'], 0.)
      model.action = dict(desiredCurvature=held['desiredCurvature'], desiredAcceleration=accel, shouldStop=should_stop_now(speed, accel))

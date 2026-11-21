"""Select a time-aligned Alpamayo trajectory; the driving action comes only from Alpamayo plans.

A plan is usable when it was ready within MAX_LATENCY of its camera query and while that query is younger than
MAX_PLAN_AGE, whatever the scheduling mode. Bad or late packets are ignored; the current plan simply ages out.
"""
import math
from collections import deque
import numpy as np

PERIOD = 500_000_000
FREE = 0  # Free-running: each prediction starts when the previous one finishes.
PERIODS = (FREE, 100_000_000, 250_000_000, 500_000_000)
FREE_WINDOW = 400_000_000  # Free-running supervisor: slot bookkeeping deadline after selection.
MAX_LATENCY = 750_000_000  # Camera query to plan ready; complex scenes reason for up to ~400 ms in the car.
MAX_PLAN_AGE = 1_500_000_000  # A plan drives while its camera query is at most this old; its horizon is 6.4 s.
# In the car a plan is ready 350-550 ms after its query, and one superseded 250 ms slot leaves a 500 ms query gap.
CONSECUTIVE_GAP = 1_000_000_000  # Consecutive plans have queries at most this far apart (two superseded slots).
HANDOFF_GRACE = 100_000_000  # modeld reads plans once per 20 Hz camera frame, after its own model run; allow two frames.
T = np.arange(65) * .1
ACTION_TIMES = np.arange(22)*.1
# Path tracking: Alpamayo corrects only about half of a heading or lateral error in its own plan, and plans drive
# 0.2-0.5 s after their camera frame. The planner therefore keeps each plan's path fixed to the road (odometry since
# its camera query) and steers to a point LOOKAHEAD_S ahead on it from the pose expected after the steering delay.
LOOKAHEAD_S = 1.0
MIN_LOOKAHEAD = 5.  # m
ODOMETRY_S = 3.  # odometry history kept, longer than the latest plan's age
# Lane assist (config `lane_assist`, off by default): while both lane lines of the local model are confident and the
# Alpamayo target stays within LANE_KEEP of the lane centre, the tracker steers to the lane centre instead. Alpamayo
# corrects only about half of its own drift; larger departures from the lane centre are Alpamayo maneuvers.
LANE_KEEP = 1.0  # m, at the lookahead distance
LANE_PROB = 0.5
LANE_WIDTH = (2.5, 4.5)  # m


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
    raise ValueError('Alpamayo planner requires a free-running, 100, 250 or 500 ms period')
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


def configured_calibration_path():
  """Learned lateral calibration file from ALPAMAYO_CONFIG `lateral_calibration` (default next to the config)."""
  import json, os
  from pathlib import Path
  path = os.environ.get('ALPAMAYO_CONFIG')
  config = json.loads(Path(path).read_text()) if path and os.path.isfile(path) else {}
  return config.get('lateral_calibration', '/data/alpamayo-calibration/lateral.json')


def configured_lane_assist():
  """Lane assist from ALPAMAYO_CONFIG `lane_assist` (default off)."""
  import json, os
  from pathlib import Path
  path = os.environ.get('ALPAMAYO_CONFIG')
  config = json.loads(Path(path).read_text()) if path and os.path.isfile(path) else {}
  value = config.get('lane_assist', False)
  if type(value) is not bool:
    raise ValueError('lane_assist must be true or false')
  return value


class PlanSelector:
  def __init__(self, calibration=None, lane_assist=False):
    self.calibration = calibration  # selfdrive.alpamayo.calibration.LateralCalibration; gain 1 without it
    self.lane_assist = lane_assist
    self.lane = None  # lane centre (x forward, y left) in the current vehicle frame, or None
    self.lane_used = False
    self.query = self.ready = self.streak = 0
    self.curve = self.path = None
    self.held = None
    self.odometry = deque()  # (ns, x, y, heading) of the vehicle; heading is the device yaw, left positive
    self.last_curvature = 0.

  def motion(self, ns, v_ego, heading):
    """Integrate vehicle odometry from speed and yaw (left positive). A constant yaw mounting offset rotates the whole
    odometry frame and cancels in the relative poses the tracker uses."""
    if not (np.isfinite(v_ego) and np.isfinite(heading)):
      return
    if self.odometry:
      t, x, y, h = self.odometry[-1]
      dt = (ns-t)/1e9
      if not 0 < dt <= .5:
        self.odometry.clear()
      else:
        heading = h+(heading-h+math.pi) % (2*math.pi)-math.pi  # unwrap
        mid = (h+heading)/2
        self.odometry.append((ns, x+v_ego*math.cos(mid)*dt, y+v_ego*math.sin(mid)*dt, heading))
        while self.odometry and self.odometry[0][0] < ns-int(ODOMETRY_S*1e9):
          self.odometry.popleft()
        return
    self.odometry.append((ns, 0., 0., heading))

  def pose_at(self, ns):
    if len(self.odometry) < 2 or not self.odometry[0][0] <= ns <= self.odometry[-1][0]+100_000_000:
      return None
    ts = [p[0] for p in self.odometry]
    i = min(max(int(np.searchsorted(ts, ns)), 1), len(ts)-1)
    a, b = self.odometry[i-1], self.odometry[i]
    f = min((ns-a[0])/max(b[0]-a[0], 1), 1.)
    return tuple(a[k]+f*(b[k]-a[k]) for k in (1, 2, 3))

  def tracked_curvature(self, now, v_ego, lat_delay, gain):
    """Pure pursuit on the newest plan's path (heading profile scaled by `gain`), or None without odometry for it."""
    query, current = self.pose_at(self.query), self.pose_at(now)
    if query is None or current is None or self.path is None:
      return None
    x, y, yaw = self.path
    if gain != 1.:
      step = np.hypot(np.diff(x), np.diff(y))
      heading = gain*(yaw[:-1]+yaw[1:])/2
      x = np.concatenate(([0.], np.cumsum(step*np.cos(heading)))); y = np.concatenate(([0.], np.cumsum(step*np.sin(heading))))
    qx, qy, qh = query
    wx, wy = qx+math.cos(qh)*x-math.sin(qh)*y, qy+math.sin(qh)*x+math.cos(qh)*y
    # Pose after the steering delay, with the last commanded curvature.
    px, py, ph = current
    d, k = v_ego*lat_delay, self.last_curvature
    px, py, ph = px+d*math.cos(ph+k*d/2), py+d*math.sin(ph+k*d/2), ph+k*d
    fx = math.cos(ph)*(wx-px)+math.sin(ph)*(wy-py)
    fy = -math.sin(ph)*(wx-px)+math.cos(ph)*(wy-py)
    arc = np.concatenate(([0.], np.cumsum(np.hypot(np.diff(fx), np.diff(fy)))))
    target = arc[int(np.argmin(np.hypot(fx, fy)))]+max(MIN_LOOKAHEAD, v_ego*LOOKAHEAD_S)
    if target > arc[-1]:
      return None
    tx, ty = np.interp(target, arc, fx), np.interp(target, arc, fy)
    self.lane_used = False
    if self.lane_assist and self.lane is not None:
      # Lane centre in the delayed-pose frame, at the same distance as the Alpamayo target.
      lx, ly = self.lane
      cx, cy, ch = current
      wx, wy = cx+math.cos(ch)*lx-math.sin(ch)*ly, cy+math.sin(ch)*lx+math.cos(ch)*ly
      ax = math.cos(ph)*(wx-px)+math.sin(ph)*(wy-py)
      ay = -math.sin(ph)*(wx-px)+math.cos(ph)*(wy-py)
      dist = math.hypot(tx, ty)
      ahead = np.hypot(ax, ay) >= dist
      if ahead.any():
        i = int(np.argmax(ahead))
        if abs(ay[i]-ty) <= LANE_KEEP:
          tx, ty, self.lane_used = ax[i], ay[i], True
    return 2*ty/max(tx*tx+ty*ty, 1.)

  def update_lane(self, model):
    """Lane centre from the local model's two ego lane lines (modelV2: y right), when both are confident."""
    self.lane = None
    try:
      probs = list(model.laneLineProbs)
      left, right = model.laneLines[1], model.laneLines[2]
      x, yl, yr = np.array(left.x), np.array(left.y), np.array(right.y)
    except (AttributeError, IndexError, ValueError):
      return
    if len(probs) < 3 or min(probs[1], probs[2]) < LANE_PROB or len(x) < 2 or not len(x) == len(yl) == len(yr):
      return
    width = np.interp(15., x, yr-yl)
    if not LANE_WIDTH[0] <= width <= LANE_WIDTH[1] or not np.isfinite([x, yl, yr]).all():
      return
    self.lane = x, -(yl+yr)/2

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
    self.path = x, y, yaw
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
    # The calibrated command stays inside the same curvature and lateral-acceleration limits as the plan itself.
    limit = min(MAX_CURVATURE, MAX_LATERAL_ACCEL_NO_ROLL/max(v_ego, 1e-3)**2)
    gain = self.calibration.gain(v_ego) if self.calibration is not None else 1.
    tracked = self.tracked_curvature(now, v_ego, lat_delay, gain)
    curvature = float(np.clip(tracked if tracked is not None and np.isfinite(tracked) else curvature*gain, -limit, limit))
    self.last_curvature = curvature
    return dict(desiredCurvature=float(curvature), desiredAcceleration=float(a), shouldStop=should_stop(v_ego, a))

  def apply(self, model, now, speed, lat_delay, long_delay):
    """Called on a fresh modelV2 builder. The action always comes from Alpamayo, never from the local model.

    Without a current plan, the last Alpamayo curvature is held with non-positive acceleration and
    alpamayoQueryMonoTime stays zero, so selfdrived blocks engagement or soft-disables.
    """
    if self.lane_assist:
      self.update_lane(model)
    candidate = self.candidate_action(now, speed, lat_delay, long_delay)
    if candidate is not None:
      model.action = candidate
      model.alpamayoQueryMonoTime = self.query
      self.held = candidate
    else:
      held = self.held or dict(desiredCurvature=0., desiredAcceleration=0., shouldStop=False)
      accel = min(held['desiredAcceleration'], 0.)
      model.action = dict(desiredCurvature=held['desiredCurvature'], desiredAcceleration=accel, shouldStop=should_stop_now(speed, accel))

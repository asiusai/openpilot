"""Select a time-aligned Alpamayo trajectory; the driving action comes from Alpamayo plans (optionally lane-centred).

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
LOOKAHEAD_S = 1.5
T_PATH = np.arange(65)*.1  # plan path times including the query pose
MIN_LOOKAHEAD = 8.  # m; a longer lookahead halves the response to plan-to-plan jitter (curvature = 2 y / L^2)
# The pose after the steering delay is predicted with the measured curvature (odometry heading change over
# CURVATURE_WINDOW). Predicting it with the tracker's own last command made consecutive commands alternate.
CURVATURE_WINDOW = .3  # s
COMMAND_TAU = .25  # s, first-order smoothing of the tracked curvature (modelV2 runs at 20 Hz)
POSE_EXTRAPOLATION_NS = 300_000_000  # odometry extrapolation after its newest sample
ODOMETRY_S = 3.  # odometry history kept, longer than the latest plan's age


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


def _config():
  import json, os
  from pathlib import Path
  path = os.environ.get('ALPAMAYO_CONFIG')
  return json.loads(Path(path).read_text()) if path and os.path.isfile(path) else {}


def configured_tuning():
  """Optional ALPAMAYO_CONFIG settings:
  `straight_lookahead_s`: path tracker lookahead on straight plans (default LOOKAHEAD_S).
  `lane_centering`: 0..1 weight that pulls the tracked plan path toward the local model's lane centre (default 0).
  `lane_center_offset_m`: lateral target inside the lane, right positive, for a device mounted off the car centre.
  `plan_curvature_bias`: constant left-positive curvature (1/m) removed from every plan path (default 0).
  The plan's steering and acceleration are used without gains."""
  config = _config()
  straight = float(config.get('straight_lookahead_s', LOOKAHEAD_S))
  if not LOOKAHEAD_S <= straight <= 4.:
    raise ValueError('straight_lookahead_s must be 1.5..4 s')
  centering = float(config.get('lane_centering', 0.))
  offset = float(config.get('lane_center_offset_m', 0.))
  bias = float(config.get('plan_curvature_bias', 0.))
  if not 0. <= centering <= 1. or not -.5 <= offset <= .5 or not abs(bias) <= 1e-3:
    raise ValueError('lane_centering must be 0..1, lane_center_offset_m -0.5..0.5 and |plan_curvature_bias| <= 0.001')
  return {'straight_lookahead_s': straight, 'lane_centering': centering, 'lane_center_offset_m': offset,
          'plan_curvature_bias': bias}


# Lane centring. On Karel's straight roads (2026-11-22, stock, zero noise, 10 steps) Alpamayo's plan kept 90-100% of
# the car's lateral offset 10-30 m ahead, followed only ~40% of the lane heading relative to the car, and bent
# ~0.2-0.3 m left by 40-60 m. The car therefore drifted, mostly left, until the driver corrected it. When the local
# model sees the ego lane, the tracked path is pulled toward its centre. A plan that leaves the lane by more than
# lane_max_dev (a turn, a fork, an obstacle) ends the blend; blinkers and unmarked roads keep the plan unchanged.
LANE_PROB = (.4, .7)  # lane line probability: no blend .. full blend
LANE_WIDTH = (2.4, 4.6)  # m, plausible ego lane widths at the car
LANE_DEFAULT_WIDTH = 3.3  # m, lane width for a single visible line until both lines have been seen
LANE_WIDTH_TAU = 5.  # s, smoothing of the measured lane width
LANE_RAMP_X = (3., 15.)  # m ahead: blend weight ramps in, so the near path stays continuous with the car
LANE_STALE_NS = 150_000_000  # lane lines older than this are ignored


def lane_max_dev(x):
  """Plan-to-centre distance (m) at x m ahead: full blend below the first value, none above the second."""
  return 1.2+.02*x, 2.0+.03*x


class PlanSelector:
  def __init__(self, straight_lookahead_s=LOOKAHEAD_S, lane_centering=0., lane_center_offset_m=0., plan_curvature_bias=0.):
    # On a straight plan the near path mostly continues the current heading; the plan's correction toward the lane shows
    # farther ahead, so the lookahead grows to straight_lookahead_s as the plan's 3 s heading change falls below 0.05 rad.
    self.straight_lookahead_s = straight_lookahead_s
    self.lane_centering = lane_centering
    self.lane_center_offset_m = lane_center_offset_m
    self.plan_curvature_bias = plan_curvature_bias
    self.lane = None  # (ns, pose at ns, forward x, left y of the lane centre, weight) from the local model
    self.lane_width = LANE_DEFAULT_WIDTH
    self.lane_width_ns = 0
    self.query = self.ready = self.streak = 0
    self.curve = self.path = None
    self.held = None
    self.odometry = deque()  # (ns, x, y, heading) of the vehicle; heading is the device yaw, left positive
    self.last_curvature = 0.
    self.tracker_status = {}  # outcome -> count, for the periodic modeld log

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
    """Odometry pose at `ns`. After the newest sample, which modeld often reads 60-110 ms late, the last segment's
    motion is extrapolated for up to POSE_EXTRAPOLATION_NS."""
    if len(self.odometry) < 2 or not self.odometry[0][0] <= ns <= self.odometry[-1][0]+POSE_EXTRAPOLATION_NS:
      return None
    ts = [p[0] for p in self.odometry]
    i = min(max(int(np.searchsorted(ts, ns)), 1), len(ts)-1)
    a, b = self.odometry[i-1], self.odometry[i]
    f = (ns-a[0])/max(b[0]-a[0], 1)
    return tuple(a[k]+f*(b[k]-a[k]) for k in (1, 2, 3))

  def measured_curvature(self, now, v_ego):
    """Vehicle curvature from the odometry heading change over the last CURVATURE_WINDOW."""
    a, b = self.pose_at(now-int(CURVATURE_WINDOW*1e9)), self.pose_at(now)
    if a is None or b is None or v_ego < 1.:
      return 0.
    return (b[2]-a[2])/(v_ego*CURVATURE_WINDOW)

  def tracked_curvature(self, now, v_ego, lat_delay):
    """Pure pursuit on the newest plan's path, or None without odometry for it.
    The result is left positive, like the plan and the odometry."""
    query, current = self.pose_at(self.query), self.pose_at(now)
    if query is None or current is None or self.path is None:
      self.count('no query pose' if query is None else 'no current pose' if current is None else 'no path')
      return None
    x, y, _ = self.path
    qx, qy, qh = query
    wx, wy = qx+math.cos(qh)*x-math.sin(qh)*y, qy+math.sin(qh)*x+math.cos(qh)*y
    # Pose after the steering delay, with the measured curvature.
    px, py, ph = current
    d, k = v_ego*lat_delay, self.measured_curvature(now, v_ego)
    px, py, ph = px+d*math.cos(ph+k*d/2), py+d*math.sin(ph+k*d/2), ph+k*d
    fx = math.cos(ph)*(wx-px)+math.sin(ph)*(wy-py)
    fy = -math.sin(ph)*(wx-px)+math.cos(ph)*(wy-py)
    fy = self.centred(fx, fy, now, (px, py, ph))
    arc = np.concatenate(([0.], np.cumsum(np.hypot(np.diff(fx), np.diff(fy)))))
    age = (now-self.query)/1e9
    turn = abs(np.interp(age+3., T_PATH, self.path[2])-np.interp(age, T_PATH, self.path[2]))
    lookahead = float(np.interp(turn, (.05, .15), (self.straight_lookahead_s, LOOKAHEAD_S)))
    target = arc[int(np.argmin(np.hypot(fx, fy)))]+max(MIN_LOOKAHEAD, v_ego*lookahead)
    if target > arc[-1]:
      self.count('short path')
      return None
    tx, ty = np.interp(target, arc, fx), np.interp(target, arc, fy)
    self.count('tracked')
    return 2*ty/max(tx*tx+ty*ty, 1.)

  def observe_lanes(self, model, now, blinker=False):
    """Keep the local model's ego lane centre (current vehicle frame, left positive) and its blend weight. With one
    confident line the centre is half the recent lane width from it."""
    self.lane = None
    if self.lane_centering <= 0. or blinker:
      return
    lines, probs = model.laneLines, model.laneLineProbs
    if len(lines) < 4 or len(probs) < 4 or len(lines[1].x) < 2 or len(lines[2].x) < 2:
      return
    pose = self.pose_at(now)
    if pose is None:
      return
    x = np.asarray(lines[1].x, dtype=float)
    left, right = np.asarray(lines[1].y, dtype=float), np.interp(x, np.asarray(lines[2].x), np.asarray(lines[2].y))
    if not np.isfinite([left, right]).all():
      return
    cl, cr = (float(np.clip((p-LANE_PROB[0])/(LANE_PROB[1]-LANE_PROB[0]), 0., 1.)) for p in (probs[1], probs[2]))
    width = right[0]-left[0]
    if cl > 0 and cr > 0 and LANE_WIDTH[0] <= width <= LANE_WIDTH[1]:
      dt = min(max((now-self.lane_width_ns)/1e9, 0.), 1.) if self.lane_width_ns else 1.
      self.lane_width += dt/(LANE_WIDTH_TAU+dt)*(width-self.lane_width)
      self.lane_width_ns = now
      centre, weight = (left+right)/2, min(cl, cr)
    elif cl > 0 and cl >= cr:
      centre, weight = left+self.lane_width/2, cl
    elif cr > 0:
      centre, weight = right-self.lane_width/2, cr
    else:
      return
    # laneLines are right positive; the tracker is left positive.
    self.lane = now, pose, x, -(centre+self.lane_center_offset_m), self.lane_centering*weight

  def centred(self, fx, fy, now, pose):
    """Plan path lateral positions (frame of `pose`) pulled toward the lane centre, or `fy` unchanged."""
    if self.lane is None or not 0 <= now-self.lane[0] <= LANE_STALE_NS:
      return fy
    self.count('lane centred')
    _, (lx0, ly0, lh0), lx, ly, weight = self.lane
    # Lane centre: lane frame -> odometry -> `pose` frame.
    wx, wy = lx0+math.cos(lh0)*lx-math.sin(lh0)*ly, ly0+math.sin(lh0)*lx+math.cos(lh0)*ly
    px, py, ph = pose
    cx = math.cos(ph)*(wx-px)+math.sin(ph)*(wy-py)
    cy = -math.sin(ph)*(wx-px)+math.cos(ph)*(wy-py)
    if not (np.diff(cx) > 0).all():
      return fy
    inside = (fx >= cx[0]) & (fx <= cx[-1])
    centre = np.interp(fx, cx, cy)
    dev = np.abs(centre-fy)
    full, none = lane_max_dev(np.maximum(fx, 0.))
    agree = np.clip((none-dev)/(none-full), 0., 1.)
    # A plan that leaves the lane stops the blend for the rest of the path. Points behind the lane lines' first point
    # (the plan starts at its older query pose) neither blend nor stop the blend.
    agree = np.minimum.accumulate(np.where(fx < cx[0], 1., agree))*inside
    ramp = np.clip((fx-LANE_RAMP_X[0])/(LANE_RAMP_X[1]-LANE_RAMP_X[0]), 0., 1.)
    w = weight*agree*ramp
    return fy+w*(centre-fy)

  def count(self, outcome):
    self.tracker_status[outcome] = self.tracker_status.get(outcome, 0)+1

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
    if self.plan_curvature_bias:
      s = np.concatenate(([0.], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
      y, yaw = y-.5*self.plan_curvature_bias*s*s, yaw-self.plan_curvature_bias*s
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
    # The tracked command stays inside the same curvature and lateral-acceleration limits as the plan itself.
    limit = min(MAX_CURVATURE, MAX_LATERAL_ACCEL_NO_ROLL/max(v_ego, 1e-3)**2)
    tracked = self.tracked_curvature(now, v_ego, lat_delay)
    if tracked is not None:
      tracked = -tracked  # openpilot desiredCurvature is right positive, like the feedforward below
    if tracked is not None and np.isfinite(tracked):
      alpha = .05/(COMMAND_TAU+.05)
      tracked = self.last_curvature+alpha*(tracked-self.last_curvature)
    curvature = float(np.clip(tracked if tracked is not None and np.isfinite(tracked) else curvature, -limit, limit))
    self.last_curvature = curvature
    return dict(desiredCurvature=float(curvature), desiredAcceleration=float(a), shouldStop=should_stop(v_ego, a))

  def apply(self, model, now, speed, lat_delay, long_delay, blinker=False):
    """Called on a fresh modelV2 builder. The action always comes from an Alpamayo plan; with lane_centering the
    local model's lane lines only pull the tracked plan path toward the lane centre.

    Without a current plan, the last Alpamayo curvature is held with non-positive acceleration and
    alpamayoQueryMonoTime stays zero, so selfdrived blocks engagement or soft-disables.
    """
    self.observe_lanes(model, now, blinker)
    candidate = self.candidate_action(now, speed, lat_delay, long_delay)
    if candidate is not None:
      model.action = candidate
      model.alpamayoQueryMonoTime = self.query
      self.held = candidate
    else:
      held = self.held or dict(desiredCurvature=0., desiredAcceleration=0., shouldStop=False)
      accel = min(held['desiredAcceleration'], 0.)
      model.action = dict(desiredCurvature=held['desiredCurvature'], desiredAcceleration=accel, shouldStop=should_stop_now(speed, accel))

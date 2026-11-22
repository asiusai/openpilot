#!/usr/bin/env python3
"""Side cameras: camera devices forward their encoded 768x432 views; the host republishes them for loggerd.

Camera role: camerad and `encoderd --side` run while the host is connected. This process forwards their
EncodeData events and answers clock probes. Host role: start the camera hotspot, authenticate camera devices,
convert frame timestamps to the host clock and publish the events. loggerd saves them as <view>camera.mp4.
"""
import collections
import json
import os
import signal
import socket
import subprocess
import threading
import time
import uuid

import openpilot.cereal.messaging as messaging
from openpilot.common.swaglog import cloudlog
from openpilot.system.asius.identity import get_device_private_key
from openpilot.system.sidecam.config import ADDRESS, CONNECTED, HELPER, HOTSPOT_MHZ, PORT, RUNTIME, VIEWS, load
from openpilot.system.sidecam.transport import FRAME, HELLO, PING, PONG, Connection

CONNECTED_GRACE = 30.  # keep cameras running through short reconnects
CLOCK_SAMPLES = 16
FRAME_NS = 50_000_000  # 20 Hz cameras


def now():
  return time.clock_gettime_ns(time.CLOCK_BOOTTIME)  # the clock of camerad timestamps and logMonoTime


def atomic_write(path, data):
  temporary = path.with_suffix('.tmp')
  temporary.write_text(data)
  os.replace(temporary, path)


def helper(config, body):
  result = subprocess.run(['sudo', config.get('helper', HELPER)], input=json.dumps({'owner': 'remote-cameras', **body}),
                          text=True, capture_output=True, timeout=45)
  reply = json.loads(result.stdout or '{}')
  if result.returncode or 'error' in reply:
    raise RuntimeError(f"Camera network helper failed: {reply.get('error', result.returncode)}")
  return reply


class Hotspot:
  """Hold the host's leased 5 GHz access point; the station connection stays up."""

  def __init__(self, config, stop):
    self.config, self.stop, self.lease = config, stop, None
    self.thread = threading.Thread(target=self.run, name='sidecam-hotspot', daemon=True)

  def run(self):
    while not self.stop.is_set():
      try:
        if self.lease is None:
          self.lease = helper(self.config, {'action': 'start', 'network': self.config['network']})['lease']
          cloudlog.warning('sidecam: hotspot started')
        else:
          helper(self.config, {'action': 'renew', 'lease': self.lease})
      except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError):
        cloudlog.exception('sidecam: hotspot lease failed')
        self.lease = None
      self.stop.wait(25 if self.lease else 5)

  def close(self):
    if self.lease:
      try:
        helper(self.config, {'action': 'stop', 'lease': self.lease})
      except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        cloudlog.exception('sidecam: hotspot stop failed')


class ViewStats:
  def __init__(self):
    self.frames = self.missing = 0
    self.last_frame = None
    self.latency = collections.deque(maxlen=200)  # sensor EOF to host publish, ms
    self.last_ns = 0

  def add(self, frame_id, eof_ns, latency_ms):
    # Count gaps by sensor time: frame ids restart with camerad.
    if self.last_frame is not None:
      self.missing += max(0, round((eof_ns - self.last_frame) / FRAME_NS) - 1)
    self.last_frame = eof_ns
    self.frames += 1
    self.latency.append(latency_ms)
    self.last_ns = now()

  def summary(self):
    values = sorted(self.latency)

    def pick(q):
      return round(values[min(len(values) - 1, int(q * len(values)))], 1) if values else None
    return {'frames': self.frames, 'missing': self.missing, 'latency_p50_ms': pick(.5), 'latency_p95_ms': pick(.95),
            'latency_max_ms': pick(1.), 'age_s': round((now() - self.last_ns) / 1e9, 2) if self.last_ns else None}


class Host:
  def __init__(self, config, identity, stop):
    self.config, self.identity, self.stop = config, identity, stop
    self.views = {peer: set(views) for peer, views in config['peers'].items()}
    self.pm = messaging.PubMaster([f'{v}EncodeData' for v in VIEWS])
    self.lock = threading.Lock()
    self.clocks = {}  # (peer, boot id) -> deque of (rtt, offset)
    self.stats = {v: ViewStats() for v in VIEWS}
    self.devices = {}

  def offset(self, key):
    with self.lock:
      samples = self.clocks.get(key)
      return min(samples)[1] if samples else None

  def ping(self, conn, key, closed):
    count = 0
    while not closed.is_set() and not self.stop.is_set():
      try:
        conn.send(PING, now().to_bytes(8, 'little'))
      except (OSError, ValueError):
        return
      count += 1
      closed.wait(.05 if count < CLOCK_SAMPLES else 1.)

  def handle(self, sock):
    conn = Connection(sock, self.identity)
    closed = threading.Event()
    try:
      conn.accept(self.views)
      kind, payload = conn.receive()
      device = json.loads(payload)
      if kind != HELLO or not isinstance(device, dict) or not isinstance(device.get('boot_id'), str):
        raise ValueError('Expected camera hello')
      key = (conn.peer, device['boot_id'])
      with self.lock:
        self.devices[conn.peer] = device
      cloudlog.warning(f"sidecam: camera {device.get('serial')} connected for {sorted(self.views[conn.peer])}")
      threading.Thread(target=self.ping, args=(conn, key, closed), daemon=True).start()
      while not self.stop.is_set():
        kind, payload = conn.receive()
        received = now()
        if kind == PONG:
          t1, t2 = int.from_bytes(payload[:8], 'little'), int.from_bytes(payload[8:16], 'little')
          rtt = received - t1
          if 0 <= rtt <= 50_000_000:
            with self.lock:
              self.clocks.setdefault(key, collections.deque(maxlen=CLOCK_SAMPLES)).append((rtt, t1 + rtt // 2 - t2))
        elif kind == FRAME:
          self.publish(conn.peer, key, payload, received)
        else:
          raise ValueError('Unexpected camera message kind')
    except (OSError, ValueError, EOFError, RuntimeError, KeyError, json.JSONDecodeError) as error:
      cloudlog.warning(f'sidecam: camera connection ended ({conn.peer}): {type(error).__name__}: {str(error)[:160]}')
    finally:
      closed.set()
      conn.close()

  def publish(self, peer, key, payload, received):
    offset = self.offset(key)
    if offset is None:
      return  # timestamps cannot be placed on the host clock yet
    reader = messaging.log_from_bytes(payload)
    which = reader.which()
    if not which.endswith('EncodeData') or which[:2] not in self.views[peer]:
      raise ValueError(f'Unauthorized camera view {which}')
    event = reader.as_builder()
    data = getattr(event, which)
    idx = data.idx
    idx.timestampSof += offset
    idx.timestampEof += offset
    event.logMonoTime = received
    data.unixTimestampNanos = time.time_ns() - (now() - idx.timestampEof)
    with self.lock:
      self.pm.send(which, event)
      self.stats[which[:2]].add(idx.frameId, idx.timestampEof, (now() - idx.timestampEof) / 1e6)

  def report(self):
    last_log = 0.
    while not self.stop.wait(1):
      with self.lock:
        status = {'role': 'host', 'views': {v: s.summary() for v, s in self.stats.items() if s.frames},
                  'clocks': {f'{p[:8]}:{b[:8]}': {'offset_ms': min(c)[1] / 1e6, 'rtt_ms': min(c)[0] / 1e6}
                             for (p, b), c in self.clocks.items() if c},
                  'devices': {p[:8]: {k: d.get(k) for k in ('serial', 'commit', 'views')} for p, d in self.devices.items()}}
      atomic_write(RUNTIME / 'status.json', json.dumps(status))
      if time.monotonic() - last_log > 10:
        last_log = time.monotonic()
        cloudlog.warning(f"sidecam: {json.dumps(status['views'])}")

  def serve(self):
    threading.Thread(target=self.report, daemon=True).start()
    with socket.socket() as server:
      server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      while not self.stop.is_set():
        try:
          server.bind((ADDRESS, PORT))
          break
        except OSError:
          self.stop.wait(1)  # the hotspot address appears after the access point starts
      server.listen(8)
      server.settimeout(.5)
      while not self.stop.is_set():
        try:
          sock, _ = server.accept()
        except TimeoutError:
          continue
        threading.Thread(target=self.handle, args=(sock,), daemon=True).start()


class Camera:
  def __init__(self, config, identity, stop):
    self.config, self.identity, self.stop = config, identity, stop
    self.views = [config['wide'], config['road']]
    serial = f"{int(open('/sys/devices/soc0/serial_number').read()):08x}"
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                            cwd=os.path.dirname(__file__)).stdout.strip()
    self.device = {'serial': serial, 'boot_id': open('/proc/sys/kernel/random/boot_id').read().strip(),
                   'commit': commit, 'views': self.views}
    self.conn = None
    self.last_connected = 0.

  def connect(self):
    sock = socket.create_connection((ADDRESS, PORT), timeout=3)
    conn = Connection(sock, self.identity)
    try:
      conn.initiate(self.config['host'])
      conn.send(HELLO, json.dumps(self.device).encode())
    except BaseException:
      conn.close()
      raise
    threading.Thread(target=self.answer, args=(conn,), daemon=True).start()
    return conn

  def answer(self, conn):
    try:
      while True:
        kind, payload = conn.receive()
        if kind != PING or len(payload) != 8:
          raise ValueError('Unexpected host message')
        conn.send(PONG, payload + now().to_bytes(8, 'little'))
    except (OSError, ValueError, EOFError, RuntimeError):
      if self.conn is conn:
        self.conn = None
      conn.close()

  def mark_connected(self):
    if self.conn is not None:
      self.last_connected = time.monotonic()
    if time.monotonic() - self.last_connected < CONNECTED_GRACE:
      atomic_write(CONNECTED, str(time.monotonic()))
    else:
      CONNECTED.unlink(missing_ok=True)

  def serve(self):
    poller = messaging.Poller()
    for v in self.views:
      messaging.sub_sock(f'{v}EncodeData', poller=poller, conflate=False)
    network = Network(self.config, self.stop)
    last_mark = 0.
    while not self.stop.is_set():
      if self.conn is None:
        try:
          network.join_if_visible()
        except (subprocess.SubprocessError, OSError):
          cloudlog.exception('sidecam: camera network check failed')
        try:
          self.conn = self.connect()
          cloudlog.warning('sidecam: connected to host')
        except (OSError, ValueError, EOFError, RuntimeError):
          self.mark_connected()
          self.stop.wait(1)
          continue
      elif self.conn.needs_renewal():
        # Make before break: the old session keeps sending until the new one is ready.
        try:
          old, self.conn = self.conn, self.connect()
          old.close()
        except (OSError, ValueError, EOFError, RuntimeError):
          pass
      if time.monotonic() - last_mark > 1:
        last_mark = time.monotonic()
        self.mark_connected()
      for sock in poller.poll(100):
        while (data := sock.receive(non_blocking=True)) is not None:
          conn = self.conn
          if conn is None:
            continue  # drop frames while disconnected; stale video is useless
          try:
            conn.send(FRAME, data)
          except (OSError, ValueError, RuntimeError):
            if self.conn is conn:
              self.conn = None
            conn.close()
    CONNECTED.unlink(missing_ok=True)


class Network:
  """Keep a persistent camera-network profile and prefer it whenever the host's hotspot is visible."""
  UUID = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'asius-sidecam'))
  PATH = '/etc/NetworkManager/system-connections/asius-sidecam.nmconnection'

  def __init__(self, config, stop):
    self.ssid, self.stop, self.next_scan, self.scans = config['network']['ssid'], stop, 0., 0
    escaped = config['network']['password'].replace('\\', '\\\\').replace(' ', '\\s')
    # `hidden` makes NetworkManager probe for this SSID in its own scans.
    profile = '\n'.join([
      '[connection]', 'id=Asius side cameras', f'uuid={self.UUID}', 'type=wifi', 'interface-name=wlan0',
      'autoconnect=true', 'autoconnect-priority=100',
      '[wifi]', 'mode=infrastructure', 'band=a', 'hidden=true', 'powersave=2', f'ssid={";".join(str(b) for b in self.ssid.encode())};',
      '[wifi-security]', 'key-mgmt=wpa-psk', f'psk={escaped}',
      '[ipv4]', 'method=auto', 'never-default=true', '[ipv6]', 'method=disabled', '',
    ])
    try:
      current = subprocess.run(['sudo', 'cat', self.PATH], capture_output=True, text=True, timeout=10).stdout
    except subprocess.SubprocessError:
      current = None
    if current != profile:
      try:
        # The password travels on stdin, never in arguments.
        subprocess.run(['sudo', 'install', '-m', '600', '/dev/stdin', self.PATH], input=profile, text=True, check=True, timeout=10)
        subprocess.run(['sudo', 'nmcli', 'connection', 'load', self.PATH], capture_output=True, timeout=15)
      except (subprocess.SubprocessError, OSError):
        cloudlog.exception('sidecam: cannot write the camera network profile')

  def join_if_visible(self):
    if time.monotonic() < self.next_scan:
      return
    self.next_scan = time.monotonic() + 10
    active = subprocess.run(['nmcli', '-g', 'GENERAL.CON-UUID', 'device', 'show', 'wlan0'], capture_output=True, text=True, timeout=10)
    if active.stdout.strip() == self.UUID:
      return
    if self.visible():
      cloudlog.warning('sidecam: joining the camera network')
      subprocess.run(['sudo', 'nmcli', '--wait', '15', 'connection', 'up', 'uuid', self.UUID], capture_output=True, timeout=20)

  def visible(self):
    """Probe for the hotspot as root: the device user cannot trigger NetworkManager scans, so its list goes stale.
    Most probes cover only the hotspot channel to keep the current link usable; every sixth probe covers all channels."""
    self.scans += 1
    channel = [] if self.scans % 6 == 0 else ['freq', str(HOTSPOT_MHZ)]
    scan = subprocess.run(['sudo', 'iw', 'dev', 'wlan0', 'scan', *channel, 'ssid', self.ssid],
                          capture_output=True, text=True, timeout=20).stdout
    if not scan:  # busy with another scan: use its results
      scan = subprocess.run(['sudo', 'iw', 'dev', 'wlan0', 'scan', 'dump'], capture_output=True, text=True, timeout=10).stdout
    return f'\tSSID: {self.ssid}\n' in scan


def main():
  config, identity, stop = load(), get_device_private_key(), threading.Event()
  signal.signal(signal.SIGTERM, lambda *_: stop.set())
  signal.signal(signal.SIGINT, lambda *_: stop.set())
  RUNTIME.mkdir(mode=0o700, parents=True, exist_ok=True)
  if config['role'] == 'host':
    hotspot = Hotspot(config, stop)
    hotspot.thread.start()
    try:
      Host(config, identity, stop).serve()
    finally:
      hotspot.close()
  else:
    Camera(config, identity, stop).serve()


if __name__ == '__main__':
  main()

"""Supervise shadow-only capture and inference without stopping openpilot."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import shutil


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--bundle', type=Path, default=Path('/data/alpamayo-eval'))
  parser.add_argument('--openpilot', type=Path, default=Path('/data/openpilot'))
  parser.add_argument('--serial', required=True, help='Exact expected value from the target SoC serial file')
  args = parser.parse_args()
  os.umask(0o077)
  assert os.getuid() != 0, 'Run as comma'
  assert Path('/sys/devices/soc0/serial_number').read_text().strip() == args.serial
  lock = (args.bundle / 'shadow.lock').open('w')
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  # Check the actual driving-model process, not only persistent parameter values.
  model_pids = []
  for proc in Path('/proc').glob('[0-9]*'):
    try:
      command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ')
      name = (proc / 'comm').read_text().strip()
      is_model = name == 'modeld' or b'selfdrive.modeld.modeld' in command or b'selfdrive/modeld/modeld.py' in command
      if is_model:
        # modeld's setproctitle wipes its original /proc environment. Its manager
        # parent retains the environment inherited by the forked model process.
        status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines())
        parent = Path('/proc') / status['PPid'].strip()
        assert b'manager.py' in (parent / 'cmdline').read_bytes(), 'Unexpected model supervisor'
        checked = subprocess.run(['sudo', '-n', 'grep', '-z', '-q', '-x', 'DISABLE_CHESTNUT=1', str(parent / 'environ')])
        assert checked.returncode == 0, 'Driving-model manager must disable Chestnut first'
        assert 'os.environ.get("DISABLE_CHESTNUT") != "1" and chestnut_compiled()' in (args.openpilot / 'openpilot/selfdrive/modeld/modeld.py').read_text()
        model_pids.append(int(proc.name))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
      continue
  assert len(model_pids) == 1, f'Expected one small-model process, found {model_pids}'
  assert shutil.disk_usage(args.bundle).free > 2*1024**3, 'Need 2 GiB disk reserve'
  output = args.bundle / 'shadow' / time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
  output.mkdir(parents=True)
  import hashlib
  sources = {name: hashlib.sha256((args.bundle / name).read_bytes()).hexdigest()
             for name in ('run-shadow.py', 'shadow-capture.py', 'shadow-worker.py', 'benchmark-fast.py')}
  (output / 'runtime-sources.json').write_text(json.dumps(sources, indent=2)+'\n')
  env = dict(os.environ, ALPAMAYO_SHADOW_DIR=str(output), OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
  children = []
  logs = []
  def stop(*_):
    raise KeyboardInterrupt
  signal.signal(signal.SIGTERM, stop)
  signal.signal(signal.SIGINT, stop)
  try:
    cpus = ','.join(str(c) for c in sorted(os.sched_getaffinity(0)) if c < 6)
    assert cpus
    capture_env = dict(env, PYTHONPATH=f'{args.bundle}/capture-deps:{args.openpilot}')
    logs.append((output / 'capture.log').open('w'))
    children.append(subprocess.Popen(['nice', '-n', '10', 'ionice', '-c', '3', 'taskset', '-c', cpus,
                                     str(args.openpilot / '.venv/bin/python'), '-u', str(args.bundle / 'shadow-capture.py')],
                                     env=capture_env, cwd=args.openpilot, stdout=logs[-1], stderr=subprocess.STDOUT, start_new_session=True))
    deadline = time.monotonic()+15
    while not (output / 'pending.npz').exists():
      assert children[0].poll() is None, 'Capture failed; see capture.log'
      assert time.monotonic() < deadline, 'No valid live camera, measured motion, calibration and small-model snapshot'
      time.sleep(.1)
    worker_env = dict(env, DEV='USB+AMD:LLVM', GMMU='0', AM_POWER_LIMIT='100', TC_OPT='2',
                      ALPAMAYO_FP8_CACHE=str(args.bundle / 'fp8-cache'), CACHEDB=str(args.bundle / 'shadow-cache.db'),
                      PYTHONPATH=f'{args.bundle}/tinygrad:{args.bundle}/alpamayo')
    worker_env.pop('ALPAMAYO_USB_PCIE', None)
    worker_env.pop('ALPAMAYO_USB_SERIAL_UPLOAD', None)
    logs.append((output / 'worker.log').open('w'))
    # The authenticated experimental session keeps its own memory allocation.
    # Reduce CPU and I/O priority; leave driving-model CPUs 6-7 available.
    children.append(subprocess.Popen(['nice', '-n', '10', 'ionice', '-c', '3', 'taskset', '-c', cpus,
                                     str(args.bundle / 'upstream/.venv/bin/python'), '-u', str(args.bundle / 'shadow-worker.py'),
                                     '--root', str(output), '--checkpoint', str(args.bundle / 'model')],
                                     env=worker_env, stdout=logs[-1], stderr=subprocess.STDOUT, start_new_session=True))
    failures = 0
    print(f'Shadow output: {output}', flush=True)
    while True:
      assert all(child.poll() is None for child in children), 'Shadow process exited; see logs'
      assert Path(f'/proc/{model_pids[0]}').exists(), 'Driving model restarted; recheck Chestnut ownership'
      health = json.loads((output / 'health.json').read_text())
      healthy = (time.monotonic_ns()-health['time_ns'] < 2_000_000_000 and health['model_alive']
                 and health['model_age_s'] < .25 and health['thermal_status'] not in ('red', 'danger'))
      failures = 0 if healthy else failures+1
      assert failures < 4, f'Openpilot health guard stopped shadow: {health}'
      mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
      assert int(mem['MemAvailable'].split()[0]) > 768*1024, 'Memory reserve reached'
      assert shutil.disk_usage(output).free > 1024**3, 'Disk reserve reached'
      time.sleep(.5)
  finally:
    for child in children:
      try:
        os.killpg(child.pid, signal.SIGTERM)
      except ProcessLookupError:
        pass
    for child in children:
      try:
        child.wait(timeout=5)
      except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
      # Compiler workers must not survive their parent.
      try:
        os.killpg(child.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
    for stream in logs:
      stream.close()
    for name in ('pending.npz', 'working.npz', 'pending.tmp'):
      (output / name).unlink(missing_ok=True)
    print('Shadow stopped; openpilot was not stopped or restarted.', flush=True)


if __name__ == '__main__':
  main()

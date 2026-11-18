"""Supervise the isolated GPU worker and forward plans only in planner mode."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def planner_mode():
  return os.environ.get('ALPAMAYO_PLANNER') == '1' and os.environ.get('ALPAMAYO_SHADOW') != '1'


def cache_directory(config, output):
  from openpilot.selfdrive.alpamayo.runtime import private_output
  cache = Path(config['cache'])
  if not cache.is_absolute() or not cache.is_dir():
    raise ValueError('Configure an absolute existing persistent cache directory')
  cache = private_output(cache)
  if cache.is_relative_to(output) or output.is_relative_to(cache):
    raise ValueError('Keep persistent cache and session output in separate directories')
  with tempfile.TemporaryFile(dir=cache):
    pass
  return cache


def worker_environment(config, cache):
  repository = Path(__file__).resolve().parents[3]
  paths = [Path(config['tinygrad']), repository, repository/'opendbc_repo']
  return dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, paths)), PYTHONNOUSERSITE='1',
              DEV='USB+AMD:LLVM', GMMU='0', TC_OPT='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
              XDG_CACHE_HOME=str(cache), CACHEDB=str(cache/'compiler-cache.db'))


def stop_worker(child):
  if child is None:
    return
  try:
    os.killpg(child.pid, signal.SIGTERM)
    child.wait(timeout=1)
  except (ProcessLookupError, subprocess.TimeoutExpired):
    pass
  try:
    os.killpg(child.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  child.wait(timeout=1)


def main():
  if os.environ.get('ALPAMAYO_SHADOW') != '1' and not planner_mode():
    return
  from openpilot.selfdrive.alpamayo.runtime import private_output, ram_handoff, storage_ok
  config_path = Path(os.environ['ALPAMAYO_CONFIG']).resolve()
  config = json.loads(config_path.read_text())
  if planner_mode() and (config.get('period_ms', 500) != 500 or config['inference'].get('variant') != 'selected'):
    raise ValueError('Planner evaluation requires the selected 500 ms configuration')
  python = Path(config['python'])
  if not python.is_file() or python == Path(sys.executable):
    raise ValueError('Configure a separate, pinned Alpamayo Python environment')
  output = Path(config['output'])
  if not output.is_absolute() or not output.is_dir() or not storage_ok(output, 128*1024**2):
    raise ValueError('Configure an existing output directory with recording disk reserve')
  output = private_output(output)
  os.umask(0o077)
  cache = cache_directory(config, output)
  def stop(*_):
    raise KeyboardInterrupt
  signal.signal(signal.SIGTERM, stop)
  boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  root = output/f'{boot}-{time.monotonic_ns()}'
  root.mkdir()
  with ram_handoff() as handoff:
    supervise(config, config_path, python, root, Path(handoff), cache, boot)


def supervise(config, config_path, python, root, handoff, cache, boot):
  from openpilot.common.params import Params
  from openpilot.selfdrive.alpamayo.capture import NativeProducer, healthy
  from openpilot.selfdrive.alpamayo.frontend import Store, Unavailable
  from openpilot.selfdrive.alpamayo.runtime import FixedSlots, memory_ok, save_selection, storage_ok
  slots = FixedSlots(config.get('period_ms', 500))
  pm = None
  if planner_mode():
    from openpilot.cereal import messaging
    pm = messaging.PubMaster(['alpamayoPlan'])
  def publish(payload=None):
    if pm is not None:
      msg = messaging.new_message('alpamayoPlan', valid=payload is not None)
      if payload is not None:
        msg.alpamayoPlan = payload
      pm.send('alpamayoPlan', msg)
  store, params = Store(), Params()
  producer = NativeProducer(store)
  child = None
  last_query, last_current = 0, {}
  started = time.monotonic()
  try:
    producer.start()
    with (root/'worker.log').open('xb') as log:
      while True:
        health = producer.health
        now = time.monotonic_ns()
        if producer.error is not None:
          raise RuntimeError('Shadow input capture failed') from producer.error
        if not healthy(health, now):
          if time.monotonic()-started > 10 or child is not None:
            raise RuntimeError('Small-model or device health stopped shadow inference')
          producer.stop.wait(.05)
          continue
        if not memory_ok() or not storage_ok(root, 128*1024**2):
          raise RuntimeError('Shadow resource reserve reached')
        if child is None:
          child = subprocess.Popen([str(python), '-m', 'openpilot.selfdrive.alpamayo.shadowd', str(config_path), str(root), str(handoff)],
                                   env=worker_environment(config, cache), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if child.poll() is not None:
          raise RuntimeError(f'Shadow runtime exited ({child.returncode}); see {root / "worker.log"}')
        latest = handoff/'plan.json'
        if pm is not None and latest.exists():
          # Rename before reading: a newer result can replace the mailbox without being unlinked here.
          reading = latest.replace(handoff/'reading.json')
          publish(json.loads(reading.read_bytes()))
          reading.unlink()
        if (handoff/'skipped').exists():
          (handoff/'skipped').unlink()
          slots.skip()
          publish()
        slot = slots.take(time.monotonic_ns()) if (root/'ready.json').exists() else None
        if slot is not None:
          try:
            selected_ns = time.monotonic_ns()
            if (selected_ns >= slot['slot_deadline_ns'] or (handoff/'pending.bin').exists() or (handoff/'working.bin').exists()):
              raise Unavailable('Busy or late slot')
            selected = store.select(selected_ns, last_query, last_current)
            route = params.get('CurrentRoute')
            if not route:
              raise Unavailable('No current recording route')
            context = dict(slot, route=route.decode() if isinstance(route, bytes) else route, boot_id=boot)
            save_selection(handoff/'pending.bin', selected, context)
            last_query = selected.query_ns
            last_current = {c: selected.frames[i*5+4].sof_ns for i, c in enumerate((1, 6))}
          except Unavailable:
            slots.skip()
            publish()
        producer.stop.wait(.005 if pm is not None else .02)
  finally:
    try:
      stop_worker(child)
    finally:
      try:
        publish()
      finally:
        producer.close()


if __name__ == '__main__':
  if len(sys.argv) == 4:
    if os.environ.get('ALPAMAYO_SHADOW') != '1' and not planner_mode():
      raise RuntimeError('Explicit shadow opt-in is required')
    os.nice(10)
    cpus = {cpu for cpu in os.sched_getaffinity(0) if cpu < 6}
    if cpus:
      os.sched_setaffinity(0, cpus)
    from openpilot.selfdrive.alpamayo.runtime import worker
    worker(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], sys.argv[3])
  else:
    main()

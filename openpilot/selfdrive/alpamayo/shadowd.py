"""Supervise the isolated GPU worker and forward plans only in planner mode."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

MATCH_WAIT_NS = 100_000_000
ANCHOR_WAIT_NS = 200_000_000  # Allows two-camera priming in this slot after a short overrun.

def planner_mode():
  return (bool(os.environ.get('ALPAMAYO_CONFIG')) and os.environ.get('ALPAMAYO_PLANNER', '1') == '1'
          and os.environ.get('ALPAMAYO_SHADOW') != '1')


class WorkerStopped(RuntimeError):
  """A fresh worker can recover without restarting modeld or interrupting the local model."""


def worker_cpu_ticks(pid):
  try:
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[11])+int(fields[12])  # utime + stime, including compiler work in this process.
  except (OSError, IndexError, ValueError):
    return None


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
  for attempt in range(3):
    root = output/f'{boot}-{time.monotonic_ns()}'
    root.mkdir()
    try:
      with ram_handoff() as handoff:
        supervise(config, config_path, python, root, Path(handoff), cache, boot)
      return
    except WorkerStopped as exc:
      (root/'worker-exit.txt').write_text(str(exc)+'\n')
      if attempt == 2:
        raise
      time.sleep(2**attempt)


def supervise(config, config_path, python, root, handoff, cache, boot):
  from openpilot.common.params import Params
  from openpilot.selfdrive.alpamayo.capture import NativeProducer, device_healthy, healthy
  from openpilot.selfdrive.alpamayo.frontend import AwaitingFrame, Store, Unavailable
  from openpilot.selfdrive.alpamayo.planner import HANDOFF_GRACE, deliverable
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
        published = time.monotonic_ns()
        if deliverable(payload['queryMonoTime'],payload['readyMonoTime'],payload['slotDeadline'],published):
          msg.logMonoTime = published
        else:
          msg = messaging.new_message('alpamayoPlan', valid=False)
      pm.send('alpamayoPlan', msg)
  store, params = Store(), Params()
  producer = NativeProducer(store)
  child = None
  worker_start = working_since = None
  next_check, last_ticks = 0., None
  admission = None
  served_plan = False
  model_seen = False
  last_query, last_current = 0, {}
  try:
    producer.start()
    with (root/'worker.log').open('xb') as log:
      while True:
        health = producer.health
        now = time.monotonic_ns()
        if producer.error is not None:
          raise RuntimeError('Shadow input capture failed') from producer.error
        if not device_healthy(health, now):
          if child is not None:
            raise WorkerStopped('Device health stopped inference')
          producer.stop.wait(.05)
          continue
        model_healthy = healthy(health, now)
        if model_seen and not model_healthy:
          raise WorkerStopped('Small-model health stopped inference')
        model_seen |= model_healthy
        if not memory_ok() or not storage_ok(root, 128*1024**2):
          raise RuntimeError('Shadow resource reserve reached')
        if child is None:
          child = subprocess.Popen([str(python), '-m', 'openpilot.selfdrive.alpamayo.shadowd', str(config_path), str(root), str(handoff)],
                                   env=worker_environment(config, cache), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
          worker_start = time.monotonic()
        if child.poll() is not None:
          raise WorkerStopped(f'Inference worker exited ({child.returncode})')
        ready = (root/'ready.json').exists()
        if not ready and time.monotonic()-worker_start > 600:
          raise WorkerStopped('Inference initialization timed out')
        # Load weights alongside small-model startup. No slot or plan can be admitted until the model is healthy.
        if not model_healthy:
          producer.stop.wait(.05)
          continue
        if (handoff/'working.bin').exists() or (handoff/'pending.bin').exists():
          elapsed = time.monotonic()
          if working_since is None:
            working_since, next_check, last_ticks = elapsed, elapsed+10, None
          # A new reasoning shape can compile for minutes even after a timely straight-road plan.
          if elapsed-working_since > 600:
            raise WorkerStopped('Inference worker stopped completing slots')
          if served_plan and elapsed >= next_check:
            ticks = worker_cpu_ticks(child.pid)
            if ticks is not None and ticks == last_ticks:
              raise WorkerStopped('Inference worker stopped making progress')
            last_ticks, next_check = ticks, elapsed+10
        else:
          working_since = None
        latest = handoff/'plan.json'
        if pm is not None and latest.exists():
          # Rename before reading: a newer result can replace the mailbox without being unlinked here.
          reading = latest.replace(handoff/'reading.json')
          payload = json.loads(reading.read_bytes())
          publish(payload)
          served_plan |= payload is not None and time.monotonic_ns() < payload['slotDeadline']
          working_since = None
          reading.unlink()
        if (handoff/'skipped').exists():
          (handoff/'skipped').unlink()
          slots.skip()
          publish()
        slot = slots.take(time.monotonic_ns()) if ready else None
        if slot is not None:
          admission = slot
        slot = admission
        if slot is not None:
          try:
            selected_ns = time.monotonic_ns()
            busy = (handoff/'pending.bin').exists() or (handoff/'working.bin').exists()
            if busy and pm is not None:
              age = selected_ns-slot['slot_start_ns']
              if age >= HANDOFF_GRACE:
                # This slot loses its prediction. Its actual camera pair can still seed the next slot's history.
                slot['anchor_only'] = True
              if age < ANCHOR_WAIT_NS:
                producer.stop.wait(.005)
                continue
            admission = None
            if selected_ns >= slot['slot_deadline_ns'] or busy or (slot.get('anchor_only') and selected_ns-slot['slot_start_ns'] >= ANCHOR_WAIT_NS):
              raise Unavailable('Busy or late slot')
            try:
              selected = store.select(selected_ns, last_query, last_current, half_second_slot=slot['slot_start_ns'] if slots.period_ms == 500 else None)
            except AwaitingFrame:
              # The matched modelV2 can arrive after this slot opens. Keep the same opening, cutoff and deadline.
              wait_ns = ANCHOR_WAIT_NS if slot.get('anchor_only') else MATCH_WAIT_NS
              if slots.period_ms == 500 and selected_ns < slot['slot_start_ns']+wait_ns:
                admission = slot
                producer.stop.wait(.002)
                continue
              raise
            route = params.get('CurrentRoute')
            if not route:
              raise Unavailable('No current recording route')
            context = dict(slot, route=route.decode() if isinstance(route, bytes) else route, boot_id=boot)
            save_selection(handoff/'pending.bin', selected, context)
            last_query = selected.query_ns
            last_current = {f.camera: f.sof_ns for f in selected.frames}
          except Unavailable:
            slots.skip()
            publish()
        producer.stop.wait(.005 if pm is not None else .02)
  finally:
    try:
      publish()
    finally:
      try:
        stop_worker(child)
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

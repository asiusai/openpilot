#!/usr/bin/env python3
import datetime
import subprocess
import time
from typing import NoReturn

import openpilot.cereal.messaging as messaging
from openpilot.common.time_helpers import min_date, MAX_DATE, system_time_valid
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.gps import get_gps_location_services, latest_gps_source
from openpilot.common.hardware import ASIUS_HARDWARE


def set_time(new_time, sample_mono=None):
  if ASIUS_HARDWARE:
    try:
      subprocess.run(['sudo', '-n', '/usr/bin/vamos-clock', 'sync', '--source', 'gps',
                      '--unix-ms', str(new_time.replace(tzinfo=datetime.UTC).timestamp() * 1000),
                      '--sample-mono', str(sample_mono)], check=True, timeout=5, capture_output=True)
    except (OSError, subprocess.SubprocessError):
      cloudlog.exception('timed.failed_setting_time')
    return
  diff = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - new_time
  if abs(diff) < datetime.timedelta(seconds=10):
    cloudlog.debug(f"Time diff too small: {diff}")
    return

  cloudlog.debug(f"Setting time to {new_time}")
  try:
    subprocess.run(f"TZ=UTC date -s '{new_time}'", shell=True, check=True)
  except subprocess.CalledProcessError:
    cloudlog.exception("timed.failed_setting_time")


def main() -> NoReturn:
  """
    timed has two responsibilities:
    - getting the current time from GPS
    - publishing the time in the logs

    AGNOS will also use NTP to update the time.
  """

  params = Params()
  gps_services = get_gps_location_services(params)

  pm = messaging.PubMaster(['clocks'])
  sm = messaging.SubMaster(gps_services)
  while True:
    sm.update(1000)

    msg = messaging.new_message('clocks')
    msg.valid = system_time_valid()
    msg.clocks.wallTimeNanos = time.time_ns()
    pm.send('clocks', msg)

    gps_location_service = latest_gps_source(sm, gps_services)
    if gps_location_service is None:
      continue
    gps = sm[gps_location_service]
    sample_mono = sm.logMonoTime[gps_location_service] / 1e9
    if not sm.updated[gps_location_service] or not 0 <= time.monotonic() - sample_mono <= 2.0:
      continue
    if not gps.hasFix:
      continue
    gps_time = datetime.datetime.fromtimestamp(gps.unixTimestampMillis / 1000., datetime.UTC).replace(tzinfo=None)
    if gps_time < min_date() or gps_time > MAX_DATE:
      continue

    set_time(gps_time, sample_mono)
    time.sleep(10)

if __name__ == "__main__":
  main()

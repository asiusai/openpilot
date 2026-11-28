"""Read-only driving UI snapshots for the live WebRTC data channel."""
from __future__ import annotations

import time

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.helpers import chestnut_compiled
from openpilot.selfdrive.selfdrived.alertmanager import OFFROAD_ALERTS

CHESTNUT_ALERTS = tuple(k for k, a in OFFROAD_ALERTS.items() if k.startswith('Offroad_Chestnut') and a['severity'] >= 0)


def offroad_alerts(params) -> list[dict]:
  """The active offroad alerts as the device UI shows them, with their extra text filled in."""
  alerts = []
  for key, definition in OFFROAD_ALERTS.items():
    alert = params.get(key)
    if isinstance(alert, dict) and isinstance(alert.get('text'), str) and alert['text']:
      alerts.append({'key': key, 'text': alert['text'].replace('%1', str(alert.get('extra', ''))), 'severity': definition.get('severity', 0)})
  return alerts


def gpu_status(sm, params) -> str:
  now = time.monotonic()
  if not sm.seen['deviceState'] or not sm.valid['deviceState'] or now - sm.logMonoTime['deviceState'] / 1e9 >= 5:
    return 'unknown'
  device = sm['deviceState']
  if any(params.get(key) is not None for key in CHESTNUT_ALERTS):
    return 'failed'
  active = params.get('ChestnutActive')
  loading = params.get('ChestnutLoading') is True
  if not device.chestnutPresent:
    return 'failed' if device.started and active is not None else 'disconnected'
  if not chestnut_compiled():
    return 'uncompiled'
  if not device.started:
    return 'ready'
  if loading:
    return 'loading'
  if active is False:
    return 'failed'
  model_seen = sm.seen['modelV2']
  model_fresh = model_seen and sm.valid['modelV2'] and now - sm.logMonoTime['modelV2'] / 1e9 < 1
  if model_fresh and sm['modelV2'].big and active is True:
    return 'active'
  if active is True or model_seen:
    return 'failed'
  return 'loading'


SERVICES = ['deviceState', 'carState', 'selfdriveState', 'driverMonitoringState', 'extrinsicsCalibration',
            'modelV2', 'carOutput', 'narrowRoadCameraState', 'wideRoadCameraState']
FIELDS = {
  'deviceState': ['deviceType', 'started', 'chestnutPresent', 'usbState', 'networkType', 'thermalStatus', 'gpuTempC', 'cpuTempC', 'freeSpacePercent'],
  'carState': ['vEgo', 'vEgoCluster', 'vCruiseCluster', 'steeringAngleDeg', 'leftBlinker', 'rightBlinker'],
  'selfdriveState': ['enabled', 'state', 'alertText1', 'alertText2', 'alertSize', 'alertStatus', 'alertType', 'alertHudVisual'],
  'driverMonitoringState': ['activePolicy', 'isRHD', 'alertLevel', 'lockout', 'lockoutMinutesRemaining', 'visionPolicyState'],
  'extrinsicsCalibration': ['rpyCalib', 'wideFromDeviceEuler', 'calStatus', 'height'],
  'modelV2': ['position', 'laneLines', 'laneLineProbs', 'roadEdges', 'roadEdgeStds'],
  'narrowRoadCameraState': ['sensor'],
  'wideRoadCameraState': ['sensor'],
}


class InCarTelemetry:
  def __init__(self):
    self.params = Params()
    self.sm = messaging.SubMaster(SERVICES)
    self.was_started = False
    self.started_at = 0.0
    self.wide = False
    self.alerts = []
    self.alerts_updated = 0.0

  def snapshot(self) -> dict:
    self.sm.update(0)
    now = time.monotonic()
    services = {}
    for name, fields in FIELDS.items():
      if not self.sm.seen[name]:
        continue
      age = max(0, now - self.sm.logMonoTime[name] / 1e9)
      max_age = 5 if name == 'deviceState' else 0.5 if name in ('carState', 'selfdriveState', 'driverMonitoringState', 'modelV2') else 2
      if age > max_age or not self.sm.valid[name]:
        continue
      values = self.sm[name].to_dict()
      if name == 'deviceState':
        values['usbState'] = {'connected': self.sm[name].usbState.connected}
      if name == 'modelV2' and not all(key in values for key in FIELDS[name]):
        continue
      if name == 'extrinsicsCalibration' and not all(key in values for key in ('rpyCalib', 'wideFromDeviceEuler', 'calStatus')):
        continue
      services[name] = {key: values[key] for key in fields if key in values}
    ds = services.get('deviceState', {})
    started = ds.get('started', False)
    if started and not self.was_started:
      self.started_at = now
    model_seen = self.sm.seen['modelV2'] and self.sm.logMonoTime['modelV2'] / 1e9 > self.started_at
    model_fresh = model_seen and self.sm.valid['modelV2'] and now - self.sm.logMonoTime['modelV2'] / 1e9 < 1
    self.was_started = started
    alert = services.get('selfdriveState')
    if started and alert is None:
      ss_time = self.sm.logMonoTime['selfdriveState'] / 1e9
      if ss_time < self.started_at and now - self.started_at > 10:
        alert = {'alertText1': 'openpilot Unavailable', 'alertText2': 'Waiting to start', 'alertSize': 'mid', 'alertStatus': 'normal'}
      elif ss_time >= self.started_at and now - ss_time > 5:
        take_control = self.sm['selfdriveState'].enabled and now - ss_time < 15
        alert = {'alertText1': 'TAKE CONTROL IMMEDIATELY' if take_control else 'System Unresponsive',
                 'alertText2': 'System Unresponsive' if take_control else 'Reboot Device', 'alertSize': 'full', 'alertStatus': 'critical',
                 'alertHudVisual': 'steerRequired'}
    if alert:
      alert = {key: value for key, value in alert.items() if key.startswith('alert')}
    confidence = None
    if model_fresh:
      predictions = self.sm['modelV2'].meta.disengagePredictions
      confidence = (1 - max(predictions.brakeDisengageProbs or [1])) * (1 - max(predictions.steerOverrideProbs or [1]))
    torque = None
    if self.sm.seen['carOutput'] and self.sm.valid['carOutput'] and now - self.sm.logMonoTime['carOutput'] / 1e9 < 1:
      torque = -self.sm['carOutput'].actuatorsOutput.torque
    if now - self.alerts_updated > 1:
      self.alerts = []
      if not started:
        self.alerts = [{**alert, 'text': alert['text'][:1024]} for alert in offroad_alerts(self.params)]
        self.alerts.sort(key=lambda alert: -alert['severity'])
      self.alerts_updated = now
    car = services.get('carState', {})
    if car.get('vEgo', 0) < 5:
      self.wide = True
    elif car.get('vEgo', 0) > 10:
      self.wide = False
    camera = 'wideRoad' if self.wide else 'road'
    return {'services': services, 'gpu': gpu_status(self.sm, self.params), 'confidence': confidence, 'torque': torque,
            'alert': alert, 'offroadAlerts': self.alerts if not started else [], 'camera': camera,
            'isMetric': self.params.get_bool('IsMetric'), 'alwaysOnDM': self.params.get_bool('AlwaysOnDM')}

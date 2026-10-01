"""Read-only driving UI snapshots for the live WebRTC data channel."""
from __future__ import annotations

import time

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.helpers import chestnut_compiled
from openpilot.selfdrive.selfdrived.alertmanager import OFFROAD_ALERTS

SERVICES = ['deviceState', 'carState', 'selfdriveState', 'driverMonitoringState', 'extrinsicsCalibration',
            'modelV2', 'carOutput', 'narrowRoadCameraState', 'wideRoadCameraState']
FIELDS = {
  'deviceState': ['deviceType', 'started', 'chestnutPresent', 'networkType', 'thermalStatus', 'gpuTempC', 'cpuTempC', 'freeSpacePercent'],
  'carState': ['vEgo', 'vEgoCluster', 'vCruiseCluster', 'steeringAngleDeg', 'leftBlinker', 'rightBlinker'],
  'selfdriveState': ['enabled', 'state', 'experimentalMode', 'alertText1', 'alertText2', 'alertSize', 'alertStatus', 'alertType', 'alertHudVisual'],
  'driverMonitoringState': ['activePolicy', 'isRHD', 'alertLevel', 'visionPolicyState'],
  'extrinsicsCalibration': ['rpyCalib', 'wideFromDeviceEuler', 'calStatus', 'height'],
  'modelV2': ['position', 'laneLines', 'laneLineProbs', 'roadEdges', 'roadEdgeStds'],
  'narrowRoadCameraState': ['sensor'],
  'wideRoadCameraState': ['sensor'],
}


class InCarTelemetry:
  def __init__(self):
    self.params = Params()
    self.sm = messaging.SubMaster(SERVICES)
    self.compiled = False
    self.gpu = 'disconnected'
    self.was_started = False
    self.present_at_start = False
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
      if name == 'modelV2' and not all(key in values for key in FIELDS[name]):
        continue
      if name == 'extrinsicsCalibration' and not all(key in values for key in ('rpyCalib', 'wideFromDeviceEuler', 'calStatus')):
        continue
      services[name] = {key: values[key] for key in fields if key in values}
    ds = services.get('deviceState', {})
    started = ds.get('started', False)
    detected = ds.get('chestnutPresent', False)
    self.compiled = self.compiled or chestnut_compiled()
    if started and not self.was_started:
      self.started_at = now
      self.present_at_start = detected
    model_seen = self.sm.seen['modelV2'] and self.sm.logMonoTime['modelV2'] / 1e9 > self.started_at
    model_fresh = model_seen and self.sm.valid['modelV2'] and now - self.sm.logMonoTime['modelV2'] / 1e9 < 1
    if not started:
      self.present_at_start = detected
      self.gpu = 'ready' if detected and self.compiled else 'uncompiled' if detected else 'disconnected'
    elif not self.present_at_start:
      self.gpu = 'disconnected'
    elif not self.compiled:
      self.gpu = 'uncompiled'
    elif self.gpu == 'failed' or not detected or (model_seen and (not model_fresh or not self.sm['modelV2'].big)):
      self.gpu = 'failed'
    elif self.params.get_bool('ChestnutLoading') or not model_seen:
      self.gpu = 'loading'
    else:
      self.gpu = 'failed' if self.params.get('ChestnutActive') is False else 'active'
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
        for key, definition in OFFROAD_ALERTS.items():
          value = self.params.get(key)
          if isinstance(value, dict) and isinstance(value.get('text'), str):
            self.alerts.append({'key': key, 'text': value['text'].replace('%1', str(value.get('extra', '')))[:1024],
                                'severity': definition.get('severity', 0)})
        self.alerts.sort(key=lambda alert: -alert['severity'])
      self.alerts_updated = now
    car = services.get('carState', {})
    if car.get('vEgo', 0) < 5:
      self.wide = True
    elif car.get('vEgo', 0) > 10:
      self.wide = False
    camera = 'wideRoad' if self.wide and services.get('selfdriveState', {}).get('experimentalMode') else 'road'
    return {'services': services, 'gpu': self.gpu if ds else 'unknown', 'confidence': confidence, 'torque': torque,
            'alert': alert, 'offroadAlerts': self.alerts if not started else [], 'camera': camera,
            'isMetric': self.params.get_bool('IsMetric'), 'alwaysOnDM': self.params.get_bool('AlwaysOnDM')}

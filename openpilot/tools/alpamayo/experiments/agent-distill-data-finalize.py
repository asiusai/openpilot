"""Seal a blocked three-drive metadata cohort. Exports no model inputs or teacher targets."""
import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
ARTIFACTS = WORK/'agent-distill-data-artifacts'
PROTECTED = '0000008e--321b2ddc03'


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec); sys.modules[name] = module
  spec.loader.exec_module(module); return module


def write(path, data):
  with path.open('x') as stream: stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(); output = args.output.resolve()
  assert output.parent.is_relative_to(WORK) and output.name.startswith('agent-distill-data-')
  sys.path[:0] = [str(EXP), '/tmp/opencode/openpilot-alpamayo', '/home/john/asius/openpilot/opendbc_repo']
  import numpy as np
  import av
  audit_module = load('agent_distill_data_audit', EXP/'agent-distill-data-audit.py')
  capture = load('agent_distill_data_existing_capture', EXP/'agent-live-capture.py')
  from calibration_trials import asof, vehicle_history
  from camera_profiles import replay_module
  sha = audit_module.sha
  audit_path, nav_path = ARTIFACTS/'agent-distill-data-audit.json', ARTIFACTS/'agent-distill-data-navcheck.json'
  audit, nav = json.loads(audit_path.read_text()), json.loads(nav_path.read_text())
  assert audit['helper_sha256'] == sha(EXP/'agent-distill-data-audit.py')
  assert nav['source_sha256'][str(EXP/'agent-distill-data-navcheck.py')] == sha(EXP/'agent-distill-data-navcheck.py')
  assert nav['status'] == 'native absent navigation unsupported by both tested driving paths'
  for document in (audit.get('sources', {}), nav['source_sha256']):
    for path, digest in document.items(): assert sha(path) == digest, path
  # Task-specific initData fields only. Never read other parameter values.
  initialization_navigation = []
  for route in audit['routes']:
    paths = route['inspected_qlogs']; first = min(paths, key=lambda p: int(Path(p).parent.name.rsplit('--', 1)[1]))
    events, _, status = audit_module.load_events(Path(first), 32*1024**2)
    for event in events:
      if event.which() != 'initData': continue
      row = dict(route=route['route'], source=first, source_sha256=sha(first), fields={})
      for item in event.initData.params.entries:
        if item.key not in ('NavDestination', 'NavigationEnabled'): continue
        value = bytes(item.value)
        row['fields'][str(item.key)] = dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest(),
          nonempty=bool(value.strip()))
      initialization_navigation.append(row)
      break
  records = []
  for drive in audit['selected_drives']:
    split, route, history = drive['split'], drive['route'], drive['history']
    assert route != PROTECTED and history['parse_status']['complete']
    q = history['query_ns']; rlog = Path(history['path'])
    assert sha(rlog) == history['sha256']
    first = min((r for r in drive['qlogs'] if r['initialization']), key=lambda r: int(Path(r['path']).parent.name.rsplit('--', 1)[1]))
    initialization = first['initialization'][0]; gps = first['gps_endpoints'][0]
    # A real UTC fix agrees with initData's wall clock. The session identifiers
    # are inferred clock/started boundaries because bootlogId was not recorded.
    clock_offset = initialization['wall_ns']-initialization['monotonic_ns']
    discrepancy_ns = gps['unix_ns']-(gps['monotonic_ns']+clock_offset)
    assert abs(discrepancy_ns) < 1_000_000_000
    assert first['ignition_counts'].get('True', 0) > 0 and first['moving_car_state_count'] > 0
    date = datetime.fromtimestamp(gps['unix_ns']/1e9, timezone.utc).date().isoformat()
    states = {kind: [] for kind in capture.bridge.STATE_LIMITS}
    navigation = []
    events, _, parse = audit_module.load_events(rlog, 128*1024**2)
    for event in events:
      kind = event.which()
      if kind in ('navRoute', 'navInstruction'):
        message = getattr(event, kind)
        entry = dict(kind=kind, timestamp_ns=int(event.logMonoTime), valid=bool(event.valid))
        if kind == 'navRoute': entry['coordinate_count'] = len(message.coordinates)
        else:
          entry['nonempty_instruction'] = bool(message.maneuverPrimaryText.strip() or message.maneuverSecondaryText.strip())
        navigation.append(entry)
      if kind not in states: continue
      m = getattr(event, kind)
      stamp = int(m.timestamp) if kind == 'deviceMotion' else int(event.logMonoTime)
      if not 0 < stamp <= q: continue
      valid = bool(event.valid)
      if kind == 'deviceMotion': valid &= bool(m.inputsOK and m.orientationNED.valid and m.velocityDevice.valid)
      elif kind == 'vehicleParameters': valid &= bool(m.valid)
      elif kind == 'extrinsicsCalibration': valid &= str(m.calStatus) == 'calibrated'
      value = None
      if split != 'sealed_test' and valid: value, extra_valid = capture.state_value(kind, m); valid &= extra_valid
      states[kind].append((stamp, value, valid))
    assert parse['complete']
    supplementary = []
    if not states['carParams']:
      # The live owner retains this slow service across segment boundaries.
      # Inspect one bounded preceding segment, retaining only causal carParams.
      number = int(rlog.parent.name.rsplit('--', 1)[1]); assert number > 0
      previous_rlog = rlog.parent.with_name(route+'--'+str(number-1))/'rlog.zst'
      events, size, status = audit_module.load_events(previous_rlog, 128*1024**2)
      for event in events:
        stamp = int(event.logMonoTime)
        if event.which() != 'carParams' or not q-120_000_000_000 <= stamp <= q: continue
        valid, value = bool(event.valid), None
        if split != 'sealed_test' and valid: value, valid = capture.state_value('carParams', event.carParams)
        states['carParams'].append((stamp, value, valid))
      assert status['complete']
      supplementary.append(dict(path=str(previous_rlog), sha256=sha(previous_rlog), uncompressed_bytes=size,
        fields_read='causal carParams only; no numerical values extracted for sealed test'))
    data = {}; availability = {}
    for kind, rows in states.items():
      rows.sort(key=lambda r: r[0]); times = np.asarray([r[0] for r in rows], dtype=np.int64)
      data[kind] = times, [r[1] if r[2] else None for r in rows]
      availability[kind] = times, [r[2] for r in rows]
    def required(kind, stamp):
      value, selected = asof(availability[kind], stamp, capture.bridge.STATE_LIMITS[kind])
      assert value, f'{split}: {kind} latest state invalid'
      return stamp-selected
    required('carParams', q); required('extrinsicsCalibration', q); required('deviceMotion', q)
    ages = {kind: max(required(kind, int(t)) for t in q+np.arange(-38, 1, dtype=np.int64)*50_000_000)
            for kind in ('carState', 'vehicleParameters')}
    for frame in history['frames']: required('extrinsicsCalibration', frame['capture_ns'])
    history_check = dict(selected_state_age_checks=True, max_car_state_age_ns=ages['carState'],
      max_vehicle_parameters_age_ns=ages['vehicleParameters'], all_extracted_state_timestamps_le_query=True,
      sealed_numeric_state_values_extracted=False if split == 'sealed_test' else None)
    if split != 'sealed_test':
      motion = {t: v for t, v, valid in states['deviceMotion'] if valid}
      calibration, _ = asof(data['extrinsicsCalibration'], q, 1_000_000_000)
      xyz, rot, future, _ = replay_module().Motion(motion).sample(q, calibration['rpyCalib'])
      assert future is None
      xyz, rot, detail = vehicle_history(dict(query_ns=q, ego_history_xyz=xyz, ego_history_rot=rot), data)
      assert xyz.shape == (1, 20, 3) and rot.shape == (1, 20, 3, 3)
      assert np.isfinite(xyz).all() and np.isfinite(rot).all()
      history_check.update(original_vehicle_history_executed=True,
        xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(), rot_sha256=hashlib.sha256(rot.tobytes()).hexdigest(),
        audit={k: v for k, v in detail.items() if k.startswith('max_')}, exported_pose_arrays=False)
    videos = []
    for name in ('fcamera.mp4', 'ecamera.mp4'):
      path = rlog.with_name(name)
      with av.open(str(path)) as container:
        stream = container.streams.video[0]
        videos.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
          codec=stream.codec_context.name, width=stream.width, height=stream.height,
          pixel_format=stream.codec_context.format.name, video_decoded=False))
      assert videos[-1]['width'] == 1344 and videos[-1]['height'] == 760
    records.append(dict(provisional_split=split, route=route, drive_id='f9ad9a96:'+date+':'+str(first['started_endpoints'][0]['started_mono_ns']),
      evidenced_date_utc=date, gps_minus_init_clock_ns=discrepancy_ns,
      inferred_boot_epoch_ns=clock_offset, recorded_bootlog_id=initialization['bootlog_id'],
      started_mono_ns=first['started_endpoints'][0]['started_mono_ns'],
      physical_drive_evidence='valid GPS UTC, matching initData wall/monotonic mapping, new boot epoch, ignition on and non-standstill carState',
      source_openpilot_commit=initialization['git_commit'], source_qlog_sha256={r['path']: r['sha256'] for r in drive['qlogs']},
      source_rlog=dict(path=str(rlog), sha256=sha(rlog)), source_videos=videos,
      supplementary_state_logs=supplementary, original_rlog_navigation_metadata=navigation,
      anchor_query_ns=q, history_metadata=history, selected_history_checks=history_check,
      navigation_source=None, admissible_training_sample=False,
      sealed_policy='Pixels stay undecoded; measured motion vectors, teacher activations, future trajectories, fit loss, and stopping targets remain unopened.'
        if split == 'sealed_test' else 'Causal source states only; no future values, model outputs, or optimizer updates.',
      missing=['Causal recorded nonempty navigation instruction accepted by the native driving contract.',
        'Exact original image calibration/preparation and completed input arrays remain unexecuted.']))
  assert len(records) == 3 and len({r['evidenced_date_utc'] for r in records}) == 3
  boots = sorted(r['inferred_boot_epoch_ns'] for r in records)
  assert min(b-a for a, b in zip(boots, boots[1:])) > 12*3600*1_000_000_000
  protocol = WORK/'agent-distill-feasibility-artifacts/agent-distill-feasibility-protocol-v3.json'
  assert sha(protocol) == 'fee4b66d940d5ac8a69ecb6031e34602a5260468296ffa631b8e799bc60aa0f0'
  previous = json.loads(protocol.read_text())
  # Only checkpoint/runtime source/scales metadata. No original test-route arrays or outputs.
  source_pins = dict(previous['source_sha256'])
  source_pins.update({str(p): sha(p) for p in [Path(__file__), EXP/'agent-distill-data-audit.py',
    EXP/'agent-distill-data-navcheck.py', EXP/'calibration_trials.py', EXP/'agent-live-capture.py', EXP/'agent-live-bridge.py']})
  for path, digest in source_pins.items(): assert sha(path) == digest, path
  result = dict(schema=1, status='blocked: three independent raw drives established; no admissible navigation cohort',
    admissible=False, records=records, protected_routes=[PROTECTED], protected_content_opened=False,
    fit_records=0, validation_records=0, sealed_teacher_records=0, pixel_decode_count=0,
    training_executed=False, gpu_initialized=False,
    cohort_size_policy='Three fixed raw anchors only. No claim of an available 16/4/8 split.',
    audit_evidence=dict(path=str(audit_path), sha256=sha(audit_path)),
    navigation_evidence=dict(path=str(nav_path), sha256=sha(nav_path)),
    init_navigation_parameters=initialization_navigation,
    teacher_target_contract=previous['teacher_contract'],
    teacher_group=previous['group'], teacher_protocol=dict(path=str(protocol), sha256=sha(protocol)),
    fp8_calibration_boundary='Selected language scales were previously calibrated using 32 original-route windows. This adds no new calibration, fit, or stopping on that route.',
    missing_fact='A causal explicit recorded navigation instruction at each independent anchor, or an independently specified supported no-navigation driving contract.',
    parent_gpu_capture_command=None,
    capture_blocker='No admissible complete input arrays exist. Do not run teacher capture or the trainer with this metadata cohort.',
    source_sha256=source_pins)
  assert 'torch' not in sys.modules
  write(output, result)
  print(json.dumps(dict(output=str(output), status=result['status'], dates=[r['evidenced_date_utc'] for r in records],
    init_nav_fields=[r for r in initialization_navigation if r['fields']], teacher_capture_ready=False), indent=2))


if __name__ == '__main__': main()

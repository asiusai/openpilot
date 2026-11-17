"""Decode thirty authorized native frames and verify nine experimental command inputs on CPU."""
import argparse
from bisect import bisect_right
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('agent_distill_command_common', HERE/'agent-distill-command-common.py')
c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = c; spec.loader.exec_module(c)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, default=c.ROOT/'inputs-v1')
  args = parser.parse_args(); root = args.output_dir.resolve()
  assert root.is_relative_to(c.ROOT) and not root.exists()
  c.ROOT.mkdir(parents=True, exist_ok=True)
  assert subprocess.check_output(['findmnt', '-n', '-o', 'UUID', '-T', str(c.ARCHIVE)], text=True).strip() == 'aac28ecf-db48-4b4f-8a99-bde6a3726ab9'
  assert subprocess.check_output(['findmnt', '-n', '-o', 'TARGET', '-T', str(root.parent)], text=True).strip() == '/mnt/nvme-500'
  root.mkdir(parents=True, exist_ok=False)
  plan = c.protocol(); c.write(root/'agent-distill-command-protocol.json', plan)  # Before images or numeric states are read.
  previous = json.loads(c.OLD.read_text())
  for path, digest in previous['source_sha256'].items(): assert c.sha(path) == digest, path
  torch, helper = c.cpu_imports()
  import numpy as np
  from PIL import Image
  from alpamayo2_edge.chat_template import build_conversation
  from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS
  sys.path[:0] = ['/tmp/opencode/openpilot-alpamayo', '/home/john/asius/openpilot/opendbc_repo']
  audit = c.module('command_existing_audit', HERE/'agent-distill-data-audit.py')
  camera = c.module('command_existing_camera_profiles', HERE/'camera_profiles.py')
  capture = c.module('command_existing_capture', HERE/'agent-live-capture.py')
  fast = c.module('command_existing_fast', HERE/'fast-inputs.py')
  native = c.module('command_existing_native_warp', HERE/'native_warp.py')
  from calibration_trials import calibrated_image, cached_calibrated_coordinates
  source = json.loads(c.GEOMETRY.read_text())
  # The selected nominal 120/30 projection does not use the NVIDIA target geometry.
  # Measured rpyCalib/wideFromDeviceEuler still come separately from each actual anchor.
  target = {}
  adapter = fast.FastInputs(c.BASE/'model', camera_profile='nvidia', allow_missing_cameras=True,
    direct_byte_inputs=True, cache_resize=True)
  assert adapter.tokenize.__code__ == helper._tokenize_messages.__code__
  records, frames_count, per_drive = [], 0, []
  os.environ['XDG_CACHE_HOME'] = str(root/'cpu-cache')
  for old in previous['records']:
    split, route, q = old['provisional_split'], old['route'], old['anchor_query_ns']
    assert c.PROTECTED not in json.dumps(old)
    for pin in old['source_videos']+[old['source_rlog']]+old['supplementary_state_logs']:
      assert c.sha(pin['path']) == pin['sha256'], pin['path']
    drive = root/split; drive.mkdir()
    geometry_meta = []
    for view, filename in ((1, 'ecamera.mp4'), (6, 'fcamera.mp4')):
      path = Path(old['source_rlog']['path']).with_name(filename)
      probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,pix_fmt,color_range:stream_side_data=rotation', '-of', 'json', str(path)], text=True))['streams'][0]
      rotations = [s['rotation'] for s in probe.get('side_data_list', []) if 'rotation' in s]
      assert rotations in ([], [0], [180], [-180])
      assert (probe['codec_name'], probe['width'], probe['height'], probe['pix_fmt']) == ('hevc', 1344, 760, 'yuv420p')
      geometry_meta.append(dict(view=view, video=str(path), video_pixels_rotated=False,
        video_display_rotation_degrees=abs(rotations[0]) if rotations else 0, ffprobe=probe))
    refs = [{**r, 'video_index': r['encoding']['video_index']} for r in old['history_metadata']['frames']]
    extract_source = drive/'extract-source'; extract_source.mkdir()
    c.write(extract_source/'manifest.json', dict(export='/', camera_manifest=dict(cameras=geometry_meta), schedule=[dict(tick=0, frames=refs)]))
    raw_root = drive/'native-rgb'
    camera.extract(SimpleNamespace(source=extract_source, output=raw_root, ticks=None, recording_uuid=None))
    states = {k: [] for k in capture.bridge.STATE_LIMITS}
    logs = [Path(pin['path']) for pin in old['supplementary_state_logs']]+[Path(old['source_rlog']['path'])]
    for log_path in logs:
      events, _, status = audit.load_events(log_path, 128*1024**2)
      for event in events:
        kind = event.which()
        if kind not in states: continue
        message = getattr(event, kind)
        stamp = int(message.timestamp) if kind == 'deviceMotion' else int(event.logMonoTime)
        if not 0 < stamp <= q: continue
        value, extra = capture.state_value(kind, message) if event.valid else (None, False)
        states[kind].append(capture.bridge.State(stamp, int(event.logMonoTime), value, bool(event.valid and extra)))
      assert status['complete']
    for rows in states.values():
      rows.sort(key=lambda r: r.stamp)
      # Overlapping segment copies must agree; retain the canonical last copy.
      by_stamp = {}
      for row in rows:
        if row.stamp in by_stamp: assert row == by_stamp[row.stamp]
        by_stamp[row.stamp] = row
      rows[:] = by_stamp.values()
    frames = []
    for ref in refs:
      image = np.load(raw_root/str(ref['view'])/f'{ref["video_index"]}.npy', allow_pickle=False)
      assert image.dtype == np.uint8 and image.shape == (3, 760, 1344) and image.flags.c_contiguous
      image.flags.writeable = False
      received = max(ref['eof_ns'], ref['published_ns'])
      frames.append(capture.bridge.Frame(ref['view'], ref['frame_id'], ref['capture_ns'], ref['eof_ns'], received, received, image))
    selected_ns = max([r.received for values in states.values() for r in values]+[f.ready_ns for f in frames])
    selection = capture.bridge.Selection(q, selected_ns, tuple(frames), states, 0, None)
    # Run both actual implementations, then require byte identity for every image.
    reference = capture.bridge.Preparer(source, target, warp=calibrated_image, cache_limit=0)
    accelerated = capture.bridge.Preparer(source, target, warp=native.calibrated_image, cache_limit=0)
    arrays, detail = reference.prepare(selection); actual, actual_detail = accelerated.prepare(selection)
    assert set(arrays) == set(actual) == {'image_frames', 'camera_indices', 'absolute_timestamps', 'ego_history_xyz', 'ego_history_rot'}
    for name in arrays: np.testing.assert_array_equal(arrays[name], actual[name], err_msg='Native/reference preparation differs: '+name)
    assert arrays['image_frames'].shape == (10, 1, 3, 432, 768) and detail['camera_profile'] == 'calibrated_vehicle_history'
    assert arrays['camera_indices'].tolist() == [1]*5+[6]*5
    shared = drive/'shared-arrays'; shared.mkdir()
    for name, value in arrays.items(): np.save(shared/(name+'.npy'), value, allow_pickle=False)
    source_inputs = []
    for ref, f in zip(refs, frames, strict=True):
      cal_times = np.array([r.stamp for r in states['extrinsicsCalibration']], np.int64)
      i = bisect_right(cal_times, f.capture_ns)-1; assert i >= 0
      cal = states['extrinsicsCalibration'][i]
      assert cal.valid and f.capture_ns-cal.stamp <= 1_000_000_000
      source_inputs.append(dict(**ref, native_rgb_sha256=hashlib.sha256(f.pixels.tobytes()).hexdigest(),
        measured_calibration=dict(timestamp_ns=cal.stamp, value=cal.value)))
    per_drive.append(dict(split=split, route=route, query_ns=q, drive_id=old['drive_id'],
      native_sources=old['source_videos'], state_sources=[dict(path=str(p), sha256=c.sha(p)) for p in logs],
      frames=source_inputs, prepared_shared_sha256={p.name: c.sha(p) for p in shared.glob('*.npy')},
      preparation_audit={k: detail[k] for k in ('vehicle_history_audit', 'motion_extrapolation_ms', 'camera_profile', 'history_policy')},
      native_reference_image_bits_equal=True, sensor_input_timestamps_max_ns=max(r.stamp for values in states.values() for r in values)))
    for index, command in enumerate(c.COMMANDS):
      sample = drive/f'command-{index}'; sample.mkdir()
      for p in shared.glob('*.npy'): os.link(p, sample/p.name)
      np.save(sample/'nav_text.npy', np.array(command), allow_pickle=False)
      data = {name: torch.from_numpy(value.copy()) for name, value in arrays.items()}; data['nav_text'] = [command]
      fast.validate_available_sample(data)
      _, _, prepared = adapter.prepare(data)
      official_messages = build_conversation(dict(data, camera_pixel_bounds=CAMERA_PIXEL_BOUNDS))
      official = helper._tokenize_messages(official_messages, adapter.processor, include_closed_thinking=True)
      for name, value in official.items():
        if name == 'pixel_values':
          codes = prepared['tokenized_data'][name]
          expected = helper._normalize_patches(codes.clone(), adapter.processor.image_processor)
          assert torch.equal(value, expected), 'Official normalization differs from selected byte palette'
        else: assert torch.equal(value, prepared['tokenized_data'][name]), 'Official/selected token or grid mismatch: '+name
      packed = adapter.native_inputs(prepared)
      assert 1800 < packed[0].shape[1] <= 2042
      assert prepared['tokenized_data']['image_grid_thw'].tolist() == [[1, 20, 36]]*10
      assert prepared['tokenized_data']['pixel_values'].shape == (7200, 768)
      assert packed[1].shape == (3, 1, packed[0].shape[1])
      # The original public full driving validator still requires 22 images.
      try: helper.prepare_model_inputs(data, adapter.config, adapter.tokenizer)
      except ValueError as error: assert '22' in str(error)
      else: raise AssertionError('Official full-camera API contract changed')
      records.append(dict(id=f'{split}-command-{index}', split=split, route_id=route, drive_id=old['drive_id'],
        query_ns=q, sample=str(sample), command_index=index, conditioning=dict(plan['command_provenance'], text=command),
        input_file_sha256={p.name: c.sha(p) for p in sample.glob('*.npy')},
        actual_prompt_tokens=packed[0].shape[1], prepared=c.fingerprint(prepared), packed=c.fingerprint(packed),
        official_selected_tokens_and_image_normalization_exact=True,
        official_full_camera_api='Rejects ten images as expected; actual official template/tokenizer/processor tested without fabricating surround images.',
        complete_input=True, source_drive_record_index=len(per_drive)-1))
      print(f'Prepared {split} command {index}: {packed[0].shape[1]} actual prompt tokens', flush=True)
    frames_count += 10
  assert frames_count == 30 and len(records) == 9 and {r['command_index'] for r in records} == {0, 1, 2}
  c.assert_no_devices()
  sources = [Path(__file__), HERE/'agent-distill-command-common.py', c.GEOMETRY, HERE/'camera_profiles.py',
    HERE/'agent-live-bridge.py', HERE/'agent-live-capture.py', HERE/'calibration_trials.py', HERE/'native_warp.py',
    HERE/'native-warp.c', HERE/'fast-inputs.py', HERE.parent/'inputs.py',
    c.BASE/'upstream/src/alpamayo2_edge/helper.py', c.BASE/'upstream/src/alpamayo2_edge/chat_template.py',
    c.BASE/'model/config.json', c.BASE/'model/model.safetensors.index.json']
  manifest = dict(schema=1, protocol=plan, admissible_for_internal_layer_capture=True,
    admissible_for_driving_quality_validation=False, gpu_executed=False, training_executed=False,
    completed_native_frames=30, completed_conditioned_inputs=9, drives=per_drive, samples=records,
    source_sha256={str(p): c.sha(p) for p in sources},
    geometry_limit='Exact selected nominal source optics and 120/30 virtual projection; measured recorded extrinsics per frame. No claim of newly measured lens distortion.',
    original_data_cohort_unchanged=c.sha(c.OLD) == c.OLD_SHA,
    sealed_targets='No teacher targets yet. Future target files must use a separate sealed_test directory and remain outside fitting/early-stop manifests.')
  c.write(root/'agent-distill-command-inputs.json', manifest)
  print(json.dumps(dict(output=str(root/'agent-distill-command-inputs.json'), inputs=9, native_frames=30, gpu_executed=False), indent=2))


if __name__ == '__main__': main()

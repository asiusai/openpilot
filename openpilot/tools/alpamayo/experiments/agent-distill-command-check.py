"""CPU provenance/conditioning checks and independent capture-boundary control mocks."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('agent_distill_command_common_check', HERE/'agent-distill-command-common.py')
c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = c; spec.loader.exec_module(c)


def mock_checks():
  import numpy as np
  capture = c.module('agent_distill_command_capture_mock', HERE/'agent-distill-command-capture.py')
  class Hidden:
    shape = (1, 2048, 2048)
    def __init__(self, label): self.label = label
  class Model:
    def __init__(self, fail=None):
      self.calls, self.fail = [], fail
      self.cache = [None]*28
      self.cache[12] = np.stack([np.full((1,8,2304,128), 12, np.uint16), np.full((1,8,2304,128), 31, np.uint16)])
      self.non_target = object()
    def text_layer(self, x, cos, sin, prefix, cache=None, mask=None, cache_pos=None):
      i = int(prefix.rsplit('.',1)[1]); self.calls.append((i,x.label,cache_pos))
      if i == self.fail: raise RuntimeError('independent original-layer failure')
      return Hidden('h'+str(i+1))
    def decode(self, *args): return 'original decode'
    def greedy_decode(self, *args): return 'original greedy'
    def trajectory(self, *args): return 'original trajectory'
  seen = []
  def bits(value):
    if isinstance(value, Hidden): seen.append(value.label); return value.label
    seen.append((value.shape,int(value.reshape(-1)[0]))); return value.copy()
  cos = np.full((1,1,2048,128), 7, np.uint16); sin = np.full_like(cos, 9)
  model = Model(); originals = {k:getattr(model,k) for k in ('text_layer','decode','greedy_decode','trajectory')}
  non_target = model.non_target
  with capture.capture_boundary(model,bits) as state:
    for name in ('decode','greedy_decode','trajectory'):
      try: getattr(model,name)()
      except AssertionError: pass
      else: raise AssertionError('Forbidden driving output was not blocked')
    x = Hidden('h0')
    try:
      for i in range(28): x = model.text_layer(x,cos,sin,'vlm.model.language_model.layers.'+str(i),cache_pos=0)
    except capture.Captured: pass
    else: raise AssertionError('Expected capture stop')
  assert [i for i,_,_ in model.calls] == list(range(16)) and state['calls'] == 16
  assert state['payload']['x'] == 'h12' and state['payload']['y'] == 'h16'
  assert np.all(state['payload']['k'] == 12) and np.all(state['payload']['v'] == 31)
  assert state['payload']['k'].shape == state['payload']['v'].shape == (1,8,2048,128)
  assert state['payload']['cos'].shape == state['payload']['sin'].shape == (1,2048,128)
  assert np.all(state['payload']['cos'] == 7) and np.all(state['payload']['sin'] == 9)
  assert all(getattr(model,k) == v for k,v in originals.items()) and model.non_target is non_target
  # An unrelated original-layer exception must also restore every hook.
  broken = Model(fail=3); before = {k:getattr(broken,k) for k in originals}
  try:
    with capture.capture_boundary(broken,bits):
      x = Hidden('h0')
      for i in range(16): x = broken.text_layer(x,cos,sin,'vlm.model.language_model.layers.'+str(i),cache_pos=0)
  except RuntimeError as error: assert str(error) == 'independent original-layer failure'
  else: raise AssertionError('Original failure disappeared')
  assert all(getattr(broken,k) == v for k,v in before.items())
  assert [i for i,_,_ in broken.calls] == list(range(4))
  return dict(original_layer_calls_unchanged=True, stopped_before_layer16=True, two_cache_outputs_and_rope_associated=True,
    no_model_arithmetic_replacement=True, non_target_operation_identity_preserved=True,
    forbidden_control_outputs_rejected=True, success_and_failure_hooks_restored=True,
    boundary='Independent CPU orchestration mocks; no teacher/model inference or GPU operation executed.')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(); output = args.output.resolve()
  assert output.is_relative_to(c.ROOT) and not output.exists()
  import numpy as np
  manifest_path = c.ROOT/'inputs-v1/agent-distill-command-inputs.json'
  manifest = json.loads(manifest_path.read_text())
  assert c.sha(c.OLD) == c.OLD_SHA and manifest['protocol'] == c.protocol()
  assert manifest['completed_native_frames'] == 30 and len(manifest['samples']) == 9
  assert not manifest['admissible_for_driving_quality_validation'] and not manifest['gpu_executed'] and not manifest['training_executed']
  drive_splits, pixel_digests = {}, {}
  for row in manifest['samples']:
    assert c.PROTECTED not in json.dumps(row)
    drive_splits.setdefault(row['drive_id'],set()).add(row['split'])
    assert row['conditioning']['basis'] == 'experimental command' and not row['conditioning']['recorded_driver_intent']
    assert row['conditioning']['recorded_available_ns'] is None and row['conditioning']['recorded_navigation_source'] is None
    assert row['conditioning']['text'] == c.COMMANDS[row['command_index']]
    for name, digest in row['input_file_sha256'].items(): assert c.sha(Path(row['sample'])/name) == digest
    pixel_digests.setdefault(row['drive_id'],set()).add(row['input_file_sha256']['image_frames.npy'])
    times = np.load(Path(row['sample'])/'absolute_timestamps.npy', allow_pickle=False)
    assert times.shape == (10,) and np.all(times <= row['query_ns']//1000)
    assert all(np.all(np.diff(times[i:i+5]) > 0) for i in (0,5))
    xyz = np.load(Path(row['sample'])/'ego_history_xyz.npy', allow_pickle=False)
    rot = np.load(Path(row['sample'])/'ego_history_rot.npy', allow_pickle=False)
    assert xyz.shape == (1,20,3) and rot.shape == (1,20,3,3) and np.isfinite(xyz).all() and np.isfinite(rot).all()
    assert row['official_selected_tokens_and_image_normalization_exact'] and row['complete_input']
  assert len(drive_splits) == 3 and all(len(v) == 1 for v in drive_splits.values())
  assert all(len(v) == 1 for v in pixel_digests.values())
  for split in ('train','validation','sealed_test'):
    rows = [r for r in manifest['samples'] if r['split'] == split]
    assert len(rows) == 3 and sorted(r['command_index'] for r in rows) == [0,1,2]
    assert len({r['packed'][0]['sha256'] for r in rows}) == 3
    for key in ('image_frames.npy','camera_indices.npy','absolute_timestamps.npy','ego_history_xyz.npy','ego_history_rot.npy'):
      assert len({r['input_file_sha256'][key] for r in rows}) == 1
  for drive in manifest['drives']:
    assert drive['native_reference_image_bits_equal'] and drive['sensor_input_timestamps_max_ns'] <= drive['query_ns']
    for frame in drive['frames']:
      assert frame['capture_ns'] <= frame['requested_ns'] <= drive['query_ns']
      assert 0 <= frame['capture_ns']-frame['measured_calibration']['timestamp_ns'] <= 1_000_000_000
  result = dict(schema=1, passed=True, input_manifest_sha256=c.sha(manifest_path), independent_capture_mocks=mock_checks(),
    three_drives_disjoint=True, nine_predeclared_commands_not_outcome_selected=True,
    command_only_input_differences_within_each_drive=True, official_selected_preprocessing_exact=True,
    causal_sensor_and_calibration_checks=True, old_cohort_unchanged=True,
    gpu_executed=False, teacher_inference_executed=False, training_executed=False)
  c.assert_no_devices(); c.write(output,result)
  print(json.dumps(result,indent=2))


if __name__ == '__main__': main()

"""Actual canonical launcher/padding controls. Stops before initialize; no model inference."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('agent_distill_command_common_launchcheck', HERE/'agent-distill-command-common.py')
c = importlib.util.module_from_spec(spec); sys.modules[spec.name] = c; spec.loader.exec_module(c)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(); output = args.output.resolve()
  assert output.is_relative_to(c.ROOT) and not output.exists()
  capture = c.module('agent_distill_command_capture_launchcheck', HERE/'agent-distill-command-capture.py')
  plan, manifest = capture.preflight(capture.DEFAULT)
  ready = capture.prepared_inputs(plan, manifest)
  import numpy as np
  os.environ['DEV'] = 'CPU'; sys.path.insert(0, str(c.BASE/'tinygrad'))
  root = Path(plan['runtime_root']); exps = root/'experiments'
  # The real launcher applies its saved environment, so make only the diagnostic
  # device selection CPU before running that unchanged launcher. No initialize is called.
  recorded = json.loads(Path(plan['invocation_path']).read_text())
  recorded['environment'] = dict(recorded['environment'], DEV='CPU')
  invocation = output.with_suffix('.agent-distill-command-cpu-invocation.json')
  c.write(invocation, recorded)
  old_spec, old_run, old_argv, old_env = importlib.util.spec_from_file_location, runpy.run_path, sys.argv[:], dict(os.environ)
  observations = []
  class StopBeforeInitialize(Exception): pass
  def pipeline(path, run_name='__main__'):
    assert Path(path).resolve() == exps/'benchmark-pipeline.py'
    module = c.module('agent_distill_command_actual_pipeline_parse', path)
    parsed = module.parse_args(sys.argv[1:])
    assert parsed.samples == [r['sample'] for r in manifest['samples']]
    assert parsed.steps == 3 and parsed.pad_prompt_to == 2048 and parsed.vision_calibration is None
    assert parsed.native_manifest is parsed.native_calibrations is parsed.native_frame_cache is None
    assert parsed.warmup == 0 and parsed.runs == 9 and parsed.max_tokens == 64
    assert parsed.pixel_scale == parsed.history_pixel_scale == parsed.history_feature_scale == 1
    assert parsed.camera_profile == 'nvidia' and parsed.allow_missing_cameras
    padded = module.local_module('padded-prefill')
    online = module.local_module('online-attention')
    lookup = module.local_module('lookup-decode')
    assert Path(padded.pad_inputs.__code__.co_filename).resolve() == exps/'agent-speculation-run.py'
    assert online.install.__name__ == 'install_attention'
    assert lookup.LookupDecoder.__name__ == 'Decoder'
    from tinygrad import Device
    assert not Device._opened_devices
    for row, prepared, packed in ready:
      actual = row['actual_prompt_tokens']; p, q = padded.pad_inputs(prepared, packed, 2048)
      assert p['actual_prompt_length'] == actual and p['prefill_draft_eligible']
      np.testing.assert_array_equal(q[0][:,:actual], packed[0])
      np.testing.assert_array_equal(q[1][:,:,:actual], packed[1])
      np.testing.assert_array_equal(q[0][0,actual:actual+6], [124549,11135,3174,4286,1561,135084])
      expected = np.broadcast_to(actual+int(packed[2])+np.arange(6), (3,1,6))
      np.testing.assert_array_equal(q[1][:,:,actual:actual+6], expected)
      assert not q[0][:,actual+6:].any() and not q[1][:,:,actual+6:].any()
      assert q[2] == int(packed[2])-(2048-actual)
      np.testing.assert_array_equal(q[3], packed[3])
      observations.append(dict(id=row['id'], actual_prompt_tokens=actual, unchanged_real_prefix=True,
        original_nominal_draft_and_mrope=True, end_padding_exact=True))
    raise StopBeforeInitialize()
  stopped = False
  try:
    runpy.run_path = pipeline
    sys.argv = [str(exps/'agent-speculation-run.py'), '--output', str(output), '--invocation', str(invocation),
      '--mode', 'candidate', '--prefill-attention', 'shuffle-fma', '--pad-prompt-to', '2048', '--warmup', '0', '--runs', '9']
    namespace = dict(__file__=str(exps/'agent-speculation-run.py'), __name__='__main__')
    try: exec(compile((exps/'agent-speculation-run.py').read_text(), namespace['__file__'], 'exec'), namespace)
    except StopBeforeInitialize: stopped = True
    else: raise AssertionError('Expected stop before pipeline initialize')
  finally:
    importlib.util.spec_from_file_location, runpy.run_path, sys.argv = old_spec, old_run, old_argv
    os.environ.clear(); os.environ.update(old_env)
  assert stopped and len(observations) == 9
  c.assert_no_devices()
  c.write(output, dict(schema=1, passed=True, actual_canonical_launcher=True, actual_pipeline_parser=True,
    original_import_hooks_installed=True, original_selected_padding_on_all_real_inputs=observations,
    initialized_model=False, teacher_inference_executed=False, gpu_executed=False, training_executed=False,
    hooks_restored=importlib.util.spec_from_file_location is old_spec and runpy.run_path is old_run,
    capture_plan_sha256=c.sha(capture.DEFAULT), helper_sha256=c.sha(Path(__file__)),
    sole_launcher_environment_difference='DEV=CPU for import-only diagnostic; no initialize or kernel execution.'))
  print(json.dumps(dict(output=str(output), passed=True, real_inputs=9, stopped_before_initialize=True), indent=2))


if __name__ == '__main__': main()

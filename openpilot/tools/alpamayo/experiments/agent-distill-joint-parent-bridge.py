"""Run the unchanged teacher capture under the normal ACL and GPU-lock wrapper."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
PLAN = WORK/'agent-distill-command-artifacts/capture-v1/agent-distill-command-capture-plan.json'
PINS = {
  EXP/'agent-distill-joint-parent.py': '9782636463911ab41d142c9c473d4a5f0422dc1000fb5e6e7ea331e0ec1977ef',
  EXP/'agent-distill-joint-runtime.py': '4f098eb5ddda5ae4c5cfe0b9e01956f5dc05177aa4994fb9f8c83b86dcec939e',
  PLAN: '5cb4a9baa5c448f507b64b250686d8cae16e43405a7d990b2c48c5212bf9871c',
}


def sha(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--verify-only', action='store_true')
  options = parser.parse_args()
  if any(sha(path) != digest for path, digest in PINS.items()):
    raise ValueError('Changed reviewed joint capture source')
  root = options.output.resolve().parent
  if options.output.name != 'report.json' or not root.is_relative_to(WORK) or not root.is_dir():
    raise ValueError('Use a fresh normal wrapper output directory on the 500 GB workspace')
  if options.output.exists() or any(root.glob('agent-distill-joint-*')):
    raise ValueError('Output already contains joint capture artifacts')
  if shutil.disk_usage(WORK).free < 4 * 1024**3:
    raise ValueError('Insufficient workspace reserve')
  spec = importlib.util.spec_from_file_location('joint_capture_parent_bridge', EXP/'agent-distill-joint-parent.py')
  parent = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(parent)
  plan = json.loads(PLAN.read_text())
  seed = plan['compiler_cache_seed']
  if sha(seed['path']) != seed['sha256']:
    raise ValueError('Compiler seed changed')
  cache = root/'agent-distill-joint-kernels.db'
  shutil.copyfile(seed['path'], cache)
  for key, value in plan['environment'].items():
    if value is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = value
  temporary = root/'agent-distill-joint-tmp'
  temporary.mkdir()
  os.environ.update(CACHEDB=str(cache), TMPDIR=str(temporary), PYTHONDONTWRITEBYTECODE='1',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                    CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
  # Preserve the selected invocation, changing only owned cache/temp locations.
  invocation = json.loads(Path(plan['invocation_path']).read_text())
  invocation['environment'].update(CACHEDB=str(cache), TMPDIR=str(temporary))
  local_invocation = root/'agent-distill-joint-invocation.json'
  local_invocation.write_text(json.dumps(invocation, indent=2)+'\n')
  plan['invocation_path'] = str(local_invocation)
  runtime = parent.module('joint_capture_runtime_bridge', EXP/'agent-distill-joint-runtime.py')
  rows = parent.selected_rows(parent.INPUTS, ['train-command-0'])
  ready = parent.prepare(plan, rows)
  sys.path.insert(0, str(BASE/'tinygrad'))
  args = SimpleNamespace(output_dir=root, execute_gpu=not options.verify_only, mode='capture',
                         language_checkpoint=parent.PILOT)
  try:
    result = parent.run(args, plan, ready, runtime)
    print(json.dumps({'complete': result['complete'], 'gpu_executed': result['gpu_executed'],
                      'output': str(options.output), 'boundary': 'unchanged teacher capture; no performance claim'}))
  finally:
    if any(sha(path) != digest for path, digest in PINS.items()):
      raise RuntimeError('Reviewed joint capture source changed during execution')


if __name__ == '__main__':
  main()

"""Explicit experimenter conditioning and immutable provenance helpers; no device imports."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
ARCHIVE = Path('/mnt/nvme-256/footage/f9ad9a96')
ROOT = WORK/'agent-distill-command-artifacts'
OLD = WORK/'agent-distill-data-artifacts/agent-distill-data-cohort.json'
OLD_SHA = '8039ab8d277c140d291507f53f6e4d7172032e492adc23ac4d76d4f4e5f3e3b6'
PROTECTED = '0000008e--321b2ddc03'
COMMANDS = ('Follow the road.', 'Turn left at the next intersection.', 'Turn right at the next intersection.')
GEOMETRY = Path('/mnt/nvme-500/tmp/alpamayo-camera-crops-20261116/source-camera-geometry.json')
CACHE = WORK/'agent-expert-combined-cache-artifacts/fp8-cache'


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
  with Path(path).open('x') as stream: stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def module(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  value = importlib.util.module_from_spec(spec); sys.modules[name] = value
  spec.loader.exec_module(value); return value


def fingerprint(value):
  import numpy as np
  if 'torch' in sys.modules:
    import torch
    if isinstance(value, torch.Tensor):
      assert value.device.type == 'cpu'
      return dict(shape=list(value.shape), dtype=str(value.dtype),
        sha256=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
  if isinstance(value, np.ndarray):
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())
  if isinstance(value, dict): return {k: fingerprint(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)): return [fingerprint(v) for v in value]
  if isinstance(value, np.generic): return value.item()
  assert value is None or isinstance(value, (str, int, float, bool)), type(value)
  return value


def cpu_imports():
  os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
    CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='')
  sys.path.insert(0, str(BASE/'upstream/src'))
  import torch
  torch.set_num_threads(4); torch.set_default_device('cpu')
  # Load Transformers before the local kernels.py can shadow its optional Hub package.
  from alpamayo2_edge import helper
  from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgeModel  # noqa: F401
  sys.path[:0] = [str(EXP), str(EXP.parent)]
  return torch, helper


def assert_no_devices():
  if 'torch' in sys.modules: assert not sys.modules['torch'].cuda.is_initialized()
  for path in Path('/proc/self/fd').iterdir():
    try: target = os.readlink(path)
    except OSError: continue
    assert not target.startswith(('/dev/dri', '/dev/kfd', '/dev/nvidia', '/dev/bus/usb')), target


def protocol():
  assert sha(OLD) == OLD_SHA
  previous = json.loads(OLD.read_text())
  return dict(schema=1, protocol='experimenter-command-layer-reconstruction-v1',
    scope='Three independent raw-drive anchors times three predeclared commands; internal-layer reconstruction plumbing only.',
    supersedes_only='The experiment-specific recorded-navigation requirement for this new counterfactually conditioned cohort.',
    old_cohort=dict(path=str(OLD), sha256=OLD_SHA, remains_admissible=False),
    commands=list(COMMANDS), command_provenance=dict(basis='experimental command', issuer='experimenter/user',
      selection='All three literal commands applied to all three preselected anchors; no content/outcome/loss-based selection.',
      recorded_available_ns=None, recorded_navigation_source=None, recorded_driver_intent=False,
      future_motion_inference=False, route_availability_or_safety_claim=False),
    protected_routes=[PROTECTED], drive_split=[dict(drive_id=r['drive_id'], route=r['route'], split=r['provisional_split'],
      query_ns=r['anchor_query_ns']) for r in previous['records']],
    teacher_contract=previous['teacher_target_contract'], teacher_group=[12, 13, 14, 15],
    targets='Selected frozen FP8/BF16 h entering layer12, h leaving layer15, layer12 post-RoPE K/V and original BF16 cos/sin.',
    target_exclusions=['No trajectory, reasoning output, control output, future CAN label, or recorded driver-intent target.'],
    sealed_test='Input and teacher-target preparation allowed. Targets remain separate from fitting, validation loss, and early stopping.',
    qualification='Nine correlated conditioned examples do not establish driving/navigation accuracy, model recoverability, or pipeline latency.')

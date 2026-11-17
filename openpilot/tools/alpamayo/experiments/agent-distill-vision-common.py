"""Small source pins, original BF16 seed, and ordered route71/73 target admission."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-distill-vision-artifacts'
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
TINY = BASE/'tinygrad'
PY = BASE/'upstream/.venv/bin/python'
REV = '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
NATIVE = WORK/'agent-distill-command-artifacts/capture-v1/runtime/model.py'
TRAIN = WORK/'agent-distill-joint-teacher-v1/report.json'
TRAIN_SHA = '5b796b1490bf234cf874a007be575612cb03599d61ba4cde041302e606b4052c'
VALIDATION = WORK/'agent-distill-joint-validation-v1/report.json'
VALIDATION_SHA = 'fdfc6a47355e9f164bc63df1ec240062931be43a0e05a56794593e6ade234d7c'
FROZEN = WORK/'agent-distill-fit-artifacts/live-v4/frozen'
STATE = FROZEN/'agent-distill-fit-state.py'
LEDGER = FROZEN/'agent-distill-fit-ledger.py'
KFD = EXP/'agent-distill-kfd1-common.py'
TARGET = 'KFD:1+AMD:LLVM:gfx1100'
LIMITS = dict(target=TARGET, selected_gpu_id=27211, selected_pci='0000:0c:00.0',
              live_allocation_limit=12*1024**3, cumulative_allocation_limit=256*1024**3,
              max_single_buffer=256*1024**2, free_reserve=5*1024**3)
HYPER = dict(updates=25, lr=3.5e-6, b1=.9, b2=.999, eps=1e-8, weight_decay=.01,
             loss='FP32 mean squared error over all 720x1152 outputs', microbatch_images=1,
             selection_steps=[0, 25], order='image_index = (step - 1) % 10')


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
  with Path(path).open('x') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def module(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
  return value


def no_devices():
  for path in Path('/proc/self/fd').iterdir():
    try: target = str(path.readlink())
    except FileNotFoundError: continue
    if target.startswith(('/dev/dri/', '/dev/kfd', '/dev/nvidia', '/dev/bus/usb/')): raise RuntimeError(target)


def storage(path):
  path = Path(path).resolve()
  if not path.is_relative_to(ROOT): raise ValueError('Keep new vision artifacts in their 500GB directory')
  while not path.exists(): path = path.parent
  assert subprocess.check_output(['findmnt', '-n', '-o', 'TARGET', '-T', str(path)], text=True, timeout=5).strip() == '/mnt/nvme-500'
  fs = os.statvfs(path)
  assert fs.f_bavail*fs.f_frsize >= 5*1024**3


def graph():
  sys.path.insert(0, str(TINY))
  return module('agent_distill_vision_block', EXP/'agent-distill-vision-block.py')


def seeds():
  import numpy as np
  B = graph(); index = read(BASE/'model/model.safetensors.index.json')['weight_map']; arrays, entries = {}, {}
  for name, shape in B.shapes().items():
    full = B.PREFIX+name; path = BASE/'model'/index[full]
    with path.open('rb') as stream:
      size = struct.unpack('<Q', stream.read(8))[0]; assert 0 < size < 16*1024**2
      header = json.loads(stream.read(size)); entry = header[full]
      assert entry['dtype'] == 'BF16' and tuple(entry['shape']) == shape
      begin, end = entry['data_offsets']; assert end-begin == 2*int(np.prod(shape))
      stream.seek(8+size+begin); raw = stream.read(end-begin); assert len(raw) == end-begin
    arrays[name] = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape)
    entries[name] = dict(shape=list(shape), bf16_sha256=hashlib.sha256(raw).hexdigest())
  return arrays, entries


def capture(path, digest, split):
  """Only a completed, caller-hash-bound route71/73 capture can open numerical targets."""
  import numpy as np
  expected_path = {'train': TRAIN, 'validation': VALIDATION}[split]
  if Path(path).resolve() != expected_path or sha(path) != digest: raise ValueError('Capture path/hash mismatch')
  if digest != {'train': TRAIN_SHA, 'validation': VALIDATION_SHA}[split]: raise ValueError('Changed fixed capture')
  report = read(path)
  if not report['complete'] or not report['gpu_executed'] or report['mode'] != 'capture': raise ValueError('Incomplete teacher capture')
  if len(report['rows']) != 1: raise ValueError('Pilot expects exactly one anchor per split')
  row = report['rows'][0]
  route = {'train': '00000071--858232e52d', 'validation': '00000073--41474e402a'}[split]
  if (row['id'], row['split'], row['route_id']) != (split+'-command-0', split, route): raise ValueError('Disallowed anchor or route')
  targets = row['targets']; keys = row['vision_input_keys']
  if len(keys) != 10 or len({k['pixel_sha256'] for k in keys}) != 10: raise ValueError('Expected ten unique ordered image keys')
  if any(k['grid'] != [1, 20, 36] for k in keys): raise ValueError('Changed patch grids')
  if len(targets['vision_batches']) != 5: raise ValueError('Expected five captured batches')
  result = []
  for batch, metadata in enumerate(targets['vision_batches']):
    if metadata != dict(images=2, patches=720, layer_order=list(range(27))): raise ValueError('Incomplete native vision execution')
    pair = {}
    for field in ('x', 'y'):
      name = f'vision{batch}.group12.{field}'; info = targets['tensors'][name]
      filename = Path(path).parent/row['id']/(name+'.npy')
      if Path(info['path']).resolve() != filename or info['shape'] != [2, 720, 1152]: raise ValueError('Target path/shape mismatch')
      if info['dtype'] != 'uint16' or info['bf16_bits'] is not True or sha(filename) != info['sha256']: raise ValueError('Target encoding/hash mismatch')
      bits = np.load(filename, allow_pickle=False)
      if bits.shape != (2, 720, 1152) or bits.dtype != np.uint16: raise ValueError('Wrong target array')
      pair[field] = (bits.astype(np.uint32) << 16).view(np.float32)
      if not np.isfinite(pair[field]).all(): raise ValueError('Nonfinite teacher target')
    for member in range(2):
      index = batch*2+member
      result.append(dict(id=f'{row["id"]}/vision{batch}/image{member}', image_index=index, batch=batch, member=member,
                         anchor=row['id'], route_id=route, pixel_key=keys[index],
                         x=pair['x'][member:member+1].copy(), y=pair['y'][member:member+1].copy()))
  return result


def pins():
  files = [*sorted(EXP.glob('agent-distill-vision-*.py')), NATIVE, NATIVE.with_name('runtime.py'),
           NATIVE.parent/'experiments/vision-cache.py', NATIVE.parent/'experiments/batched-vision.py',
           STATE, LEDGER, KFD, BASE/'model/config.json', BASE/'model/model.safetensors.index.json']
  files += [TINY/name for name in ('tinygrad/tensor.py', 'tinygrad/nn/optim.py', 'tinygrad/mixin/gradient.py', 'tinygrad/mixin/elementwise.py',
                                  'tinygrad/device.py', 'tinygrad/helpers.py', 'tinygrad/runtime/ops_amd.py',
                                  'tinygrad/runtime/support/amd.py', 'tinygrad/runtime/support/hcq2.py')]
  assert subprocess.check_output(['git', '-C', str(TINY), 'rev-parse', 'HEAD'], text=True).strip() == REV
  return {str(path): sha(path) for path in files}


def verify_plan(path, digest):
  if Path(path).resolve().parent != ROOT or sha(path) != digest: raise ValueError('Plan path/hash mismatch')
  p = read(path)
  assert p['limits'] == LIMITS and p['hyperparameters'] == HYPER and p['tinygrad_revision'] == REV
  for name, expected in p['sources'].items():
    if sha(name) != expected: raise ValueError('Changed source '+name)
  arrays, entries = seeds(); assert entries == p['seed']; del arrays
  return p

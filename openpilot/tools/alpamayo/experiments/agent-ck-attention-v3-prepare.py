"""CPU-only baseline freeze, CK adapter audit, and source-bound parent manifest."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

EXP = Path(__file__).parent
OLD = EXP/'agent-ck-attention-v2'
ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-attention-v3')
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
ROOT.mkdir(exist_ok=True)
os.environ.update(DEV='CPU', CPU='1', AMD='0', ALLOW_DEVICE_USAGE='0', TMPDIR=str(ROOT), CACHEDB=str(ROOT/'compiler-cache.db'))
sys.path.insert(0, str(TINY))


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), EXP/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  target = ROOT/'prepared-final'
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp, ProgramInfo
  from tinygrad.runtime import ops_amd
  from tinygrad.engine.realize import get_call_outs_ins
  adapter = load('agent-ck-attention-v2-adapter')
  shapes = [adapter.QSHAPE, adapter.LSHAPE, adapter.QSHAPE, adapter.KVSHAPE, adapter.KVSHAPE]
  import math
  params = [UOp.param(i, dtypes.float32 if i == 1 else dtypes.bfloat16, math.prod(shape), 'NULL').reshape(shape)
            for i, shape in enumerate(shapes)]
  raw = adapter.program(*params)
  info = ProgramInfo.from_sink(raw.src[0])
  program = raw.replace(arg=info)
  assert info.outs == (0, 1) and info.ins == (2, 3, 4)
  assert info.global_size == (16, 32, 1) and info.local_size == (128, 1, 1)
  assert [p.arg.slot for p in info.params] == [0, 1, 2, 3, 4]
  call = program.call(*params)
  assert get_call_outs_ins(call) == ((0, 1), (2, 3, 4))
  words = adapter.argument_words([UOp.const(0x100000+i*0x10000, dtypes.uint64) for i in range(5)])
  packed = b''.join(w.arg if w.op.name == 'BINARY' else struct.pack('<Q', w.ssimplify()) for w in words)
  assert len(packed) == 296 and struct.unpack_from('<5Q', packed) == tuple(0x100000+i*0x10000 for i in range(5))
  assert struct.unpack_from('<IIIHHH', packed, 40) == (16, 32, 1, 128, 1, 1)
  assert struct.unpack_from('<H', packed, 104) == (3,)
  fake = SimpleNamespace(target=(12, 0, 0), iface=SimpleNamespace(props={'lds_size_in_kb': 64}))
  data, image = ops_amd._amd_program_image(fake, adapter.artifacts()[1])
  before = ops_amd.AMDComputeQueue.kernargs
  with adapter.abi_scope():
    assert ops_amd.AMDComputeQueue.kernargs is not before
  assert ops_amd.AMDComputeQueue.kernargs is before and not adapter.ACTIVE
  target.mkdir(exist_ok=False)
  (target/'cpu-audit.json').write_text(json.dumps({'gpu_executed': False, 'outputs': info.outs, 'inputs': info.ins,
    'parameter_slots': [p.arg.slot for p in info.params], 'grid': info.global_size, 'block': info.local_size,
    'argument_bytes': len(packed), 'argument_fixture_hex': packed.hex(), 'descriptor': asdict(data),
    'image_sha256': hashlib.sha256(image).hexdigest(), 'image_bytes': len(image),
    'scope_restored': True, 'lse_is_real_output_dependency': True}, indent=2)+'\n')
  baseline = load('agent-attention-rowsoftmax')
  text, blob = baseline.compiled(16, 8, 2048, 2048, True, 128, waves=8, value_layout=True, transpose='shuffle-fma')
  (target/'baseline.cpp').write_text(text)
  (target/'baseline.hsaco').write_bytes(blob)
  metadata = subprocess.check_output(['llvm-readelf-18', '--all', str(target/'baseline.hsaco')], text=True)
  assert '.workgroup_processor_mode: 0' in metadata and '.wavefront_size: 32' in metadata
  (target/'baseline-readelf.txt').write_text(metadata)
  baseline_data, _ = ops_amd._amd_program_image(fake, blob)
  sources = [EXP/name for name in ('agent-ck-attention-v2-adapter.py', 'agent-ck-attention-v2-native.cpp',
    'agent-ck-attention-v2-oracle.py', 'agent-ck-attention-v3-prepare.py', 'agent-ck-attention-v3-run.py', 'agent-ck-attention-v3-audit.py',
    'agent-attention-rowsoftmax.py', 'online-attention.py')]
  sources += [EXP.parent/'model.py', OLD/'native-overlay/source.cpp', OLD/'native-overlay/kernel.hsaco',
    OLD/'native-overlay/record.json', OLD/'overlay/ck_tile/host/device_prop.hpp',
    OLD/'oracle-r2/manifest.json', target/'baseline.cpp', target/'baseline.hsaco']
  sources += [TINY/'tinygrad'/name for name in ('runtime/ops_amd.py', 'runtime/support/hcq2.py',
    'runtime/support/compiler_amd.py', 'engine/realize.py', 'uop/ops.py', 'tensor.py', 'device.py')]
  oracle = json.loads((OLD/'oracle-r2/manifest.json').read_text())
  manifest = {'schema': 1, 'gpu_executed': False, 'timeout_seconds': 300,
    'tinygrad_commit': subprocess.check_output(['git', '-C', str(TINY), 'rev-parse', 'HEAD'], text=True).strip(),
    'tinygrad': str(TINY), 'oracle': str(OLD/'oracle-r2'), 'candidate': str(OLD/'native-overlay/kernel.hsaco'),
    'baseline': str(target/'baseline.hsaco'), 'baseline_descriptor': asdict(baseline_data),
    'baseline_configuration': {'heads':16, 'kv_heads':8, 'rows':2048, 'keys':2048, 'causal':True, 'width':128,
      'waves':8, 'value_layout':True, 'transpose':'shuffle-fma', 'mode':'CU'},
    'source_sha256': {str(path):digest(path) for path in sources}, 'limits': oracle['limits'],
    'replay_order': oracle['replay_order'],
    'boundaries': {'A': 'Resident contiguous BHSD Q/K/V to BHSD BF16 O; CK also writes full FP32 LSE. Baseline V packing included.',
      'B': 'Resident BSHD Q and BHSD KV cache with capacity2304, sliced to2048, to contiguous BSHD O; CK LSE retained. All packing and output conversion included.'},
    'boundary_B_limit': 'Explicit layout diagnostic. Cache capacity follows model.prefill(2048,64); live Q producer physical layout still requires parent trace.',
    'classification': 'Numerical variant, synthetic full-shape microprotocol only; no route or under100ms claim.'}
  assert manifest['tinygrad_commit'] == '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
  (target/'protocol.json').write_text(json.dumps(manifest, indent=2)+'\n')
  print(json.dumps({'manifest': str(target/'protocol.json'), 'cpu_audit_passed': True,
                    'candidate_descriptor': asdict(data), 'baseline_descriptor': asdict(baseline_data)}, indent=2))


if __name__ == '__main__': main()

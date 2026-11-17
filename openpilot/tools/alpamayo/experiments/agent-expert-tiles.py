"""Optional measured expert tiles. Install immediately after expert-kernels.install.

The existing packed GEMM, quantization, scale association, and bias expressions
are retained. Only two expert projection shapes select different WMMA tiles.
"""
from collections import Counter
from functools import partial
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import MethodType, ModuleType

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
REPORT = WORK/'agent-expert-gemm-fc2-v2/report.json'
REPORT_SHA = '6a4b1305af75ae154bd999e7f06f63deccbd1809a6c997d2945f4e9bf8a7b2ce'
PINS = {
  'agent-expert-gemm.py': '34980d79fb3c38e6e1ef62e33a29ab0c4c81032cc68f48dfed54f2cf4fdeaba1',
  'tiled-gemm.py': '52ca20491721d395a0eb3ca75b459c8bbb7e299647511e16751d6282a2bf2f53',
  'expert-kernels.py': '4676128ef21bd42f851311b3b5dba0563ab6cd1e54d367a3bca2e377f5a53b8b',
  'amd_gemm.py': 'a52cc1cd5331be686ef8fcd1516bba2cfbc1092ca1d64251176491add99041b2',
}
SHAPES = {'mlp.fc2': (64, 512, 3840), 'self_attn.o_proj': (64, 512, 2048)}
TILES = {'mlp.fc2': (1, 4), 'self_attn.o_proj': (2, 2)}
NAME = re.compile(r'expert\.expert\.layers\.(0|[1-9]|1[0-9]|2[0-7])\.(mlp\.fc2|self_attn\.o_proj)')


def sha(data): return hashlib.sha256(data).hexdigest()


def checked_bytes(name, data):
  assert sha(data) == PINS[name], f'Passed GPU source changed: {name}'
  return data


def verify_evidence(report=REPORT, source_dir=HERE):
  report, source_dir = Path(report), Path(source_dir)
  data = report.read_bytes()
  assert sha(data) == REPORT_SHA, 'GPU evidence bytes changed'
  result = json.loads(data)
  assert result['complete'] and result['gpu_executed'] and result['wgp']
  assert result['source_sha256'] == PINS
  assert result['tinygrad_revision'] == '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
  assert result['summary']['cases'] == 8 and result['summary']['compiled_variants'] == 56
  assert result['summary']['rejected_variants'] == 0 and not result['independent_control_rejections']
  relocated = {}
  for name in PINS:
    path = (source_dir.parent if name == 'amd_gemm.py' else source_dir)/name
    checked_bytes(name, path.read_bytes())
    relocated[name] = str(path)
  selections = []
  for case in result['cases']:
    assert case['production_gemm_exact']
    for variant in case['variants'].values():
      assert variant['status'] == 'compiled'
      assert [check['state'] for check in variant['checks']] == list(range(4))
      assert all(check['exact_bf16'] for check in variant['checks'])
    if case['case'] not in ('fc2', 'o'): continue
    suffix = 'mlp.fc2' if case['case'] == 'fc2' else 'self_attn.o_proj'
    tile = 'x'.join(map(str, TILES[suffix]))
    variant = case['variants'][tile]
    assert tuple(case['shape_mnk']) == SHAPES[suffix]
    assert variant['batch_mutated_input_scales_exact'] and variant['captured_matrix_calls'] == 40
    timing = next(record for record in case['timing'] if record['variant'] == tile)
    assert timing['matrix_calls_per_batch'] == 40 and len(timing['paired_batch_ms']) == 5
    selections.append({'projection': suffix, 'bias': case['bias'], 'tile': TILES[suffix],
      'binary_sha256': variant['binary_sha256'], 'resources': variant['resources'], 'timing': timing})
  assert len(selections) == 4
  return {'report': str(report), 'report_sha256': REPORT_SHA, 'source_sha256': PINS.copy(),
    'relocated_sources': relocated, 'selected_gpu_cases': selections, 'pipeline_validated': False}


def load_tiled():
  path = HERE/'tiled-gemm.py'
  data = checked_bytes(path.name, path.read_bytes())
  module = ModuleType('agent_expert_tiles_verified_tiled')
  module.__file__ = str(path)
  exec(compile(data, str(path), 'exec'), module.__dict__)
  return module


def gemm(quantized, weight, scales, bias, tile, tiled):
  from tinygrad import Tensor, dtypes
  from amd_gemm import pack
  output = Tensor.empty(64, 512, device=quantized.device, dtype=dtypes.bfloat16)
  return Tensor.custom_kernel(output, pack(quantized), pack(weight), *scales, *(() if bias is None else (bias,)),
    fxn=partial(tiled.kernel, tile_m=tile[0], tile_n=tile[1], row_scale=False, column_scale=False,
      local_m=1, local_n=1, swap=False, unroll_k=1, group_m=1))[0]


class Installation:
  def __init__(self, model, original, info):
    self.model, self.original, self.info, self.active, self.wrapper = model, original, info, True, None

  def close(self):
    # Later wrappers retain their own chain. An inactive hook delegates through
    # its original method, so cleanup never removes a later quantization wrapper.
    self.active = False
    if self.model.linear is self.wrapper: self.model.linear = self.original
    if getattr(self.model, 'agent_expert_tiles', None) is self: del self.model.agent_expert_tiles
    self.info['active'] = False


def install(model, *, evidence=REPORT):
  from tinygrad import Tensor, dtypes
  import amd_gemm
  assert model.fp8_wmma and model.dtype == dtypes.bfloat16
  assert not hasattr(model, 'agent_expert_tiles'), 'Already installed'
  verified = verify_evidence(evidence)
  checked_bytes('amd_gemm.py', Path(amd_gemm.__file__).read_bytes())
  original = model.linear
  origin = Path(inspect.getsourcefile(original.__func__))
  assert origin.name == 'expert-kernels.py', 'Install immediately after expert-kernels.install, before later wrappers'
  checked_bytes('expert-kernels.py', origin.read_bytes())
  tiled = load_tiled()
  info = {'active': True, 'gpu_matrix_evidence': verified, 'trace_calls': Counter(), 'quantizer_calls': 0,
    'supplied_quantized_calls': 0, 'route_exactness_validated': False, 'tiles': TILES.copy()}
  handle = Installation(model, original, info)

  def linear(self, x, name, quantized_input=None, heads=None):
    match = NAME.fullmatch(name)
    weight = self.weights.get(name+'.weight')
    if (not handle.active or match is None or x.ndim < 2 or x.dtype != dtypes.bfloat16 or heads is not None
        or weight is None or weight.dtype != dtypes.fp8e4m3 or name+'.weight' not in self.weight_scales):
      return original(x, name, quantized_input, heads)
    suffix = match[2]
    rows, columns, width = SHAPES[suffix]
    bias, ws = self.weights.get(name+'.bias'), self.weight_scales[name+'.weight']
    if (x.numel()//x.shape[-1] != rows or x.shape[-1] != width or weight.shape != (columns, width)
        or ws.dtype != dtypes.float32 or ws.numel() != 1
        or (bias is not None and (bias.dtype != dtypes.bfloat16 or bias.shape != (columns,)))):
      return original(x, name, quantized_input, heads)
    quantized, scale = self.quantize(x) if quantized_input is None else quantized_input
    assert isinstance(quantized, Tensor) and quantized.dtype == dtypes.fp8e4m3 and quantized.shape == (rows, width)
    assert isinstance(scale, Tensor) and scale.dtype == dtypes.float32 and scale.numel() == 1
    info['trace_calls'][name] += 1
    info['quantizer_calls' if quantized_input is None else 'supplied_quantized_calls'] += 1
    return gemm(quantized, weight, (scale, ws), bias, TILES[suffix], tiled).reshape(*x.shape[:-1], columns)

  handle.wrapper = MethodType(linear, model)
  model.linear = handle.wrapper
  model.agent_expert_tiles = handle
  return handle

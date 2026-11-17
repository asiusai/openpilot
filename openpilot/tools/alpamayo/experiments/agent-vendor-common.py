"""Shared provenance and numerical diagnostics for a numerical vendor candidate."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

EXP = Path(__file__).parent
ROOT = Path('/tmp/opencode')
CHECKPOINT = ROOT/'alpamayo-asius-20261114/model'
CALIBRATION = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/prefill-calibration/report.prefill-scales.json')
METADATA = ROOT/'alpamayo-vendor-tuned-selected.json'
IMAGE = ROOT/'alpamayo-vendor-fp8.hsaco'
SOLUTIONS = (27854,28279)
SHAPES = ((2112,9216,2048),(2112,2048,9216),(2112,4096,2048))
PINS = {IMAGE:'9dbb05bf273cd6a2a3b36c65fc2254ac95b5275804ae58940783e672f3f3ebc0',
        METADATA:'ada4182c5187a405395edc6b138c742a13eb592db423d60c624b092bef8d1626',
        EXP/'vendor-gemm.py':'aed3ad5bd725e6b7755c8f5860f1c5e2c61181d241445fad36512f76268e7450'}


def sha(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream,'sha256').hexdigest()


def module(name):
  spec = importlib.util.spec_from_file_location('agent_vendor_'+name.replace('-','_'),EXP/(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def _byte_layout(out,source,*,mode,rows,width,renderer=None):
  """Explicit byte pointer loads/stores avoid FP8 conversion in movement kernels."""
  from tinygrad import Device,dtypes
  from tinygrad.codegen.opt.postrange import Scheduler
  from tinygrad.uop.ops import UOp,KernelInfo,AxisType
  from dataclasses import replace
  # Retype only the callee's pointer parameters. Buffer addresses and byte extents
  # remain unchanged, and JIT argument slots remain ordinary dynamic parameters.
  def byte_parameter(value):
    base = value.base
    return value.substitute({base:base.replace(arg=replace(base.arg,dtype=dtypes.uint8))},walk=True)
  source,out = byte_parameter(source),byte_parameter(out)
  index = UOp.range(rows*width,0)
  row,column = index//width,index%width
  packed = (row//16*(width//16)+column//16)*256+(column%16//4)*64+(row%16)*4+column%4
  src_index,dst_index = (packed,index) if mode == 'unpack' else (index,packed) if mode == 'pack' else (index,index)
  value = source.flatten().index(src_index).load()
  store = out.flatten().index(dst_index).store(value)
  renderer = Device[Device.DEFAULT].renderer if renderer is None else renderer
  schedule = Scheduler(store.end(index).sink(arg=KernelInfo()),renderer)
  schedule.convert_loop_to_global()
  if renderer.has_local:
    from tinygrad.codegen.opt import Opt,OptOps
    schedule.apply_opt(Opt(OptOps.SPLIT,0,(4,AxisType.UPCAST)))
    local = next(count for count in (64,32,16,8,4) if (rows*width//4)%count == 0)
    global_range = next(value for value in schedule.rngs if value.axis_type is AxisType.GLOBAL)
    schedule.apply_opt(Opt(OptOps.SPLIT,schedule.rngs.index(global_range),(local,AxisType.LOCAL)))
  return schedule.get_optimized_ast(name_override='agent_vendor_byte_layout_'+mode)


def byte_layout(value,mode='rowmajor',rows=None):
  import functools
  from tinygrad import Tensor,dtypes
  if value.dtype not in (dtypes.fp8e4m3,dtypes.uint8) or value.ndim != 2:
    raise TypeError('Expected a two-dimensional FP8 or byte matrix')
  rows,width = (value.shape[0] if rows is None else rows),value.shape[1]
  if width%16 or (mode == 'pack' and rows%16):
    raise ValueError('These bounded shapes require complete 16-byte tiles')
  output = Tensor.empty(rows,width,device=value.device,dtype=dtypes.fp8e4m3)
  return Tensor.custom_kernel(output,value,fxn=functools.partial(_byte_layout,mode=mode,rows=rows,width=width))[0].realize()


def materialize_rowmajor(value):
  return byte_layout(value)


def materialize_packed_weight(logical_weight):
  from amd_gemm import pack
  # The pinned model stores unpack(packed_storage) as a logical weight view.
  # Recover that physical packed source before byte copying; otherwise the
  # custom-call frontend may first materialize an FP8 movement numerically.
  packed = pack(logical_weight).realize()
  return byte_layout(packed,'unpack',logical_weight.shape[0])


def pack_bytes(value):
  return byte_layout(value,'pack')


def unpack_activation(value,rows):
  return byte_layout(value,'unpack',rows)


def assign_bytes(destination,source):
  import functools
  from tinygrad import Tensor
  from tinygrad import dtypes
  if destination.shape != source.shape or destination.dtype != source.dtype:
    raise ValueError('Byte assignment requires the same shape and dtype')
  if destination.dtype == dtypes.fp8e4m3:
    Tensor.custom_kernel(destination,source,fxn=functools.partial(_byte_layout,mode='rowmajor',rows=destination.shape[0],width=destination.shape[1]))[0].realize()
  else:
    destination.bitcast(dtypes.uint8).assign(source.bitcast(dtypes.uint8)).realize()


def provenance(checkpoint=CHECKPOINT,calibration=CALIBRATION):
  """Hash local artifacts without opening a device or downloading anything."""
  pins = {str(path):{'sha256':sha(path),'bytes':path.stat().st_size} for path in PINS}
  for path,expected in PINS.items():
    if pins[str(path)]['sha256'] != expected:
      raise ValueError(f'Pinned vendor artifact changed: {path}')
  checkpoint = Path(checkpoint)
  index = json.loads((checkpoint/'model.safetensors.index.json').read_text())
  checkpoint_files = ['config.json','model.safetensors.index.json',*sorted(set(index['weight_map'].values()))]
  checkpoint_hashes = {name:sha(checkpoint/name) for name in checkpoint_files}
  scales = json.loads(Path(calibration).read_text())['scales']
  if not scales or not all(np.isfinite(value) and value > 0 for value in scales.values()):
    raise ValueError('Fixed prefill scales must be finite and positive')
  expected = 4*json.loads((checkpoint/'config.json').read_text())['vlm_config']['text_config']['num_hidden_layers']
  if len(scales) != expected:
    raise ValueError('Incomplete fixed prefill calibration')
  records = json.loads(METADATA.read_text())
  chosen = {record['solution']['index']:record for record in records if record['solution']['index'] in SOLUTIONS}
  if set(chosen) != set(SOLUTIONS):
    raise ValueError('Missing reviewed solution metadata')
  cache_index = Path(os.environ.get('ALPAMAYO_FP8_CACHE','/dev/shm/alpamayo-9060-fp8'))/'index.json'
  cache = json.loads(cache_index.read_text()) if cache_index.exists() else None
  return {'classification':'NUMERICAL CANDIDATE; vendor reduction and scale application may round differently',
          'vendor_artifacts':pins,'solution_records':chosen,
          'package_manifest_sha256':sha(ROOT/'alpamayo-hipblaslt-7.0.2/manifest.json'),
          'package_version':'hipBLASLt 1.0.0.70002-56 / ROCm 7.0.2',
          'checkpoint':str(checkpoint),'checkpoint_file_sha256':checkpoint_hashes,
          'fixed_prefill_calibration':str(calibration),'fixed_prefill_calibration_sha256':sha(calibration),
          'fixed_scale_count':len(scales),
          'fp8_cache_index_sha256':sha(cache_index) if cache else None,
          'fp8_cache_declared_weights_sha256':cache.get('sha256') if cache else None,
          'source_sha256':{path.name:sha(path) for path in [*EXP.glob('agent-vendor*.py'),EXP.parent/'amd_gemm.py',
                                                         EXP/'vendor-gemm.py',EXP/'quantize-packed.py']}}


@contextmanager
def runtime():
  """One fresh vendor module and one already-open USB AMD runtime per process."""
  from tinygrad import Device
  from tinygrad.runtime.ops_amd import AMDComputeQueue
  if any(name.split(':')[0] == 'AMD' for name in Device._opened_devices):
    raise RuntimeError('Start this experiment in a fresh process before opening AMD')
  previous = os.environ.get('ALPAMAYO_VENDOR_SOLUTIONS')
  os.environ['ALPAMAYO_VENDOR_SOLUTIONS'] = str(METADATA)
  vendor = module('vendor-gemm')
  original = AMDComputeQueue.kernargs
  try:
    for index in SOLUTIONS:
      vendor.selected(index)
    device = Device[Device.DEFAULT]
    if not device.is_usb or device.arch != 'gfx1200':
      raise RuntimeError('This reviewed ABI requires USB gfx1200')
    module('prefill-kernels').fix_dispatch_packet()
    with vendor.argument_adapter(device):
      yield device,vendor
  finally:
    bad_cleanup = AMDComputeQueue.kernargs is not original or vendor.ACTIVE
    try:
      if bad_cleanup:
        AMDComputeQueue.kernargs = original
        vendor.ACTIVE = False
    finally:
      if previous is None:
        os.environ.pop('ALPAMAYO_VENDOR_SOLUTIONS',None)
      else:
        os.environ['ALPAMAYO_VENDOR_SOLUTIONS'] = previous
    if bad_cleanup:
      raise RuntimeError('Vendor ABI adapter needed forced restoration; results are invalid')


def bf16_round(values):
  values = np.asarray(values,dtype=np.float32)
  bits = values.view(np.uint32)
  return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)


def bf16_float(bits):
  return (np.asarray(bits,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)


def distribution(values):
  values = np.asarray(values)
  return {'count':int(values.size),'mean':float(np.mean(values,dtype=np.float64)) if values.size else None,
          'p50':float(np.quantile(values,.5)) if values.size else None,
          'p95':float(np.quantile(values,.95)) if values.size else None,
          'p99':float(np.quantile(values,.99)) if values.size else None,
          'max':float(np.max(values)) if values.size else None}


def differences(original_bits,candidate_bits):
  a,b = bf16_float(original_bits),bf16_float(candidate_bits)
  if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
    raise ValueError('Nonfinite or incorrectly shaped GEMM output')
  changed = original_bits != candidate_bits
  delta = np.abs(a-b)
  nonzero = np.abs(a) > 0
  relative = delta[nonzero]/np.abs(a[nonzero])
  def ordered(bits):
    bits = bits.astype(np.int32)
    return np.where(bits & 0x8000,0x8000-(bits & 0x7fff),0x8000+bits)
  ulps = np.abs(ordered(original_bits)-ordered(candidate_bits))
  return {'exact_vs_original':not bool(changed.any()),'differing_bf16_values':int(changed.sum()),
          'total_values':int(changed.size),'differing_fraction':float(changed.mean()),
          'absolute_delta_all':distribution(delta),'absolute_delta_changed':distribution(delta[changed]),
          'relative_delta_nonzero_original':distribution(relative),
          'relative_delta_changed_nonzero_original':distribution(delta[changed & nonzero]/np.abs(a[changed & nonzero])),
          'original_zero_candidate_nonzero':int(np.count_nonzero((~nonzero)&(b != 0))),
          'bf16_ulp_delta_changed':distribution(ulps[changed]),
          'normalized_rms_delta':float(np.sqrt(np.mean(delta.astype(np.float64)**2))/max(np.sqrt(np.mean(a.astype(np.float64)**2)),1e-300)),
          'acceptance':'No numerical acceptance threshold is applied here. Route quality gates remain required.'}


def fp64_samples(original_bits,candidate_bits,x,w,x_scale,w_scale,bias,limit=512,seed=9060):
  original,candidate = bf16_float(original_bits),bf16_float(candidate_bits)
  changed = np.flatnonzero(original_bits.ravel() != candidate_bits.ravel())
  randomizer = np.random.default_rng(seed)
  populations = {'differing_values':changed,'uniform_output':None}
  reports = {}
  for name,population in populations.items():
    count = min(limit,original.size if population is None else population.size)
    selected = randomizer.choice(original.size,count,replace=False) if population is None else randomizer.choice(population,count,replace=False)
    rows,columns = np.divmod(selected,original.shape[1])
    dots = np.empty(count,dtype=np.float64)
    for start in range(0,count,32):
      rr,cc = rows[start:start+32],columns[start:start+32]
      dots[start:start+32] = np.sum(x[rr].astype(np.float64)*w[cc].astype(np.float64),axis=1,dtype=np.float64)
    exact = dots*np.float64(x_scale)*np.float64(w_scale)+bias[columns].astype(np.float64)
    fp32_product = dots*np.float64(np.float32(np.float32(x_scale)*np.float32(w_scale)))+bias[columns].astype(np.float64)
    a,b = original[rows,columns].astype(np.float64),candidate[rows,columns].astype(np.float64)
    ae,be = np.abs(a-exact),np.abs(b-exact)
    reports[name] = {'sample_count':count,'seed':seed,'original_error':distribution(ae),'candidate_error':distribution(be),
                     'candidate_better':int(np.count_nonzero(be < ae)),'candidate_worse':int(np.count_nonzero(be > ae)),
                     'equal_error':int(np.count_nonzero(be == ae)),
                     'original_error_with_fp32_scale_product':distribution(np.abs(a-fp32_product)),
                     'candidate_error_with_fp32_scale_product':distribution(np.abs(b-fp32_product))}
  return {'oracle':'FP64 sum of actual quantized FP8 operands; FP64 scale product and FP32 scale-product variants are both reported.',
          **reports}

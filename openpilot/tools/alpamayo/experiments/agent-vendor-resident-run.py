"""Parent-only bounded GPU residency smoke. This is not a performance benchmark.

The CPU check is agent-vendor-resident-check.py. Run this file only through the
parent's serialized GPU wrapper, in a fresh process with a hard child timeout.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def module(path):
  spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-', '_'), path)
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def small_integer_bits(value):
  value = np.asarray(value)
  assert np.issubdtype(value.dtype,np.integer) and value.min() >= -2 and value.max() <= 2
  return np.array([0xc0,0xb8,0x00,0x38,0x40],dtype=np.uint8)[value+2]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',required=True,type=Path)
  parser.add_argument('--solution',type=int,choices=(27854,28279),default=27854)
  parser.add_argument('--upload-seconds',type=float,default=30.)
  args = parser.parse_args()
  if not 1 <= args.upload_seconds <= 60: parser.error('upload-seconds must be within 1..60')
  root = Path(__file__).parent
  check = module(root/'agent-vendor-resident-check.py')
  resident = module(root/'agent-vendor-resident.py')
  bench = module(check.FROZEN/'agent-vendor-bench.py')
  common = bench.common
  report = {'classification':'Resident upload smoke only; no GEMM performance result or route acceptance',
            'artifact_pins':check.provenance(), 'solution':args.solution, 'shape_m_n_k':[64,128,64],
            'calls_per_graph':40, 'checks':[], 'timing_samples':[],
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path(__file__),root/'agent-vendor-resident.py',root/'agent-vendor-resident-check.py')}}
  def save(): args.output.write_text(json.dumps(report,indent=2)+'\n')
  save()
  from tinygrad import Tensor,TinyJit,dtypes,Device
  from tinygrad.device import Compiled
  from tinygrad.helpers import GlobalCounters,PROFILE
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  assert not Device._opened_devices and not PROFILE
  original_abi,original_builder,original_cache = ops_amd.AMDComputeQueue.kernargs,ops_amd.amd_build_program,hcq2.hcq_compile_cache
  Compiled.wait_timeout_ms = 10000
  pool = None
  try:
    with common.runtime() as (device,vendor):
      report['device'] = {'arch':device.arch,'usb':device.is_usb,'cu_count':device.cu_cnt}
      with resident.install(device,vendor,(args.solution,),seconds=args.upload_seconds) as pool:
        report['residency'] = pool.stats
        save()
        rng = np.random.default_rng(905027854)
        m,n,k = report['shape_m_n_k']
        host_inputs = [rng.integers(-2,3,(m,k),dtype=np.int32) for _ in range(40)]
        host_w = rng.integers(-2,3,(n,k),dtype=np.int32)
        host_bias = ((np.arange(n)%9)-4).astype(np.float32)/32
        def fp8(value):
          return Tensor(small_integer_bits(value),device=device.device).bitcast(dtypes.fp8e4m3).contiguous().realize()
        inputs = [fp8(value) for value in host_inputs]
        assert len({value.uop.buffer for value in inputs}) == 40
        weight = fp8(host_w)
        sx,sw = .125,.25
        xs = Tensor([sx],device=device.device,dtype=dtypes.float32).realize()
        ws = Tensor([sw],device=device.device,dtype=dtypes.float32).realize()
        ones = Tensor.ones(n,device=device.device,dtype=dtypes.float32).contiguous().realize()
        bias = Tensor(host_bias,device=device.device).realize()
        zero = Tensor.zeros(m,n,device=device.device,dtype=dtypes.bfloat16).contiguous().realize()
        def forty(*parameters):
          assert len(parameters) == 43
          w,s,t = parameters[-3:]
          # Retain all outputs for an independent integer oracle. No time is
          # divided by 40, and no serialization/performance claim is made.
          return tuple(vendor.gemm(a,w,(s,t),ones,bias,zero,index=args.solution).realize() for a in parameters[:-3])
        graph = TinyJit(forty)
        previous = None
        for stage in ('eager40','capture40','replay40','mutated_replay40'):
          print('Resident smoke stage',stage,flush=True)
          if stage == 'mutated_replay40':
            host_inputs = [rng.integers(-2,3,(m,k),dtype=np.int32) for _ in range(40)]
            host_w = rng.integers(-2,3,(n,k),dtype=np.int32)
            for target,source in zip(inputs,host_inputs,strict=True): common.assign_bytes(target,fp8(source))
            common.assign_bytes(weight,fp8(host_w))
            sx,sw = .25,.0625
            xs.assign(Tensor([sx],device=device.device,dtype=dtypes.float32)).realize()
            ws.assign(Tensor([sw],device=device.device,dtype=dtypes.float32)).realize()
          pool.guard()
          before = GlobalCounters.kernel_count
          outputs = graph(*inputs,weight,xs,ws)
          device.synchronize()
          executed = GlobalCounters.kernel_count-before
          assert executed >= 40 and len(outputs) == 40
          if stage != 'eager40': assert bench.captured_gemm_count(graph,args.solution) == 40
          digests = []
          for index,(out,host_x) in enumerate(zip(outputs,host_inputs,strict=True)):
            expected = common.bf16_round((host_x.astype(np.int64)@host_w.astype(np.int64).T).astype(np.float64)*sx*sw+host_bias)
            bits = out.bitcast(dtypes.uint16).numpy()
            np.testing.assert_array_equal(bits,expected,err_msg=f'{stage} output {index}')
            digests.append(hashlib.sha256(bits.tobytes()).hexdigest())
          if stage == 'mutated_replay40': assert all(a != b for a,b in zip(previous,digests,strict=True))
          elif previous is not None: assert digests == previous
          previous = digests
          assert pool.stats['image_uploads'] == 1 and pool.stats['uploaded_bytes'] == 67880700
          report['checks'].append({'stage':stage,'counted_kernels':executed,'all_40_integer_outputs_exact':True,
                                   'x_scale':sx,'weight_scale':sw,'nonzero_bias':True,'image_uploads_so_far':1})
          save()
          print(json.dumps(report['checks'][-1]),flush=True)
      assert pool.closed and pool.stats['builder_and_compile_cache_restored']
    assert ops_amd.AMDComputeQueue.kernargs is original_abi and not vendor.ACTIVE
    assert ops_amd.amd_build_program is original_builder and hcq2.hcq_compile_cache is original_cache
    report['abi_restored'] = report['passed'] = True
  except BaseException as exc:
    report['failure'] = f'{type(exc).__name__}: {exc}'
    report['abi_restored'] = ops_amd.AMDComputeQueue.kernargs is original_abi
    report['builder_and_cache_restored'] = ops_amd.amd_build_program is original_builder and hcq2.hcq_compile_cache is original_cache
    raise
  finally: save()


if __name__ == '__main__': main()

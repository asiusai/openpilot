"""Bounded 40-GEMM USB experiment; classify floating differences as numerical.

Only --inspect is CPU-only. The parent must run GPU mode through its trial wrapper.
"""
import argparse
import functools
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location('agent_vendor_common',Path(__file__).with_name('agent-vendor-common.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def forty_calls(fn,*values):
  if len(values) != 43:
    raise ValueError('Expected 40 independent activation parameters plus weight and two scales')
  weight,s,t = values[-3:]
  result = None
  for activation in values[:-3]:
    result = fn(activation,weight,s,t).realize()
  return result


def captured_gemm_count(jit,index):
  from tinygrad.uop.ops import Ops
  names = []
  for call in jit.captured._linear.src:
    call = call.without_after
    if call.op is not Ops.CALL: raise AssertionError(f'Unexpected captured operation: {call.op}')
    aux = getattr(call.arg,'aux',None)
    if hasattr(aux,'kernels'):
      names.extend(str(entry[1]) for entry in aux.kernels)
    elif call.body.op is Ops.PROGRAM:
      names.append(str(call.body.src[0].arg.name))
  prefix = 'alpamayo_fp8_gemm' if index is None else f'alpamayo_vendor_{index}_'
  return sum(name.startswith(prefix) for name in names)


def inspect(report):
  import os
  from tinygrad import Device
  from tinygrad.runtime.ops_amd import _amd_program_image
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.helpers import Target
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad import dtypes
  from tinygrad.codegen import to_program
  before = set(Device._opened_devices)
  previous = os.environ.get('ALPAMAYO_VENDOR_SOLUTIONS')
  os.environ['ALPAMAYO_VENDOR_SOLUTIONS'] = str(common.METADATA)
  try:
    vendor = common.module('vendor-gemm')
    descriptions = []
    for index in common.SOLUTIONS:
      blob = vendor.binary(index)
      data,image = _amd_program_image(SimpleNamespace(iface=SimpleNamespace(props={'lds_size_in_kb':64}),target=(12,0,0)),blob)
      descriptions.append({'solution':index,'aliased_elf_sha256':hashlib.sha256(blob).hexdigest(),
                           'loaded_image_sha256':hashlib.sha256(image).hexdigest(),'loaded_image_bytes':len(image),
                           'descriptor_kernargs':data.kernargs_segment_size,'metadata_kernargs':176,
                           'lds_bytes':data.group_segment_size,'wave32':data.wave32})
    report['cpu_descriptor_checks'] = descriptions
    renderer = AMDLLVMRenderer(Target(device='AMD',renderer='LLVM',arch='gfx1200'))
    compiled = []
    for mode in ('rowmajor','pack','unpack'):
      for width in (2048,9216):
        out = UOp.placeholder((2112,width),dtypes.fp8e4m3,0)
        source = UOp.placeholder((2112,width),dtypes.fp8e4m3,1)
        ast = common._byte_layout(out,source,mode=mode,rows=2112,width=width,renderer=renderer)
        program = to_program(ast,renderer)
        binary = next(value.arg for value in program.src if value.op is Ops.BINARY)
        source_text = next(value.arg for value in program.src if value.op is Ops.SOURCE)
        compiled.append({'mode':mode,'shape':[2112,width],'binary_bytes':len(binary),
                         'binary_sha256':hashlib.sha256(binary).hexdigest(),
                         'source_sha256':hashlib.sha256(source_text.encode()).hexdigest(),
                         'contains_fp8_conversion_intrinsics':'@f32_to_fp8' in source_text or 'cvt.f32.fp8' in source_text})
    if any(value['contains_fp8_conversion_intrinsics'] for value in compiled):
      raise AssertionError('Byte layout compiler introduced a floating conversion')
    report['cpu_gfx1200_byte_kernel_compilation'] = compiled
  finally:
    if previous is None:
      os.environ.pop('ALPAMAYO_VENDOR_SOLUTIONS',None)
    else:
      os.environ['ALPAMAYO_VENDOR_SOLUTIONS'] = previous
  if set(Device._opened_devices) != before:
    raise AssertionError('CPU descriptor inspection unexpectedly opened a runtime device')
  report['opened_devices_during_inspection'] = []


def small_controls(device,vendor):
  from tinygrad import Tensor,TinyJit,dtypes
  randomizer = np.random.default_rng(71854)
  records = []
  for shape in ((64,128,64),(145,144,80)):
    rows,columns,width = shape
    for index in common.SOLUTIONS:
      host_x = randomizer.integers(-2,3,size=(rows,width)).astype(np.float32)
      host_w = randomizer.integers(-2,3,size=(columns,width)).astype(np.float32)
      host_bias = ((np.arange(columns)%9)-4).astype(np.float32)/32
      x = Tensor(host_x,device=device.device).cast(dtypes.fp8e4m3).contiguous().realize()
      w = Tensor(host_w,device=device.device).cast(dtypes.fp8e4m3).contiguous().realize()
      xs = Tensor([.125],device=device.device,dtype=dtypes.float32).realize()
      ws = Tensor([.25],device=device.device,dtype=dtypes.float32).realize()
      ones = Tensor.ones(columns,device=device.device,dtype=dtypes.float32).contiguous().realize()
      bias = Tensor(host_bias,device=device.device).realize()
      zero = Tensor.zeros(rows,columns,device=device.device,dtype=dtypes.bfloat16).contiguous().realize()
      run = TinyJit(lambda a,b,s,t:vendor.gemm(a,b,(s,t),ones,bias,zero,index=index).realize())
      for mutation in range(3):
        if mutation:
          host_x = randomizer.integers(-2,3,size=host_x.shape).astype(np.float32)
          host_w = randomizer.integers(-2,3,size=host_w.shape).astype(np.float32)
          common.assign_bytes(x,Tensor(host_x,device=device.device).cast(dtypes.fp8e4m3).realize())
          common.assign_bytes(w,Tensor(host_w,device=device.device).cast(dtypes.fp8e4m3).realize())
        output = run(x,w,xs,ws)
        device.synchronize()
        expected = common.bf16_round((host_x.astype(np.float64)@host_w.astype(np.float64).T)*.03125+host_bias)
        np.testing.assert_array_equal(output.bitcast(dtypes.uint16).numpy(),expected)
      records.append({'solution':index,'shape_m_n_k':shape,'mutations':3,'integer_fp64_oracle_exact':True,
                      'scale_product':.03125,'nonzero_column_bias':True})
  return records


def gpu_benchmark(args,report):
  from tinygrad import Tensor,TinyJit,dtypes
  from tinygrad.device import Compiled
  from tinygrad.helpers import GlobalCounters
  from amd_gemm import _gemm
  Compiled.wait_timeout_ms = 10000
  randomizer = np.random.default_rng(906027854)
  with common.runtime() as (device,vendor):
    common.module('wgp_compiler').install(device)
    report['device'] = {'arch':device.arch,'cu_count':device.cu_cnt,'usb':device.is_usb}
    report['integer_controls'] = small_controls(device,vendor)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for shape in common.SHAPES:
      rows,columns,width = shape
      print('40-call shape',shape,flush=True)
      host_x = randomizer.standard_normal((rows,width),dtype=np.float32)*8
      host_w = randomizer.standard_normal((columns,width),dtype=np.float32)*8
      initial_x,initial_w = host_x,host_w
      base = Tensor(host_x,device=device.device).cast(dtypes.fp8e4m3).contiguous().realize()
      row_inputs = [common.materialize_rowmajor(base) for _ in range(40)]
      packed_inputs = [common.pack_bytes(value) for value in row_inputs]
      w = Tensor(host_w,device=device.device).cast(dtypes.fp8e4m3).contiguous().realize()
      packed_w = common.pack_bytes(w)
      xs = Tensor([.125],device=device.device,dtype=dtypes.float32).realize()
      ws = Tensor([.25],device=device.device,dtype=dtypes.float32).realize()
      ones = Tensor.ones(columns,device=device.device,dtype=dtypes.float32).contiguous().realize()
      bias = Tensor.zeros(columns,device=device.device,dtype=dtypes.float32).contiguous().realize()
      zero = Tensor.zeros(rows,columns,device=device.device,dtype=dtypes.bfloat16).contiguous().realize()
      baseline = lambda a,b,s,t:Tensor.custom_kernel(Tensor.empty(rows,columns,device=device.device,dtype=dtypes.bfloat16),
                                                     a,b,s,t,fxn=functools.partial(_gemm,packed_input=True))[0]
      for boundary in ('gemm_ready','packed_activation_bridge'):
        if boundary != 'gemm_ready':
          common.assign_bytes(base,Tensor(initial_x,device=device.device).cast(dtypes.fp8e4m3).realize())
          for row,packed in zip(row_inputs,packed_inputs,strict=True):
            common.assign_bytes(row,base)
            common.assign_bytes(packed,common.pack_bytes(row))
          common.assign_bytes(w,Tensor(initial_w,device=device.device).cast(dtypes.fp8e4m3).realize())
          common.assign_bytes(packed_w,common.pack_bytes(w))
          xs.assign(Tensor([.125],device=device.device,dtype=dtypes.float32)).realize()
          ws.assign(Tensor([.25],device=device.device,dtype=dtypes.float32)).realize()
        functions = {None:baseline}
        for index in common.SOLUTIONS:
          def candidate(a,b,s,t,index=index):
            row = a if boundary == 'gemm_ready' else common.unpack_activation(a,rows)
            return vendor.gemm(row,b,(s,t),ones,bias,zero,index=index)
          functions[index] = candidate
        graphs = {}
        timings = {index:[] for index in functions}
        parameters = {}
        for index,fn in functions.items():
          graphs[index] = TinyJit(functools.partial(forty_calls,fn))
          inputs = packed_inputs if index is None or boundary != 'gemm_ready' else row_inputs
          parameters[index] = (*inputs,packed_w if index is None else w,xs,ws)
          for _ in range(2):
            graphs[index](*parameters[index])
            device.synchronize()
          if captured_gemm_count(graphs[index],index) != 40:
            raise AssertionError('Captured graph does not contain exactly 40 intended GEMMs')
        # Interleave methods to avoid assigning all later thermal states to one solution.
        counters = {}
        for repeat in range(args.replays):
          order = list(functions)
          randomizer.shuffle(order)
          for index in order:
            device.synchronize()
            before = GlobalCounters.kernel_count
            started = time.perf_counter()
            graphs[index](*parameters[index])
            device.synchronize()
            elapsed = (time.perf_counter()-started)*1000
            executed = GlobalCounters.kernel_count-before
            if executed < 40:
              raise AssertionError(f'40-call graph executed only {executed} counted kernels')
            timings[index].append(elapsed/40)
            counters[index] = executed
        stage = {'shape_m_n_k':shape,'boundary':boundary,'calls_per_graph':40,'warmup_calls_per_method':80,
                 'measured_graph_replays':args.replays,
                 'timing_boundary':('One host graph invocation plus final USB/GPU synchronization, divided by 40. '
                                    'Weights, input preparation, upload, quantization, capture, and readbacks excluded. '
                                    'packed_activation_bridge includes every activation unpack/contiguous operation.'),
                 'input_buffers':'40 distinct parameter buffers; equal contents per mutation. Weights repeat across calls.',
                 'methods':[],'mutated_inputs':[]}
        for index in functions:
          stage['methods'].append({'solution':index,'median_ms_per_gemm':statistics.median(timings[index]),
                                   'samples_ms_per_gemm':timings[index],'counted_kernels_per_graph':counters[index],
                                   'captured_gemms_per_graph':captured_gemm_count(graphs[index],index)})
        # Correctness and mutation checks occur after timing. They never enter the speed samples.
        previous_hashes = {}
        for mutation in range(3):
          if mutation:
            host_x = randomizer.standard_normal((rows,width),dtype=np.float32)*(4 if mutation == 1 else 12)
            host_w = randomizer.standard_normal((columns,width),dtype=np.float32)*(12 if mutation == 1 else 4)
            common.assign_bytes(base,Tensor(host_x,device=device.device).cast(dtypes.fp8e4m3).realize())
            for row,packed in zip(row_inputs,packed_inputs,strict=True):
              common.assign_bytes(row,base)
              common.assign_bytes(packed,common.pack_bytes(row))
            common.assign_bytes(w,Tensor(host_w,device=device.device).cast(dtypes.fp8e4m3).realize())
            common.assign_bytes(packed_w,common.pack_bytes(w))
          sx,sw = ((.125,.25),(.25,.0625),(.0037,.0143))[mutation]
          xs.assign(Tensor([sx],device=device.device,dtype=dtypes.float32)).realize()
          ws.assign(Tensor([sw],device=device.device,dtype=dtypes.float32)).realize()
          actual_x,actual_w = base.float().numpy(),w.float().numpy()
          outputs = {}
          for index in functions:
            first = graphs[index](*parameters[index])
            device.synchronize()
            bits = first.bitcast(dtypes.uint16).numpy().copy()
            repeated = graphs[index](*parameters[index])
            device.synchronize()
            np.testing.assert_array_equal(repeated.bitcast(dtypes.uint16).numpy(),bits)
            digest = hashlib.sha256(bits.tobytes()).hexdigest()
            if mutation and digest == previous_hashes[index]:
              raise AssertionError('Mutated inputs produced an unchanged complete output')
            previous_hashes[index] = digest
            outputs[index] = bits
          for index in common.SOLUTIONS:
            stage['mutated_inputs'].append({'mutation':mutation,'solution':index,'x_scale':float(np.float32(sx)),
                                            'w_scale':float(np.float32(sw)),'repeat_bits_exact':True,
                                            **common.differences(outputs[None],outputs[index]),
                                            'fp64':common.fp64_samples(outputs[None],outputs[index],actual_x,actual_w,
                                                                       np.float32(sx),np.float32(sw),np.zeros(columns,dtype=np.float32),
                                                                       args.fp64_samples,9060+mutation)})
          del outputs,actual_x,actual_w
        report['benchmarks'].append(stage)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        del graphs,parameters
        gc.collect()
      del row_inputs,packed_inputs,base,w,packed_w,ones,bias,zero
      gc.collect()
    report['abi_adapter_restoration_expected_on_exit'] = True
  report['abi_adapter_restored'] = True


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',required=True,type=Path)
  parser.add_argument('--inspect',action='store_true')
  parser.add_argument('--checkpoint',type=Path,default=common.CHECKPOINT)
  parser.add_argument('--prefill-calibration',type=Path,default=common.CALIBRATION)
  parser.add_argument('--replays',type=int,default=5)
  parser.add_argument('--fp64-samples',type=int,default=512)
  args = parser.parse_args()
  if not 3 <= args.replays <= 9 or not 1 <= args.fp64_samples <= 1024:
    parser.error('Bounded experiment requires 3-9 graph replays and 1-1024 FP64 samples')
  args.output.parent.mkdir(parents=True,exist_ok=True)
  report = {'provenance':common.provenance(args.checkpoint,args.prefill_calibration),'benchmarks':[],
            'acceptance':'NUMERICAL experiment. No strict arithmetic parity claim or altered model-quality threshold.'}
  if args.inspect:
    inspect(report)
  else:
    gpu_benchmark(args,report)
  args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
  main()

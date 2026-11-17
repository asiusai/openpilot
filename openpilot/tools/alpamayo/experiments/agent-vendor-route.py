"""Numerical hipBLASLt candidate for fixed-scale LANGUAGE PREFILL MLPs only.

The parent owns microbenchmark review, route comparisons, and adoption.
This program preserves the existing packed quantizer and every target check.
"""
import argparse
from contextlib import contextmanager,ExitStack
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import MethodType
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('agent_vendor_common',Path(__file__).with_name('agent-vendor-common.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


@contextmanager
def install_prefill(model,packed_type,vendor,fc1_solution=28279,fc2_solution=27854):
  from tinygrad import Tensor,dtypes
  if fc1_solution not in common.SOLUTIONS or fc2_solution not in common.SOLUTIONS:
    raise ValueError('Only reviewed solutions 27854 and 28279 are supported')
  original_linear,original_layer = model.linear,model.text_layer
  weights,biases,ones,zero = {},{},{},{}
  eligible = {}
  for name,weight in model.weights.items():
    if not name.startswith('vlm.model.language_model.layers.') or not name.endswith('.weight'):
      continue
    linear = name[:-7]
    expected = (9216,2048) if linear.endswith('.mlp.fc1') else (2048,9216) if linear.endswith('.mlp.fc2') else None
    if expected is None:
      continue
    if weight.shape != expected or weight.dtype != dtypes.fp8e4m3 or name not in model.weight_scales:
      raise ValueError(f'Unsupported numerical prefill weight: {name}')
    # model.weights stores a logical unpacked view of packed FP8 bulk storage.
    # Materialize that logical row-major view once. Never pass packed bytes as row-major.
    weights[linear] = common.materialize_packed_weight(weight)
    columns = weight.shape[0]
    bias = model.weights.get(linear+'.bias')
    biases[linear] = (Tensor.zeros(columns,device=weight.device,dtype=dtypes.float32) if bias is None else bias.float()).contiguous().realize()
    if columns not in ones:
      ones[columns] = Tensor.ones(columns,device=weight.device,dtype=dtypes.float32).contiguous().realize()
      zero[columns] = Tensor.zeros(2112,columns,device=weight.device,dtype=dtypes.bfloat16).contiguous().realize()
    eligible[linear] = fc1_solution if expected == (9216,2048) else fc2_solution
  if not eligible:
    raise ValueError('No eligible language MLP weights')
  state = {'prefill':False}
  stats = {'classification':'NUMERICAL; never exact','scope':'Language prefill fc1/fc2 only, M=2112',
           'installed_linears':eligible,'rowmajor_weight_bytes':sum(weight.numel()*weight.dtype.itemsize for weight in weights.values()),
           'python_trace_build_calls':{},
           'quantization':'Existing PackedActivation producer unchanged; exact unpack permutation; scalar scales passed once to vendor GEMM.',
           'quality_status':'UNVALIDATED. Parent must run its paired 64-window and full-route quality gates.'}
  def linear(self,x,name,quantized_input=None,heads=None):
    rows,width = x.numel()//x.shape[-1],x.shape[-1]
    if not state['prefill'] or name not in eligible:
      return original_linear(x,name,quantized_input,heads)
    if rows != 2112 or x.dtype != dtypes.bfloat16:
      raise ValueError('Numerical prefill dispatch requires padded 2112-row BF16 inputs')
    packed,scale = self.quantize(x) if quantized_input is None else quantized_input
    if not isinstance(packed,packed_type) or (packed.rows,packed.width) != (rows,width):
      raise TypeError('Expected the existing quantizer PackedActivation; refusing an ambiguous layout')
    if packed.value.dtype != dtypes.fp8e4m3 or packed.value.shape != (2112,width):
      raise ValueError('Invalid packed activation extent or dtype')
    rowmajor = common.unpack_activation(packed.value,rows)
    weight = weights[name]
    columns = weight.shape[0]
    # Vendor applies activation and weight scales. Do not dequantize first or rescale its output.
    value = vendor.gemm(rowmajor,weight,(scale,self.weight_scales[name+'.weight']),
                        ones[columns],biases[name],zero[columns],index=eligible[name])
    calls = stats['python_trace_build_calls']
    calls[name] = calls.get(name,0)+1
    return value.reshape(*x.shape[:-1],columns)
  def text_layer(self,x,*args,**kwargs):
    prefix = args[2] if len(args) >= 3 else kwargs['prefix']
    position = kwargs.get('cache_pos',args[5] if len(args) >= 6 else None)
    previous = state['prefill']
    state['prefill'] = (prefix.startswith('vlm.model.language_model.layers.') and x.shape[1] >= 128
                        and isinstance(position,int) and position == 0)
    try:
      return original_layer(x,*args,**kwargs)
    finally:
      state['prefill'] = previous
  model.linear,model.text_layer = MethodType(linear,model),MethodType(text_layer,model)
  try:
    yield stats
  finally:
    model.linear,model.text_layer = original_linear,original_layer


def value(argv,flag,default=None):
  return argv[argv.index(flag)+1] if flag in argv else default


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',required=True,type=Path)
  parser.add_argument('--invocation',required=True,type=Path)
  parser.add_argument('--micro-report',required=True,type=Path,help='Completed parent 40-call report; integer controls must pass')
  parser.add_argument('--fc1-solution',type=int,choices=common.SOLUTIONS,default=28279)
  parser.add_argument('--fc2-solution',type=int,choices=common.SOLUTIONS,default=27854)
  parser.add_argument('--runs',type=int)
  parser.add_argument('--warmup',type=int)
  args = parser.parse_args()
  saved = json.loads(args.invocation.read_text())
  command = saved['command'][1:]
  if Path(command[0]).resolve() != Path(__file__).with_name('benchmark-pipeline.py').resolve():
    raise ValueError('Only the adjacent pinned benchmark pipeline is supported')
  if '--packed-quantize' not in command or not value(command,'--prefill-calibration') or value(command,'--pad-prompt-to') != '2112':
    raise ValueError('Numerical route test requires the original fixed-scale packed quantizer and padding to 2112')
  if '--fused-qkv' in command or '--row-quantize' in command:
    raise ValueError('This MLP-only candidate does not implement fused or per-row scale protocols')
  if '--validation-reference' in command:
    raise ValueError('The saved strict arithmetic validation reference is incompatible with a numerical candidate; use the parent quality fixture invocation')
  micro = json.loads(args.micro_report.read_text())
  controls = micro.get('integer_controls',[])
  if {item['solution'] for item in controls if item.get('integer_fp64_oracle_exact')} != set(common.SOLUTIONS):
    raise ValueError('Both reviewed solutions need completed independent integer controls')
  if len(micro.get('benchmarks',[])) != 6 or not micro.get('abi_adapter_restored'):
    raise ValueError('The bounded microbenchmark or ABI cleanup did not complete')
  for key,setting in saved['environment'].items():
    if setting is None:
      os.environ.pop(key,None)
    else:
      os.environ[key] = setting
  for path in reversed(os.environ.get('PYTHONPATH','').split(os.pathsep)):
    if path:
      sys.path.insert(0,path)
  sys.path.insert(0,str(Path(__file__).parent))
  checkpoint,calibration = value(command,'--checkpoint'),value(command,'--prefill-calibration')
  provenance = common.provenance(checkpoint,calibration)
  if micro['provenance']['vendor_artifacts'] != provenance['vendor_artifacts']:
    raise ValueError('Vendor artifacts changed since the microbenchmark')
  performance = {}
  for stage in micro['benchmarks']:
    if stage['boundary'] != 'packed_activation_bridge':
      continue
    shape = tuple(stage['shape_m_n_k'])
    solution = args.fc1_solution if shape == (2112,9216,2048) else args.fc2_solution if shape == (2112,2048,9216) else None
    if solution is None:
      continue
    measured = {entry['solution']:entry['median_ms_per_gemm'] for entry in stage['methods']}
    performance[str(shape)] = {'solution':solution,'baseline_ms':measured[None],'candidate_ms':measured[solution],
                               'hypothesis_supported':measured[solution] < measured[None]}
  if len(performance) != 2:
    raise ValueError('Missing target-shape bridge timings')
  # Performance is reported, never converted into an arithmetic or quality gate.
  command[command.index('--output')+1] = str(args.output)
  for name,setting in (('runs',args.runs),('warmup',args.warmup)):
    if setting is not None:
      if setting < (1 if name == 'runs' else 0):
        raise ValueError('Invalid bounded prediction count')
      flag = '--'+name
      if flag in command:
        command[command.index(flag)+1] = str(setting)
      else:
        command += [flag,str(setting)]
  if args.output.exists():
    raise FileExistsError('Use a new route output path')
  args.output.parent.mkdir(parents=True,exist_ok=True)
  pipeline = common.module('benchmark-pipeline')
  original_loader = pipeline.local_module
  installed = []
  with common.runtime() as (device,vendor), ExitStack() as stack:
    def load(name):
      result = original_loader(name)
      if name == 'quantize-packed':
        original_install = result.install
        @functools.wraps(original_install)
        def install(model,*values,**kwargs):
          fixed = kwargs.get('fixed_scales',values[0] if values else None)
          if not fixed:
            raise ValueError('The numerical candidate requires explicit fixed prefill scales')
          count = original_install(model,*values,**kwargs)
          installed.append(stack.enter_context(install_prefill(model,result.PackedActivation,vendor,args.fc1_solution,args.fc2_solution)))
          return count
        result.install = install
      return result
    stack.enter_context(patch.object(pipeline,'local_module',load))
    stack.enter_context(patch.object(sys,'argv',command))
    pipeline.main()
    if len(installed) != 1 or not installed[0]['python_trace_build_calls']:
      raise AssertionError('Language prefill vendor dispatch was never traced')
    report = json.loads(args.output.read_text())
    report['vendor_numerical_candidate'] = {'provenance':provenance,'micro_report_sha256':common.sha(args.micro_report),
                                            'performance_hypothesis':performance,'dispatch':installed[0],
                                            'resolved_command':command,
                                            'unchanged_paths':['vision','expert','serial decode','draft target verification','quantization rounding'],
                                            'required_parent_gates':['paired 64-window ADE/FDE','full-route ADE/FDE','trajectory shape','speed','slowing'],
                                            'quality_passed':None,'classification':'NUMERICAL; never relabel as exact'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
  # ABI cleanup completes after every model call and candidate method restoration.
  report = json.loads(args.output.read_text())
  report['vendor_numerical_candidate']['abi_adapter_restored'] = True
  args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
  main()

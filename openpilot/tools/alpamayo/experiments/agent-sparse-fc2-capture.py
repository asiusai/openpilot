#!/usr/bin/env python3
"""Parent-only one-query ORIGINAL fc2 capture. --prepare/--preflight are CPU only."""
import argparse
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import sys
from types import MethodType
from unittest.mock import patch
import numpy as np
import importlib.util

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('sparse_fc2_common',HERE/'agent-sparse-fc2-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


class Capture:
  def __init__(self,model,binding,output):
    self.model,self.binding,self.original,self.output=model,binding,model.linear,output
    outer=inspect.getclosurevars(self.original.__func__).nonlocals
    inner=inspect.getclosurevars(outer['old_linear'].__func__).nonlocals
    self.packed_weights=inner['weights']
    self.records={};self.valid=None;self.bytes=0
    def linear(this,x,name,quantized_input=None,heads=None):
      prefix=name.removesuffix('.mlp.fc2')
      layer=int(prefix.rsplit('.',1)[-1]) if prefix.startswith('vlm.model.language_model.layers.') and name.endswith('.mlp.fc2') else None
      if layer not in C.LAYERS or not binding.state['prefill'] or x.numel()//x.shape[-1]<128:
        return self.original(x,name,quantized_input,heads)
      if layer in self.records or quantized_input is not None or binding.state['prefix']!=prefix or self.valid is None:
        raise ValueError('Capture must observe each declared original prefill fc2 exactly once')
      if (x.numel()//x.shape[-1],this.weights[name+'.weight'].shape[0],x.shape[-1])!=C.SHAPE:
        raise ValueError('Actual fc2 shape differs from predeclared 2112x2048x9216')
      if name+'.bias' in this.weights:raise ValueError('Original selected language fc2 must have no bias')
      quantized=binding.quantize_once(x)
      packed,scale=quantized
      from tinygrad import dtypes
      w=self.packed_weights[name];ws=this.weight_scales[name+'.weight']
      if packed.value.dtype!=w.dtype or packed.value.dtype!=dtypes.fp8e4m3 or scale.dtype!=ws.dtype or ws.numel()!=1:
        raise ValueError('Original consumed tensor dtype/scale changed')
      # Read-only diagnostics. Original numerical operands and original quantized
      # tuple enter the original linear implementation without modification.
      a_bytes=packed.value.bitcast(dtypes.uint8).numpy()
      w_bytes=w.bitcast(dtypes.uint8).numpy()
      xs=scale.numpy().reshape(1);wws=ws.numpy().reshape(1)
      if np.any((a_bytes&127)==127) or np.any((w_bytes&127)==127) or not np.isfinite(xs).all() or not np.isfinite(wws).all():
        raise ValueError('Capture requires finite original FP8 operands/scales')
      value=self.original(x,name,quantized,heads).realize()
      if value.dtype!=dtypes.bfloat16:raise ValueError('Original fc2 output must remain BF16')
      bits=value.bitcast(dtypes.uint16).numpy().reshape(C.SHAPE[:2])
      arrays=dict(activation_packed=a_bytes,weight_packed=w_bytes,activation_scale=xs,weight_scale=wws,original_bf16=bits)
      record=dict(layer=layer,name=name,shape=list(C.SHAPE),actual_prompt_rows=self.valid,
        input_layout='amd_gemm.pack',weight_layout='amd_gemm.pack',output_layout='row-major BF16 bits',
        original_quantizer_calls=1,actual_fp8_tensor_consumed=True,files={})
      for key,array in arrays.items():
        array=np.ascontiguousarray(array)
        self.bytes+=array.nbytes
        if self.bytes>C.PAYLOAD:raise ValueError('Capture payload bound exceeded')
        path=self.output/f'layer-{layer:02d}-{key}.npy'
        with path.open('xb') as stream:np.save(stream,array,allow_pickle=False)
        path.chmod(0o444)
        record['files'][key]=dict(path=str(path),sha256=C.sha(path),shape=list(array.shape),dtype=array.dtype.str,bytes=path.stat().st_size)
      self.records[layer]=record
      return value
    model.linear=MethodType(linear,model)

  def restore(self):self.model.linear=self.original


def execute(args,plan):
  if args.output.exists():raise FileExistsError('Use a new capture output')
  overlay,runner,root=C.verify_plan(plan)
  disk=C.guard_disk(args.output)
  if '--vision-calibration' in plan['baseline_plan']['numerical_options']:
    raise ValueError('Capture requires the declared original dynamic-vision recipe')
  for key,value in plan['baseline_plan']['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  for path in reversed(os.environ['PYTHONPATH'].split(':')):sys.path.insert(0,path)
  pipeline=runner.local('benchmark-pipeline')
  opts=dict(plan['baseline_plan']['numerical_options'])
  opts.update({'--samples':[C.SAMPLE],'--warmup':['0'],'--runs':['1'],'--output':[str(args.output.with_suffix('.pipeline.json'))]})
  argv=[str(root/'benchmark-pipeline.py'),*[item for key,values in opts.items()for item in (key,*values)]]
  directory=args.output.parent/'tensors'
  C.guard_disk(directory);directory.mkdir(exist_ok=False)
  report=dict(schema='agent-sparse-fc2-capture-v1',complete=False,paired_prediction_bits_exact=False,
    plan=plan,plan_identity=plan['identity_sha256'],sample=C.SAMPLE,layers=list(C.LAYERS),disk=disk,
    timing_diagnostic_only=True,queries=[],records=[])
  capture=None
  try:
    with ExitStack() as stack:
      original_loader=pipeline.local_module
      def local(name):
        module=original_loader(name)
        if name=='fast-inputs':
          original=module.FastInputs.prepare
          def prepare(adapter,data):
            config,tokenizer,prepared=original(adapter,data)
            return config,tokenizer,dict(prepared,_activation_sparsity_source={
              key:data[key]for key in ('camera_indices','absolute_timestamps','nav_text')})
          stack.enter_context(patch.object(module.FastInputs,'prepare',prepare))
        return module
      stack.enter_context(patch.object(pipeline,'local_module',local))
      initialize=pipeline.initialize
      def init(*a,**kw):
        nonlocal capture
        device_guard=runner.local('agent-activation-sparsity-device')
        components=runner.initialize_checked(initialize,device_guard,report,*a,**kw)
        model=components.model
        if model.padded_prefill.jit.cnt!=0 or model.cache is not None:raise ValueError('Capture requires fresh model/decode/cache state')
        binding=overlay.Binding(model,root)
        capture=Capture(model,binding,directory);stack.callback(capture.restore)
        original=components.runner
        def run(model,settings,config,tokenizer,prepared,packed,seed,inputs):
          if report['queries']:raise ValueError('Only the single predeclared original query is authorized')
          capture.valid=int(prepared['actual_prompt_length'])
          prepared_id=runner.prepared_identity(prepared,packed)
          if prepared_id!=plan['original_query']['prepared']:raise ValueError('Actual IDs/pixels/history/timestamps/navigation changed')
          prediction=original(model,settings,config,tokenizer,prepared,packed,seed,inputs)
          identity=runner.prediction_identity(prediction)
          if identity!=plan['original_query']['prediction']:raise ValueError('Original paired prediction bits differ')
          report['queries'].append(dict(sample=C.SAMPLE,prepared=prepared_id,prediction=identity))
          report['paired_prediction_bits_exact']=True
          return prediction
        components.runner=run
        report['quantizer_binding']=binding.evidence
        return components
      stack.enter_context(patch.object(pipeline,'initialize',init))
      stack.enter_context(patch.object(sys,'argv',argv))
      pipeline.main()
      if capture is None or set(capture.records)!=set(C.LAYERS) or len(report['queries'])!=1:
        raise ValueError('Incomplete one-query three-layer original capture')
      report['records']=[capture.records[layer]for layer in C.LAYERS]
      report['raw_payload_bytes']=capture.bytes
      if capture.bytes!=C.PAYLOAD:raise ValueError('Unexpected capture payload size')
      report['complete']=True
  except BaseException as error:
    report['error']=dict(type=type(error).__name__,message=str(error));raise
  finally:
    if capture is not None:report['records']=[capture.records[layer]for layer in C.LAYERS if layer in capture.records]
    C.write(args.output,report)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path)
  p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true')
  p.add_argument('--parent-gpu',action='store_true');a=p.parse_args()
  if a.prepare:
    if a.output.exists():raise FileExistsError('Use a new sparse-fc2 plan')
    C.write(a.output,C.prepare_plan());print('CPU-only source/input/model plan prepared');return
  if a.plan is None:p.error('--plan is required')
  plan=json.loads(a.plan.read_text())
  if a.preflight:C.verify_plan(plan);print('CPU-only sparse-fc2 provenance verified');return
  if not a.parent_gpu:p.error('Only the parent may run the GPU capture')
  execute(a,plan)


if __name__=='__main__':main()

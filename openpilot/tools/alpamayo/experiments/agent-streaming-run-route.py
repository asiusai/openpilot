"""Parent-only, bounded serial correctness replay. No future labels or draft reuse."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stream_route_common',HERE/'agent-streaming-run-common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode',choices=('baseline','forced-full','streaming'),required=True)
  parser.add_argument('--limit',type=int,default=32)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--baseline',type=Path)
  parser.add_argument('--full-control',type=Path)
  parser.add_argument('--parent-gpu',action='store_true')
  parser.add_argument('--dry-run',action='store_true')
  options=parser.parse_args()
  if not 1<=options.limit<=32:parser.error('Bounded chronological prefix: 1..32')
  common.verify_sources();sources=common.sources()
  paths=json.loads((common.ROOT/'smoke32.json').read_text())[:options.limit]
  if options.dry_run:
    print(json.dumps(dict(mode=options.mode,samples=paths,configuration=str(common.ROOT/'configuration.json'),
      prototype_sources=sources,classification='optional approximate recurrent correctness trial; GPU not executed'),indent=2));return
  if not options.parent_gpu:parser.error('GPU execution belongs to the parent; use --parent-gpu under flock and timeout')
  if options.output.exists() or options.output.with_suffix('.jsonl').exists():parser.error('Use a new result path')
  common.parent_gpu_environment()
  if options.limit>4:
    bounded=common.validate_report(common.ROOT/'streaming4.json','route-streaming')
    if len(bounded['runs'])!=4 or not any(r['streaming']['reused_images'] for r in bounded['runs']):
      raise ValueError('The four-input recurrent trial must pass before a larger smoke')
  baseline=None
  if options.mode!='baseline':
    if options.baseline is None:parser.error('Paired baseline required')
    baseline=common.validate_report(options.baseline,'route-baseline')
    if [r['sample'] for r in baseline['runs']]!=paths:raise ValueError('Exact paired sample sequence required')
  if options.mode=='streaming':
    if options.full_control is None:parser.error('Passing forced-full control required')
    control=common.validate_report(options.full_control,'route-forced-full')
    if not control.get('exact_control_passed') or not control.get('cache_control_exact') or len(control['runs'])<4:
      raise ValueError('At least four exact output/logit/active-KV full-refresh controls required')
  import numpy as np
  pipeline=common.load('benchmark-pipeline',common.EXP)
  runtime=common.load('agent-streaming-run-runtime')
  kernels=common.load('agent-streaming-run-kernels')
  original_load=pipeline.local_module;owners=[]
  def local_module(name):
    module=original_load(name)
    if name=='benchmark-fast' and options.mode!='baseline':
      import benchmark
      original_model=benchmark.EdgeModel
      def create(*a,**kw):
        model=original_model(*a,**kw);owners.append(runtime.StreamingPrefill(model));return model
      benchmark.EdgeModel=create
    if name=='online-attention':
      install=module.install
      def selected_install(model,*a,**kw):
        result=install(model,*a,**kw)
        kernels.selected().install(model,target='prefill',transpose='shuffle-fma',waves=8)
        return result
      module.install=selected_install
    return module
  pipeline.local_module=local_module
  arguments=json.loads((common.ROOT/'configuration.json').read_text())['args']
  args=pipeline.parse_args(arguments);args.samples=paths;args.output=options.output;args.runs=len(paths)
  context=pipeline.initialize(args,sources=sources,source_dir=common.ROOT)
  owner=owners[0] if owners else None
  if owner:owner.install_padded()
  capture_control=options.mode=='forced-full' or options.mode=='baseline' and options.limit<=4
  current_cache_digest=[None]
  if capture_control:
    from tinygrad import dtypes
    original_prefill=context.model.padded_prefill
    def prefill_control(x,cos,sin,max_tokens,length):
      logits=original_prefill(x,cos,sin,max_tokens,length)
      def digest(value):return hashlib.sha256(value.contiguous().bitcast(dtypes.uint8).numpy().tobytes()).hexdigest()
      current_cache_digest[0]=dict(logits=digest(logits),active_kv=[digest(cache[:,:,:,:length]) for cache in context.model.cache])
      return logits
    context.model.padded_prefill=prefill_control
  report=common.report_base('route-'+options.mode)
  report.update(passed=False,mode=options.mode,classification='approximate recurrent candidate' if owner else 'serial paired baseline',
    quality_passed=None,full_pipeline_qualified=False,serial_reasoning=True,nominal_draft_verified=False,
    diagnostic_prefill_readback=capture_control,
    timing_boundary='sample RGB load, CPU preparation, upload, current inference, decoded path, bank commit; excludes camera capture/ISP/video decode',
    startup_and_jit_included=True,original_configuration=context.report,runs=[])
  options.output.parent.mkdir(parents=True,exist_ok=True)
  options.output.write_text(json.dumps(report,indent=2)+'\n')
  inputs=None;provenance=common.sha(common.ROOT/'sources.json')
  try:
    for index,path in enumerate(paths):
      start=time.perf_counter();data=common.read_sample(path);loaded=time.perf_counter()
      config,tokenizer,prepared=context.adapter.prepare(data)
      packed=context.adapter.native_inputs(prepared)
      prepared,packed=context.pad_inputs(prepared,packed,2048)
      context.vision_cache.prepare(prepared)
      meta=common.metadata(data,prepared,path,context.vision_cache.keys,provenance)
      if inputs is None or not inputs.update(context.settings,config,prepared,packed):
        if inputs is not None:
          inputs.upload_jit.reset()
          for graph in inputs.selective_jits.values():graph.reset()
        inputs=context.input_class(context.settings,config,prepared,packed)
      # Bound selective-upload graph owners independently of the 48-image bank.
      while len(inputs.selective_jits)>=8:
        key=next(iter(inputs.selective_jits));inputs.selective_jits.pop(key).reset()
      inputs.selective_cache=context.vision_cache
      if owner:owner.prepare(meta,inputs.spans,prepared['actual_prompt_length'],force_full=options.mode=='forced-full')
      staged=time.perf_counter()
      prediction=context.runner(context.model,context.settings,config,tokenizer,prepared,packed,seed=0,inputs=inputs)
      inferred=time.perf_counter();event=owner.commit() if owner else None;done=time.perf_counter()
      entry=dict(sample=path,query_timestamp_us=meta['query_us'],image_clock_us=meta['timestamp_us'],
        load_ms=(loaded-start)*1000,prepare_upload_buffers_ms=(staged-loaded)*1000,
        inference_ms=(inferred-staged)*1000,commit_ms=(done-inferred)*1000,pipeline_ms=(done-start)*1000,
        prediction=prediction,streaming=event,vision_hits=context.vision_cache.hits,vision_misses=context.vision_cache.misses,
        prefill_control=current_cache_digest[0],
        uploaded_pixel_bytes=inputs.uploaded_pixel_bytes,allocated_bytes_by_device=dict(context.GlobalCounters.mem_used_per_device),
        graph_counts=dict(selective_upload=len(inputs.selective_jits),vision_stitch=len(context.vision_cache.stitchers),
          vision_encode=len(context.vision_cache.encoders),trajectory=len(context.model.trajectory_jits),
          decoder=len(context.runner.__self__.jits),sparse=0 if owner is None else len(owner.sparse_jits)))
      assert entry['vision_hits']+entry['vision_misses']==10
      if baseline:
        expected=baseline['runs'][index]['prediction']
        delta=np.abs(np.asarray(prediction['xyz'])-np.asarray(expected['xyz']))
        exact=(prediction['generated_token_ids']==expected['generated_token_ids'] and
          np.array_equal(prediction['xyz'],expected['xyz']) and np.array_equal(prediction['rot'],expected['rot']))
        entry['paired_difference']=dict(exact=exact,tokens_equal=prediction['generated_token_ids']==expected['generated_token_ids'],
          max_xyz_m=float(delta.max()),mean_xyz_m=float(delta.mean()))
        if options.mode=='forced-full':
          cache_exact=(entry['prefill_control'] is not None and entry['prefill_control']==baseline['runs'][index]['prefill_control'])
          entry['paired_difference']['prefill_logits_and_active_kv_exact']=cache_exact
          if not exact or not cache_exact:raise ValueError('Forced-full changed original output/cache: '+str(entry['paired_difference']))
      report['runs'].append(entry)
      with options.output.with_suffix('.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
      print(json.dumps(dict(index=index,sample=path,pipeline_ms=entry['pipeline_ms'],streaming=event)),flush=True)
    report['passed']=True
    report['exact_control_passed']=options.mode=='forced-full'
    report['cache_control_exact']=options.mode=='forced-full'
    totals=[r['pipeline_ms'] for r in report['runs']]
    report['serial_correctness_timing_ms']=dict(median=float(np.median(totals)),p95=float(np.percentile(totals,95)),maximum=max(totals))
  except BaseException as error:
    if owner:owner.abort()
    report['failure']=repr(error)
    raise
  finally:options.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

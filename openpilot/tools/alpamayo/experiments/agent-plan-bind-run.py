"""Parent-only metadata capture for one actual serial count1 Alpamayo plan.

Uses the frozen factory, original speculative launcher, and first current input.
No full64 replay, full startup, stored tensor data, or persistent-plan execution.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
QUALIFIED=WORK/'agent-startup-key-fix1-replay/report.json'
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PINS={
 'tinygrad/engine/jit.py':'bdf5f3029e9e98faaddb0e83a51296189e7cf6ebf845d84fb46b8414f2a9424b',
 'tinygrad/engine/realize.py':'d8229adf90f86e8f21f2e9b6081d6eb0604c42058faa52cde4826c55cf08292f',
 'tinygrad/runtime/support/hcq2.py':'dbab7aef4f8c226d43ca5d138381d5d757a9d8683dfca381ac32e6d826734480',
 'tinygrad/uop/ops.py':'357ce1fd8b9cc3d04789aacbed3480944aeb11e8b280c5fd80cf57839a8cb16d',
 'tinygrad/schedule/memory.py':'5dd7860dda424845af5111329de84a15e62a078f14e99733f36828bfa3270631',
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
  return value


def under_work(path):
  path=Path(path).resolve()
  if not path.is_relative_to(WORK): raise ValueError('All generated artifacts and caches must remain under WORK')
  return path


def first_dataset(replay,qualified):
  """Verify metadata and load only row zero. Other63 image arrays are never read."""
  import numpy as np
  samples_path=WORK/'quality-samples.json';manifest_path=replay.SOURCE/'manifest.json'
  provenance=qualified['protocol']['dataset']
  if sha(samples_path)!=replay.SAMPLES_SHA256 or sha(manifest_path)!=replay.MANIFEST_SHA256:
    raise ValueError('Recorded sample/manifest identity changed')
  samples=json.loads(samples_path.read_text());manifest=json.loads(manifest_path.read_text())
  if len(samples)!=64 or len(set(samples))!=64 or samples[0]!=qualified['rows'][0]['sample']:
    raise ValueError('Qualified current sample changed')
  by_path={row['sample']:row for row in manifest['schedule'] if row.get('sample') in set(samples)}
  rows=[{key:copy.deepcopy(by_path[sample][key]) for key in
    ('sample','query_ns','frames','navigation_text','ego_history_xyz','ego_history_rot','vehicle_history_audit')} for sample in samples]
  if replay.digest(rows)!=provenance['metadata_sha256']: raise ValueError('Recorded input metadata changed')
  calibration=manifest_path.with_name('render-calibrations.json');export=Path(manifest['export'])
  if sha(calibration)!=provenance['calibration_sha256'] or sha(export/'clips/native-timestamps.json')!=provenance['native_timestamps_sha256']:
    raise ValueError('Camera calibration or capture metadata changed')
  from calibration_trials import asof
  from camera_profiles import replay_module
  calibrations=json.loads(calibration.read_text());_,cameras=replay_module().camera_records(export)
  native={c:(np.asarray([r['capture_ns'] for r in cameras[c]],np.int64),cameras[c]) for c in (1,6)}
  replay.Dataset.validate_native(rows[0],native,(np.asarray([r['ns'] for r in calibrations],np.int64),calibrations),asof)
  return SimpleNamespace(samples=samples,provenance=copy.deepcopy(provenance),rows=rows[:1],load=replay.Dataset.load)


def capture_one(c,args,arrays,report,output,expected,reference,replay):
  import numpy as np
  import torch
  from tinygrad import Tensor,dtypes
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.uop.ops import UOp
  audit=local(ROOT/'agent-plan-bind-audit.py')
  warm=local(WORK/'startup-cache-parent-source/agent-startup-spec-warm.py')
  if sha(warm.__file__)!='3b7f2d7d5e1888243271fa34b7bb8f91d7b7f7c5dd3789503bec1071a3e2900c':
    raise ValueError('Qualified restoration helper changed')
  model,decoder=c.model,c.decoder
  if c.inputs is not None or decoder.jits or c.startup is not None or any(c.startup_active):
    raise ValueError('Require a fresh current-input factory before any serial graph exists')
  if args.vision_calibration is not None or args.pad_prompt_to!=2048 or args.max_tokens!=64:
    raise ValueError('Require the qualified dynamic-vision pad2048 max64 recipe')
  before_history=copy.deepcopy({name:getattr(decoder,name) for name in ('history','statistics','prefill_draft_records')})
  data={k:list(v) if k=='nav_text' else torch.from_numpy(v.copy()) for k,v in arrays.items()}
  config,tokenizer,prepared=c.adapter.prepare(data)
  if config is not c.config or tokenizer is not c.tokenizer: raise AssertionError('Initialized owners changed')
  packed=c.adapter.native_inputs(prepared)
  prepared,packed=c.pad_inputs(prepared,packed,args.pad_prompt_to)
  if not prepared.get('prefill_draft_eligible') or not np.all(prepared['tokenized_data']['image_grid_thw'].numpy()==(1,20,36)):
    raise ValueError('First current input has an unreviewed layout')
  c.vision_cache.prepare(prepared)
  owner=c.input_class(c.settings,config,prepared,packed);owner.selective_cache=c.vision_cache;c.inputs=owner
  if len(owner.batches)!=10: raise ValueError('Require ten complete current front-camera history images')
  if replay.tree(prepared)!=reference['prepared'] or replay.tree(packed)!=reference['packed']:
    raise AssertionError('Actual first prepared/packed model input differs from the qualified input')
  pixels,positions,ids,cos,sin,dc,ds,*_=owner.upload(c.settings.initial_noise)
  embedding=model.encode(pixels,positions,ids,owner.batches,owner.spans)
  prompt=int(prepared['actual_prompt_length'])
  prefill=lambda:model.padded_prefill(embedding,cos,sin,c.settings.max_tokens,prompt)
  logits=prefill();Tensor.realize(logits,dc,ds)
  model.synchronize()
  snapshots=[t.clone().realize() for t in model.cache]
  first=logits.clone().realize();payload=model.padded_prefill.last_targets.clone().realize()
  def identity(): return tuple(id(t._buffer().base) for t in model.cache)
  allocation=identity()
  def exact(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype: raise AssertionError('State shape/dtype changed')
    dt={1:dtypes.uint8,2:dtypes.uint16,4:dtypes.uint32}[a.dtype.itemsize]
    if (a.bitcast(dt)!=b.bitcast(dt)).any().item(): raise AssertionError('State bits changed')
  def reset_kv():
    for target,saved in zip(model.cache,snapshots,strict=True): target.assign(saved).realize()
    model.synchronize()
  key=(dc.shape,1,model.cache[0].shape[3]);record=None
  report.update(current_inputs=1,future_inputs=0,prompt=prompt,count=1,cos_shape=list(dc.shape),cache_shape=list(model.cache[0].shape),
    snapshot_bytes=sum(t.numel()*t.dtype.itemsize for t in snapshots),predictions_published=0,forced_tokens=0,
    first_prepared_and_packed_exact=True,other63_image_arrays_read=False)
  try:
    # Normal eager pass retains original compiler/cache history and provides a reference.
    eager=decoder.decode(logits,dc,ds,prompt,0,1)
    Tensor.realize(*eager);model.synchronize()
    eager_saved=tuple(t.clone().realize() for t in eager)
    reset_kv()
    pos=UOp.variable('block_position',0,model.cache[0].shape[3]-1).bind(prompt)
    step=UOp.variable('block_step',0,dc.shape[2]-1).bind(0)
    input_uops,*_=_prepare_jit_inputs((logits,dc,ds,pos,step),{})
    with audit.capture_boundary(input_uops) as record:
      captured=decoder.decode(logits,dc,ds,prompt,0,1)
    Tensor.realize(*captured);model.synchronize()
    for a,b in zip(captured,eager_saved,strict=True): exact(a,b)
    reset_kv()
    replayed=decoder.decode(logits,dc,ds,prompt,0,1);Tensor.realize(*replayed)
    for a,b in zip(replayed,eager_saved,strict=True): exact(a,b)
    # The ordinary serial fallback feeds its own output back as an input.
    aliased_logits=replayed[0]
    Tensor.realize(*decoder.decode(aliased_logits,dc,ds,prompt,0,1))
    model.synchronize()
    jit=decoder.jits[key]
    registry=audit.model_registry(model,(logits,dc,ds),jit.captured.ret,
      bulk_index=Path(os.environ['ALPAMAYO_FP8_CACHE'])/'index.json',vocab=args.reduced_vocab,checkpoint=args.checkpoint)
    report['capture']=audit.export_capture(record,jit,registry,output.parent/'agent-plan-bind-capture',expected)
    report.update(eager_capture_replay_bits_equal=True,linked_alias_variants=len(jit.captured._linears),serial_decoder_calls=4,
                  python_model_executions=2,
                  persistent_plan_executed=False,poison_reload_proved=False)
  finally:
    restored=warm.restore_prefill(model.cache,snapshots,prefill,identity=identity,equal=exact,
                                 copy_into=lambda target,saved:target.assign(saved).realize())
    exact(restored,first);exact(model.padded_prefill.last_targets,payload);model.synchronize()
    if identity()!=allocation or c.inputs is not owner: raise AssertionError('Current input or KV owner changed')
    after_history={name:getattr(decoder,name) for name in before_history}
    if after_history!=before_history: raise AssertionError('Diagnostic accepted a suffix or changed generation history')
    report.update(full_kv_bits_restored=True,actual_first_logits_bits_restored=True,target_payload_bits_restored=True,
                  current_owner_retained=True,generation_history_unchanged=True)


def main(argv=None):
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--cache-db',type=Path,required=True)
  parser.add_argument('--execute-gpu',action='store_true')
  cli=parser.parse_args(argv)
  output=under_work(cli.output);cache=under_work(cli.cache_db)
  if output.exists(): raise FileExistsError(output)
  if not cli.execute_gpu: raise ValueError('Parent must explicitly authorize --execute-gpu')
  for path in (os.environ.get('TMPDIR',''),os.environ.get('CACHEDB','')): under_work(path)
  if Path(os.environ['CACHEDB']).resolve()!=cache: raise ValueError('Compiler cache does not match its recorded location')
  for name,expected in PINS.items():
    if sha(TG/name)!=expected: raise ValueError('Pinned plan boundary changed: '+name)
  sys.dont_write_bytecode=True
  sys.path[:0]=[str(SOURCE),str(SOURCE/'experiments')]
  policy=local(SOURCE/'experiments/agent-paced-dynamic-policy.py');policy.verify_snapshot()
  with policy.imports(): replay=local(SOURCE/'experiments/agent-live-replay.py')
  pipeline=local(SOURCE/'experiments/benchmark-pipeline.py')
  qualified=json.loads(QUALIFIED.read_text())
  dataset=first_dataset(replay,qualified)
  normalized,_,protocol=replay.normalize(SOURCE.parent/'pipeline-invocation.json',dataset,pipeline,output.with_suffix('.benchmark.json'))
  if qualified['protocol']!=protocol or qualified['startup_key']['qualified'] is not True:
    raise ValueError('Selected recipe differs from the qualified count1 source')
  expected=qualified['startup_cache']['serial_signatures']['1']
  # Relocate only storage paths. Numeric/runtime target settings stay byte-for-byte unchanged.
  normalized['environment'].update(CACHEDB=str(cache),TMPDIR=os.environ['TMPDIR'],PYTHONDONTWRITEBYTECODE='1')
  invocation=output.with_suffix('.agent-plan-bind-invocation.json')
  with invocation.open('x') as stream: json.dump(normalized,stream,indent=2);stream.write('\n')
  report=dict(schema=1,diagnostic_only=True,qualified=False,executable_plan=False,gpu_execution_requested=True,
    qualified_report_sha256=sha(QUALIFIED),original_invocation_sha256=sha(SOURCE.parent/'pipeline-invocation.json'),
    relocated_invocation_sha256=sha(invocation),protocol=protocol,source_hashes={p.name:sha(p) for p in ROOT.glob('agent-plan-bind-*.py')},
    runtime_sources=PINS,artifact_root=str(output.parent),compiler_cache=str(cache),stages=[])
  report['dataset_contract']='Historical64 provenance identifies the recipe; only the first current input is freshly prepared and compared'
  begin=time.perf_counter()
  def stage(name):
    report['stages'].append(dict(name=name,elapsed_ms=(time.perf_counter()-begin)*1000))
    print(json.dumps({'agent_plan_bind_stage':report['stages'][-1]}),flush=True)
  def run_pipeline(path,run_name):
    if Path(path).resolve()!=SOURCE/'experiments/benchmark-pipeline.py' or run_name!='__main__': raise ValueError('Unexpected launcher target')
    selected=local(path);args=selected.parse_args()
    if args.warm_current_shapes or args.warmup_graphs or args.pipeline_inputs or args.late_inputs: raise ValueError('Unexpected broader warm/prefetch')
    sources,source_dir=selected.snapshot_sources(output.with_suffix('.factory.json'))
    components=selected.initialize(args,sources=sources,source_dir=source_dir);stage('factory_initialized')
    report['factory_runtime_environment']=components.report['runtime_environment']
    if report['factory_runtime_environment']!=qualified['initialized']['runtime_environment']: raise ValueError('Numerical runtime environment changed')
    arrays=dataset.load(dataset.rows[0]);stage('first_current_input_loaded')
    capture_one(components,args,arrays,report,output,expected,qualified['rows'][0],replay);stage('count1_captured_and_real_prefill_restored')
    report['gpu_diagnostic_completed']=True
    return {}
  try:
    launcher=local(SOURCE/'experiments/agent-speculation-run.py')
    argv=[str(SOURCE/'experiments/agent-speculation-run.py'),'--invocation',str(invocation),'--output',str(output.with_suffix('.launcher.json')),
          '--mode','candidate','--warmup','0','--runs','1','--pad-prompt-to','2048','--prefill-attention','shuffle-fma']
    with patch.dict(os.environ),patch.object(sys,'argv',argv),patch.object(sys,'path',list(sys.path)), \
         patch.object(importlib.util,'spec_from_file_location',importlib.util.spec_from_file_location),patch.object(runpy,'run_path',run_pipeline):
      launcher.main()
    if report['capture']['qualified_numerical_identities_equal'] is not True:
      raise AssertionError('Numerical program identity changed; retained metadata is diagnostic evidence only')
  except BaseException as error:
    report['error']=repr(error)
    raise
  finally:
    report['elapsed_ms']=(time.perf_counter()-begin)*1000
    with output.open('x') as stream: json.dump(report,stream,indent=2);stream.write('\n')


if __name__=='__main__': main()

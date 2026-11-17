"""Parent-only four-query exact group4/group2 parity; timings are diagnostic."""
import argparse
import ast
import copy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
from types import MethodType

import numpy as np

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('agent_vision_group4_run_common',HERE/'agent-vision-group4-common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def tensor_bits(tensor):
  from tinygrad import dtypes
  kinds={1:dtypes.uint8,2:dtypes.uint16,4:dtypes.uint32,8:dtypes.uint64}
  data=tensor.bitcast(kinds[tensor.dtype.itemsize]).numpy()
  return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(data.tobytes()).hexdigest())


def key_record(key):return [list(key[0]),key[1].hex()]


def controls(cache,pixels,positions,record,barrier):
  """Use only the first genuine query's first four images and real encoder JITs.

  Input buffers, feature-cache entries, language caches, and history stay unchanged.
  Read hashes only after completed JIT calls, outside captured bodies.
  """
  from tinygrad import Tensor,TinyJit
  originals=[(cache.image_view(pixels,i*720,(i+1)*720),cache.image_view(positions,i*720,(i+1)*720)) for i in range(4)]
  logical_before=[key_record(key) for key in cache.cache]
  for count in (1,2,3,4):
    def encode(pairs):
      result=[]
      for start in range(0,len(pairs),cache.group_limit):
        chunk=pairs[start:start+cache.group_limit];xs=[x for pair in chunk for x in pair]
        Tensor.realize(*xs);shape=tuple(t.shape for t in xs)
        if shape not in cache.encoders:cache.encoders[shape]=TinyJit(cache.batch_encoder)
        result.extend(tensor_bits(t) for t in cache.encoders[shape](*xs))
        barrier('control graph')
      return result
    started=time.perf_counter();before=[encode(originals[:count]) for _ in range(3)]
    c.require(before[0]==before[1]==before[2],f'Batch{count} initial/capture/replay changed feature bits')
    mutated=list(originals[:count]);p,v=mutated[-1]
    mutated[-1]=((p*2).contiguous().realize(),v)
    after=[encode(mutated) for _ in range(2)]
    c.require(after[0]==after[1],f'Batch{count} mutation replay changed bits')
    c.require(before[0][:-1]==after[0][:-1],f'Batch{count} mutation coupled earlier images')
    c.require(before[0][-1]!=after[0][-1],f'Batch{count} mutation did not affect last image')
    restored=encode(originals[:count]);c.require(restored==before[0],f'Batch{count} stale graph inputs after restore')
    record.append(dict(images=count,original=before[0],mutated=after[0],restored=restored,
      initial_capture_replay_exact=True,mutation_replay_exact=True,earlier_images_independent=True,
      seconds_including_compile_capture_and_reads=time.perf_counter()-started))
  c.require(logical_before==[key_record(key) for key in cache.cache],'Controls changed logical feature cache')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--arm',choices=('group2','group4'),required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--execute-gpu',action='store_true')
  args=parser.parse_args()
  plan=c.read(c.PLAN)
  for relative,digest in plan['source_sha256'].items():c.require(c.sha(c.SNAP/relative)==digest,'Frozen source changed: '+relative)
  for path,digest in plan['input_files_sha256'].items():c.require(c.sha(path)==digest,'Input/protocol changed: '+path)
  c.require(args.execute_gpu,'Parent must explicitly pass --execute-gpu; CPU preparation never opens a device')
  c.require(Path(__file__).resolve()==c.EXP/Path(__file__).name,'Execute the frozen helper only')
  c.require(not args.output.exists(),'Never overwrite a GPU result')
  invocation_path=c.ROOT/('agent-vision-group4-'+args.arm+'-invocation.json')
  invocation=c.read(invocation_path)
  original_env=os.environ.copy();original_path=sys.path[:]
  for key,value in invocation['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  sys.path[:0]=[str(c.SNAP),str(c.EXP)]
  command=[str(c.EXP/'agent-speculation-run.py'),'--invocation',str(invocation_path),
    '--output',str(args.output),'--mode','candidate','--prefill-attention','shuffle-fma','--pad-prompt-to','2048','--warmup','0','--runs','4']
  audit=dict(schema=1,arm=args.arm,timings_valid_for_throughput=False,queries=[],controls=[],binaries={},
    plan_sha256=c.sha(c.PLAN),source_sha256=plan['source_sha256'],peak_live_buffer_bytes=0,
    memory_limit_bytes=7*1024**3,memory_limit_boundary='tinygrad live buffer accounting at runtime construction and barriers; allocator reservation/driver overhead excluded',
    no_future_input_warming=True,full_model_gpu_parity_passed=False)
  from tinygrad import Device
  from tinygrad.helpers import GlobalCounters
  from tinygrad.engine import realize
  from tinygrad.uop.ops import Ops
  phase=[None];owners=[];decoders=[];original_spec=importlib.util.spec_from_file_location
  original_runtime=realize.get_runtime;original_argv=sys.argv[:];restores=[];restoration_checks=[]
  binaries=args.output.parent/'agent-vision-group4-binaries';binaries.mkdir(exist_ok=False)

  def memory():
    used=int(GlobalCounters.mem_used)
    audit['peak_live_buffer_bytes']=max(audit['peak_live_buffer_bytes'],used)
    c.require(used<=audit['memory_limit_bytes'],'Live buffers exceeded7GiB; stop before further inference')

  def barrier(label):Device[Device.DEFAULT].synchronize();memory()

  def runtime(device,program,*values,**options):
    memory()
    if phase[0] and program.op is Ops.PROGRAM and str(device).startswith('USB'):
      key=program.key.hex()
      if key not in audit['binaries']:
        source=program.src[2].arg;binary=program.src[3].arg
        c.require(isinstance(source,str) and isinstance(binary,bytes),'Unexpected PROGRAM layout')
        (binaries/(key+'.source')).write_text(source);(binaries/(key+'.hsaco')).write_bytes(binary)
        audit['binaries'][key]=dict(name=program.src[0].arg.name,target=str(program.arg.target),device=str(device),
          source_sha256=hashlib.sha256(source.encode()).hexdigest(),binary_sha256=hashlib.sha256(binary).hexdigest(),
          binary_bytes=len(binary),resources=c.resources(binary),first_phase=phase[0],
          local_size=list(program.arg.local_size),global_size=[str(x) for x in program.arg.global_size])
    return original_runtime(device,program,*values,**options)

  def patch_module(module,path):
    if path.name=='vision-cache.py':
      original_class=module.VisionCache;original_init=original_class.__init__
      def init(owner,*values,**options):
        original_init(owner,*values,**options);owners.append(owner)
        expected=int(args.arm[-1]);c.require(owner.group_limit==expected and owner.capacity==48,'Effective group/capacity differs')
        old=owner.encode
        def encode(pixels,positions,ids,batches,spans):
          c.require(len(batches)==10 and all(b-a==720 and n==1 for a,b,n in batches),'Require ten full720patch images')
          phase[0]='vision-controls' if not audit['controls'] else 'vision'
          try:
            if not audit['controls']:controls(owner,pixels,positions,audit['controls'],barrier)
            phase[0]='vision';begin=time.perf_counter();result=old(pixels,positions,ids,batches,spans);barrier('vision completed')
            audit['queries'][-1]['vision']=dict(keys=[key_record(key) for key in owner.keys],
              features=[tensor_bits(owner.cache[key]) for key in owner.keys],
              stitched_embedding=tensor_bits(result),hits=owner.hits,misses=owner.misses,
              lru_keys=[key_record(key) for key in owner.cache],seconds_including_diagnostic_fences=time.perf_counter()-begin,
              encoder_shapes=[[[int(v) for v in shape] for shape in key] for key in owner.encoders])
            return result
          finally:phase[0]=None
        owner.encode=encode;restores.append(lambda owner=owner,old=old:setattr(owner,'encode',old))
        restoration_checks.append(lambda owner=owner,old=old:owner.encode is old)
      module.VisionCache.__init__=init
      restores.append(lambda:setattr(original_class,'__init__',original_init))
      restoration_checks.append(lambda:original_class.__init__ is original_init)
    if path.name=='lookup-decode.py':
      original_class=module.LookupDecoder;original_init=original_class.__init__
      def init(decoder,*values,**options):
        original_init(decoder,*values,**options);decoders.append(decoder)
        old=decoder.run
        def run(self,model,settings,config,tokenizer,prepared,packed,seed,inputs):
          helper=c.module(c.EXP/'agent-live-replay.py','agent_vision_group4_readonly_fingerprints')
          row=dict(index=len(audit['queries']),prepared=helper.tree(prepared),packed=helper.tree(packed),
            cpu=[dict(shape=list(t.shape),dtype=str(t.dtype),sha256=hashlib.sha256(bytes(t._buffer().as_memoryview(allow_zero_copy=True))).hexdigest()) for t in inputs.cpu],
            batches=helper.tree(inputs.batches),spans=helper.tree(inputs.spans),history_before=copy.deepcopy(self.history),
            owner_reused=bool(audit['queries']) and id(inputs)==audit.get('_input_owner'),
            actual_prompt=int(prepared['actual_prompt_length']),padded_prompt=int(packed[0].shape[1]))
          audit['_input_owner']=id(inputs);audit['queries'].append(row)
          expected=plan['cpu_queries'][row['index']]
          c.require(row['prepared']==expected['prepared'] and row['packed']==expected['packed'],'Actual prepared input differs from CPU-qualified canonical input')
          c.require(all(a['shape']==b['shape'] and a['sha256']==b['sha256'] for a,b in zip(row['cpu'],expected['arrays'],strict=True)),'Actual upload-owner bytes differ')
          c.require(row['padded_prompt']==2048 and row['actual_prompt']+6<=2048,'Effective prompt bound does not fit genuine six-target prefill')
          c.require(settings.diffusion_steps==3 and settings.vision_batch==1 and settings.max_tokens==64,'Effective inference recipe differs')
          result=old(model,settings,config,tokenizer,prepared,packed,seed,inputs);barrier('prediction completed')
          valid=row['actual_prompt']+len(result['generated_token_ids'])
          row.update(history_after=copy.deepcopy(self.history),statistics=copy.deepcopy(self.statistics[-1]),
            spec_record=copy.deepcopy(self.prefill_draft_records[-1]),pixel_bytes=inputs.uploaded_pixel_bytes,
            kv_valid=[tensor_bits(t[:,:,:,:valid]) for t in model.cache],kv_shapes=[list(t.shape) for t in model.cache],
            prediction_bits={key:hashlib.sha256(np.asarray(result[key],np.float32).tobytes()).hexdigest() for key in ('xyz','rot')},
            prediction={key:result[key] for key in ('prompt_tokens','prefill_tokens','generated_tokens','generated_token_ids','reasoning','xyz','rot')})
          c.require(len(row['kv_valid'])==28 and all(shape[3]==2304 for shape in row['kv_shapes']),'Language cache layout changed')
          return result
        decoder.run=MethodType(run,decoder);restores.append(lambda decoder=decoder,old=old:setattr(decoder,'run',old))
        restoration_checks.append(lambda decoder=decoder,old=old:decoder.run is old)
      module.LookupDecoder.__init__=init
      restores.append(lambda:setattr(original_class,'__init__',original_init))
      restoration_checks.append(lambda:original_class.__init__ is original_init)

  def hooked_spec(name,location,*values,**options):
    spec=original_spec(name,location,*values,**options);path=Path(location).resolve()
    if path.parent!=c.EXP or path.name not in ('vision-cache.py','lookup-decode.py'):return spec
    loader=spec.loader
    class Loader:
      def create_module(self,value):return loader.create_module(value)
      def exec_module(self,value):loader.exec_module(value);patch_module(value,path)
    spec.loader=Loader();return spec

  tree=ast.parse((c.EXP/'benchmark-pipeline.py').read_text())
  init=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='initialize')
  audit['protected_initializer_ast_sha256']=hashlib.sha256(ast.dump(init,include_attributes=False).encode()).hexdigest()
  started=time.perf_counter()
  try:
    realize.get_runtime=runtime;importlib.util.spec_from_file_location=hooked_spec
    sys.argv=command;runpy.run_path(command[0],run_name='__main__')
    c.require(len(audit['queries'])==4 and len(owners)==1 and len(decoders)==1,'Missing parent parity queries/instances')
    c.require(bool(audit['binaries']),'No actual USB vision binaries were captured')
    audit['completed']=True
  except BaseException as error:
    audit['error']=repr(error);raise
  finally:
    for restore in reversed(restores):restore()
    realize.get_runtime=original_runtime;importlib.util.spec_from_file_location=original_spec;sys.argv=original_argv
    sys.path[:]=original_path
    os.environ.clear();os.environ.update(original_env)
    audit.pop('_input_owner',None);audit['elapsed_s']=time.perf_counter()-started
    audit['postcleanup']=dict(import_hook_restored=importlib.util.spec_from_file_location is original_spec,
      runtime_hook_restored=realize.get_runtime is original_runtime,argv_restored=sys.argv==original_argv,
      instance_and_class_methods_restored=all(check() for check in restoration_checks),
      protected_initializer_never_replaced=True,initializer_source_unchanged=c.sha(c.EXP/'benchmark-pipeline.py')==plan['source_sha256']['experiments/benchmark-pipeline.py'])
    c.write(args.output.with_suffix('.group4-audit.json'),audit)
    if args.output.exists():
      report=c.read(args.output);report['vision_group4_audit']=dict(arm=args.arm,sidecar=str(args.output.with_suffix('.group4-audit.json')),timings_valid_for_throughput=False)
      report.setdefault('summary',{})['profiling_enabled']=True
      args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

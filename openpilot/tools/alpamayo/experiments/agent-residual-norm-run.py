"""Plan or run one bounded parent-owned residual/norm/packing microcheck.

No GPU work occurs in plan mode. Parent serialization and timeout are mandatory.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import statistics
import struct
import sys
import time

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TRIAL=WORK/'current-profile-dynamic-2048'
FROZEN=TRIAL/'report-sources'
CHECKPOINT=Path('/tmp/opencode/alpamayo-asius-20261114/model')
LAYERS=(0,13,27)
BATCH=16


def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_')+'_residual_micro',path)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def setarg(command,key,value):command[command.index('--'+key)+1]=str(value)


def plan(output,source_bundle=None):
  p=json.loads((TRIAL/'report.current-profile-plan.json').read_text())
  r=json.loads((TRIAL/'report.json').read_text())
  control=p['current_control_invocation']
  command=control['command'][:]
  assert '--vision-calibration' not in command and '--profile-kernels' not in command
  assert command[command.index('--pad-prompt-to')+1]=='2048' and command[command.index('--steps')+1]=='3'
  original_root=Path(command[1]).parent.parent
  core_sources={path:expected for path,expected in p['source_sha256'].items() if Path(path).parent==original_root}
  for name,expected in r['source_sha256'].items():
    if sha((FROZEN/name).read_bytes())!=expected:raise ValueError('Frozen source changed: '+name)
  for path,expected in p['artifact_sha256'].items():
    if sha(Path(path).read_bytes())!=expected:raise ValueError('Original input artifact changed: '+path)
  for path,expected in p['tinygrad']['source_sha256'].items():
    if sha(Path(path).read_bytes())!=expected:raise ValueError('Pinned tinygrad source changed: '+path)
  if source_bundle is None:
    root=output.with_suffix('.agent-residual-norm-source')
    root.mkdir(parents=True,exist_ok=False)
    (root/'experiments').mkdir();(root/'agents').mkdir()
    for path,expected in core_sources.items():
      data=Path(path).read_bytes()
      if sha(data)!=expected:raise ValueError('Original core source changed: '+path)
      (root/Path(path).name).write_bytes(data)
    for name,expected in r['source_sha256'].items():
      if name=='model.py':
        assert sha((root/name).read_bytes())==expected
      else:(root/'experiments'/name).write_bytes((FROZEN/name).read_bytes())
    owned={path.name:sha(path.read_bytes()) for path in HERE.glob('agent-residual-norm-*.py')}
    for name in owned:(root/'agents'/name).write_bytes((HERE/name).read_bytes())
    write(root/'agent-residual-norm-manifest.json',dict(original_core_source_sha256=core_sources,
      original_frozen_source_sha256=r['source_sha256'],owned_source_sha256=owned,
      source_profile_plan_sha256=sha((TRIAL/'report.current-profile-plan.json').read_bytes())))
  else:root=source_bundle.resolve()
  manifest=json.loads((root/'agent-residual-norm-manifest.json').read_text())
  assert manifest['original_core_source_sha256']==core_sources and manifest['original_frozen_source_sha256']==r['source_sha256']
  for path,expected in core_sources.items():
    if sha((root/Path(path).name).read_bytes())!=expected:raise ValueError('Frozen core copy changed: '+path)
  for name,expected in r['source_sha256'].items():
    target=root/name if name=='model.py' else root/'experiments'/name
    if sha(target.read_bytes())!=expected:raise ValueError('Frozen experiment copy changed: '+name)
  for name,expected in manifest['owned_source_sha256'].items():
    if sha((root/'agents'/name).read_bytes())!=expected or sha((HERE/name).read_bytes())!=expected:
      raise ValueError('Owned source differs from its frozen manifest: '+name)
  command[1]=str(root/'experiments/benchmark-pipeline.py')
  setarg(command,'warmup',8);setarg(command,'runs',1)
  setarg(command,'output',output.with_suffix('.capture.json'))
  environment=dict(control['environment'],PROFILE='0')
  environment['PYTHONPATH']=environment['PYTHONPATH'].split(os.pathsep)[0]+os.pathsep+str(root)
  result=dict(schema=1,mode='diagnostic microcheck',gpu_executed=False,complete=False,
    parent_timeout_seconds=300,source_profile_plan_sha256=sha((TRIAL/'report.current-profile-plan.json').read_bytes()),
    original_frozen_source_sha256=r['source_sha256'],original_core_source_sha256=core_sources,
    input_artifact_sha256=p['artifact_sha256'],
    owned_source_sha256={str(path):sha(path.read_bytes()) for path in sorted(HERE.glob('agent-residual-norm-*.py'))},
    command=command,environment=environment,core_root=str(root),experiment_root=str(root/'experiments'),
    source_bundle_manifest_sha256=sha((root/'agent-residual-norm-manifest.json').read_bytes()),tinygrad=p['tinygrad'],
    layers=list(LAYERS),shape=[1,2048,2048],
    target_query=p['native_prefix']['queries'][8],batch_calls=BATCH,rounds=3,
    capture='Eight original chronological warmups; reset only padded-prefill trace for target query; stop after layer 27 slot 2.',
    timing_boundary='Residual BF16 ADD + original RMSNorm + fixed FP8 conversion/layout + both output stores, 16 dependent calls.',
    timing_limits=['Host-inclusive synchronized batch; warm resident buffers/weights; not model streaming or service throughput.',
      'Model load, diagnostic captures, input uploads, compilation and comparison readbacks are excluded from micro timing.',
      'The 4.93ms affected-chain budget is a ceiling, not a forecast.'])
  return result


class Captured(Exception):pass


def checkpoint_norm_bytes(name):
  index=json.loads((CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map']
  with (CHECKPOINT/index[name]).open('rb') as stream:
    length=struct.unpack('<Q',stream.read(8))[0]
    if length>16*1024*1024:raise ValueError('Unexpected checkpoint header')
    header=json.loads(stream.read(length));entry=header[name]
    assert entry['dtype']=='BF16' and entry['shape']==[2048]
    begin,end=entry['data_offsets'];stream.seek(8+length+begin);return stream.read(end-begin)


def capture_inputs(p,output):
  import numpy as np
  from tinygrad import dtypes
  from tinygrad.uop.ops import Ops
  install=load(HERE/'agent-residual-norm-install.py')
  old_spec,old_run,old_argv=importlib.util.spec_from_file_location,runpy.run_path,sys.argv
  captured={};restores=[];context={}
  experiments=Path(p['experiment_root'])

  def attach(components):
    model=components.model
    method=install.find(model.quantize,'quantize-packed.py','quantize_input')
    state=inspect.getclosurevars(method.__func__).nonlocals
    globals_=method.__func__.__globals__;original_q=globals_['quantize'];original_n=model.norm
    active=[False];pending=[None];calls=[0]
    def norm(x,name,rms=False,eps=None):
      prefix=name.removesuffix('.post_attention_layernorm')
      selected=name.endswith('.post_attention_layernorm') and prefix in {install.PREFIX+str(i) for i in LAYERS}
      if not active[0] or not selected:return original_n(x,name,rms=rms,eps=eps)
      assert state['state']==dict(prefill=True,prefix=prefix,index=2)
      assert x.uop.op is Ops.ADD and x.shape==(1,2048,2048)
      from tinygrad import Tensor
      a,b=(Tensor(node) for node in x.uop.src)
      weight=model.weights[name+'.weight'];scale=state['fixed'][prefix+'/2']
      def bits(t):return t.bitcast(dtypes.uint16).numpy().copy()
      sample=dict(a=bits(a),b=bits(b),weight=bits(weight),scale=scale.numpy().copy(),residual=bits(x))
      assert sample['weight'].tobytes()==checkpoint_norm_bytes(name+'.weight')
      expected=np.float32(json.loads(Path(p['command'][p['command'].index('--prefill-calibration')+1]).read_text())['scales'][prefix+'/2'])
      assert sample['scale'].tobytes()==expected.tobytes()
      normalized=original_n(x,name,rms=rms,eps=eps)
      sample['normalized']=bits(normalized)
      captured[prefix]=sample;pending[0]=(prefix,normalized)
      return normalized
    def quantize(x,scale=None):
      result=original_q(x,scale)
      if pending[0] is not None and x is pending[0][1]:
        prefix=pending[0][0];pending[0]=None
        assert state['state']==dict(prefill=True,prefix=prefix,index=3)
        captured[prefix]['packed']=result[0].bitcast(dtypes.uint8).numpy().copy()
        if prefix==install.PREFIX+'27':raise Captured()
      return result
    model.norm=norm;globals_['quantize']=quantize
    restores.extend((lambda:setattr(model,'norm',original_n),lambda:globals_.__setitem__('quantize',original_q)))
    original_runner=components.runner
    def runner(*args,**kwargs):
      if calls[0]==8:
        assert components.vision_cache.hits>=0
        model.padded_prefill.jit.reset()
        active[0]=True
      try:return original_runner(*args,**kwargs)
      finally:
        if active[0]:context.update(vision_hits=components.vision_cache.hits,vision_misses=components.vision_cache.misses)
        calls[0]+=1
    components.runner=runner

  def pipeline(path,run_name):
    spec=old_spec('agent_residual_norm_frozen_pipeline',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=module.initialize
    def initialize(*args,**kwargs):
      components=original(*args,**kwargs);attach(components);return components
    module.initialize=initialize;module.main()
  try:
    runpy.run_path=pipeline
    sys.argv=[str(experiments/'agent-speculation-run.py'),'--output',str(output.with_suffix('.capture.json')),
      '--invocation',str(output.with_suffix('.agent-residual-norm-invocation.json')),'--mode','candidate',
      '--prefill-attention','shuffle-fma','--pad-prompt-to','2048','--warmup','8','--runs','1']
    namespace={'__file__':str(experiments/'agent-speculation-run.py'),'__name__':'__main__'}
    try:exec(compile((experiments/'agent-speculation-run.py').read_text(),namespace['__file__'],'exec'),namespace)
    except Captured:pass
    else:raise AssertionError('Expected bounded original-prefill capture stop')
  finally:
    for restore in reversed(restores):restore()
    importlib.util.spec_from_file_location=old_spec;runpy.run_path=old_run;sys.argv=old_argv
  assert len(captured)==3 and context==dict(vision_hits=6,vision_misses=4)
  assert sum(value.nbytes for sample in captured.values() for value in sample.values())<128*1024*1024
  np.savez(output.with_suffix('.agent-residual-norm-inputs.npz'),**{prefix+'/'+key:value for prefix,sample in captured.items() for key,value in sample.items()})
  return captured,context


@contextmanager
def observe(proofs,label):
  from tinygrad.runtime.support import hcq2
  from tinygrad.runtime.support.hcq2 import unwrap_lane
  from tinygrad.uop.ops import Ops
  from tinygrad.engine.realize import get_call_outs_ins
  original=hcq2._build_queues
  def build(ctx):
    queues=original(ctx)
    calls=[call for call,_,_ in ctx.batch]
    if len(calls) in (BATCH,3*BATCH):
      if label=='candidate':
        assert len(calls)==BATCH and len(queues)==1
        for call in calls:
          outs,ins=get_call_outs_ins(call)
          assert tuple(outs)==(0,1) and 2 in ins
        def span(arg):
          base,lane,offset=unwrap_lane(arg);assert base.op is Ops.PARAM and lane is None
          return base.arg.slot,offset,arg.max_numel()*arg.dtype.itemsize
        outputs=[span(call.src[1]) for call in calls]
        inputs=[span(call.src[3]) for call in calls]
        assert inputs[1:]==outputs[:-1], 'Missing running-residual data dependence'
        ordered=[]
        for command in next(iter(queues.values())):
          if command.op is Ops.CALL and command.body.op is Ops.PROGRAM:ordered.append(command)
          elif command.op is Ops.CALL and command.body.op is Ops.CUSTOM_FUNCTION and command.body.arg.name=='exec_barrier':ordered.append('barrier')
        assert ordered==[v for i,call in enumerate(calls) for v in ((call,) if i==0 else ('barrier',call))]
      else:assert len(calls)==3*BATCH
      proofs.append(dict(label=label,kernel_calls=len(calls),dependent_operations=BATCH,actual_queue_builder=True))
    return queues
  hcq2._build_queues=build
  try:yield
  finally:hcq2._build_queues=original


def gpu_micro(samples,result,output):
  import numpy as np
  from tinygrad import Tensor,TinyJit,Device,dtypes
  dev=Device[Device.DEFAULT];assert dev.arch=='gfx1200'
  kernel=load(HERE/'agent-residual-norm-kernel.py')
  norm=load(FROZEN/'fused-norm.py');quant=load(FROZEN/'quantize-packed.py')
  cpu=load(HERE/'agent-residual-norm-check.py')
  source,binary=kernel.compiled()
  norm_source,norm_binary=norm.compiled(2048,True,1e-5)
  result['compiled']=dict(candidate_source_sha256=sha(source.encode()),candidate_binary_sha256=sha(binary),
                         frozen_norm_source_sha256=sha(norm_source.encode()),frozen_norm_binary_sha256=sha(norm_binary))
  def baseline(a,b,w,s):
    residual=(a+b).realize()
    value=norm.norm(residual,w,rms=True,eps=1e-5)
    return residual,quant.quantize(value,s)[0]
  def tensors(sample):
    return tuple(Tensor(sample[key],device=dev.device).bitcast(dtypes.bfloat16).realize() for key in ('a','b','weight'))+(Tensor(sample['scale'],device=dev.device).realize(),)
  def mutate(inputs,sample):
    for target,value in zip(inputs,tensors(sample)):target.assign(value).realize()
    dev.synchronize()
  def bits(values):
    return tuple(value.bitcast(dtypes.uint16 if value.dtype==dtypes.bfloat16 else dtypes.uint8).numpy().copy() for value in values)
  def exact(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):
      np.testing.assert_array_equal(x,y)
      if x.dtype==np.uint16:assert np.isfinite(cpu.floats(x)).all(), 'Nonfinite BF16 output'
      else:assert np.all((x&127)!=127), 'Nonfinite FP8 output'
  def batch(fn):
    def run(a,b,w,s):
      results=[]
      for _ in range(BATCH):a,p=fn(a,b,w,s);results.append(p)
      return (a,*results)
    return TinyJit(run)
  jobs=[];result['gates']=[]
  for prefix,sample in samples.items():
    inputs=tensors(sample)
    ref,can=TinyJit(baseline),TinyJit(kernel.fused)
    controls=[('current',sample)]
    for name in ('a','b','weight','scale','zeros-ties-outliers'):
      changed={key:sample[key].copy() for key in ('a','b','weight','scale')}
      if name in ('a','b'):changed[name][:,:64,:8]=0x4280 if name=='a' else 0xc200
      elif name=='weight':changed[name]=cpu.bf16(cpu.floats(changed[name])*.5)
      elif name=='scale':changed[name]=np.asarray(changed[name]*np.float32(2),np.float32)
      else:
        changed['a'].fill(0);changed['b'].fill(0);changed['a'][:,0]=0x3f80;changed['b'][:,0]=0x3b80
        changed['a'][:,1,:64]=0x7f00;changed['b'][:,2,:64]=0x8000
      controls.append((name,changed))
    for index,(name,control) in enumerate(controls):
      mutate(inputs,control)
      # First execution, capture, and actual replay must each pass.
      for replay in range(3 if index==0 else 1):
        expected=bits(ref(*inputs));actual=bits(can(*inputs));exact(actual,expected)
        if index==0:exact(expected,(sample['residual'],sample['packed']))
      result['gates'].append(dict(prefix=prefix,case=name,all_bf16_residual_bits=True,all_fp8_packed_bytes=True))
      write(output,result)
    mutate(inputs,sample)
    refs,candidates=batch(baseline),batch(kernel.fused);proofs=[]
    for label,jit in (('baseline',refs),('candidate',candidates)):
      jit(*inputs)
      with observe(proofs,label):jit(*inputs)
    exact(bits(candidates(*inputs)),bits(refs(*inputs)))
    assert {row['label'] for row in proofs}=={'baseline','candidate'}
    # Mutate real input bytes after graph capture, then restore and check every batch output.
    mutate(inputs,controls[1][1]);exact(bits(candidates(*inputs)),bits(refs(*inputs)))
    mutate(inputs,sample);exact(bits(candidates(*inputs)),bits(refs(*inputs)))
    jobs.append((prefix,inputs,refs,candidates,proofs))
  # No candidate receives timing until every current-input and synthetic gate passes.
  result['timings']=[]
  for prefix,inputs,ref,can,proofs in jobs:
    times={'baseline':[],'candidate':[]}
    for round_ in range(3):
      for label,jit in ((('baseline',ref),('candidate',can)) if round_%2==0 else (('candidate',can),('baseline',ref))):
        dev.synchronize();start=time.perf_counter();jit(*inputs);dev.synchronize()
        times[label].append((time.perf_counter()-start)*1000/BATCH)
    exact(bits(can(*inputs)),bits(ref(*inputs)))
    result['timings'].append(dict(prefix=prefix,per_call_ms=times,median_ms={key:statistics.median(value) for key,value in times.items()},proofs=proofs))
    write(output,result)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--mode',choices=('plan','run'),default='plan')
  parser.add_argument('--source-bundle',type=Path)
  args=parser.parse_args();output=args.output.resolve()
  if output.exists():raise FileExistsError(output)
  p=plan(output,args.source_bundle);output.parent.mkdir(parents=True,exist_ok=True)
  invocation=output.with_suffix('.agent-residual-norm-invocation.json')
  write(invocation,dict(command=p['command'],environment=p['environment']))
  write(output.with_suffix('.agent-residual-norm-plan.json'),p)
  if args.mode=='plan':print(json.dumps(dict(mode='plan',gpu_executed=False,output=str(output))));return
  if any(name=='tinygrad' or name.startswith('tinygrad.') for name in sys.modules):raise RuntimeError('Fresh process required')
  original_env=dict(os.environ);original_path=sys.path[:]
  try:
    for key,value in p['environment'].items():
      if value is None:os.environ.pop(key,None)
      else:os.environ[key]=value
    sys.path[:0]=[p['experiment_root'],p['core_root'],*p['environment']['PYTHONPATH'].split(os.pathsep)]
    samples,context=capture_inputs(p,output)
    result=dict(schema=1,gpu_executed=True,complete=False,diagnostic_only=True,
      plan_sha256=sha(output.with_suffix('.agent-residual-norm-plan.json').read_bytes()),capture_context=context,
      captured_sha256={prefix:{key:sha(value.tobytes()) for key,value in sample.items()} for prefix,sample in samples.items()},
      timings_valid_for_model_throughput=False)
    write(output,result)
    try:gpu_micro(samples,result,output)
    except Exception as error:
      result['failure']=dict(type=type(error).__name__,message=str(error));write(output,result);raise
    result['complete']=True;write(output,result);print(json.dumps(result,indent=2))
  finally:
    sys.path[:]=original_path;os.environ.clear();os.environ.update(original_env)


if __name__=='__main__':main()

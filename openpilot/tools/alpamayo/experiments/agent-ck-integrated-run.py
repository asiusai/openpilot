"""One fresh eight-sample arm. Default is CPU verification; parent explicitly enables execution."""
import argparse
from contextlib import nullcontext
import copy
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
from types import MethodType

spec=importlib.util.spec_from_file_location('ck_integrated_common',Path(__file__).with_name('agent-ck-integrated-common.py'))
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def configure(arm):
  invocation=json.loads((C.PREPARED/f'{arm}-invocation.json').read_text())
  for key,value in invocation['environment'].items():
    if value is None: os.environ.pop(key,None)
    else: os.environ[key]=value
  sys.path[:0]=[str(C.TINY),str(C.BASE),str(C.BASE/'experiments')]
  return invocation

def run(plan,arm,directory):
  H=C.load('agent-ck-integrated-hook');instances=[];started=time.monotonic()
  report={'completed':False,'gpu_executed':False,'arm':arm,'numerical_variant':arm=='candidate',
    'adoption_allowed':False,'timings_valid_for_throughput':False,'source_protocol_sha256':C.sha(C.PREPARED/'protocol.json'),
    'reused_live_evidence_sha256':C.LIVE_SHA,'reused_micro_evidence_sha256':C.MICRO_SHA,
    'indices':C.INDICES,'initial_decoder_history':[],'phase':'startup','readback_bytes':0}
  old_spec,old_argv,old_path=importlib.util.spec_from_file_location,sys.argv,sys.path[:]
  def spec_hook(name,location,*args,**kwargs):
    value=old_spec(name,location,*args,**kwargs)
    if Path(location).resolve()!=C.BASE/'experiments/lookup-decode.py': return value
    loader=value.loader
    class Loader:
      def create_module(self,spec): return loader.create_module(spec)
      def exec_module(self,module):
        loader.exec_module(module);original=module.LookupDecoder
        class ScreenDecoder(original):
          def __init__(self,model):
            super().__init__(model);assert self.history==[]
            origins={name:getattr(model,name).__func__.__code__.co_filename for name in ('text_layer','attention')}
            assert Path(origins['text_layer'])==C.BASE/'experiments/quantize-packed.py'
            assert Path(origins['attention'])==C.BASE/'experiments/agent-attention-rowsoftmax.py'
            assert type(model.padded_prefill).__name__=='SpeculativePrefill'
            report.update(gpu_executed=True,phase='fresh_screen8',installed_method_origins=origins)
            instance=H.Integration(C,model,self,arm,report);instances.append(instance)
            original_run=self.run
            def observed_run(obj,*args,**kwargs):
              before=copy.deepcopy(obj.history);value=original_run(*args,**kwargs)
              report.setdefault('decoder_histories',[]).append({'before':before,'after':copy.deepcopy(obj.history)})
              return value
            self.run=MethodType(observed_run,self)
            instance.decoder_run_restore=lambda:delattr(self,'run')
        module.LookupDecoder=ScreenDecoder
    value.loader=Loader();return value
  def timeout(*args): raise TimeoutError('Integrated screen reached its 580-second internal deadline')
  old_alarm=signal.signal(signal.SIGALRM,timeout);signal.alarm(580)
  scope=H.native_scope(C,report) if arm=='candidate' else nullcontext()
  try:
    with scope:
      try:
        importlib.util.spec_from_file_location=spec_hook
        sys.argv=[str(C.BASE/'experiments/agent-speculation-run.py'),'--invocation',str(C.PREPARED/f'{arm}-invocation.json'),
          '--output',str(directory/'pipeline.json'),'--mode','candidate','--prefill-attention','shuffle-fma',
          '--pad-prompt-to','2048','--warmup','0','--runs','8','--samples-json',str(C.PREPARED/'samples.json')]
        runpy.run_path(sys.argv[0],run_name='__main__')
        assert len(instances)==1
        owner=instances[0];owner.device.synchronize()
        screen=[r for r in report['prefills'] if r['phase'].startswith('screen_')]
        assert len(screen)==8 and owner.original_jit.cnt>=8 and owner.control_done
        assert len(report['decoder_histories'])==8 and report['decoder_histories'][0]['before']==[]
        if arm=='candidate': assert report['model_controls']['passed'] and report['native_kernarg_builds']>=28
        pipeline=json.loads((directory/'pipeline.json').read_text())
        assert not pipeline['warmup'] and [e['sample'] for e in pipeline['runs']]==json.loads((C.PREPARED/'samples.json').read_text())
        pipeline['summary']['profiling_enabled']=True
        pipeline['ck_screen']={'arm':arm,'numerical_variant':arm=='candidate','timings_valid_for_throughput':False,
          'reason':'Per-layer audits, pointer observations, and capture-time mutation controls are included.'}
        (directory/'pipeline.json').write_text(json.dumps(pipeline,indent=2,allow_nan=False)+'\n')
        report.update(completed=True,phase='screen8_complete',pipeline=str(directory/'pipeline.json'),pipeline_sha256=C.sha(directory/'pipeline.json'),
          full64_allowed=False,remaining='Fresh paired 5% aggregate/window comparison and manual text/speed/slowing/turn review; then parent decision.')
      finally:
        for instance in reversed(instances):
          try: instance.close()
          finally: instance.decoder_run_restore()
        report['hooks_restored']=all(i.closed for i in instances)
        report['model_renderer_unchanged']=all(i.device.renderer is i.renderer and i.renderer.compiler is i.compiler for i in instances)
  except BaseException as error:
    report['failure']={'type':type(error).__name__,'message':str(error),'phase':report['phase']};raise
  finally:
    importlib.util.spec_from_file_location=old_spec;sys.argv=old_argv;sys.path[:]=old_path
    signal.alarm(0);signal.signal(signal.SIGALRM,old_alarm)
    report['elapsed_seconds']=time.monotonic()-started
    C.write(directory/'report.json',report)
  return report

def main():
  parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('baseline','candidate'),required=True)
  parser.add_argument('--parent-run',action='store_true');args=parser.parse_args()
  assert not sys.flags.optimize
  plan=C.verify();directory=C.ROOT/args.arm
  assert not directory.exists(),'This frozen arm already has an output directory'
  if not args.parent_run:
    print(json.dumps({'verified':True,'gpu_executed':False,'arm':args.arm,'protocol':C.sha(C.PREPARED/'protocol.json')}));return
  directory.mkdir();configure(args.arm);os.environ['ALLOW_DEVICE_USAGE']='1'
  run(plan,args.arm,directory)

if __name__=='__main__': main()

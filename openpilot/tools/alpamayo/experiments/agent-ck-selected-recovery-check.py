"""CPU postrun test: real composition, coverage, PV close/observer exit and final validators."""
import argparse
import copy
import difflib
import importlib.util
import inspect
import os
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch

s=importlib.util.spec_from_file_location('ck_selected_recovery',Path(__file__).with_name('agent-ck-selected-recovery-common.py'))
R=importlib.util.module_from_spec(s);s.loader.exec_module(R);S=R.S

def main(output):
  assert not output.exists();os.environ.update(ALLOW_DEVICE_USAGE='0',DEV='CPU')
  A,C,N=R.configure()
  from tinygrad import Device
  assert not Device._opened_devices
  original=S.read(R.FAILED/'report.json');proof=R.recover();assert proof==S.read(R.ROOT/'receipt-v1.json')
  rejected=[];events=[];live=[]
  def reject(name,fn):
    try:fn()
    except (AssertionError,ValueError,RuntimeError,KeyError):rejected.append(name)
    else:raise AssertionError('Negative control passed: '+name)
  # Reuse the original fake neural operations, but run the genuine selected Composition created by production execute.
  source=(S.EXP/'agent-expert-combined-check.py').read_text()
  pre=source[source.index('  c,manifest ='):source.index('  for candidate in (False,True):')]
  setup=source[source.index('    mlps,visions,quants ='):source.index("    comp = Composition(")]
  body=source[source.index("          core ="):source.index("          stream = C.load(")]
  body=S.Q.replace_once(body,'model = fixtures.FakeModel(core)',
    'fixtures.FakeModel.text_layer = core.EdgeModel.text_layer\n          model = fixtures.FakeModel(core)')
  body=S.Q.replace_once(body,"assert decoder.history == [] and comp.lookup_initial[0]['vision_cache_empty']",
    'assert decoder.history == [] and not cache.cache and not cache.encoders and not cache.stitchers')
  body=S.Q.replace_once(body,'h.original_matmul,h.originals = actual_matmul,original_trajectory',
    '''# CPU capture metadata comes from the immutable GPU result; cleanup is never synthesized.
          h.programs.update(copy.deepcopy(_original['selected_pv_cleanup']['runtime_programs']))
          records=[]
          for name in (h.expected['name'],'alpamayo_attention_scores','alpamayo_attention_softmax'):
            digest=_original['selected_pv_cleanup']['captures'][0]['binaries'][name][0]
            key=next(k for k,v in h.programs.items() if v['name']==name and v['binary_sha256']==digest)
            records.extend([('AMD',name,None,(),bytes.fromhex(key),(),((0,),(1,2)))]*84)
          call=SimpleNamespace(arg=SimpleNamespace(aux=SimpleNamespace(kernels=records)))
          call.without_after=call
          model.trajectory_jits['cpu_capture']=SimpleNamespace(captured=SimpleNamespace(_linear=SimpleNamespace(src=(call,))))
          h.originals['trajectory']=lambda *args,**kwargs:model._trajectory(*args,**kwargs)
          for _ in range(64):model.trajectory(None,None,None,2033,3,6)
          h.original_matmul,h.originals = actual_matmul,original_trajectory''')
  # The same actual storage installer has constructor restoration. Exercise its callback with a CPU constructor stand-in.
  # Cache I/O is not repeated here; its immutable evidence gap is explicit in the recovery receipt.
  tail='''
          def captured(names):
            call=SimpleNamespace(arg=SimpleNamespace(aux=SimpleNamespace(kernels=[('AMD',n) for n in names])))
            call.without_after=call
            return SimpleNamespace(_linear=SimpleNamespace(src=(call,)))
          model.prefill_jit.captured=captured(['agent_packed_epilogue_cpu']*28)
          model.prefill_jit.cnt=64
          for batch in (1,2):
            j=type(model.prefill_jit)();j.cnt=64
            j.captured=captured(['agent_vision_tiles_3x3_cpu']*25+['agent_vision_tiles_3x6_cpu']*25)
            cache.encoders[((1,720,768),(1,720,1152))*batch]=j
          assert comp.pv_handle.info['active'] and not comp.pv_handle.info['cleanup_exact']
          _reject('old_final_validator_inside_active_scope',lambda:_N.validate_pv(h.info,True,64,h.info['evidence']))
          _events.append('actual_composition_populated_active')
          _live.extend([comp,model,h,cache])
'''
  def dedent(text,n):return '\n'.join(line[n:] if line.strip() else '' for line in text.splitlines())+'\n'
  # Setup returns dispatch doubles. populate runs inside production execute's already-entered Composition.
  code=dedent(pre,2)+'\ncandidate=True\n'+dedent(setup,4)
  code+='\nglobal_before = renderer,renderer.compiler,PARALLEL.value,dict(to_program_cache)\n'
  code+='\ndef populate(comp):\n  with ExitStack() as mocks:\n'
  code+="    mocks.enter_context(patch.object(Fake,'ndim',property(lambda self:len(self.shape)),create=True))\n"
  code+='    mocks.enter_context(patch.object(comp.vision,\'Tensor\',Fake));mocks.enter_context(patch.object(comp.vision,\'gemm\',vision_gemm))\n'
  code+='\n'.join('    '+line for line in dedent(body+tail,10).splitlines())+'\n'
  env=ModuleType('recovery_neural_fixture');env.__file__=str(S.EXP/'agent-expert-combined-check.py')
  # Retain original test imports without executing main or its trailing unrelated comparator tests.
  exec(compile(source[:source.index('def main():')],env.__file__,'exec'),env.__dict__)
  env.C.MLP=S.BASE
  env._original,env._reject,env._N,env._events,env._live=original,reject,N,events,live
  exec(compile(code,'<recovery-postrun-neural-fixture>','exec'),env.__dict__)
  actual_loader=C.load
  installed=[]
  def load(path,name):
    m=actual_loader(path,name)
    if Path(path)==C.OLD/'agent-expert-live-storage-install.py':
      real=m.factory
      def factory(*args,**kw):
        cls=real(*args,**kw)
        def construct(mode):
          h=cls(mode,device_factory=lambda:SimpleNamespace(arch='gfx1200',device='AMD'),mlp_dispatch=env.dispatch)
          installed.append(h.composition);return h
        return construct
      m.factory=factory
    return m
  validate=N.validate_pv
  def final_validator(info,*args):
    events.append('final_validator_after_exit' if not info['active'] else 'validator_while_active')
    return validate(info,*args)
  def run(plan,arm,folder):
    assert len(installed)==1;comp=installed[0]
    # Trigger the actual storage installer's restore_constructor callback, with no weights or device opened.
    class Model:
      def __init__(self):pass
    init=Model.__init__
    fake=SimpleNamespace(EdgeModel=Model,load=init,original_init=init,Path=Path,tempfile=__import__('tempfile'))
    comp.after_load(fake,S.BASE/'experiments/benchmark-fast.py')
    env.populate(comp)
    result=copy.deepcopy(original);result.update(completed=True,pipeline=str(R.FAILED/'pipeline.json'),pipeline_sha256=R.PINS['pipeline.json'],
      CPU_TEST_ONLY=True,gpu_fields_are_saved_evidence_playback=True,native_execution_in_this_test=False)
    result.pop('failure')
    # Saved actual CK data is replayed only as CPU test input. This output is explicitly never GPU evidence.
    S.C.selected_coverage(result);events.append('production_postrun_coverage_returned')
    S.write(folder/'report.json',result);return result
  Q,text=R.quality_module()
  Q.S=SimpleNamespace(**vars(S));Q.S.configure=R.configure
  old_runner=Q.S.runner
  Q.S.runner=lambda common:SimpleNamespace(configure=lambda arm:None,run=run)
  folder=output.parent/(output.stem+'-postrun');assert not folder.exists();folder.mkdir()
  old_spec=importlib.util.spec_from_file_location
  from tinygrad.runtime.support import hcq2
  old_queues=hcq2._build_queues
  try:
    with patch.object(C,'load',load),patch.object(N,'validate_pv',final_validator),patch.object(type(Device),'__getitem__',
        lambda *a:(_ for _ in ()).throw(AssertionError('CPU test cannot open hardware'))):
      result=Q.execute(S.verify(),folder)
  finally:
    for name in ('model','benchmark','inputs','native_warp','stream_inputs'):__import__('sys').modules.pop(name,None)
  assert result['completed'] and result['selected_exact']['exact'] and result['selected_constructor_restored']
  comp,model,h,cache=live
  assert comp.session.stats['methods_restored'] and not h.active and h.info['cleanup_exact'] and h.info['observer_cleanup_exact']
  assert importlib.util.spec_from_file_location is old_spec and hcq2._build_queues is old_queues
  assert not hasattr(model,'agent_expert_pv_qualified') and not hasattr(model,'agent_vision_tiles') and not cache.encoders
  assert events.index('production_postrun_coverage_returned')<events.index('final_validator_after_exit')
  assert not Device._opened_devices
  for field,value in (('cleanup_exact',False),('observer_cleanup_exact',False),('active',True),('bound_program_calls',0)):
    bad=copy.deepcopy(h.info);bad[field]=value
    reject('original_PV_'+field,lambda:validate(bad,True,64,bad['evidence']))
  bad=copy.deepcopy(original);bad['prefills'][63]['all28_audit_counts'][27][0]=1
  reject('last_layer_final_audit',lambda:R.recorded_arm(bad))
  bad=copy.deepcopy(original);bad['raw_native_pointers'][0][0]['offset_bytes']=0
  reject('physical_output_offset',lambda:R.ck_evidence(bad))
  bad=copy.deepcopy(original);bad['capture_inspections'][62]['graphs'][1]['native']=27
  reject('last_linked_CK_count',lambda:R.ck_evidence(bad))
  reject('wrong_failed_SHA',lambda:R.gate(R.FAILED/'report.json','0'*64,R.ROOT/'receipt-v1.json',S.sha(R.ROOT/'receipt-v1.json')))
  reject('wrong_receipt_SHA',lambda:R.gate(R.FAILED/'report.json',R.FAILED_SHA,R.ROOT/'receipt-v1.json','0'*64))
  assert S.sha(R.FAILED/'report.json')==R.FAILED_SHA and S.read(R.FAILED/'report.json')['completed'] is False
  diff=''.join(difflib.unified_diff((S.EXP/'agent-ck-selected-quality.py').read_text().splitlines(True),text.splitlines(True),
    fromfile='frozen-quality.py',tofile='recovery-quality-overlay.py'))
  (R.ROOT/'quality-order.diff').write_text(diff)
  S.write(output,dict(complete=True,gpu_executed=False,actual_production_execute_postcoverage_finally=True,
    actual_composition_and_PV_close_and_observer_exit=True,original_final_validators_unmodified=True,
    exact_original64_recomputed=True,events=events,rejected=rejected,
    test_doubles=['neural arithmetic','captured GPU metadata replayed from immutable report','weight constructor without cache I/O','CK output report playback'],
    original_failure_unchanged=True))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)

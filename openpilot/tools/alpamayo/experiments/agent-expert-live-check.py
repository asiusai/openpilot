"""CPU proof of saved-reference reuse, live PV ownership, storage admission, and cleanup."""
import argparse
import copy
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import pickle
import runpy
import socket
import sys
from types import MethodType,SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('expert_live_check_common',HERE/'agent-expert-live-common.py')
N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)


def forbidden(*args,**kwargs): raise AssertionError('Device opening and network access forbidden')


def admission():
  N.configure(); N.paths()
  # Retain the fully checked cache and actual-loader dry-run implementation unchanged.
  frozen = N.read(N.CACHE_PROTOCOL)
  for name in ('common','preflight'):
    path = N.CACHE_ROOT/f'agent-expert-combined-cache-{name}.py'
    assert N.sha(path) == frozen['files'][str(path)]
  cache = N.load(N.CACHE_ROOT/'agent-expert-combined-cache-common.py','expert_live_frozen_cache')
  checked = cache.cache_check()
  preflight = N.load(N.CACHE_ROOT/'agent-expert-combined-cache-preflight.py','expert_live_frozen_loader_probe')
  shim = SimpleNamespace(paths=N.paths,BASE=N.MANIFEST,read=N.read,sha=N.sha,load=N.load)
  preflight.N = SimpleNamespace(original=lambda:shim,LOADER=N.LIVE/'experiments/benchmark-fast.py',
    read=N.read,CACHE=N.WEIGHTS,ARTIFACTS=N.ARTIFACTS)
  invocation = N.read(N.LIVE.parent/'pipeline-invocation.json')['command']
  checkpoint = Path(invocation[invocation.index('--checkpoint')+1])
  with patch.object(socket.socket,'connect',forbidden): factory = preflight.probe(checkpoint)
  from tinygrad import Device
  with patch.object(type(Device),'__getitem__',forbidden),patch.object(socket.socket,'connect',forbidden):
    namespace = runpy.run_path(str(N.LIVE/'experiments/benchmark-pipeline.py'),run_name='expert_live_actual_runpy_probe')
    exec('import wgp_compiler',namespace)
    module = namespace['wgp_compiler']; assert Path(module.__file__).resolve() == N.LIVE/'experiments/wgp_compiler.py'
    compiler = module.WGPCompiler('gfx1200'); assert type(pickle.loads(pickle.dumps(compiler))) is type(compiler)
  assert not Device._opened_devices
  return {'complete':True,'gpu_executed':False,'cache':checked,'factory':factory,
    'actual_runpy_import_and_pickle':True,'reserve':N.reserve(N.ARTIFACTS),'actual_storage_bindings':N.storage()}


def controls():
  N.configure(); N.paths(); evidence = N.evidence(); m,manifest = N.live(); P,pv_evidence = N.pv()
  from tinygrad import Device,dtypes
  from tinygrad.helpers import PARALLEL
  from tinygrad.codegen import to_program_cache
  from tinygrad.engine.realize import _get_call_to_compile
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import UOp,Ops
  assert not Device._opened_devices
  installer = N.load(HERE/'agent-expert-live-install.py','expert_live_cpu_install')
  Hooks = installer.factory(N); hooks = Hooks('consumer')
  oldchecks = N.load(N.EXP/'agent-exact-live-check.py','expert_live_saved_cpu_fixtures'); oldchecks.m = m
  fixtures = N.load(m.c.MLP/'experiments/agent-fused-mlp-check.py','expert_live_model_fixtures')
  Fake,original_model = fixtures.FakeTensor,fixtures.FakeModel
  renderer = P.usb().renderers()['parent-wgp']
  class FakeDevice:
    DEFAULT = 'USB+AMD:LLVM'
    def __class_getitem__(cls,key): return SimpleNamespace(renderer=renderer)
  rejected=[]
  def reject(label,fn):
    try: fn()
    except (AssertionError,ValueError,RuntimeError,KeyError): rejected.append(label)
    else: raise AssertionError('Invalid live control passed: '+label)
  initial_spec,initial_queue = importlib.util.spec_from_file_location,hcq2._build_queues
  before_global = renderer,renderer.compiler,PARALLEL.value,dict(to_program_cache)
  closed=[]
  with ExitStack() as doubles:
    doubles.enter_context(patch.object(type(Device),'__getitem__',forbidden))
    doubles.enter_context(patch.object(socket.socket,'connect',forbidden))
    try:
      with hooks:
        reject('nested live hooks',lambda:Hooks('consumer').__enter__())
        replay = m.load(m.ROOT/'experiments/agent-live-replay.py','expert_live_cpu_replay')
        dataset = replay.Dataset(m.WORK/'quality-samples.json',replay.SOURCE/'manifest.json')
        pipeline = m.load(m.ROOT/'experiments/benchmark-pipeline.py','expert_live_cpu_pipeline')
        normalized,args,protocol = replay.normalize(m.INVOCATION,dataset,pipeline,N.ARTIFACTS/'unused-report.json')
        reference,consumer = N.read(N.REFERENCE),N.read(N.CONSUMER)
        m.equal(protocol,reference['protocol'],'saved_complete_protocol')
        assert (args.warmup,args.runs) == (0,2) and args.samples == manifest['selected_samples']
        assert normalized['environment']['ALPAMAYO_FP8_CACHE'] == str(N.WEIGHTS)
        assert args.vision_calibration is None and args.steps == 3 and args.pad_prompt_to == 2048
        for i,row in enumerate(reference['rows']):
          assert row['sample'] == dataset.samples[i] and row['query_ns'] == dataset.rows[i]['query_ns']
        wrong = SimpleNamespace(samples=dataset.samples[::-1],provenance=dataset.provenance)
        reject('changed order',lambda:replay.normalize(m.INVOCATION,wrong,pipeline,N.ARTIFACTS/'unused-report.json'))
        actual_load = m.load
        def fake_model(core):
          model = original_model(core)
          model.config['expert_config'] = {'llm_config':{'num_hidden_layers':28}}
          model.trajectory_jits={}; model.cache=[Fake(shape=(2,1,8,2304,128),dtype=dtypes.bfloat16)]*28
          for name in ('trajectory','_trajectory','expert_step'): setattr(model,name,MethodType(getattr(core.EdgeModel,name),model))
          expert = actual_load(m.ROOT/'experiments/expert-kernels.py','expert_live_cpu_expert')
          doubles.enter_context(patch.object(expert,'pack',lambda x:x)); doubles.enter_context(patch.object(expert,'unpack',lambda x:x))
          with patch('tinygrad.Device',FakeDevice): expert.install(model)
          model._cpu_expert = expert
          return model
        doubles.enter_context(patch.object(fixtures,'FakeModel',fake_model))
        def loading(path,name):
          return fixtures if Path(path) == m.c.MLP/'experiments/agent-fused-mlp-check.py' else actual_load(path,name)
        with patch.object(m,'load',loading): model = oldchecks.fake_factory(hooks,doubles)
        h = hooks.composition.pv_handle
        assert before_global == (renderer,renderer.compiler,PARALLEL.value,dict(to_program_cache))
        assert hooks.composition.session.frame is None and not hooks.composition.session.state['prefill']
        shapes=P.usb().load('agent-expert-pv-compile').specs(P.SHAPE)
        buffers=[UOp.placeholder(s,dtypes.bfloat16,i) for i,s in enumerate(shapes)]
        original_matmul,originals=h.original_matmul,h.originals.copy()
        h.original_matmul=lambda *a,**kw:h.prepared['current']['ast']
        def trace(*args):
          for step in range(3):
            for layer in range(28):
              model._cpu_expert._matmul(*buffers,shape=(1,16,8,64,2368,128),scaled=True)
              selected=model._cpu_expert._matmul(*buffers,shape=P.SHAPE)
              assert selected is h.prepared['candidate']['program'] and selected.op is Ops.PROGRAM
        h.originals['_trajectory']=trace
        model._trajectory(None,None,None,2033,3,6)
        assert h.info['traces']==[{'pv':84,'qk':84}] and h.info['bound_program_calls']==84
        params=[UOp.param(i,x.dtype,x.numel(),device='AMD') for i,x in enumerate(buffers)]
        assert _get_call_to_compile(h.prepared['candidate']['program'].call(*params)) is None
        assert model._cpu_expert._matmul(*buffers,shape=P.SHAPE) is h.prepared['current']['ast']
        h.depth=1
        for shape,scaled,causal in ((P.SHAPE,True,False),(P.SHAPE,False,True),((1,16,8,32,128,2368),False,False)):
          assert model._cpu_expert._matmul(*buffers,shape=shape,scaled=scaled,causal=causal) is h.prepared['current']['ast']
        h.depth=0; h.original_matmul,h.originals=original_matmul,originals
        before_startup=replay.execute_existing_launcher.__globals__['_startup_before_audit']
        owner=SimpleNamespace(_startup_spec={'ready':True}); seen=[]
        with patch.object(hooks.warm,'initialize',lambda c,s,a,**kw:seen.append((c,a)) or {'ready':True}):
          before_startup(owner,None,SimpleNamespace(rows=[1],load=lambda row:{'current':True}),'consumer')
        assert seen==[(owner,{'current':True})]
        binding=SimpleNamespace(bind_components=lambda c,*a,**kw:SimpleNamespace(inputs=c.inputs,close=lambda:closed.append(True)))
        hooks.runtime(binding)
        component=SimpleNamespace(model=model,inputs=object(),_startup_spec={'ready':True},vision_cache=hooks.composition.cache,
          decoder=SimpleNamespace(model=model,history=[],statistics=[],prefill_draft_records=[]))
        bad=copy.copy(component); bad.model=object(); reject('foreign model owner',lambda:binding.bind_components(bad,None))
        component.decoder.history=[[1]]; reject('nonempty initial history',lambda:binding.bind_components(component,None)); component.decoder.history=[]
        assert binding.bind_components(component,None).inputs is component.inputs
        reject('duplicate Consumer',lambda:binding.bind_components(component,None))
        proof=hooks.coverage(); assert proof['consumer_bindings']==1
        missing=hooks.composition.vision_info['trace_calls'].pop('vlm.model.visual.encoder.layers.26.mlp.fc2/batch2')
        reject('missing vision projection',hooks.coverage)
        hooks.composition.vision_info['trace_calls']['vlm.model.visual.encoder.layers.26.mlp.fc2/batch2']=missing
        # Inspect paced insertion without starting a producer, clock, socket, or Consumer.
        paced=m.load(m.ROOT/'experiments/agent-paced-replay.py','expert_live_cpu_paced')
        before_pace=paced.model_run.__globals__['_startup_before_binding']
        reject('source clock already started',lambda:before_pace(None,None,None,SimpleNamespace(thread=None),SimpleNamespace(anchor=1)))
        m.equal(hooks.transforms,manifest['expected_transforms'],'original_startup_transform_identity')
        raise KeyboardInterrupt('Deliberate CPU interrupted cleanup')
    except KeyboardInterrupt: pass
  assert importlib.util.spec_from_file_location is initial_spec and hcq2._build_queues is initial_queue
  assert h.info['cleanup_exact'] and h.info['observer_cleanup_exact'] and not h.active
  assert closed and hooks.composition.session.stats['methods_restored'] and not hasattr(model,'agent_vision_tiles')
  assert model._cpu_expert._matmul is original_matmul and model.trajectory == originals['trajectory']
  # Exercise the new report gate with explicitly synthetic metadata over the saved TWO-query Consumer.
  data={'reference_admission':{'synthetic':True},'evidence':evidence}
  candidate=copy.deepcopy(consumer)
  prior_pv=N.read(N.WORK/'agent-expert-combined-cache-native-candidate/report.json')['agent_expert_combined']['pv']
  candidate['agent_expert_live']={'protocol_sha256':'CPU','mode':'consumer','passed':True,'hooks_restored':True,
    'reference_sha256':N.REFERENCE_SHA,'saved_reference_admission':data['reference_admission'],'pv':prior_pv,
    'transforms':{k:v for k,v in manifest['expected_transforms'].items() if k!='paced'},
    'coverage':consumer['agent_exact_live']['coverage']}
  N.qualify(candidate,data,'CPU')
  for field in ('kv28','owner','history','sixth_target','floatbits','signedzero','nonfinite','startup','stage','binary','qk','pv83','cleanup'):
    changed=copy.deepcopy(candidate)
    if field=='kv28': changed['rows'][0]['defined_kv_prefix_bits'][27]['sha256']='changed'
    elif field=='owner': changed['rows'][0]['consumer_protocol']['owned_input_objects']=2
    elif field=='history': changed['rows'][0]['history_before']['history']=[[1]]
    elif field=='sixth_target': changed['rows'][0]['speculation']['targets'][5]+=1
    elif field in ('floatbits','signedzero','nonfinite'):
      values=changed['rows'][0]['prediction']['xyz']
      while isinstance(values[0],list): values=values[0]
      values[0]=float('nan') if field=='nonfinite' else -0.0 if field=='signedzero' else values[0]+1e-7
    elif field=='startup': changed['startup_spec']['target_payload_bits_restored']=False
    elif field=='stage': changed['startup_spec']['stages'].pop()
    elif field=='binary': changed['agent_expert_live']['pv']['captures'][0]['pv_full_description']['binary_sha256']='changed'
    elif field=='qk': changed['agent_expert_live']['pv']['captures'][0]['binaries']['alpamayo_attention_scores']=['changed']
    elif field=='pv83': changed['agent_expert_live']['pv']['captures'][0]['pv_calls']=83
    else: changed['agent_expert_live']['pv']['cleanup_exact']=False
    reject('report/'+field,lambda:N.qualify(changed,data,'CPU'))
  assert not Device._opened_devices
  for name in ('model','benchmark','inputs','native_warp','stream_inputs'): sys.modules.pop(name,None)
  return {'complete':True,'gpu_executed':False,'rejected':rejected,'transforms':hooks.transforms,
    'reference_admission':{'source_protocol_inputs_order_empty_history_and_ownership_proven':True,
      'reference_sha256':N.REFERENCE_SHA,'previous_consumer_sha256':N.CONSUMER_SHA,
      'normalized_protocol_exact':True,'full_dataset_and_two_selected_input_snapshots_exact':True,
      'selected_indices':[0,16],'initial_history':'empty','owner_behavior':'unchanged frozen bind and Consumer; compared again at runtime',
      'full_startup_transform_bytes_exact':True,'only_numeric_addition':'Qualified complete WGP 2x2 expert PV PROGRAM',
      'storage_only_changes':N.storage(),'original_CU_program_identity_source':'Actual prior USB capture and exact composed quality4/native36 reference'},
    'integration':{'all28_mlp':True,'all50_vision_batch1_and2':True,'pv84_qk84':True,'complete_program_no_worker_recompile':True,
      'global_compiler_parallel_cache_unchanged':True,'interrupted_cleanup_exact':True},'fresh_gpu_equality_established':False}


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); p.add_argument('--admit',action='store_true')
  args=p.parse_args(); assert not args.output.exists()
  result=admission() if args.admit else controls()
  result['source_sha256']={str(p):N.sha(p) for p in HERE.glob('agent-expert-live-*.py')}
  N.write(args.output,result,new=True)
  print({'complete':True,'gpu_executed':False,'rejections':len(result.get('rejected',[]))})

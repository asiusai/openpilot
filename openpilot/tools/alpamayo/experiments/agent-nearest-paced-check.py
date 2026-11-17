"""CPU admission mutations and real received-only adapters; no inference/device access."""
import copy, importlib.util, json, socket, sys
from pathlib import Path
from types import SimpleNamespace, MethodType
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_paced_check_admit',HERE/'agent-nearest-paced-admit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def reject(fn):
  try: fn()
  except (ValueError, AssertionError, RuntimeError, KeyError): return True
  raise AssertionError('Mutation or invalid admission was accepted')

def mutations(c,plan,startup,protocol,reference,saved):
  checks={}
  changes=(
    ('reference_history',lambda r,s:r['rows'][1]['history_before']['history'].append([123])),
    ('consumer_history',lambda r,s:s['rows'][1]['history_before']['history'].append([123])),
    ('reference_result_bit',lambda r,s:r['rows'][1]['prediction']['xyz'][0][0].__setitem__(0,r['rows'][1]['prediction']['xyz'][0][0][0]+1e-12)),
    ('consumer_result_bit',lambda r,s:s['rows'][3]['prediction']['rot'][0][0][0].__setitem__(0,s['rows'][3]['prediction']['rot'][0][0][0][0]+1e-12)),
    ('nonfinite_result',lambda r,s:r['rows'][0]['prediction']['xyz'][0][0].__setitem__(0,float('nan'))),
    ('input_hash',lambda r,s:r['rows'][0]['cpu_buffers'][0].update(sha256='0'*64)),
    ('statistics',lambda r,s:r['rows'][1]['decoder_statistics'].update(serial_tokens=99)),
    ('speculation',lambda r,s:r['rows'][0]['speculation'].update(accepted=False)),
    ('completion',lambda r,s:r['rows'][0].update(completed=False)),
    ('ownership',lambda r,s:r['reference_ownership'][0].update(owned_input_objects=2)),
    ('startup_kv',lambda r,s:r['startup_spec'].update(full_kv_bits_restored=False)),
    ('startup_history_scope',lambda r,s:r['startup_spec'].update(future_inputs=1)),
    ('startup_graphs',lambda r,s:r['startup_spec']['graph_counts'].update(jits=0)),
    ('factory_source',lambda r,s:r['initialized']['source_sha256'].update(**{'model.py':'0'*64})),
    ('runner_source',lambda r,s:r['source_sha256'].update(**{'agent-nearest-order-run.py':'0'*64})),
    ('wrong_saved_hash',lambda r,s:r.update(saved_consumer_sha256='0'*64)),
    ('old_failure_relabeling',lambda r,s:s.update(predictions_qualified=True)),
    ('cpu_as_gpu',lambda r,s:r.update(gpu_executed=False)),
  )
  for name,change in changes:
    r,s=copy.deepcopy(reference),copy.deepcopy(saved);change(r,s)
    checks[name]=reject(lambda:candidate(c,plan,startup,protocol,r,s))
  checks['actual_source_pin']=reject(lambda:a.check_pins({HERE/'agent-nearest-paced-run.py':'0'*64}))
  return checks

def candidate(*args): return a.pair(*args)

def admission_components(saved):
  """Host objects for the new read-only bind guard; no fake inference result."""
  settings=SimpleNamespace(**copy.deepcopy(saved['initialized']['settings']))
  decoder=SimpleNamespace(history=[],statistics=[],prefill_draft_records=[],prefill_draft_old_history=[])
  def run(self,*args,**kwargs): raise AssertionError('CPU checks must not infer')
  decoder.run=MethodType(run,decoder)
  config=SimpleNamespace();tokenizer=object();owner=object()
  return SimpleNamespace(report=copy.deepcopy(saved['initialized']),settings=settings,_startup_spec=copy.deepcopy(saved['startup_spec']),
    startup_active=[False],startup=None,decoder=decoder,vision_cache=SimpleNamespace(cache={},hits=0,misses=0,verify=False),
    inputs=owner,runner=decoder.run,adapter=SimpleNamespace(config=config,tokenizer=tokenizer),config=config,tokenizer=tokenizer,
    pad_inputs=lambda *args:None,profiler=None,provenance={'CPU_host_fixture':True},vision_provenance={},model=object(),input_class=object)

def composition(c,plan,policy,startup,dynamic,saved,run):
  warm=a.load(Path(plan['startup_runner']).with_name('agent-startup-spec-warm.py'))
  records=dict(bindings=[],compiler_cache_relocation=[]);transforms={}
  output=HERE/'agent-nearest-paced-cpu-fixture/report.json'
  with policy.imports(c.BASE),run.observer(c,plan,saved,output,records),startup.imports(c.BASE,warm,{},transforms),dynamic.imports():
    paced=dynamic.local('agent-paced-replay')
    adapted_local=paced.model_run.__globals__['local']
    replay=adapted_local('agent-live-replay');runtime=adapted_local('agent-live-runtime')
    data=SimpleNamespace(samples=saved['qualified_factory_protocol']['replay_order'],provenance=saved['qualified_factory_protocol']['dataset'])
    normalized,parsed,protocol=replay.normalize(Path(plan['invocation']),data,c.local('benchmark-pipeline'),output)
    assert protocol==saved['qualified_factory_protocol']
    assert normalized['environment']['CACHEDB']==str(output.parent/'compiler-cache.db')
    before=copy.deepcopy(normalized['environment']);before['CACHEDB']=records['compiler_cache_relocation'][0]['before']
    assert before==protocol['environment']
    live=dynamic.local('agent-live-consumer')
    Store,Preparer,Consumer=paced.bridge.Store,paced.bridge.Preparer,live.Consumer
    originalStore,originalPreparer,originalConsumer=Store.__bases__[0],Preparer.__bases__[0],Consumer.__bases__[0]
    for name in ('add_frame','add_state','bounds'): assert getattr(Store,name) is getattr(originalStore,name)
    for name in ('predict','synchronize','_bind_inputs','_check_bounds','_guard_startup','close'):
      assert getattr(Consumer,name) is getattr(originalConsumer,name)
    checks=a.load(c.FROZEN/'agent-nearest-live-check.py')
    availability=checks.independent(policy,paced.bridge,Store)
    arrays=replay.Dataset.load(dict(sample=plan['cases'][0]['sample']));meta=copy.deepcopy(plan['cases'][0]['metadata'])
    Consumer._data(SimpleNamespace(last_query_ns=0),arrays,meta,meta['prepared_ns'])
    metadata_controls={}
    for name,change in (
      ('unreceived',lambda m:m['frames'][0].update(received_ns=m['selected_ns']+1,ready_ns=m['selected_ns']+1)),
      ('not_ready',lambda m:m['frames'][0].update(ready_ns=m['selected_ns']+1)),
      ('future_capture',lambda m:m['frames'][0].update(capture_ns=m['query_ns']+1)),
      ('wrong_capture_timestamp',lambda m:m['frames'][0].update(capture_ns=m['frames'][0]['capture_ns']+1000)),
      ('future_calibration',lambda m:m['frames'][0].update(image_calibration_ns=m['frames'][0]['capture_ns']+1)),
      ('wrong_policy',lambda m:m.update(history_policy='all-offline-captures'))):
      bad=copy.deepcopy(meta);change(bad)
      metadata_controls[name]=reject(lambda:Consumer._data(SimpleNamespace(last_query_ns=0),arrays,bad,bad['prepared_ns']))
    components=admission_components(saved);clock=SimpleNamespace(anchor=None)
    # The existing binder executes all ownership/settings guards. Only the final
    # Consumer constructor is a host fixture, because this test owns no GPU tensors.
    class HostConsumer:
      def __init__(self,**kwargs):
        self.inputs=kwargs['inputs'];self.provenance={'nearest_live_policy':dict(policy=policy.POLICY,source_sha256=a.sha(c.FROZEN/'agent-nearest-live-policy.py'))}
    bound_args=replay.binding_args(parsed);components.vision_cache.verify=bound_args.verify_vision_cache
    with patch.object(runtime,'local',return_value=SimpleNamespace(Consumer=HostConsumer)):
      consumer=runtime.bind_components(components,bound_args,clock=clock)
    assert consumer.inputs is components.inputs and len(records['bindings'])==1
    binding_mutations={}
    for name,change in (
      ('decoder_history',lambda v:v.decoder.history.append([1])),
      ('logical_vision_cache',lambda v:v.vision_cache.cache.update(stale=True)),
      ('startup_restoration',lambda v:v._startup_spec.update(full_kv_bits_restored=False)),
      ('source',lambda v:v.report['source_sha256'].update(**{'model.py':'0'*64}))):
      bad=admission_components(saved);change(bad)
      binding_mutations[name]=reject(lambda:run.before_binding(c,bad,clock,saved,replay))
    binding_mutations['source_clock_started']=reject(lambda:run.before_binding(c,admission_components(saved),SimpleNamespace(anchor=1),saved,replay))
    simulation=schedule(c,paced,Consumer)
  assert paced.bridge.Store is originalStore and paced.bridge.Preparer is originalPreparer and live.Consumer is originalConsumer
  return dict(import_composition_exact=True,startup_transforms=transforms,original_methods_unchanged=True,
    original_module_aliases_restored=True,normalizer_changes_only_compiler_cache_path=True,
    real_binder_guards_with_host_constructor=True,availability=availability,metadata_rejections=metadata_controls,
    binding_rejections=binding_mutations,scheduling=simulation)

def schedule(c,paced,Consumer):
  """One400ms injected-service scenario with the actual recorded timestamps and scheduler."""
  import numpy as np
  fixture=c.local('agent-paced-replay-check');bridge=paced.bridge
  trace=paced.Trace(paced.SOURCE/'manifest.json',paced.LOGS,35.,8.)
  assert len(trace.frames)==450
  host=fixture.HostClock();clock=paced.Clock(trace.start,host)
  store=bridge.Store(frame_limit=64,byte_limit=384*1024**2)
  class RGB(fixture.RGBDouble):
    def __call__(self,ref):
      # IO takes real simulated elapsed time AFTER receipt, never at publication.
      if clock.anchor is not None: host.advance(200_000)
      return super().__call__(ref)
  producer=paced.Producer(trace,store,RGB(),clock)
  prep=bridge.Preparer(trace.manifest['source_camera_geometry'],trace.manifest['target_camera_geometry'],warp=fixture.fake_warp,clock=clock)
  nav=paced.Navigation(prep,trace);runner=bridge.Bridge(store,nav,hz=10.,clock=clock,require_small_model=False);nav.runner=runner
  recorded=[];selections=[]
  class HostConsumer:
    def __init__(self): self.inputs=SimpleNamespace(uploaded_pixel_bytes=0);self.last_query_ns=0;self.calls=self.completions=0
    def predict(self,arrays,meta):
      Consumer._data(self,arrays,meta,clock())  # Actual nearest Consumer timestamp/shape guard.
      self.calls+=1;self.last_query_ns=meta['query_ns'];selections.append(copy.deepcopy(meta))
      before=arrays['absolute_timestamps'].copy()
      assert runner.active.locked()
      old_reads=producer.rgb.reads
      assert runner.run_once(self) is None  # No second preparation or future query while active.
      assert producer.rgb.reads==old_reads
      host.advance(400_000_000);producer.pump()
      np.testing.assert_array_equal(arrays['absolute_timestamps'],before)
      return dict(query_ns=meta['query_ns'],prediction=dict(xyz=np.zeros((1,64,3)).tolist(),CPU_fixture=True))
    def synchronize(self): self.completions+=1
  consumer=HostConsumer();components=SimpleNamespace(vision_cache=SimpleNamespace(hits=0,misses=10),decoder=SimpleNamespace())
  producer.prime_history();assert producer.prime['latest_release_ns']<=trace.start
  # Metadata exists for the full window; the store has only admitted past/current frames.
  assert all(f.ready_ns<=trace.start for rows in store.frames.values() for f in rows)
  clock.start()
  def idle():
    options=[trace.end]
    if runner.deadline>clock(): options.append(runner.deadline)
    if producer.index<len(trace.events) and trace.events[producer.index].release_ns>clock(): options.append(trace.events[producer.index].release_ns)
    host.advance(max(1,min(options)-clock()))
  summary=paced.observe(trace,producer,runner,consumer,components,clock=clock,budget_seconds=180.,emit=recorded.append,idle=idle,before_tick=producer.pump)
  rows=[r for r in recorded if r['kind']=='model_completion']
  assert len(rows)==consumer.calls==consumer.completions>0
  assert summary['maximum_active_inference']==1 and summary['queued_prediction_snapshots']==0 and summary['requested_hz']==10
  arrivals={(r['view'],r['frame_id']):r for r in producer.arrivals}
  for row in rows:
    meta=row['input'];q=meta['query_ns']
    available=[r for r in producer.arrivals if r['ready_ns']<=meta['selected_ns'] and r['capture_ns']<=q]
    for i,ref in enumerate(meta['frames']):
      source=arrivals[(ref['view'],ref['frame_id'])]
      for key in ('capture_ns','received_ns','ready_ns'): assert ref[key]==source[key]
      assert source['published_ns']<=source['release_ns']<=ref['received_ns']<=ref['ready_ns']<=meta['selected_ns']
      wanted=q+bridge.OFFSETS_NS[i%5];choices=[r for r in available if r['view']==ref['view']]
      expected=max(choices,key=lambda r:r['capture_ns']) if i%5==4 else min(choices,key=lambda r:(abs(r['capture_ns']-wanted),r['capture_ns']))
      assert ref['frame_id']==expected['frame_id']
    assert row['timing']['query_to_output_ms']==(row['completed_ns']-q)/1e6
  assert any(r['ready_ns']>r['received_ns'] for r in producer.arrivals if not r['primed'])
  assert runner.counts['busy_calls']==len(rows)
  assert summary['small_model_guard']=='not asserted; PC recorded replay'
  return dict(actual_trace_frames=450,requested_ceiling_hz=10,one_active=True,no_queued_preparation=True,
    actual_admission_timestamps_preserved=True,all_selections_received_only=True,explicit_PC_small_model_limit=True,
    limit='Injected400ms host consumer and identity-image fixture; not a throughput or output-quality measurement')

def main():
  a.setup_cpu();c,plan,policy,startup,dynamic,protocol,reference,saved,admission=a.verify()
  run=a.load(HERE/'agent-nearest-paced-run.py')
  from tinygrad import Device
  def blocked(*args,**kwargs): raise AssertionError('CPU checks prohibit device opens and network calls')
  with patch.object(type(Device),'__getitem__',blocked),patch.object(socket.socket,'connect',blocked),patch.object(socket.socket,'connect_ex',blocked):
    negatives=mutations(c,plan,startup,protocol,reference,saved)
    adapters=composition(c,plan,policy,startup,dynamic,saved,run)
  assert not Device._opened_devices
  # The original frozen gates and both original results remain unchanged after imports/tests.
  a.verify()
  report=dict(passed=True,cpu_only=True,gpu_executed=False,opened_devices=[],source_sha256=a.sources(),
    admission=admission,admission_rejections=negatives,adapters=adapters,automatic_adoption=False,
    limit='CPU controls validate admission and scheduling adapters only; the optional parent run has not executed')
  a.save(HERE/'agent-nearest-paced-checks.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()

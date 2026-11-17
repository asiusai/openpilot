"""Real canonical/Consumer CPU adapters, host-storage doubles, and exact gate controls."""
import copy,importlib.util,json,socket,sys
from pathlib import Path
from types import SimpleNamespace,MethodType
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_order_check_common',HERE/'agent-nearest-order-common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.setup_cpu()
import numpy as np
import torch
from tinygrad import Device
torch.set_num_threads(1)

def check():
  plan,policy,startup,dynamic=c.verify_assets();saved=c.validate_saved(plan,startup)
  run=c.load(HERE/'agent-nearest-order-run.py');diagnostic=c.load(HERE/'agent-nearest-order-diagnose.py').diagnose()
  replay,normalized,args,factory,changes=run.recipe(plan,dynamic,HERE/'agent-nearest-order-cpu-artifacts/canonical.json')
  cpu=c.local('agent-live-consumer-check');fast=c.local('fast-inputs');compact=c.local('compact-inputs').CompactInputs
  Vision=c.local('vision-cache').VisionCache;speculation=c.local('agent-speculation-policy');live=c.local('agent-live-consumer')
  Consumer=policy.consumer_class(live)
  immutable=(live.Consumer.predict,live.Consumer.synchronize,live.Consumer._data,compact.update,compact.arrays)
  rows=[dict(sample=x['sample'],query_ns=x['query_ns']) for x in plan['cases']]
  dataset=SimpleNamespace(rows=rows,samples=[r['sample'] for r in rows],inputs=[replay.tree(replay.Dataset.load(r)) for r in rows])
  folder=HERE/'agent-nearest-order-cpu-artifacts';folder.mkdir(exist_ok=False)
  fixture_records=[]

  def fixture():
    clock=cpu.Clock();model=cpu.fake_model(clock);model.padded_prefill.last_targets=None
    adapter=fast.FastInputs(args.checkpoint,camera_profile='nvidia',allow_missing_cameras=True,cache_resize=True,direct_byte_inputs=True)
    direct=fast.FastInputs(args.checkpoint,camera_profile='nvidia',allow_missing_cameras=True,cache_resize=True,direct_byte_inputs=True)
    settings=SimpleNamespace(max_tokens=64,vision_batch=1,byte_pixels=True,cache_rotary_inputs=False,coarse_timing=False,
      greedy=True,diffusion_steps=3,initial_noise=np.load(args.initial_noise,allow_pickle=False))
    padding=lambda p,k,l:speculation.padded_inputs(p,k,l,speculation.CHECKPOINT_DRAFT_IDS,adapter.config.traj_ids['future_start'])
    vision=Vision(model,capacity=144);base=cpu.cpu_inputs_class(compact);state={'ready':False,'created':0}
    class Inputs(base):
      def __init__(self,*a,**kw):
        if state['ready']:raise RuntimeError('CPU owner was discarded after initialization')
        super().__init__(*a,**kw);state['created']+=1
        if not self.cache_rotary_inputs:self.gpu_rotary={}
    class Decoder:
      def __init__(self):
        self.model=model;self.history=[];self.prefill_draft_old_history=[];self.prefill_draft_records=[];self.statistics=[]
        self.jits={};self.verifiers={};self.select_jits={};self.warmup_graphs=False;self.before_diffusion=lambda:None;self.calls=0
      def generate(self,*a,**kw):raise AssertionError('CPU fixture cannot execute inference')
      def run(self,m,setting,config,tokenizer,prepared,packed,seed,inputs):
        index=self.calls;self.calls+=1
        assert inputs is owner and state['created']==1 and seed==0
        arrays=replay.Dataset.load(rows[index]);ep,ek,eu,es=cpu.prepare_reference(direct,arrays,settings,compact,padding,2048)
        cpu.exact(prepared,ep)
        for left,right in zip(packed,ek,strict=True):cpu.exact(left,right)
        for left,right in zip(inputs.cpu,eu,strict=True):cpu.exact(left.value,right)
        assert inputs._signature==es
        assert replay.history(self)==saved['rows'][index]['history_before']
        missing=tuple(i for i,key in enumerate(vision.keys) if key not in vision.cache)
        assert len(missing)==10
        def upload(*sources):
          for slot,source in zip(missing,sources[:len(missing)],strict=True):
            begin,end,_=inputs.batches[slot];cpu.exact(source.value,eu[0][begin:end])
          for j,source in enumerate(sources[len(missing):],1):cpu.exact(source.value,settings.initial_noise if j==len(inputs.cpu)-1 else eu[j])
          return inputs.pixel_storage
        inputs.selective_jits[missing]=upload;inputs.upload(settings.initial_noise)
        for key in vision.keys:vision.cache[key]='CPU shape fixture';vision.cache.move_to_end(key)
        vision.hits=0;vision.misses=10
        # Shape-only output fixture. It never qualifies model output or changes a GPU target.
        output=copy.deepcopy(saved['rows'][index]['prediction']);tokens=output['generated_token_ids']
        self.history=c.remember(self.history,tokens)
        self.prefill_draft_records.append(copy.deepcopy(saved['rows'][index]['speculation']))
        self.statistics.append(copy.deepcopy(saved['rows'][index]['decoder_statistics']))
        return output
    decoder=Decoder()
    first=replay.Dataset.load(rows[0]);ep,ek,eu,_=cpu.prepare_reference(adapter,first,settings,compact,padding,2048)
    owner=Inputs(settings,adapter.config,ep,ek,arrays=eu);owner.selective_cache=vision;state['ready']=True
    components=SimpleNamespace(adapter=adapter,model=model,settings=settings,input_class=Inputs,vision_cache=vision,
      decoder=decoder,runner=decoder.run,profiler=None,startup=None,startup_active=[False],vision_provenance={},
      report={'warmup':[],'runs':[]},benchmark=None,Device=SimpleNamespace(DEFAULT='CPU'),
      GlobalCounters=SimpleNamespace(mem_used_per_device={}),config=adapter.config,tokenizer=adapter.tokenizer,pad_inputs=padding,inputs=owner)
    return components,clock,state

  def canonical(adapted):
    components,clock,state=fixture();audit=replay.Audit(dataset);audit.install(components);observed=components.runner
    def complete(bound,*a,**kw):
      result=observed(*a,**kw);components.model.synchronize();audit.complete();return result
    components.runner=MethodType(complete,components.decoder)
    pipeline=c.local('benchmark-pipeline');effective=copy.copy(args)
    effective.output=folder/('canonical.json' if adapted else 'negative-unadapted.json')
    pipeline.parse_args=lambda:effective;pipeline.initialize=lambda *a,**kw:components
    pipeline.snapshot_sources=lambda output:({'CPU_fixture':True},folder)
    original_module=pipeline.local_module
    def local(name):
      if name=='padded-prefill':return SimpleNamespace(pad_inputs=components.pad_inputs)
      return original_module(name)
    pipeline.local_module=local
    transformation=c.transform_main(pipeline) if adapted else None
    pipeline.main()
    assert components.decoder.calls==4 and state['created']==1
    return audit.rows,transformation

  with patch.object(type(Device),'__getitem__',cpu.forbidden),patch.object(socket.socket,'connect',cpu.forbidden):
    try:canonical(False)
    except RuntimeError as error:assert str(error)=='CPU owner was discarded after initialization'
    else:raise AssertionError('Unadapted canonical loop unexpectedly kept the warmed input')
    reference,transformation=canonical(True)
    components,clock,state=fixture();audit=replay.Audit(dataset);audit.install(components)
    consumer=Consumer(adapter=components.adapter,model=components.model,runner=components.runner,settings=components.settings,
      config=components.config,input_class=components.input_class,vision_cache=components.vision_cache,pad_inputs=components.pad_inputs,
      pad_length=2048,initializer_provenance={'CPU_fixture':True},decoder=components.decoder,inputs=components.inputs,
      speculation_policy=speculation,speculation_draft=speculation.CHECKPOINT_DRAFT_IDS,clock=clock)
    for case,row in zip(plan['cases'],rows,strict=True):
      clock.now=case['metadata']['prepared_ns'];result=consumer.predict(replay.Dataset.load(row),copy.deepcopy(case['metadata']))
      consumer.synchronize();audit.complete(consumer_record=result['live_consumer'])
    consumer.close()
    assert consumer.constructed_inputs==0 and consumer.reused_inputs==4 and state['created']==1
  comparison=c.compare_rows(reference,audit.rows);assert comparison['exact']
  saved_comparison=c.compare_rows(reference,saved['rows']);assert saved_comparison['exact'],saved_comparison
  for case,row in zip(plan['cases'],reference,strict=True):
    for key,value in case['expected_cpu_hashes'].items():assert row[key]==value
  assert immutable==(live.Consumer.predict,live.Consumer.synchronize,live.Consumer._data,compact.update,compact.arrays)
  negative={}
  for label,mutate in (
      ('history',lambda x:x[1]['history_before']['history'].clear()),
      ('statistics',lambda x:x[1]['decoder_statistics'].update(serial_tokens=23)),
      ('speculation',lambda x:x[1]['speculation'].update(accepted=True)),
      ('input_hash',lambda x:x[1]['cpu_buffers'][0].update(sha256='changed')),
      ('completion',lambda x:x[1].update(completed=False)),
      ('one_float_bit',lambda x:x[1]['prediction']['xyz'][0][0].__setitem__(0,float(np.nextafter(x[1]['prediction']['xyz'][0][0][0],np.inf)))),
      ('nonfinite',lambda x:x[1]['prediction']['rot'][0][0][0].__setitem__(0,float('nan')))):
    bad=copy.deepcopy(reference);mutate(bad)
    try:value=c.compare_rows(bad,saved['rows'])
    except ValueError:negative[label]=True
    else:negative[label]=not value['exact']
    assert negative[label]
  for label,mutate in (
      ('startup_kv_restore',lambda x:x['startup_spec'].update(full_kv_bits_restored=False)),
      ('warmed_owner',lambda x:x['rows'][0]['consumer_protocol'].update(input_reused=False)),
      ('initial_history',lambda x:x['rows'][0]['history_before']['history'].append([1])),
      ('source_pin',lambda x:x['initialized']['source_sha256'].update(**{'lookup-decode.py':'changed'}))):
    bad=copy.deepcopy(saved);mutate(bad)
    try:c.validate_saved(plan,startup,report=bad)
    except ValueError:negative[label]=True
    else:raise AssertionError('Saved evidence gate accepted mutation: '+label)
  assert not Device._opened_devices
  return dict(passed=True,cpu_only=True,gpu_executed=False,opened_devices=[],source_sha256=c.own_sources(),pins=c.PINS,
    canonical_main_transform=transformation,bounded_recipe_changes=changes,canonical_lookup_diagnosis=diagnostic,
    real_adapter_queries_per_arm=4,real_canonical_main_executed=True,real_consumer_predict_and_synchronize=True,
    warmed_owner_reused_all_four=True,actual_CPU_hashes_match_saved=True,CPU_comparison=comparison,
    original_methods_immutable=True,negative_controls=negative,
    limitation='Host storage and decoded-output fixtures only. No GPU inference, numerical parity or quality qualification.',
    automatic_pace=False,automatic_adoption=False)

if __name__=='__main__':
  result=check();c.save(HERE/'agent-nearest-order-checks.json',result)
  print(json.dumps({k:v for k,v in result.items() if k not in ('canonical_lookup_diagnosis','source_sha256','bounded_recipe_changes')},indent=2))

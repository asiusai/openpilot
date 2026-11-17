"""CPU qualification of actual36 native metadata/worker path and frozen75% contracts."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_token75_timing_check_common',HERE/'agent-token75-timing-common.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def main():
  c.require(os.environ.get('DEV')=='CPU','Set DEV=CPU explicitly')
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',XDG_CACHE_HOME=str(c.ROOT/'agent-token75-timing-cpu-cache'))
  plan=c.read(c.PROTOCOL);c.verify(plan,require_cpu=False)
  sys.path.insert(0,str(c.SNAP));sys.path.insert(0,str(c.FEXP))
  import torch
  torch.set_num_threads(1)
  from tinygrad import Device
  c.require(Device.DEFAULT=='CPU','CPU devices only')
  factory=c.module(c.FEXP/'agent-token-current-runtime.py','agent_token75_timing_cpu_factory')
  policy=factory.policy_module;runtime=factory.make_runtime(c.POLICY)
  route=c.module(c.FEXP/'agent-token-current-route.py','agent_token75_timing_cpu_route')
  route.verify_qualification(Path(plan['relocated_cpu_report']),c.FEXP/'agent-token-current-hook-report.json',c.FEXP/'agent-token-current-compiler-report.json')
  stream=c.module(c.FEXP/'stream-inputs.py','agent_token75_timing_native_stream')
  fast=c.module(c.FEXP/'fast-inputs.py','agent_token75_timing_fast_inputs')
  padded=c.module(c.FEXP/'padded-prefill.py','agent_token75_timing_cpu_padded')
  compact=c.module(c.FEXP/'compact-inputs.py','agent_token75_timing_cpu_compact')
  vision=c.module(c.FEXP/'vision-cache.py','agent_token75_timing_cpu_vision')
  invocation=c.read(plan['invocations']['candidate']);command=invocation['command'];checkpoint=command[command.index('--checkpoint')+1]
  Native=stream.NativeFrames
  class Owner(runtime.NumericalPruner):
    def __init__(self):self.require_packed=True;self.events=[];self.decoders=[];self.metadata=None
  owner=Owner();model=SimpleNamespace(_agent_token_pruner=owner)
  class Decoder:
    def __init__(self,model):self.model=model
    def run(self,model,args,config,tokenizer,prepared,packed,seed,inputs):
      prompt,frames=owner.metadata
      plan=policy.select(np.ones(prompt),prompt,2304,frames,c.POLICY)
      independent_latest={camera:max(frame.capture_ns for frame in frames if frame.camera==camera) for camera in (1,6)}
      for frame in frames:
        kept=int(((plan.slots>=frame.begin)&(plan.slots<frame.end)).sum())
        assert kept==(180 if frame.capture_ns==independent_latest[frame.camera] else 135)
      for layer in range(28):
        prefix=set(range(prompt)) if layer<2 else set(map(int,plan.slots))
        for position in (prompt,prompt+6,prompt+65):
          expected=prefix|set(range(prompt,position+1))
          assert set(np.flatnonzero(np.isfinite(plan.language_mask(layer,[position])[0])))==expected
        expected=prefix|set(range(prompt,prompt+6))|set(range(2304,2368))
        assert set(np.flatnonzero(np.isfinite(plan.expert_mask(layer,prompt+6))))==expected
      self.last_plan=plan
      owner.events.append(dict(policy=c.POLICY,classification='CPU structural contract only; injected equal saliency',
        retained_prompt_tokens=len(plan.slots),prompt_prefix_invariance_pass=False,quality_passed=None))
      return dict(contract_only=True)
  def initializer():raise AssertionError('No production model initializer in CPU fixture')
  modules={'stream-inputs':stream,'fast-inputs':fast,'lookup-decode':SimpleNamespace(LookupDecoder=Decoder)}
  pipeline=SimpleNamespace(ThreadPoolExecutor=ThreadPoolExecutor,local_module=lambda name:modules[name],initialize=initializer)
  records=[]
  with ExitStack() as stack:
    route.install_pipeline_hooks(pipeline,runtime,stack,[])
    native=pipeline.local_module('stream-inputs').NativeFrames(plan['native_manifest'],plan['native_calibrations'],plan['raw_cache'])
    adapter=pipeline.local_module('fast-inputs').FastInputs(checkpoint,camera_profile='nvidia',allow_missing_cameras=True,cache_resize=True,direct_byte_inputs=True)
    decoder=pipeline.local_module('lookup-decode').LookupDecoder(model)
    def prepare_input(sample):
      folder=Path(sample)
      data={p.stem:[str(np.load(p,allow_pickle=False).item())] if p.stem=='nav_text' else torch.from_numpy(np.load(p,allow_pickle=False,mmap_mode='c'))
            for p in folder.glob('*.npy') if p.stem!='image_frames'}
      pixels,timing=native.prepare(sample)
      np.testing.assert_array_equal(pixels,np.load(folder/'image_frames.npy',allow_pickle=False,mmap_mode='r'))
      data['image_frames']=torch.from_numpy(pixels)
      config,tokenizer,prepared=adapter.prepare(data);packed=adapter.native_inputs(prepared)
      prepared,packed=padded.pad_inputs(prepared,packed,2112)
      prompt=prepared['actual_prompt_length']
      assert packed[0].shape[1]==2112 and prompt<2112
      arrays=compact.CompactInputs.arrays(SimpleNamespace(max_tokens=64,byte_pixels=True),prepared,packed)
      keys=vision.VisionCache.input_keys(prepared)
      assert arrays[0].dtype==np.uint8 and len(keys)==10
      assert arrays[2].shape==(1,2112) and arrays[3].shape==(1,1,2112,128)
      assert arrays[5].shape==(1,1,64,128) and arrays[7].shape==(1,1,128,128)
      assert all(array.dtype==np.uint16 for array in arrays[3:9])
      image=np.flatnonzero(packed[0][0]==config.vlm_config.image_token_id)
      spans=tuple((int(x[0]),int(x[-1])+1) for x in np.split(image,np.flatnonzero(np.diff(image)!=1)+1))
      return config,tokenizer,prepared,packed,SimpleNamespace(spans=spans),timing
    with pipeline.ThreadPoolExecutor(max_workers=1) as pool:
      pending=pool.submit(prepare_input,plan['samples'][0])
      for index,sample in enumerate(plan['samples']):
        config,tokenizer,prepared,packed,inputs,timing=pending.result()
        pending=pool.submit(prepare_input,plan['samples'][index+1]) if index+1<36 else None
        result=decoder.run(model,None,config,tokenizer,prepared,packed,0,inputs)
        assert result['contract_only'] and result['token_prune_policy']['query_ns']==plan['queries_ns'][index]
        current=owner.metadata
        if pending is not None:pending.result()
        assert owner.metadata==current,'Future worker changed current controller metadata'
        observations=prepared['_agent_token_prune_observations']
        assert len(observations.frames)==10 and all(stamp<=observations.query_ns for _,stamp in observations.frames)
        assert prepared['tokenized_data']['image_grid_thw'].tolist()==[[1,20,36]]*10
        records.append(dict(sample=sample,query_ns=observations.query_ns,
          canonical_frames=list(observations.frames),selected_prompt_tokens=len(decoder.last_plan.slots),
          actual_prompt=prepared['actual_prompt_length'],padded_rows=packed[0].shape[1],compact_byte_inputs_qualified=True,
          native_pixels_match_existing=True,future_worker_did_not_change_current=True,native_cache_counts=timing))
        print(json.dumps(dict(native_cpu_input=index+1,total=36)),flush=True)
  # Execute inherited actual model/decode/expert contracts with small CPU projections,75% only.
  frozen_check=c.module(c.FEXP/'agent-token-current-check.py','agent_token75_timing_frozen_contract')
  class Policy75:
    POLICIES={c.POLICY:.75}
    def __getattr__(self,name):return getattr(policy,name)
  contract=frozen_check.actual_contract(factory,Policy75())
  assert list(contract['policies'])==[c.POLICY]
  assert all(name.split(':')[0] in ('CPU','NPY','PYTHON') for name in Device._opened_devices)
  c.verify(plan,require_cpu=False)
  report=dict(cpu_only=True,cpu_qualified=True,actual_native_inputs=36,native_pixel_controls=360,
    independent_layer_mask_cases=36*28*4,records=records,actual_frozen_controller_contract=contract,
    cache_and_masks_and_current_input_qualified=True,source_bytes_match_passed75_GPU=True,
    ordinary_lookup_pad2112=True,no_nominal_prefill_introduced=True,compatibility_gpu_gate_required=False,
    compatibility_limit='No new arithmetic/kernel/decoder change; nativeGPU execution remains parent-owned and unverified here',
    opened_devices=sorted(Device._opened_devices),protocol_sha256=c.sha(c.PROTOCOL),gpu_executed=False,
    classification='CPU qualification only; injected structural scores and small fake projections are not route quality')
  c.write_new(Path(plan['cpu_qualification']),report)
  print(json.dumps(dict(cpu_qualified=True,native_inputs=36,opened_devices=report['opened_devices'],gpu_executed=False),indent=2))


if __name__=='__main__':main()

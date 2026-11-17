"""CPU controls. GPU backends and network connections are always forbidden."""
import argparse
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.update(DEV='CPU', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode = True
import numpy as np
from tinygrad import Device

ROOT = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


def forbidden(*args, **kwargs):
  raise AssertionError('Device/network access forbidden in startup CPU checks')


warm = local('agent-startup-spec-warm')


class Proofs(unittest.TestCase):
  def test_actual_speculative_instance_acceptance_and_fallback_after_restore(self):
    spec, policy = local('agent-speculation-prefill'), local('agent-speculation-policy')
    draft = list(policy.CHECKPOINT_DRAFT_IDS)
    anchor = draft[-1]
    for accepted in (True, False):
      cache = [np.arange(2*80, dtype=np.uint16).reshape(2,80)+i*1000 for i in range(3)]
      snapshots = [x.copy() for x in cache]
      prefix = 12
      target = draft[:] if accepted else [42, *draft[1:]]
      payload = np.array([*target, *([1]*6)], dtype=np.int32)
      model = SimpleNamespace(cache=cache, padded_prefill=SimpleNamespace(last_prompt=prefix, last_targets=payload, prefix_audit=None))
      class Decoder:
        history, statistics = [], []
        def __init__(self):
          self.history, self.statistics, self.previous = [], [], None
          self.before_diffusion = forbidden
          self.serial_calls = 0
        def read_tokens(self, value): return value
        def generate(self, *args):
          self.serial_calls += 1
          for x in cache: x[:,prefix:prefix+2] = 60000
          result = [42, anchor]
          self.history.append(result)
          self.statistics.append({'serial_tokens': 2})
          return result
      decoder = Decoder()
      spec.install_decoder(decoder, draft, anchor)
      generate = decoder.generate
      history = decoder.history
      with warm.decoder_state(decoder):
        # Independent overwrite model covers every required tail/verifier extent.
        for count in (*range(1,7),16,32,48,64):
          for x in cache: x[:,prefix:prefix+count] = 50000+count
        decoder.history = [[9,9,9]]
        decoder.statistics.append({'warm': True})
        decoder.prefill_draft_records.append({'warm': True})
        decoder.before_diffusion()
      self.assertIs(decoder.history, history)
      self.assertEqual(decoder.history, [])
      self.assertEqual(decoder.statistics, [])
      self.assertEqual(decoder.prefill_draft_records, [])
      def actual_prefill():
        for x,saved in zip(cache,snapshots): x[:,:prefix+6] = saved[:,:prefix+6]
        model.padded_prefill.last_targets = payload.copy()
        return np.array([7,2], np.uint16)
      warm.restore_prefill(cache, snapshots, actual_prefill, identity=lambda:tuple(map(id,cache)),
        equal=np.testing.assert_array_equal, copy_into=lambda a,b:np.copyto(a,b))
      result = decoder.generate(model, None, None, None, None, None, None, prefix)
      self.assertIs(decoder.generate, generate)
      self.assertEqual(result, draft if accepted else [42,anchor])
      self.assertEqual(decoder.serial_calls, int(not accepted))
      self.assertEqual(decoder.prefill_draft_records[-1]['accepted'], accepted)
      for x,saved in zip(cache,snapshots):
        expected = saved.copy()
        if not accepted: expected[:,prefix:prefix+2] = 60000
        np.testing.assert_array_equal(x,expected)

  def test_restoration_detects_prefix_suffix_tail_and_allocation_corruption(self):
    for index in (0,12,17,79):
      cache = [np.arange(80,dtype=np.uint16)]
      saved = [cache[0].copy()]
      def bad_prefill(): cache[0][index] ^= 1
      with self.assertRaises(AssertionError):
        warm.restore_prefill(cache,saved,bad_prefill,identity=lambda:tuple(map(id,cache)),
          equal=np.testing.assert_array_equal,copy_into=np.copyto)
    cache = [np.arange(80,dtype=np.uint16)]
    def replaced(): cache[0] = cache[0].copy()
    with self.assertRaisesRegex(AssertionError,'allocations'):
      warm.restore_prefill(cache,[cache[0].copy()],replaced,identity=lambda:tuple(map(id,cache)),
        equal=np.testing.assert_array_equal,copy_into=np.copyto)

  def test_state_and_prefetch_restored_on_exception(self):
    decoder = SimpleNamespace(history=[[1,2]], statistics=[{'a':1}], previous=[2], before_diffusion=forbidden)
    old = decoder.history
    with self.assertRaisesRegex(RuntimeError,'injected'):
      with warm.decoder_state(decoder):
        decoder.history[0].append(3)
        decoder.statistics = []
        decoder.before_diffusion()
        raise RuntimeError('injected')
    self.assertIs(decoder.history,old)
    self.assertEqual(decoder.history,[[1,2]])
    self.assertEqual(decoder.statistics,[{'a':1}])
    self.assertIs(decoder.before_diffusion,forbidden)

  def test_graph_admission_and_owner_replacement_fail_before_work(self):
    jit = object()
    graphs = warm.ClosedGraphs({'existing':jit})
    graphs['existing'] = jit
    graphs.move_to_end('existing')
    with self.assertRaisesRegex(RuntimeError,'Unwarmed'): graphs['new'] = object()
    with self.assertRaisesRegex(RuntimeError,'Unwarmed'): graphs['existing'] = object()
    calls = []
    class Base:
      @staticmethod
      def signature(*args): return 'signature'
      def __init__(self): calls.append('allocation')
    no_upload = SimpleNamespace(StartupWarmup=lambda active:SimpleNamespace(patch_inputs=lambda cls:None))
    state = {'ready':False}
    cls = warm.input_class(Base,no_upload,state)
    cls()
    state['ready'] = True
    with self.assertRaisesRegex(RuntimeError,'owner'): cls()
    self.assertEqual(calls,['allocation'])
    with self.assertRaisesRegex(ValueError,'non-speculative'): cls.signature(None,None,{},None)

  def test_actual_dynamic_upload_dispatch_all_1024_subsets(self):
    helper = local('agent-pipeline-warmup')
    class Base:
      def upload(self,noise): raise AssertionError('Wrong upload branch')
    helper.StartupWarmup(lambda:False).patch_inputs(Base)
    data = np.arange(60,dtype=np.uint8).reshape(20,3)
    keys = tuple(range(10))
    owner = Base()
    owner.batches = tuple((i*2,i*2+2,1) for i in range(10))
    owner.cpu = [SimpleNamespace(shape=data.shape,dtype=data.dtype)]
    owner.noise = np.zeros((1,64,2),np.float32)
    owner.selective_cache = SimpleNamespace(keys=keys,verify=False,cache={})
    counts = set()
    def selection(current, missing):
      self.assertIs(current,owner)
      counts.add(len(missing))
      def call(): return tuple(data[a:b].copy() for i,(a,b,_) in enumerate(owner.batches) if i in missing)
      return call, ()
    with patch.object(helper,'selection_call',selection):
      for mask in range(1024):
        owner.selective_cache.cache = {i:True for i in keys if not mask&(1<<i)}
        outputs = owner.upload(np.full_like(owner.noise,mask))
        missing = [i for i in keys if mask&(1<<i)]
        self.assertEqual(len(outputs),len(missing))
        self.assertEqual(owner.uploaded_pixel_bytes,len(missing)*6)
        for value,i in zip(outputs,missing): np.testing.assert_array_equal(value,data[i*2:i*2+2])
        self.assertTrue(np.all(owner.noise==mask))
    self.assertEqual(counts,set(range(11)))


def live_tests():
  """Use the real Consumer and CPU CompactInputs; GPU execution stays a double."""
  fixtures = local('agent-live-consumer-check')
  replay = local('agent-live-replay')
  fixtures.ARGS = SimpleNamespace(checkpoint=Path('/tmp/opencode/alpamayo-asius-20261114/model'),
    manifest=replay.SOURCE/'manifest.json', raw_frames=WORK/'agent-paced-replay-cache-35-43/rgb',
    prediction_report=WORK/'full-native-5hz-fixed-byte/report.json')
  fixtures.Checks.setUpClass()
  case = fixtures.Checks(methodName='test_completion_ownership_and_idempotent_sync')
  def bind_ready():
    runtime = case.setup_runtime(speculation=True,length=2048)
    old = runtime.consumer
    old.close()
    prepared, packed, arrays, _ = fixtures.prepare_reference(case.reference_adapter,case.cases[0][0],
      runtime.settings,case.compact,runtime.padding,2048)
    no_upload = SimpleNamespace(StartupWarmup=lambda active:SimpleNamespace(patch_inputs=lambda cls:None))
    state = {'ready':False}
    cls = warm.input_class(old.input_class,no_upload,state)
    owner = cls(runtime.settings,old.config,prepared,packed,arrays=arrays)
    state['ready'] = True
    runtime.consumer = case.live.Consumer(adapter=old.adapter,model=runtime.model,runner=runtime.decoder.run,
      settings=runtime.settings,config=old.config,input_class=cls,vision_cache=runtime.vision,
      pad_inputs=runtime.padding,pad_length=2048,decoder=runtime.decoder,inputs=owner,
      speculation_policy=case.policy,speculation_draft=case.policy.CHECKPOINT_DRAFT_IDS,
      initializer_provenance={'cpu_control':'prebound current owner'},clock=runtime.clock)
    return runtime,owner
  runtime, owner = bind_ready()
  first,_ = case.run_case(runtime,case.cases[0],complete=False)
  assert first['live_consumer']['input_reused'] and runtime.consumer.inputs is owner
  assert not first['live_consumer']['completed']
  runtime.consumer.synchronize()
  assert first['live_consumer']['completed'] and runtime.model.suffix == 'accepted speculative suffix'
  runtime.decoder.targets[0] = 42
  second,_ = case.run_case(runtime,case.cases[1])
  assert runtime.consumer.inputs is owner and runtime.model.suffix == 'serial overwritten suffix'
  assert not runtime.decoder.prefill_draft_records[-1]['accepted']
  runtime.consumer.close()
  assert runtime.consumer.inputs is None
  runtime,owner = bind_ready()
  case.run_case(runtime,case.cases[0],complete=False)
  runtime.model.fail_sync = True
  try: runtime.consumer.synchronize()
  except OSError: pass
  else: raise AssertionError('Failed completion was accepted')
  runtime.consumer.close()
  assert runtime.consumer.inputs is owner
  return {'accepted_and_fallback':True,'current_owner_reused':True,'success_releases_owner':True,
          'failed_sync_retains_owner':True,'real_adapters_and_consumer':True,'model_execution':'CPU doubles only'}


def cpu_storage():
  """Actual pinned Tensor/TinyJit, integer bit checks, and same-allocation writes."""
  from tinygrad import Tensor, TinyJit, dtypes
  original = type(Device).__getitem__
  def only_cpu(owner, key):
    if key not in ('CPU','NPY'): raise AssertionError('Only CPU and NumPy storage are permitted')
    return original(owner,key)
  with patch.object(type(Device),'__getitem__',only_cpu), patch.object(socket.socket,'connect',forbidden):
    initial = np.arange(2*80*8,dtype=np.uint16).reshape(2,1,1,80,8)
    cache = [Tensor(initial.copy(),device='CPU').bitcast(dtypes.bfloat16).realize() for _ in range(2)]
    reference = Tensor(np.full((2,1,1,18,8),16000,dtype=np.uint16),device='CPU').bitcast(dtypes.bfloat16).realize()
    def actual_prefill(x):
      for tensor in cache: tensor[:,:,:,:18].assign(x).realize()
      return x[:,:,:,-1:].clone().realize()
    prefill = TinyJit(actual_prefill)
    for _ in range(3): prefill(reference)
    snapshots = [x.clone().realize() for x in cache]
    identity = lambda: tuple(id(x._buffer().base) for x in cache)
    consumer_identity = tuple(id(x.uop.base) for x in cache)
    before = identity()
    for x in cache:
      x[:,:,:,12:76].assign(Tensor.zeros(2,1,1,64,8,device='CPU',dtype=dtypes.bfloat16)).realize()
    def exact(left,right):
      assert not (left.bitcast(dtypes.uint16)!=right.bitcast(dtypes.uint16)).any().item()
    warm.restore_prefill(cache,snapshots,lambda:prefill(reference),identity=identity,equal=exact,
                         copy_into=lambda a,b:a.assign(b).realize())
    assert identity()==before
    assert tuple(id(x.uop.base) for x in cache)==consumer_identity
    for value,saved in zip(cache,snapshots):
      np.testing.assert_array_equal(value.bitcast(dtypes.uint16).numpy(),saved.bitcast(dtypes.uint16).numpy())
  return dict(real_pinned_tensor_and_jit=True,full_storage_bits_exact=True,buffer_and_consumer_uop_identities_stable=True,
              backend='CPU',gpu_executed=False)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path)
  args = parser.parse_args()
  with patch.object(type(Device),'__getitem__',forbidden), patch.object(socket.socket,'connect',forbidden), \
       patch.object(socket.socket,'connect_ex',forbidden):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Proofs))
    if not result.wasSuccessful(): raise SystemExit(1)
    lifecycle = live_tests()
    adapters = local('agent-startup-spec-run').main(['--task','check-adapters'])
  if Device._opened_devices: raise AssertionError('Unexpected backend initialization')
  storage = cpu_storage()
  if any(name not in ('CPU','NPY') for name in Device._opened_devices): raise AssertionError('Unexpected non-CPU backend')
  report = dict(passed=True,cpu_only=True,gpu_executed=False,tests=result.testsRun,lifecycle=lifecycle,
    adapters=adapters,cpu_storage=storage,devices=list(Device._opened_devices),sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
      for p in ROOT.glob('agent-startup-spec-*.py')})
  if args.output:
    with args.output.open('x') as stream: json.dump(report,stream,indent=2); stream.write('\n')
  print(json.dumps(report,indent=2))


if __name__ == '__main__': main()

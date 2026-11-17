"""CPU integration check for agent-pipeline; never load the real model benchmark."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch


def check_dynamic_selection(Tensor, Device):
  import numpy as np
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp
  spec = importlib.util.spec_from_file_location('agent_pipeline_warmup',Path(__file__).with_name('agent-pipeline-warmup.py'))
  warm = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(warm)
  randomizer = np.random.default_rng(9060)
  pixels = randomizer.integers(0,256,size=(20,8),dtype=np.uint8)
  palette = Tensor((np.arange(256,dtype=np.float32)-127.5)/127.5,device='CPU').cast(dtypes.bfloat16).realize()
  palette_bits = palette.bitcast(dtypes.uint16).numpy()
  positions = Tensor(np.zeros(1,dtype=np.uint16),device='CPU').realize()
  class Owner(types.SimpleNamespace):
    def upload(self,noise):
      raise AssertionError('Original selective path should be replaced')
    def warmup_selections(self):
      raise AssertionError('Original selection warmup should be replaced')
  owner = Owner(cpu=[Tensor(pixels.copy(),device='CPU').realize(),positions,
                                      Tensor(np.array([[1,5]],dtype=np.int32),device='PYTHON').realize(),
                                      Tensor.zeros(1,64,2,device='CPU').realize()],
                                pixel_table_gpu=palette,pixel_storage=Tensor.full((20,8),-2,device='CPU',dtype=dtypes.bfloat16).realize(),
                                positions={'fixed':positions},layout_key='fixed',gpu_rotary={},selective_jits={},
                                batches=tuple((index*2,index*2+2,1) for index in range(10)))
  def rows(tensor,begin,end):
    width = tensor.shape[1]
    buffer = tensor._buffer().view((end-begin)*width*tensor.dtype.itemsize,begin*width*tensor.dtype.itemsize).ensure_allocated()
    return Tensor(UOp.from_buffer(buffer,tensor.dtype)).reshape(end-begin,width)
  owner.rows = rows
  sentinel = Tensor.full((20,8),-2,device='CPU',dtype=dtypes.bfloat16).realize()
  sentinel_bits = sentinel.bitcast(dtypes.uint16).numpy()
  for mask in range(1024):
    # Mutate the actual host input buffer. Captured graphs must read new bytes.
    pixels[:] = randomizer.integers(0,256,size=pixels.shape,dtype=np.uint8)
    destination = np.frombuffer(owner.cpu[0]._buffer().as_memoryview(allow_zero_copy=True),dtype=np.uint8).reshape(pixels.shape)
    destination[:] = pixels
    owner.pixel_storage.assign(sentinel).realize()
    missing = tuple(index for index in range(10) if mask & (1 << index))
    jit, arguments = warm.selection_call(owner,missing)
    output = jit(*arguments)
    Tensor.realize(*output)
    expected = sentinel_bits.copy()
    for index in missing:
      expected[index*2:index*2+2] = palette_bits[pixels[index*2:index*2+2]]
    np.testing.assert_array_equal(output[0].bitcast(dtypes.uint16).numpy(),expected)
  assert len(owner.selective_jits) == 11
  warm.StartupWarmup(lambda:True).patch_inputs(Owner)
  owner.warmup_selections()
  assert all(jit.cnt >= 2 for jit in owner.selective_jits.values())
  owner.selective_cache = types.SimpleNamespace(keys=tuple(range(10)),cache={0:None,2:None,4:None},verify=False)
  owner.noise = np.frombuffer(owner.cpu[-1]._buffer().as_memoryview(allow_zero_copy=True),dtype=np.float32).reshape(1,64,2)
  owner.pixel_storage.assign(sentinel).realize()
  output = owner.upload(np.ones((1,64,2),dtype=np.float32))
  expected = sentinel_bits.copy()
  for index in (1,3,5,6,7,8,9):
    expected[index*2:index*2+2] = palette_bits[pixels[index*2:index*2+2]]
  np.testing.assert_array_equal(output[0].bitcast(dtypes.uint16).numpy(),expected)
  np.testing.assert_array_equal(output[-1].numpy(),np.ones((1,64,2),dtype=np.float32))
  assert owner.uploaded_pixel_bytes == 7*2*8
  return {'masks_checked':1024,'graph_variants':11,'all_source_bytes_mutated_each_case':True,
          'all_selection_graphs_captured':True,'installed_upload_interface_checked':True,
          'checks':'Exact BF16 palette bits at selected rows; untouched rows retain sentinel bits.',
          'limit':'CPU backend only. AMD symbolic-store lowering and USB transfer behavior remain unverified.'}


def check_decoder_warmup(Tensor):
  import numpy as np
  from tinygrad import TinyJit, dtypes
  from tinygrad.uop.ops import UOp
  spec = importlib.util.spec_from_file_location('agent_pipeline_warmup',Path(__file__).with_name('agent-pipeline-warmup.py'))
  warm = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(warm)
  spec = importlib.util.spec_from_file_location('agent_pipeline_block',Path(__file__).with_name('block-decode.py'))
  block = importlib.util.module_from_spec(spec)
  with patch.dict(sys.modules,{'benchmark':types.SimpleNamespace(timed=None,UnicycleAccelCurvatureCourseActionSpace=None),
                               'runtime':types.SimpleNamespace(download=lambda value:value.numpy())}):
    spec.loader.exec_module(block)
  class Model:
    def __init__(self):
      self.cache = [Tensor(np.arange(1024,dtype=np.float32).reshape(1,1,1,1024,1),device='CPU').realize()]
    def _greedy_decode(self,logits,cos,sin,position,step):
      token = logits.reshape(-1).argmax().float().realize()
      self.cache[0][:,:,:,position:position+1,:].assign(token.reshape(1,1,1,1,1)).realize()
      return (logits+1).realize(),token
  class Decoder(block.BlockDecoder):
    def __init__(self,model):
      super().__init__(model,6)
      self.fallback_block = 6
      self.compact_io = True
      self.verifiers, self.select_jits = {}, {}
      self.choose_jit = TinyJit(lambda value:value.reshape(-1).argmax().float().realize())
    def first_token(self,logits):
      return int(self.choose_jit(logits).item())
    def verify(self,ids,cos,sin,position,step):
      count = ids.shape[0]
      self.model.cache[0][:,:,:,position:position+count,:].assign(ids.reshape(1,1,1,count,1).float()).realize()
      values = (ids.reshape(1,count,1).float()+Tensor.arange(8).reshape(1,1,8)).realize()
      return values,values.argmax(axis=-1).realize()
    def select_logits(self,values,index):
      count = values.shape[1]
      if count not in self.select_jits:
        self.select_jits[count] = TinyJit(lambda value,row:value[:,row:row+1].clone().realize())
      return self.select_jits[count](values,UOp.variable('selected_draft_row',0,count-1).bind(index))
    def generate(self,model,args,config,tokenizer,logits,cos,sin,prompt):
      return self.decode(logits,cos,sin,prompt,0,1)
  model, args = Model(),types.SimpleNamespace(max_tokens=519)
  decoder = Decoder(model)
  logits = Tensor(np.arange(8,dtype=np.float32).reshape(1,1,8),device='CPU').realize()
  cos = Tensor(np.ones((3,1,519,1),dtype=np.float32),device='CPU').realize()
  sin = Tensor(np.zeros((3,1,519,1),dtype=np.float32),device='CPU').realize()
  prefix = model.cache[0].numpy()[:,:,:,:16].copy()
  plan = warm.warm_decoder(decoder,model,args,None,logits,cos,sin,16)
  np.testing.assert_array_equal(model.cache[0].numpy()[:,:,:,:16],prefix)
  assert sorted(key[1] for key in decoder.jits) == list(range(1,7))
  assert sorted(key[0] for key in decoder.verifiers) == [16,32,48,64]
  assert sorted(decoder.select_jits) == [16,32,48,64]
  assert all(len(jit.captured._linears) >= 2 for jit in decoder.jits.values())
  output, tokens = decoder.decode(logits,cos,sin,16,0,6)
  np.testing.assert_array_equal(output.numpy(),logits.numpy()+6)
  np.testing.assert_array_equal(tokens.numpy(),np.full(6,7,dtype=np.float32))
  # Reallocate equal-capacity caches. The helper must discard captured old owners.
  events = []
  warm.StartupWarmup(lambda:False,lambda kind,**values:events.append(kind)).patch_decoder(Decoder)
  decoder.generate(model,args,None,None,logits,cos,sin,16)
  original_cache = model.cache[0]
  for current in (Tensor.zeros(*original_cache.shape,device='CPU').realize(),original_cache):
    model.cache = [current]
    old_jit = decoder.jits[(cos.shape,1,1024)]
    output, token = decoder.generate(model,args,None,None,logits,cos,sin,16)
    Tensor.realize(output,token)
    assert decoder.jits[(cos.shape,1,1024)] is not old_jit
    assert model.cache[0].numpy()[0,0,0,16,0] == 7
  assert events.count('conditioning_cache_reallocated') == 2
  return {**plan,'reasoning_budget_preserved':519,'prefix_unchanged':True,
          'serial_alias_variants_warmed':True,'cache_reallocation_cases':2,
          'limit':'Synthetic CPU language operations. Genuine model verification, AMD lowering, and route parity remain required.'}


def check_vision_warmup(Tensor):
  import numpy as np
  spec = importlib.util.spec_from_file_location('agent_pipeline_warmup',Path(__file__).with_name('agent-pipeline-warmup.py'))
  warm = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(warm)
  class Vision:
    def __init__(self):
      self.grouped, self.group_limit, self.views = True,2,False
      self.keys = tuple(((1,2,1),bytes([index])) for index in range(10))
      self.encoders, self.cache = {},{}
      self.batch_encoder = lambda *values:tuple((a+b).realize() for a,b in zip(values[::2],values[1::2]))
      self.result = object()
    def encode(self,pixels,positions,ids,batches,spans):
      return self.result
  warm.StartupWarmup(lambda:True).patch_vision(Vision)
  vision = Vision()
  pixels = Tensor(np.arange(80,dtype=np.float32).reshape(20,4),device='CPU').realize()
  positions = Tensor(np.arange(80,dtype=np.float32).reshape(20,4)+1,device='CPU').realize()
  before_pixels, before_positions = pixels.numpy().copy(),positions.numpy().copy()
  batches = tuple((index*2,index*2+2,1) for index in range(10))
  result = vision.encode(pixels,positions,None,batches,None)
  assert result is vision.result
  assert sorted(len(shape)//2 for shape in vision.encoders) == [1,2]
  assert all(jit.cnt >= 2 for jit in vision.encoders.values())
  assert vision.cache == {}
  np.testing.assert_array_equal(pixels.numpy(),before_pixels)
  np.testing.assert_array_equal(positions.numpy(),before_positions)
  return {'current_image_graph_sizes':[1,2],'input_arrays_unchanged':True,'future_features_cached':0,
          'limit':'Synthetic CPU vision arithmetic. Actual vision kernels and route parity remain unverified.'}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',required=True,type=Path)
  args = parser.parse_args()
  if os.environ.get('DEV') != 'CPU':
    parser.error('Set DEV=CPU explicitly for this check')
  from tinygrad import Tensor, TinyJit, Device
  spec = importlib.util.spec_from_file_location('agent_pipeline',Path(__file__).with_name('agent-pipeline.py'))
  helper = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(helper)
  results = []
  original_spec = importlib.util.spec_from_file_location
  benchmark_path = Path(__file__).with_name('benchmark-pipeline.py').resolve()
  with tempfile.TemporaryDirectory(prefix='agent-pipeline-cpu-',dir='/tmp/opencode') as directory:
    directory = Path(directory)
    for validation_count in (0,2):
      for defer in (False,True):
        fake = types.ModuleType('agent_pipeline_cpu_fixture')
        fake.ThreadPoolExecutor = helper.ThreadPoolExecutor
        class Decoder:
          def __init__(self):
            self.jit = TinyJit(lambda value:(value*3+1).realize())
            self.before_diffusion = lambda:None
          def run(self,model,args,config,tokenizer,prepared,packed,seed,inputs):
            value = Tensor([prepared],device='CPU').realize()
            result = int(self.jit(value).item())
            self.before_diffusion()
            return result
        class Vision:
          def __init__(self,model):
            self.encoders = {}
        def local_module(name):
          return types.SimpleNamespace(**({'LookupDecoder':Decoder} if name == 'lookup-decode' else {'VisionCache':Vision}))
        fake.local_module = local_module
        def benchmark_main():
          parsed = helper.flag_value(sys.argv,'--output')
          warmup = int(helper.flag_value(sys.argv,'--warmup'))
          expected_budget = int(helper.flag_value(sys.argv,'--max-tokens'))
          assert expected_budget == 519
          model, inputs = types.SimpleNamespace(cache=None), types.SimpleNamespace()
          vision = fake.local_module('vision-cache').VisionCache(model)
          decoder = fake.local_module('lookup-decode').LookupDecoder()
          values = list(range(8))
          outputs = []
          with fake.ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(lambda value:value,values[0])
            for index, value in enumerate(values):
              if index == warmup+validation_count:
                gc.collect()
                gc.freeze()
                gc.get_freeze_count()
              prepared = pending.result()
              pending = None
              def queue_next():
                nonlocal pending
                if index+1 < len(values):
                  pending = pool.submit(lambda value:value,values[index+1])
              decoder.before_diffusion = queue_next
              outputs.append(decoder.run(model,None,None,None,prepared,None,seed=0,inputs=inputs))
          assert vision is not None
          Path(parsed).write_text(json.dumps({'outputs':outputs}))
        fake.main = benchmark_main
        class FakeLoader:
          def create_module(self,spec):
            return fake
          def exec_module(self,module):
            pass
        def specification(name,path,*values,**kwargs):
          if Path(path).resolve() == benchmark_path:
            return importlib.util.spec_from_loader(name,FakeLoader())
          return original_spec(name,path,*values,**kwargs)
        invocation = directory/'invocation.json'
        output = directory/f'v{validation_count}-defer{int(defer)}.json'
        command = [sys.executable,str(benchmark_path),'--warmup','3','--runs','5','--max-tokens','519',
                   '--pipeline-inputs','--late-inputs','--freeze-warmup-gc','--output','unused']
        if validation_count:
          reference = directory/'validation.json'
          reference.write_text(json.dumps({f'past-{index}':{} for index in range(validation_count)}))
          command += ['--validation-reference',str(reference)]
        invocation.write_text(json.dumps({'command':command,'environment':{'DEV':'CPU'}}))
        replay_args = types.SimpleNamespace(invocation=invocation,output=output,defer_first_measured=defer,
                                           event_capacity=8192,reasoning_budget=None,warmup_graphs=False,
                                           warm_current_shapes=False,graph_budget=32,warmup=None,runs=None)
        # The real freeze is unnecessary for checking scheduling and hook restoration.
        with patch.object(importlib.util,'spec_from_file_location',specification), patch.object(gc,'freeze',lambda:None):
          helper.replay(replay_args)
        outputs = json.loads(output.read_text())['outputs']
        assert outputs == [value*3+1 for value in range(8)]
        diagnostics = json.loads(output.with_suffix('.pipeline.json').read_text())
        events = diagnostics['events']
        assert diagnostics['counts']['prediction_begin'] == 8
        assert diagnostics['counts']['prediction_end'] == 8
        assert diagnostics['counts']['jit_cold_begin'] == 2
        assert diagnostics['counts']['jit_cold_end'] == 2
        assert diagnostics['counts']['get_freeze_count_end'] == 1
        assert diagnostics['status'] == 'complete'
        assert diagnostics['dropped_events'] == 0
        boundary = next(row['t_ns'] for row in events if row['kind'] == 'get_freeze_count_end')
        if defer:
          start = next(row['t_ns'] for row in events if row['kind'] == 'input_started' and row['ordinal'] == 3+validation_count)
          assert start >= boundary
        results.append({'validation_inputs':validation_count,'defer_first_measured':defer,
                        'outputs':outputs,'cold_jit_calls':diagnostics['counts']['jit_cold_begin'],
                        'events':len(events)})
  dynamic_selection = check_dynamic_selection(Tensor,Device)
  decoder_warmup = check_decoder_warmup(Tensor)
  vision_warmup = check_vision_warmup(Tensor)
  opened = sorted(Device._opened_devices)
  assert all(name.split(':')[0] in ('CPU','NPY','PYTHON') for name in opened),opened
  args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps({'status':'passed','opened_devices':opened,'cases':results,
                                    'dynamic_selection':dynamic_selection,
                                    'decoder_warmup':decoder_warmup,
                                    'vision_warmup':vision_warmup,
                                    'limit':'Real TinyJit on CPU with synthetic arithmetic. No model or target-verification claim.'},indent=2)+'\n')


if __name__ == '__main__':
  main()

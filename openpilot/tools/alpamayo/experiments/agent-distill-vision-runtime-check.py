"""CPU numerical, full720 lazy mapping, native/batched lifecycle, and actual-runner first-call checks."""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = Path('/tmp/opencode/alpamayo-asius-20261114')
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-distill-vision-runtime-artifacts'
NATIVE = WORK/'agent-distill-command-artifacts/capture-v1/runtime'


def load(name, path):
  spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec)
  sys.modules[name] = value; spec.loader.exec_module(value); return value


def main():
  parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(); root = args.output.resolve().parent
  if not root.is_relative_to(ROOT) or root.exists(): raise ValueError('Use a new CPU-check directory under the runtime artifacts root')
  root.mkdir(parents=True); (root/'tmp').mkdir()
  os.environ.update(DEV='CPU', TC='0', ALLOW_TF32='0', BEAM='0', JITBEAM='0', PARALLEL='2',
                    CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                    OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                    CACHEDB=str(root/'cpu-cache.db'), TMPDIR=str(root/'tmp'))
  def hardware_guard(event, values):
    if event == 'open' and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(('/dev/kfd', '/dev/dri/', '/dev/nvidia', '/dev/bus/usb/')):
      raise RuntimeError('CPU runtime check attempted hardware access')
  sys.addaudithook(hardware_guard)
  sys.path[:0] = [str(BASE/'tinygrad'), str(NATIVE)]
  import numpy as np
  from tinygrad import Tensor, TinyJit, Device, dtypes
  from tinygrad.helpers import Context
  H = load('vision_runtime_check_hook', HERE/'agent-distill-vision-runtime-hook.py')
  D = load('vision_runtime_check_driver', HERE/'agent-distill-vision-runtime-run.py')
  P = load('vision_runtime_check_parent', HERE/'agent-distill-joint-parent.py')
  plan = D.checked_plan(); row = P.selected_rows(P.INPUTS, ['train-command-0'])[0]
  ready = P.prepare(plan, [row])[0]  # Actual ten history images; no model or GPU loading.
  Native = load('vision_runtime_check_native', NATIVE/'model.py').EdgeModel
  B = load('vision_runtime_check_block', HERE/'agent-distill-vision-block.py')
  CONFIG = H.read(BASE/'model/config.json')
  weights, selection = H.load_selected(); seed, seed_receipt = H.load_seed()
  selected_sources = {str(path): H.sha(path) for path in HERE.glob('agent-distill-vision-runtime-*.py')}

  def model(dtype=dtypes.float32):
    value = Native.__new__(Native); value.config = deepcopy(CONFIG); value.dtype = dtype
    value.fp8_wmma = value.cuda_kernels = False
    value.cache = None; value.prompt_length = value.greedy_shape = None
    value.weights = {H.PREFIX+'12.'+k: Tensor(v.copy(), device='CPU').cast(dtype).realize() for k,v in seed.items()}
    value.weight_scales = {}; value.trajectory_jits = {}; value.encode_jits = {}
    value.prefill_jit = TinyJit(value._language); value.decode_jit = TinyJit(value._language); value.greedy_decode_jit = TinyJit(value._greedy_decode)
    return value

  rejected = []
  def reject(name, fn):
    try: fn()
    except (ValueError, AssertionError): rejected.append(name)
    else: raise AssertionError('Accepted '+name)

  C = load('vision_runtime_check_capture', HERE/'agent-distill-vision-common.py')
  examples = C.capture(C.TRAIN, C.TRAIN_SHA, 'train')
  x = np.concatenate([examples[i]['x'][:, :3] for i in (0,1)], axis=0)
  trained = B.Block(weights); expected = trained(Tensor(x)).numpy()
  numeric = {}
  for dtype, label in ((dtypes.float32, 'fp32'), (dtypes.bfloat16, 'bf16')):
    value = model(dtype); old_weights = dict(value.weights); old_scales = dict(value.weight_scales); original_config = value.config
    with H.vision_pilot(value, weights, selection=selection, allow_short_cpu_fixture=True) as receipt:
      actual = value.vision_layer(Tensor(x).cast(dtype), 12).float().numpy()
      assert actual.shape == (2,3,1152) and np.isfinite(actual).all()
      if label == 'fp32': np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)
      assert not any(k.startswith(H.PREFIX+'27.') for k in value.weight_scales)
      numeric[label] = dict(max_abs_vs_fp32_training=float(np.abs(actual-expected).max()), finite=True,
                            parameter_count=sum(v.size for v in weights.values()), dtype=str(dtype))
      reject('double_install_'+label, lambda: H.vision_pilot(value, weights, selection=selection).__enter__())
    assert value.config is original_config and value.weights == old_weights and value.weight_scales == old_scales
  # The checkpoint must differ from its seed, and the explicit seed control cannot activate silently.
  assert any(not np.array_equal(seed[k], weights[k]) for k in weights)
  reject('unapproved_seed', lambda: H.vision_pilot(model(), seed, selection=seed_receipt).__enter__())
  old_sha = H.REPORT_SHA; H.REPORT_SHA = '0'*64
  reject('wrong_completed_report_hash', H.load_selected); H.REPORT_SHA = old_sha

  # Native complete vision graph at real B2,T720 width: retain original method bodies and record every attention shape.
  value = model(); aliases = dict(value.weights)
  for i in range(27):
    for name in H.shapes(): value.weights[H.PREFIX+str(i)+'.'+name] = aliases[H.PREFIX+'12.'+name]
  value.weights['vlm.model.visual.embeddings.patch_embedding.weight'] = Tensor.empty(1152,588)
  value.weights['vlm.model.visual.embeddings.patch_embedding.bias'] = Tensor.zeros(1152)
  for name in ('vlm.model.visual.post_layernorm', 'vlm.model.projector.norm'):
    value.weights[name+'.weight'] = Tensor.ones(1152); value.weights[name+'.bias'] = Tensor.zeros(1152)
  # Read original projector shapes from safetensor metadata, without materializing those weights.
  import struct
  index = H.read(BASE/'model/model.safetensors.index.json')['weight_map']
  for name in index:
    if name.startswith('vlm.model.projector.linear_fc'):
      with (BASE/'model'/index[name]).open('rb') as stream:
        size = struct.unpack('<Q',stream.read(8))[0]; entry = json.loads(stream.read(size))[name]
      value.weights[name] = Tensor.empty(*entry['shape'])
  calls, attention_shapes, tail = [], [], []
  original_layer, original_attention, original_norm = value.vision_layer, value.attention, value.norm
  def layer(self, x, i): calls.append(i); return original_layer(x, i)
  def attention(self, q, k, v, *args, **kwargs):
    attention_shapes.append([list(t.shape) for t in (q,k,v)]); return original_attention(q,k,v,*args,**kwargs)
  def norm(self, x, name, *args, **kwargs):
    if name in ('vlm.model.visual.post_layernorm','vlm.model.projector.norm'): tail.append(name)
    return original_norm(x,name,*args,**kwargs)
  value.vision_layer = MethodType(layer,value); value.attention = MethodType(attention,value); value.norm = MethodType(norm,value)
  with H.vision_pilot(value, weights, selection=selection):
    with patch.object(Tensor, 'realize', lambda self, *args, **kwargs: self):
      output = value._vision(Tensor.empty(2,720,588), Tensor.empty(2,720,1152))
      assert output.shape == (360,2048)
    assert calls == list(H.SOURCES) and attention_shapes == [[[2,16,720,72]]*3]*24
    assert tail == ['vlm.model.visual.post_layernorm', 'vlm.model.projector.norm']
  lazy = dict(shape=[2,720,1152], output_shape=[360,2048], actual_native_vision_graph=True,
              layer_sources=calls, all24_attention_shapes=[2,16,720,72], final_norm_projector_preserved=True, executed=False)

  # Both nesting orders retain the enclosing text/expert architecture and restore the original configuration.
  J = load('vision_runtime_check_joint', HERE/'agent-distill-joint-runtime.py')
  for vision_outer in (True, False):
    value = model(); original_config = value.config
    for prefix in (J.LANGUAGE,J.EXPERT): value.weights[prefix+'12.input_layernorm.weight'] = Tensor.ones(1)
    small = {'input_layernorm.weight':np.ones(1,np.float32)}
    a = lambda: H.vision_pilot(value,weights,selection=selection)
    b = lambda: J.joint_pilot(value,small,small,allow_untrained_expert=True)
    outer, inner = (a,b) if vision_outer else (b,a)
    with outer():
      outer_config = value.config
      with inner():
        assert value.config['vlm_config']['vision_config']['num_hidden_layers'] == 24
        assert value.config['vlm_config']['text_config']['num_hidden_layers'] == value.config['expert_config']['llm_config']['num_hidden_layers'] == 25
      assert value.config is outer_config
    assert value.config is original_config and not hasattr(value,'vision_depth_mapping') and not hasattr(value,'joint_depth_mapping')

  value=model(); original_config=value.config; original_weights=dict(value.weights)
  warmed=TinyJit(lambda t:(t+1).realize()); probe=Tensor.zeros(3).realize()
  warmed(probe); warmed(probe); assert warmed.cnt==2
  value.encode_jits['teacher']=warmed; value.prefill_jit=warmed
  value.padded_prefill=SimpleNamespace(shape=(1,),last_targets=probe,last_prompt=7,prefix_audit={},
                                      baseline=SimpleNamespace(shape=(1,),last_targets=probe,last_prompt=7))
  reset_depths=[]; actual_reset=H.reset_state
  def recorded_reset(model,*args,**kwargs):
    reset_depths.append(model.config['vlm_config']['vision_config']['num_hidden_layers'])
    return actual_reset(model,*args,**kwargs)
  try:
    with patch.object(H,'reset_state',recorded_reset), H.vision_pilot(value,weights,selection=selection):
      assert warmed.cnt==0 and not value.encode_jits
      assert value.padded_prefill.last_targets is value.padded_prefill.last_prompt is None
      reject('cropped_patch_input',lambda:value.vision_layer(Tensor.zeros(1,719,1152),12))
      reject('old27_loop_index',lambda:value.vision_layer(Tensor.zeros(1,720,1152),24))
      raise RuntimeError('intentional context failure')
  except RuntimeError as exc: assert str(exc)=='intentional context failure'
  assert reset_depths==[27,24] and value.config is original_config and value.weights==original_weights and warmed.cnt==0
  value.weight_scales[H.PREFIX+'27.self_attn.q_proj.weight']=Tensor.ones(1)
  reject('invented_student_scale',lambda:H.vision_pilot(value,weights,selection=selection).__enter__())

  # Actual batched encoder, VisionCache, and inherited decoder.run execute the first candidate prediction.
  # Expensive layer math, language generation, and trajectory integration are explicit CPU mocks.
  batched = load('vision_runtime_check_batched', NATIVE/'experiments/batched-vision.py')
  cache_module = load('vision_runtime_check_cache', NATIVE/'experiments/vision-cache.py')
  trace = dict(layer_calls=[], projector_calls=[], prefill=0, trajectory=0, inputs=0, upload=0, speculation=[])
  used = {}
  def initialize(options, **unused):
    value = model(); original_config = value.config
    value.synchronize = lambda: None
    def linear(self, x, name, quantized_input=None, heads=None):
      if name == 'vlm.model.visual.embeddings.patch_embedding': return x.float().mean(-1,keepdim=True).expand(*x.shape[:-1],1152)
      if name == 'vlm.model.projector.linear_fc1': trace['projector_calls'].append(name); return x
      if name == 'vlm.model.projector.linear_fc2': trace['projector_calls'].append(name); return x[:,:2048]
      raise AssertionError(name)
    value.linear = MethodType(linear,value)
    value.norm = lambda x,name,*a,**k: (trace['projector_calls'].append(name) or x)
    def mock_layer(self,x,i): trace['layer_calls'].append((x.shape[0],i)); return x + (i+1)*.001
    value.vision_layer = MethodType(mock_layer,value)
    value.weights['vlm.model.language_model.embed_tokens.weight'] = Tensor.zeros(1,2048)
    encode = batched.install(value)
    def projector(x):
      outputs=[]
      for i in range(x.shape[0]):
        h=value.norm(x[i:i+1],'vlm.model.visual.post_layernorm')
        h=value.norm(h,'vlm.model.projector.norm').reshape(-1,4608)
        h=value.linear(h,'vlm.model.projector.linear_fc1').float().gelu(approximate='none').cast(value.dtype)
        outputs.append(value.linear(h,'vlm.model.projector.linear_fc2').realize())
      return tuple(outputs)
    value.batched_projector = projector
    cache = cache_module.VisionCache(value,capacity=144,grouped=True,views=False,batch_encoder=encode,group_limit=2)
    # Import after the actual launcher installs its hooks. Exercise selected padding and decoder installation.
    padding = load('vision_runtime_check_selected_padding', NATIVE/'experiments/padded-prefill.py')
    decoder_module = load('vision_runtime_check_selected_decoder', NATIVE/'experiments/lookup-decode.py')
    policy = load('vision_runtime_check_selected_policy', NATIVE/'experiments/agent-speculation-policy.py')
    value.output_token_mapping = Tensor.arange(1).realize()
    decoder = decoder_module.LookupDecoder(value)
    original_generate = decoder.generate
    def generate(self,*args,**kwargs):
      generated = original_generate(*args,**kwargs)
      trace['speculation'].append(deepcopy(self.prefill_draft_records[-1]))
      return generated
    decoder.generate = MethodType(generate,decoder)
    def prefill(embedding,*args):
      trace['prefill']+=1; assert embedding.shape==(1,2048,2048)
      assert value.config['vlm_config']['text_config']['num_hidden_layers']==28
      prefill.last_prompt=int(args[-1]); prefill.prefix_audit=None
      prefill.last_targets=Tensor(np.array([*policy.CHECKPOINT_DRAFT_IDS,*([1]*6)],np.int32)).realize()
      value.cache=[Tensor.zeros(1) for _ in range(28)]; return Tensor.zeros(1,1,1)
    value.padded_prefill=prefill; value.prefill=prefill
    def trajectory(actions,*args):
      trace['trajectory']+=1; assert value.config['expert_config']['llm_config']['num_hidden_layers']==28
      return actions*0
    value.trajectory=trajectory
    class Inputs:
      def __init__(self,settings,config,prepared,packed):
        trace['inputs']+=1; pixels=prepared['tokenized_data']['pixel_values'].numpy().astype(np.float32)
        self.pixels=Tensor(pixels).realize(); self.vpos=Tensor.zeros(7200,1152).realize()
        self.ids=Tensor.zeros(*packed[0].shape,dtype=dtypes.int32).realize()
        indices=np.flatnonzero(packed[0][0]==CONFIG['vlm_config']['image_token_id'])
        self.spans=tuple((int(s[0]),int(s[-1])+1) for s in np.split(indices,np.flatnonzero(np.diff(indices)!=1)+1))
        self.batches=tuple((i*720,(i+1)*720,1) for i in range(10))
      def upload(self,noise):
        trace['upload']+=1
        return (self.pixels,self.vpos,self.ids,*[Tensor.ones(1,1,1,128) for _ in range(6)],Tensor(noise.astype(np.float32)))
    reference=H.read(D.TEACHERS[row['id']][0]); settings=SimpleNamespace(**reference['settings']['settings'])
    settings.initial_noise=np.load(options.initial_noise,allow_pickle=False)
    config=SimpleNamespace(traj_ids=CONFIG['traj_ids'],expert_config=SimpleNamespace(action_space_cfg=CONFIG['expert_config']['action_space_cfg']))
    components=SimpleNamespace(model=value,decoder=decoder,vision_cache=cache,settings=settings,config=config,
      tokenizer=SimpleNamespace(decode=lambda tokens:'CPU mocked reasoning'),input_class=Inputs,runner=decoder.run,
      pad_inputs=padding.pad_inputs,report=reference['settings'])
    used.update(model=value,cache=cache,decoder=decoder,config=original_config,projector=projector)
    # Preexisting graph and feature state must be discarded before the first candidate call.
    cache.cache['stale']=Tensor.zeros(1); cache.reference_model=object(); cache.reference_encoders['stale']=TinyJit(lambda x:x+1)
    cache.encoders['stale']=TinyJit(lambda x:x+1); cache.stitchers['stale']=TinyJit(lambda x:x+1)
    value.encode_jits['stale']=TinyJit(lambda x:x+1); decoder.history.append([999])
    return components
  options=SimpleNamespace(output=root/'runner-report.json')
  result=D.execute_prediction(options,plan,ready,weights,selection,cpu_mock=True,initialize=initialize)
  assert trace['layer_calls']==[(2,i) for _ in range(5) for i in H.SOURCES]
  expected_tail=['vlm.model.visual.post_layernorm','vlm.model.projector.norm','vlm.model.projector.linear_fc1','vlm.model.projector.linear_fc2']
  assert trace['projector_calls']==expected_tail*10
  assert trace['inputs']==trace['upload']==trace['prefill']==trace['trajectory']==1
  assert len(trace['speculation'])==1 and trace['speculation'][0]['accepted'] and trace['speculation'][0]['actual_prompt']==row['actual_prompt_tokens']
  assert used['model'].config is used['config'] and used['model'].batched_projector is used['projector']
  assert not used['cache'].cache and not used['cache'].encoders and not used['cache'].stitchers and not used['cache'].reference_encoders
  assert used['cache'].reference_model is None and not used['model'].encode_jits and not used['decoder'].history
  # Native independent-image callsite also maps24, preserving the ordinary final normalization/projector path.
  value=used['model']; trace['layer_calls'].clear(); trace['projector_calls'].clear()
  with Context(JIT=0), H.vision_pilot(value,weights,selection=selection,decoder=used['decoder'],vision_cache=used['cache']):
    native_output=value._vision(Tensor.zeros(1,720,588),Tensor.zeros(1,720,1152)).numpy()
    assert native_output.shape==(180,2048) and np.isfinite(native_output).all()
  assert trace['layer_calls']==[(1,i) for i in H.SOURCES] and trace['projector_calls']==expected_tail
  P.no_devices(); assert all(d.startswith(('CPU','NPY','DISK')) for d in Device._opened_devices),Device._opened_devices
  assert {str(path):H.sha(path) for path in HERE.glob('agent-distill-vision-runtime-*.py')}==selected_sources
  report=dict(complete=True,gpu_executed=False,optimizer_updates=0,selection=selection,numerical=numeric,full720_lazy=lazy,
    nested_joint25_both_orders=True,actual_ten_history_images=True,actual_native_independent_callsite=True,
    actual_batched_encoder_and_packed_projector=True,actual_vision_cache_and_complete_runner_first_call=True,
    actual_selected_speculative_padding_and_generate=True,
    runner_prediction_calls=1,runner_vision_groups=5,images_per_group=2,language_depth=28,expert_depth=28,
    stale_jits_features_references_cleared=True,warmed_jit_reset_before_architecture_change=True,
    exception_context_restoration=True,contexts_restored=True,rejected=rejected,
    full_model_math_mocked=True,quality_qualified=False,opened_devices=sorted(Device._opened_devices),sources=selected_sources)
  args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k!='sources'},indent=2))


if __name__=='__main__':main()

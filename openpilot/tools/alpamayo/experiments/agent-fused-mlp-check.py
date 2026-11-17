"""CPU fake-dispatch checks of the actual pinned text layer and packed installer."""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(path):
  spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-','_')+'_fused_check',path)
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


class FakeUOp:
  def __init__(self,tensor): self.tensor = tensor
  def store(self,other): return ('cache-store',self,other)
  def after(self,store): return self


class FakeTensor:
  calls = []
  def __init__(self, data=None, *, shape=None,dtype=None,device='AMD',tag='value'):
    from tinygrad import dtypes
    if isinstance(data,FakeUOp):
      old = data.tensor
      self.shape,self.dtype,self.device,self.tag,self.data = old.shape,old.dtype,old.device,old.tag,old.data
    else:
      self.shape = tuple(shape) if shape is not None else (len(data),)
      self.dtype,self.device,self.tag,self.data = dtype or dtypes.float32,device,tag,data
    self.uop = FakeUOp(self)
  def numel(self): return math.prod(self.shape)
  def realize(self): return self
  def reshape(self,*shape):
    if len(shape) == 1 and isinstance(shape[0],tuple): shape = shape[0]
    if -1 in shape: shape = tuple(self.numel()//math.prod([s for s in shape if s != -1]) if s == -1 else s for s in shape)
    assert math.prod(shape) == self.numel()
    return FakeTensor(self.data,shape=shape,dtype=self.dtype,device=self.device,tag=self.tag)
  def transpose(self,a,b):
    shape = list(self.shape); shape[a],shape[b] = shape[b],shape[a]
    return FakeTensor(shape=shape,dtype=self.dtype,device=self.device,tag=self.tag)
  def __getitem__(self,index):
    index = index if isinstance(index,tuple) else (index,)
    shape = []
    for i,n in enumerate(self.shape):
      item = index[i] if i < len(index) else slice(None)
      if isinstance(item,slice): shape.append(len(range(*item.indices(n))))
      else:
        assert isinstance(item,int)
        if item >= n or item < -n: raise IndexError(item)
    return FakeTensor(shape=shape,dtype=self.dtype,device=self.device,tag=self.tag)
  def cat(self,other,dim):
    shape = list(self.shape); shape[dim] += other.shape[dim]
    return FakeTensor(shape=shape,dtype=self.dtype,device=self.device,tag=self.tag)
  def __add__(self,other):
    assert self.shape == other.shape
    return FakeTensor(shape=self.shape,dtype=self.dtype,device=self.device,tag='residual')
  def relu(self): FakeTensor.calls.append(('ordinary-relu',self.shape)); return self
  def square(self): FakeTensor.calls.append(('ordinary-square',self.shape)); return self
  @staticmethod
  def stack(*values):
    return FakeTensor(shape=(len(values),*values[0].shape),dtype=values[0].dtype,tag='cache-stack')
  @staticmethod
  def empty(*shape,device='AMD',dtype=None):
    if len(shape) == 1 and isinstance(shape[0],tuple): shape = shape[0]
    return FakeTensor(shape=shape,dtype=dtype,device=device,tag='gemm-output')
  @staticmethod
  def custom_kernel(out,*args,fxn):
    FakeTensor.calls.append(('original-gemm',args[1].tag,args,out.shape))
    return (out,)


class FakeModel:
  def __init__(self,core):
    from tinygrad import dtypes
    self.dtype,self.fp8_wmma,self.cuda_kernels = dtypes.bfloat16,True,False
    self.config = {'vlm_config':{'text_config':{'num_hidden_layers':28}}}
    self.weights,self.weight_scales,self.events = {},{},[]
    class FakeJit:
      def __init__(self): self.cnt,self.captured,self.resets = 0,None,0
      def reset(self): self.cnt,self.captured,self.resets = 0,None,self.resets+1
    self.prefill_jit,self.decode_jit = FakeJit(),FakeJit()
    for prefix,width,hidden in [*(('vlm.model.language_model.layers.'+str(i),2048,9216) for i in range(28)),
                                ('expert.expert.layers.0',512,3840)]:
      for suffix,shape in (('self_attn.q_proj',(2048,width)),('self_attn.k_proj',(1024,width)),
                          ('self_attn.v_proj',(1024,width)),('self_attn.o_proj',(width,2048)),
                          ('mlp.fc1',(hidden,width)),('mlp.fc2',(width,hidden))):
        key = prefix+'.'+suffix+'.weight'
        self.weights[key] = FakeTensor(shape=shape,dtype=dtypes.fp8e4m3,tag=key)
        self.weight_scales[key] = FakeTensor([.125],dtype=dtypes.float32,tag=key+'-scale')
    self.text_layer = MethodType(core.EdgeModel.text_layer,self)
  def quantize(self,x):
    from tinygrad import dtypes
    self.events.append(('dynamic-quantize',x.shape))
    return FakeTensor(shape=(x.numel()//x.shape[-1],x.shape[-1]),dtype=dtypes.fp8e4m3),FakeTensor([.5],dtype=dtypes.float32)
  def linear(self,x,name,quantized_input=None,heads=None):
    self.events.append(('fallback-linear',name))
    if quantized_input is None: self.quantize(x)
    width = self.weights[name+'.weight'].shape[0]
    return FakeTensor(shape=(*x.shape[:-1],width),dtype=x.dtype)
  def norm(self,x,name,**kwargs): self.events.append(('norm',name)); return x
  def attention(self,q,k,v,mask,causal): self.events.append(('attention',causal)); return q
  def rope(self,q,k,cos,sin): self.events.append(('rope',)); return q,k


def raises(kind,fn):
  try: fn()
  except kind: return
  raise AssertionError(f'Expected {kind.__name__}')


def check():
  from tinygrad import dtypes,Device
  assert not Device._opened_devices
  hook,kernel,packed,core = (load(HERE/name) for name in ('agent-fused-mlp.py','agent-packed-epilogue.py','quantize-packed.py','../model.py'))
  checks,dispatches,quantizations = [],[],[]
  device = SimpleNamespace(arch='gfx1200',device='AMD')
  fixed = {f'{hook.PREFIX}{i}/{j}':.0037*(1+i*4+j) for i in range(28) for j in range(4)}
  def pack(x): return FakeTensor(shape=x.shape,dtype=x.dtype,tag='packed:'+x.tag)
  def quantize(x,scale=None):
    assert type(x) is FakeTensor, 'Original quantization must never receive the fused token'
    rows,width = x.numel()//x.shape[-1],x.shape[-1]
    quantizations.append((x,scale))
    return FakeTensor(shape=((rows+15)//16*16,width),dtype=dtypes.fp8e4m3,tag='quantized'),scale
  def dispatch(x,w,s,ws,out_scale,rows,variant):
    assert variant == 'transpose4x4'
    dispatches.append((x,w,s,ws,out_scale,rows))
    return FakeTensor(shape=((rows+15)//16*16,9216),dtype=dtypes.fp8e4m3,tag='fused-packed')
  def run(model,rows,prefix=None,past=0,keywords=False):
    prefix = prefix or hook.PREFIX+'0'
    width = 512 if prefix.startswith('expert.') else 2048
    x = FakeTensor(shape=(1,rows,width),dtype=dtypes.bfloat16)
    cache = FakeTensor(shape=(2,1,8,2304,128),dtype=dtypes.bfloat16)
    if keywords: return model.text_layer(x,cos=None,sin=None,prefix=prefix,cache=cache,mask=None,cache_pos=past)
    return model.text_layer(x,None,None,prefix,cache,None,past)
  with patch.object(packed,'Tensor',FakeTensor),patch.object(packed,'pack',pack),patch.object(packed,'quantize',quantize),patch.object(core,'Tensor',FakeTensor):
    model = FakeModel(core)
    packed.install(model,fixed_scales=fixed)
    # An unrelated outer installer must remain present after our scope.
    q,l,t = model.quantize,model.linear,model.text_layer
    def outer_q(self,x): return q(x)
    def outer_l(self,x,name,quantized_input=None,heads=None): return l(x,name,quantized_input,heads)
    def outer_t(self,x,*args,**kwargs): return t(x,*args,**kwargs)
    model.quantize,model.linear,model.text_layer = (MethodType(f,model) for f in (outer_q,outer_l,outer_t))
    originals = {name:getattr(model,name) for name in ('quantize','linear','text_layer')}
    scales_object = model.prefill_activation_scales
    with hook.install(model,packed,kernel,device,fixed,dispatch=dispatch,qualified_rows=hook.SUPPORTED_ROWS) as session:
      previous = {'prefill':False,'prefix':'outer-state','index':19}
      session.state.update(previous)
      for rows in hook.SUPPORTED_ROWS:
        for i in range(28):
          start_q,start_c = len(quantizations),len(FakeTensor.calls)
          output = run(model,rows,hook.PREFIX+str(i),keywords=bool(i%2))
          assert output.shape == (1,rows,2048) and session.state == previous and session.frame is None
          assert len(quantizations)-start_q == 3, 'Only QKV, O, and FC1 inputs use the ordinary quantizer'
          events = FakeTensor.calls[start_c:]
          assert not any(e[0].startswith('ordinary-') for e in events)
          mlps = [e for e in events if e[0] == 'original-gemm' and '.mlp.' in e[1]]
          assert len(mlps) == 1 and '.mlp.fc2.' in mlps[0][1]
          assert mlps[0][2][0].tag == 'fused-packed'
          assert mlps[0][2][2] is session.fixed[f'{hook.PREFIX}{i}/3']
          assert dispatches[-1][2] is session.fixed[f'{hook.PREFIX}{i}/2']
          assert dispatches[-1][4] is session.fixed[f'{hook.PREFIX}{i}/3']
      assert session.stats['python_trace_fc1'] == session.stats['python_trace_fc2'] == session.stats['python_trace_fused_quantize'] == 84
      assert set(session.stats['layer_hits'].values()) == {3}
      checks.append('all_28_layers_three_rows_actual_text_layer_and_actual_packed_fc2_exact_dispatch')
      before = len(dispatches)
      for rows,prefix,past in ((64,hook.PREFIX+'0',5),(2112,hook.PREFIX+'0',1),(128,hook.PREFIX+'0',0),(64,'expert.expert.layers.0',None)):
        run(model,rows,prefix,past)
        assert len(dispatches) == before and session.state == previous
      # Standalone vision and other linears pass through without a text-layer scope.
      x = FakeTensor(shape=(1,2112,2048),dtype=dtypes.bfloat16)
      model.linear(x,hook.PREFIX+'0.mlp.fc1')
      assert len(dispatches) == before
      vision_name = 'vlm.model.visual.encoder.layers.0.mlp.fc1'
      model.weights[vision_name+'.weight'] = FakeTensor(shape=(4304,1152),dtype=dtypes.fp8e4m3)
      model.linear(FakeTensor(shape=(2,720,1152),dtype=dtypes.bfloat16),vision_name)
      assert len(dispatches) == before
      checks.append('decode_expert_other_rows_nonzero_cache_and_outside_layer_passthrough')
      assert model.prefill_activation_scales is scales_object and not scales_object
      checks.append('calibration_capture_dictionary_untouched_and_nested_quantizer_state_restored')
      # Runtime scale tensors are reused, not copied or rebuilt from host values.
      out_scale = session.fixed[hook.PREFIX+'0/3']
      out_scale.data = [1.03125]
      run(model,2112)
      assert dispatches[-1][4] is out_scale and dispatches[-1][4].data == [1.03125]
      checks.append('actual_output_scale_tensor_identity_preserved_after_mutation')
      # Exercise token misuse gates in the same pinned quantizer state.
      frame = {'prefix':hook.PREFIX+'0','rows':2112,'active':True,'token':None,'consumed':False}
      session.frame = frame
      session.state.update(prefill=True,prefix=frame['prefix'],index=1)
      raises(AssertionError,lambda:model.linear(x,frame['prefix']+'.mlp.fc1'))
      session.state['index'] = 2
      token = model.linear(x,frame['prefix']+'.mlp.fc1')
      raises(AssertionError,token.square)
      raises(TypeError,lambda:token.numpy())
      raises(AssertionError,lambda:model.quantize(token))
      token.relu()
      raises(AssertionError,token.relu)
      token.square()
      raises(AssertionError,lambda:model.linear(token,hook.PREFIX+'1.mlp.fc2'))
      model.linear(token,frame['prefix']+'.mlp.fc2')
      raises(AssertionError,lambda:model.linear(token,frame['prefix']+'.mlp.fc2'))
      frame['active'] = False
      raises(RuntimeError,lambda:token.shape)
      session.frame = None
      session.state.update(previous)
      checks.append('strict_relu_square_fc2_single_use_prefix_and_escape_gates')
      # Missing/moved calibration slots and replacement weights fail explicitly.
      key = hook.PREFIX+'0.mlp.fc1.weight'
      old = model.weight_scales[key]
      model.weight_scales[key] = FakeTensor([.5],dtype=dtypes.float32)
      raises(AssertionError,lambda:run(model,2112))
      model.weight_scales[key] = old
      assert session.state == previous and session.frame is None
      checks.append('weight_scale_replacement_rejected_and_exception_state_restored')
      model.prefill_jit.cnt,model.prefill_jit.captured = 2,object()
      record = dict(session.stats)
    assert all(getattr(model,name) is value for name,value in originals.items())
    assert not hasattr(model,'_agent_fused_mlp_active') and session.stats['methods_restored']
    assert model.prefill_jit.cnt == 0 and model.prefill_jit.captured is None and model.prefill_jit.resets == 1
    assert model.decode_jit.resets == 0
    record['methods_restored'] = session.stats['methods_restored']
    checks.append('original_outer_installer_methods_restored')
    checks.append('captured_prefill_graph_invalidated_on_exit_and_decoder_jit_untouched')
    for label,mutation,expected_error in (
      ('capture_scales',lambda m:packed.install(m,fixed_scales=fixed,capture_scales=True),AssertionError),
      ('dynamic_scales',lambda m:packed.install(m),AssertionError),
      ('bias',lambda m:m.weights.__setitem__(hook.PREFIX+'0.mlp.fc1.bias',FakeTensor(shape=(9216,),dtype=dtypes.bfloat16)),AssertionError)):
      other = FakeModel(core)
      packed.install(other,fixed_scales=fixed)
      mutation(other)
      raises(expected_error,lambda:hook.Session(other,packed,kernel,device,fixed,dispatch=dispatch))
      checks.append(label+'_rejected_before_install')
    try:
      with hook.install(model,packed,kernel,device,fixed,dispatch=dispatch): raise KeyboardInterrupt('CPU cleanup check')
    except KeyboardInterrupt: pass
    assert all(getattr(model,name) is value for name,value in originals.items())
    checks.append('keyboard_interrupt_finally_restoration')
    model.prefill_jit.cnt = 1
    raises(AssertionError,lambda:hook.Session(model,packed,kernel,device,fixed,dispatch=dispatch))
    model.prefill_jit.cnt = 0
    checks.append('installation_after_prefill_tracing_rejected')
    for corrupt in (dict(fixed,**{hook.PREFIX+'0/3':float('nan')}),{k:v for k,v in fixed.items() if k != hook.PREFIX+'0/3'}):
      raises(AssertionError,lambda:hook.Session(model,packed,kernel,device,corrupt,dispatch=dispatch))
    checks.append('missing_or_nonfinite_fixed_scale_metadata_rejected')
    before = len(dispatches)
    with hook.install(model,packed,kernel,device,fixed,dispatch=dispatch) as default_session:
      run(model,2048)
      assert 2048 not in default_session.rows and len(dispatches) == before
    raises(AssertionError,lambda:hook.Session(model,packed,kernel,device,fixed,dispatch=dispatch,qualified_rows=(2049,)))
    checks.append('2048_disabled_by_default_and_unknown_qualified_rows_rejected')
  runner = load(HERE/'agent-fused-mlp-run.py')
  micro = runner.micro_gate(runner.MICRO)
  assert micro['variant'] == 'transpose4x4' and not micro['old_fused_enabled']
  raises(AssertionError,lambda:runner.micro_gate(runner.MICRO,(2048,)))
  checks.append('historical_2027_2112_GPU_evidence_cannot_qualify_2048')
  checks.append('passed_gpu_micro_report_sources_and_selected_variant_verified_on_cpu')
  fixture = {'warmup':[],'runs':[{'sample':'immutable-sample','input_latest_timestamp_us':1,
    'prediction':{'timings_ms':{'total':1.},'prompt_tokens':2027,'prefill_tokens':2112,'generated_tokens':1,
                  'generated_token_ids':[3],'reasoning':'text','xyz':[[[0.0,1.0,2.0]]],'rot':[[[1.0]]],
                  'stage_timing_kind':'complete'}}]}
  changed = copy.deepcopy(fixture)
  changed['runs'][0]['prediction']['timings_ms']['total'] = 9.
  assert runner.compare(changed,fixture)['float_bits_exact']
  mutations = [lambda p:p['xyz'][0][0].__setitem__(0,-0.0),lambda p:p['xyz'][0][0].__setitem__(1,1.0000000000000002),
               lambda p:p['rot'][0][0].__setitem__(0,0.0),lambda p:p['generated_token_ids'].__setitem__(0,4),
               lambda p:p.__setitem__('prefill_tokens',2111),lambda p:p.__setitem__('future_field',True),
               lambda p:p['xyz'][0][0].__setitem__(0,float('nan'))]
  for mutate in mutations:
    changed = copy.deepcopy(fixture); mutate(changed['runs'][0]['prediction'])
    raises(AssertionError,lambda:runner.compare(changed,fixture))
  changed = copy.deepcopy(fixture); changed['runs'][0]['sample'] = 'other'
  raises(AssertionError,lambda:runner.compare(changed,fixture))
  checks.append('full_prediction_parity_rejects_one_bit_signed_zero_rotation_tokens_lengths_fields_nan_and_input_changes')
  manifest = runner.make_manifest(Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/full-native-5hz-padded/invocation.json'))
  runner.verify_manifest(manifest)
  corrupted = copy.deepcopy(manifest)
  corrupted['files'][str(HERE/'agent-fused-mlp-run.py')] = '0'*64
  raises(AssertionError,lambda:runner.verify_manifest(corrupted))
  checks.append('recorded_invocation_manifest_preparation_and_source_change_rejection')
  nested = runner.make_manifest(Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-speculation-quality-reference/invocation.json'))
  runner.verify_manifest(nested)
  assert len(nested['chain']) == 2
  command = nested['chain'][0]['recorded']['command']
  assert runner.option(command,'warmup') == '2' and runner.option(command,'runs') == '62'
  checks.append('frozen_nested_speculation_invocation_preserves_64_prediction_schedule')
  assert not Device._opened_devices
  return {'passed':True,'gpu_executed':False,'opened_devices':[],'checks':checks,'normal_actual_text_layers_checked':84,
          'trace_test_stats_including_negative_controls':record,
          'source_sha256':{p.name:hook.sha(p) for p in (HERE/'agent-fused-mlp.py',HERE/'agent-fused-mlp-run.py',Path(__file__))},
          'numerical_claim':'Dispatch/state tests only. Existing GPU kernel byte checks and parent pipeline parity are separate.'}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',required=True,type=Path)
  args = parser.parse_args()
  report = check()
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'passed':True,'checks':len(report['checks']),'opened_devices':report['opened_devices']}))


if __name__ == '__main__': main()

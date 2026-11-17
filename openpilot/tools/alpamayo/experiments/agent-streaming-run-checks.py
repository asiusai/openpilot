"""CPU controls. GPU arithmetic and checkpoint output parity remain parent gates."""
import importlib.util,json,sys
from pathlib import Path
from types import MethodType,SimpleNamespace
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stream_checks_common',HERE/'agent-streaming-run-common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
common.verify_sources();common.prepare_cpu_environment()
import numpy as np
import torch
from tinygrad import Tensor,TinyJit,Device,dtypes
from tinygrad.uop.ops import UOp,KernelInfo
kernels=common.load('agent-streaming-run-kernels')
runtime=common.load('agent-streaming-run-runtime')
rng=np.random.default_rng(1749)

def tensor(value,dtype=None):return Tensor(np.asarray(value),device='CPU',dtype=dtype).realize()
def array(value):return value.float().numpy()

def copies():
  # Exercise the actual kernels through TinyJit capture and replay. Poison all
  # unused storage, mutate maps and values, and compare with independent indexing.
  bank_np=np.arange(2*1*2*11*4,dtype=np.float32).reshape(2,1,2,11,4)
  bank=tensor(bank_np);stage=Tensor.zeros(2,1,2,3,4,device='CPU').realize()
  def build(q,qr,br,frames):
    full=kernels.assemble(q,bank,qr,br).realize()
    stage.assign(kernels.gather(full,frames)).realize()
    return full
  def commit(dest):return kernels.scatter(bank,stage,dest).realize()
  build_jit,commit_jit=TinyJit(build),TinyJit(commit)
  for iteration in range(6):
    q_np=rng.normal(size=(2,1,2,5,4)).astype(np.float32)+iteration
    qr=np.array([0,-1,1,-1,2,3,-1,-1],np.int32)
    br=np.array([-1,(iteration+1)%11,-1,7,-1,-1,10,-1],np.int32)
    frames=np.array([6,1,4],np.int32);dest=np.array([7,-1,2],np.int32)
    expected=np.zeros((2,1,2,8,4),np.float32)
    for slot in range(8):
      if qr[slot]>=0:expected[...,slot,:]=q_np[...,qr[slot],:]
      elif br[slot]>=0:expected[...,slot,:]=bank_np[...,br[slot],:]
    result=build_jit(tensor(q_np),tensor(qr),tensor(br),tensor(frames))
    np.testing.assert_array_equal(array(result),expected)
    np.testing.assert_array_equal(array(stage),expected[...,frames,:])
    old=bank_np.copy()
    for source,destination in enumerate(dest):
      if destination>=0:bank_np[...,destination,:]=expected[...,frames[source],:]
    commit_jit(tensor(dest))
    np.testing.assert_array_equal(array(bank),bank_np)
    assert not np.array_equal(old,bank_np)
  # Out-of-bounds reads and disabled writes must neither fetch poison nor wrap.
  poisoned=tensor(np.full((1,3,4),np.nan,np.float32))
  np.testing.assert_array_equal(array(kernels.gather(poisoned,tensor(np.array([-1,3,999],np.int32)))),np.zeros((1,3,4)))
  out=tensor(np.ones((1,3,4),np.float32))
  np.testing.assert_array_equal(array(kernels.scatter(out,poisoned,tensor(np.array([-1,3,999],np.int32)))),np.ones((1,3,4)))
  return dict(jit_iterations=6,read_then_commit_alias_mutation=True,invalid_gather_zero=True,invalid_scatter_unchanged=True)

def reference(q,k,v,slots,valid):
  # Deliberately scalar per query: no runtime kernel or compact-row causal rule.
  out=np.zeros_like(q,dtype=np.float64)
  for head in range(q.shape[1]):
    for row,slot in enumerate(slots):
      if slot<0:continue
      kh=head//(q.shape[1]//k.shape[1]);end=min(slot+1,valid)
      score=k[0,kh,:end].astype(np.float64)@q[0,head,row].astype(np.float64)/np.sqrt(q.shape[-1])
      p=np.exp(score-score.max());p/=p.sum();out[0,head,row]=p@v[0,kh,:end]
  return out

def causality():
  q,k,v=(rng.normal(size=s).astype(np.float32) for s in ((1,4,8,16),(1,2,19,16),(1,2,19,16)))
  slots=np.array([0,2,4,7,11,14,-1,-1],np.int32);expected=reference(q,k,v,slots,15)
  mask=np.full((8,19),-np.inf,np.float32)
  for row,slot in enumerate(slots):
    if slot>=0:mask[row,:min(slot+1,15)]=0
  # Independent dense PyTorch CPU implementation, only a test oracle.
  kt=np.repeat(k,2,axis=1);vt=np.repeat(v,2,axis=1)
  result=torch.nn.functional.scaled_dot_product_attention(torch.from_numpy(q),torch.from_numpy(kt),torch.from_numpy(vt),
    attn_mask=torch.from_numpy(mask)).numpy()
  np.testing.assert_allclose(result,expected,atol=3e-7,rtol=3e-7)
  poisoned_k,poisoned_v=k.copy(),v.copy();poisoned_k[:,:,15:]=1e20;poisoned_v[:,:,15:]=-1e20
  np.testing.assert_array_equal(reference(q,poisoned_k,poisoned_v,slots,15),expected)
  changed=k.copy();changed[:,:,8:]=1000
  np.testing.assert_array_equal(reference(q,changed,v,slots,15)[:,:,:4],expected[:,:,:4])
  assert not np.allclose(reference(q,changed,v,slots,15)[:,:,4:6],expected[:,:,4:6])
  wrong=reference(q,k,v,np.arange(8),15)
  assert not np.allclose(wrong[:,:,:6],expected[:,:,:6])
  return dict(independent_torch_cpu_reference=True,original_slot_causality=True,padded_poison=True,future_mutation=True,
    compact_index_negative_control_detected=True)

def quantized_full_control():
  from model import EdgeModel
  packed=common.load('quantize-packed',common.EXP)
  prefix='vlm.model.language_model.layers.0';names=('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.fc1','mlp.fc2')
  # Run the actual packed-quantizer wrapper and conversion. Replace its GPU-only
  # packing/GEMM with a deterministic diagonal CPU operator, in this test only.
  packed.pack=lambda x:x
  def cpu_diagonal(out,x,weight,scale,weight_scale,*,packed_input=False):
    i=UOp.range(out.max_numel(),0);width=x.shape[-1];columns=out.shape[-1]
    value=x.flatten()[i//columns*width+i%columns%width].cast(dtypes.float32)*scale.flatten()[0]*weight_scale.flatten()[0]
    return out.flatten()[i].store(value.cast(out.dtype)).end(i).sink(arg=KernelInfo(name='stream_test_diagonal'))
  packed._gemm=cpu_diagonal
  fixed={f'{prefix}/{i}':s for i,s in enumerate((.02,.03,.04,.05))}
  records=[];old_quantize=packed.quantize
  def record(x,scale=None):
    records.append(float(scale.item()) if scale is not None else None);return old_quantize(x,scale)
  packed.quantize=record
  def make(stream):
    model=SimpleNamespace(config={'vlm_config':{'text_config':{'num_hidden_layers':28}},'expert_config':{'llm_config':{'num_hidden_layers':28}}},
      weights={},weight_scales={},dtype=dtypes.bfloat16,cuda_kernels=False,fp8_wmma=False)
    for suffix in names:
      width=1024 if suffix in ('self_attn.k_proj','self_attn.v_proj') else 2048
      model.weights[prefix+'.'+suffix+'.weight']=Tensor.zeros(width,2048,device='CPU',dtype=dtypes.fp8e4m3).realize()
      model.weight_scales[prefix+'.'+suffix+'.weight']=Tensor([.25],device='CPU').reshape(()).realize()
    for suffix in ('input_layernorm','post_attention_layernorm'):
      model.weights[prefix+'.'+suffix+'.weight']=Tensor.ones(2048,device='CPU',dtype=dtypes.bfloat16).realize()
    for name in ('norm','linear','quantize','attention','text_layer'):setattr(model,name,MethodType(getattr(EdgeModel,name),model))
    model.rope=EdgeModel.rope
    owner=runtime.StreamingPrefill(model) if stream else None
    packed.install(model,fixed_scales=fixed)
    return model,owner
  original,_=make(False);candidate,owner=make(True)
  data=rng.normal(size=(1,128,2048)).astype(np.float32)
  x=tensor(data,dtypes.bfloat16)
  angles=np.arange(128,dtype=np.float32)[None,None,:,None]*np.linspace(.001,.1,128,dtype=np.float32)[None,None,None,:]
  cos,sin=tensor(np.cos(angles),dtypes.bfloat16),tensor(np.sin(angles),dtypes.bfloat16)
  c1,c2=(Tensor.zeros(2,1,8,256,128,device='CPU',dtype=dtypes.bfloat16).realize() for _ in range(2))
  frames=tensor(np.array([5,21,100],np.int32));owner.pending_kv=[Tensor.zeros(2,1,8,3,128,device='CPU',dtype=dtypes.bfloat16).realize()]
  expected=original.text_layer(x,cos,sin,prefix,c1,None,cache_pos=0);first=records.copy();records.clear()
  with owner.stage('full',frames):result=candidate.text_layer(x,cos,sin,prefix,c2,None,cache_pos=0)
  second=records.copy();records.clear()
  np.testing.assert_array_equal(array(result),array(expected));np.testing.assert_array_equal(array(c1),array(c2))
  np.testing.assert_allclose(first,list(fixed.values()),rtol=1e-7,atol=0);assert first==second
  # Raw capture is pre-RoPE; rebasing it at these positions recovers active keys.
  raw=owner.pending_kv[0]
  selected_cos,selected_sin=kernels.gather(cos,frames),kernels.gather(sin,frames)
  _,rebased=EdgeModel.rope(raw[0],raw[0],selected_cos,selected_sin)
  np.testing.assert_array_equal(array(rebased),array(kernels.gather(c2[0],frames)))
  np.testing.assert_array_equal(array(raw[1]),array(kernels.gather(c2[1],frames)))
  assert not np.array_equal(array(raw[0]),array(rebased))
  # Restore quantizer state after the layer; out-of-prefill calls must pass through.
  candidate.quantize(x);assert records==[]
  # Exercise the actual sparse layer using a CPU-only attention oracle. No dense
  # implementation is present in the runtime; only this test replaces HIP launch.
  full_angles=np.arange(2048,dtype=np.float32)[None,None,:,None]*np.linspace(.001,.1,128,dtype=np.float32)[None,None,None,:]
  fc,fs=tensor(np.cos(full_angles),dtypes.bfloat16),tensor(np.sin(full_angles),dtypes.bfloat16)
  slots=np.arange(100,228,dtype=np.int32);slot_tensor=tensor(slots)
  qr=np.full(2048,-1,np.int32);qr[slots]=np.arange(128)
  br=np.arange(2048,dtype=np.int32);br[slots]=-1;br[2026:]=-1
  bank_values=rng.normal(0,.25,size=(2,1,8,2048,128)).astype(np.float32)
  owner.bank=[tensor(bank_values,dtypes.bfloat16)]
  active=Tensor.full((2,1,8,2304,128),99,device='CPU',dtype=dtypes.bfloat16).realize()
  attention_original=runtime.kernels.attention;oracle_calls=[]
  def oracle(q,k,v,s,l):
    positions=s.numpy();valid=int(l.numpy()[0]);qv,kv,vv=array(q),array(k),array(v)
    mask=np.where(np.arange(k.shape[2])[None,:]<=positions[:,None],0,-np.inf).astype(np.float32)
    mask[:,valid:]=-np.inf
    output=torch.nn.functional.scaled_dot_product_attention(torch.from_numpy(qv),
      torch.from_numpy(np.repeat(kv,2,axis=1)),torch.from_numpy(np.repeat(vv,2,axis=1)),attn_mask=torch.from_numpy(mask)).numpy()
    oracle_calls.append(dict(slots=positions.copy(),k=kv.copy(),v=vv.copy()))
    return tensor(output,dtypes.bfloat16)
  runtime.kernels.attention=oracle
  try:
    with owner.stage('sparse',frames,query_slots=slot_tensor,query_read=tensor(qr),bank_read=tensor(br),
                     full_cos=fc,full_sin=fs,limits=tensor(np.array([2026,128],np.int32))):
      sparse_result=candidate.text_layer(x,kernels.gather(fc,slot_tensor),kernels.gather(fs,slot_tensor),prefix,active,None,cache_pos=0)
    assert np.isfinite(array(sparse_result)).all()
    assert records==first and len(oracle_calls)==1
    assert np.array_equal(oracle_calls[0]['slots'],slots)
    bank=owner.bank[0];_,expected_keys=EdgeModel.rope(bank[0],bank[0],fc,fs)
    reused=np.flatnonzero(br>=0)
    np.testing.assert_array_equal(oracle_calls[0]['k'][:,:,reused],array(expected_keys)[:,:,reused])
    np.testing.assert_array_equal(oracle_calls[0]['v'][:,:,reused],array(bank[1])[:,:,reused])
    np.testing.assert_array_equal(oracle_calls[0]['v'][:,:,2026:],0)
    np.testing.assert_array_equal(array(active[0,:,:,:2048]),oracle_calls[0]['k'])
    np.testing.assert_array_equal(array(active[1,:,:,:2048]),oracle_calls[0]['v'])
    np.testing.assert_array_equal(array(active[:,:,:,2048:]),99)
  finally:runtime.kernels.attention=attention_original
  return dict(actual_original_text_layer=True,actual_fixed_quantizer_wrapper=True,fixed_scale_sequence=first,
    output_and_active_cache_exact=True,pre_rope_capture_and_current_rebase_exact=True,
    sparse_fixed_scale_sequence_identical=True,sparse_current_positions_and_assembled_active_cache=True,
    limit='One actual layer control with CPU diagonal GEMM; checkpoint, packed AMD GEMM, selected HIP and 28-layer parity require parent')

def prepared_inputs():
  fast=common.load('fast-inputs',common.EXP);vision=common.load('vision-cache',common.EXP)
  adapter=fast.FastInputs('/tmp/opencode/alpamayo-asius-20261114/model',pixel_scale=1,cache_resize=True,
    camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True)
  planner=common.Planner();events=[];metas=[];spans_first=None
  paths=json.loads((common.ROOT/'smoke32.json').read_text())
  for path in paths:
    data=common.read_sample(path);config,_,prepared=adapter.prepare(data);ids,pos,delta,_=adapter.native_inputs(prepared)
    indices=np.flatnonzero(ids[0]==config.vlm_config.image_token_id)
    spans=[(int(a[0]),int(a[-1])+1) for a in np.split(indices,np.flatnonzero(np.diff(indices)!=1)+1)]
    meta=common.metadata(data,prepared,path,vision.VisionCache.input_keys(prepared),'frozen')
    plan=planner.begin(meta,spans,ids.shape[1]);event=plan['event'];events.append(event);metas.append(meta)
    assert np.array_equal(plan['query_slots'][:plan['last']+1],np.flatnonzero(plan['bank_read'][:ids.shape[1]]<0))
    assert len(set(plan['destinations'][plan['destinations']>=0].tolist()))==np.count_nonzero(plan['destinations']>=0)
    assert all(i not in event['reused_images'] for i in (4,9))
    if plan['event']['reset']:assert not (plan['bank_read']>=0).any()
    assert len(planner.bank)<=48 and len(plan['after'])<=48
    planner.commit();spans_first=spans if spans_first is None else spans_first
    print('CPU prepared',Path(path).name,event['reset'],event['query_tokens'],flush=True)
  assert any(e['reused_images'] for e in events) and any(e['reset']=='periodic_2s' for e in events)
  # Every invalidation is tested without consuming future outputs.
  reset_checks={}
  seed=metas[0]
  for label,changes in [('time_reset',{}),('input_gap',{'timestamp_us':seed['timestamp_us']+600001}),
      ('navigation_change',{'timestamp_us':seed['timestamp_us']+250000,'nav':'changed current nav'}),
      ('provenance_change',{'timestamp_us':seed['timestamp_us']+250000,'provenance':'changed'}),
      ('periodic_2s',{})]:
    p=common.Planner();p.begin(seed,spans_first,events[0]['original_prompt_tokens']);p.commit()
    if label=='periodic_2s':
      for step in range(1,8):
        p.begin(dict(seed,timestamp_us=seed['timestamp_us']+step*250000),spans_first,events[0]['original_prompt_tokens']);p.commit()
      changes={'timestamp_us':seed['timestamp_us']+2000000}
    result=p.begin(dict(seed,**changes),spans_first,events[0]['original_prompt_tokens'])
    assert result['event']['reset']==label;reset_checks[label]=True;p.abort();assert not p.bank and p.epoch is None
  # Same capture stamp with mutated image content cannot hit the bank.
  p=common.Planner();p.begin(seed,spans_first,events[0]['original_prompt_tokens']);p.commit()
  identities=list(seed['identities']);key=identities[0];identities[0]=(*key[:-1],'mutated-pixels')
  altered=dict(seed,timestamp_us=seed['timestamp_us']+100000,identities=tuple(identities))
  result=p.begin(altered,spans_first,events[0]['original_prompt_tokens']);assert 0 not in result['event']['reused_images']
  # Metadata rejects current-sample future capture and stale history.
  data=common.read_sample(paths[0]);_,_,prepared=adapter.prepare(data);keys=vision.VisionCache.input_keys(prepared)
  rejections=[]
  for change in (200000,-500000):
    bad=dict(data);bad['absolute_timestamps']=data['absolute_timestamps'].clone();bad['absolute_timestamps'].reshape(-1)[0]+=change
    try:common.metadata(bad,prepared,paths[0],keys,'frozen')
    except ValueError:rejections.append(True)
  assert len(rejections)==2
  return dict(samples=32,events=events,maximum_bank_entries=max(e['bank_entries'] for e in events),
    reset_controls=reset_checks,changed_image_same_stamp_misses=True,invalid_capture_rejected=True,
    original_positions_untouched=True,current_motion_and_nav=True)

def main():
  report=common.report_base('cpu-controls');report['passed']=False
  calibration=json.loads((common.ROOT/'fixed-language-scales.json').read_text())['scales']
  assert set(calibration)=={f'vlm.model.language_model.layers.{layer}/{index}' for layer in range(28) for index in range(4)}
  assert all(np.isfinite(value) and value>0 for value in calibration.values())
  report['fixed_language_scales']=dict(count=112,sha256=common.sha(common.ROOT/'fixed-language-scales.json'))
  for name,check in [('copies',copies),('causality',causality),('forced_full_layer',quantized_full_control),('current_inputs',prepared_inputs)]:
    print('CPU control:',name,flush=True);report[name]=check()
  report['opened_devices']=sorted(Device._opened_devices);assert not any('AMD' in d or 'USB' in d for d in report['opened_devices'])
  report.update(passed=True,gpu_executed=False,quality_passed=None)
  (common.ROOT/'cpu-checks.json').write_text(json.dumps(report,indent=2)+'\n')
  print('CPU controls passed',report['opened_devices'],flush=True)

if __name__=='__main__':main()

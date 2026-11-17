"""One short, post-capture check on actual model tensors. Decoder history is never advanced here."""
import copy
import hashlib

def run(owner,ret):
  import numpy as np
  from tinygrad import Tensor,dtypes
  H=owner.common.load('agent-ck-integrated-hook')
  assert owner.arm=='candidate' and owner.original_jit.captured is not None
  history=copy.deepcopy(owner.decoder.history);args=owner.last_args
  captured=owner.original_jit.captured;linked=captured._linear.key
  row=owner.layers[27]
  def read_outputs(): return tuple(owner.download(row[x]) for x in ('out','lse'))
  before=read_outputs();head_before=tuple(owner.download(x) for x in ret)
  events=[]
  def replay(label,new_args=args):
    count=owner.original_jit.cnt;value=owner.original_jit(*new_args)
    assert owner.original_jit.captured is captured and captured._linear.key==linked
    owner.after_prefill(label,count)
    assert owner.decoder.history==history
    events.append(label)
    return value,read_outputs()
  def exact(a,b):
    for x,y in zip(a,b,strict=True): np.testing.assert_array_equal(x,y)
  def head_exact(value): exact(head_before,tuple(owner.download(x) for x in value))
  def float_o(bits): return (bits.astype(np.uint32)<<16).view(np.float32)
  # Same live input, cache owners, and captured graph; every guarded output is poisoned again by the graph.
  value,again=replay('control_same_input_poison_replay');exact(before,again);head_exact(value)
  # All capacity tails are preserved and restored. Decode is paused before it can consume cache suffixes.
  tails=[c[:,:,:,2048:].clone().realize() for c in owner.model.cache]
  try:
    for cache in owner.model.cache: cache[:,:,:,2048:].assign(float('nan')).realize()
    value,tail=replay('control_all28_capacity_tail_nan');exact(before,tail);head_exact(value)
    checks=Tensor.stack(*[(c[:,:,:,2048:].isnan()==False).cast(dtypes.int32).sum() for c in owner.model.cache]).realize()
    assert not owner.download(checks).any(),'Capacity tails were overwritten or silently cleared'
  finally:
    for cache,saved in zip(owner.model.cache,tails,strict=True): cache[:,:,:,2048:].assign(saved).realize()
  # Mutate one actual language-input row. Fixed activation scales preserve the causal prefix across all 28 layers.
  changed=args[0].clone().realize()
  changed[:,1536:1537].assign(changed[:,1536:1537]+0.5).realize()
  changed_words=(changed.bitcast(dtypes.uint16)!=args[0].bitcast(dtypes.uint16)).cast(dtypes.int32).sum().reshape(1).realize()
  assert owner.download(changed_words).item()>0
  _,mutated=replay('control_actual_language_row1536_mutation',(changed,*args[1:]))
  exact((before[0][:,:,:1536],before[1][:,:,:1536]),(mutated[0][:,:,:1536],mutated[1][:,:,:1536]))
  model_delta=float(np.abs(float_o(mutated[0])-float_o(before[0])).max())
  assert model_delta>=0.0001,('Stale model input replay',model_delta)
  value,restored=replay('control_original_model_input_restored');exact(before,restored);head_exact(value)
  # Use actual restored layer-27 Q/K/V, the same normalized output helper, and a fresh guard allocation.
  q,k,v=row['packed'];vm=v.clone().realize();vm[:,2:3,128:].assign(vm[:,2:3,128:]+0.125).realize()
  stores=(owner.io.allocate('AMD',(H.OC+2*H.G,),dtypes.bfloat16),owner.io.allocate('AMD',(H.LC+2*H.G,),dtypes.float32))
  for store in stores: H.poison(owner.common,store)
  out,lse=owner.helper.outputs('candidate',True,q,k,vm,*stores);out.realize(lse)
  for store,result in zip(stores,(out,lse),strict=True): assert not owner.download(H.audit(store,result)).any()
  gqa=owner.download(out);gqa_lse=owner.download(lse);exact((before[1],),(gqa_lse,))
  allowed=np.zeros(before[0].shape,bool);allowed[:,4:6,128:]=True
  assert np.array_equal(gqa[~allowed],before[0][~allowed]),'Cross-head or causal GQA contamination'
  gqa_delta=float(np.abs(float_o(gqa)-float_o(before[0])).max())
  assert gqa_delta>=0.0001,('Stale live V mutation',gqa_delta)
  # Recompute the audit after deliberate corruption; cached zero flags cannot qualify this test.
  negatives=[]
  def patch_word(tensor,index,value,dt):
    # A typed physical Buffer view avoids unsupported STORE(BITCAST(...)) and works with device-only returned storage.
    from tinygrad.uop.ops import UOp
    buffer=tensor.uop.buffer;size=dt.itemsize;count=buffer.nbytes//size
    assert 0<=index<count and tensor.dtype.itemsize==size
    lo,hi=max(0,index-1),min(count,index+2)
    neighbors=Tensor(UOp.from_buffer(buffer.view((hi-lo)*size,lo*size).ensure_allocated(),dt))
    old=owner.download(neighbors).reshape(-1)
    target=buffer.view(size,index*size).ensure_allocated()
    assert target.base is buffer.base and target.offset==buffer.offset+index*size
    assert target.get_storage().buf==buffer.get_storage().buf+index*size
    Tensor(UOp.from_buffer(target,dt)).assign(value).realize()
    current=owner.download(neighbors).reshape(-1);expected=old.copy();expected[index-lo]=value
    np.testing.assert_array_equal(current,expected)
    return {'offset_bytes':target.offset,'base_bytes':target.base.nbytes,'address':int(target.get_storage().buf),
      'bytes':size,'neighbors_exact':True,'host_mapped':target.get_storage().host is not None}
  for which,store,result in zip(('O','LSE'),row['stores'],(row['out'],row['lse']),strict=True):
    dt=dtypes.uint16 if which=='O' else dtypes.uint32
    nan,guard=(0x7fc1,0x5aa5) if which=='O' else (0x7fc12345,0x5aa55aa5)
    patch=patch_word(store,H.G,nan,dt)
    flags=owner.download(H.audit(store,result));assert flags[0]>0 and flags[1]>0 and flags[4]>0
    negatives.append({'output':which,'kind':'guarded_nan_poison','counts':flags.tolist(),'patch':patch})
    patch=patch_word(store,0,guard^1,dt)
    flags=owner.download(H.audit(store,result));assert flags[2]>0
    negatives.append({'output':which,'kind':'redzone_bit','counts':flags.tolist(),'patch':patch})
    expected=before[0 if which=='O' else 1].reshape(-1)
    patch=patch_word(result,result.numel()-1,int(expected[-1])^1,dt)
    flags=owner.download(H.audit(store,result));assert flags[4]>0
    negatives.append({'output':which,'kind':'returned_last_word_bit','counts':flags.tolist(),'patch':patch})
  value,restored=replay('control_poison_redzones_and_returned_bits_restored');exact(before,restored);head_exact(value)
  assert owner.decoder.history==history
  owner.report['model_controls']={'passed':True,'phases':events,'model_input_max_delta':model_delta,
    'live_GQA_V_max_delta':gqa_delta,'GQA_unchanged_heads_and_causal_prefix_exact':True,
    'all28_capacity_tail_nan_ignored':True,'last_layer_full_bits_and_head_outputs_restored':True,
    'same_captured_graph':True,'decoder_history_unchanged':True,'negative_controls':negatives,
    'full_layer27_output_sha256':[hashlib.sha256(x.tobytes()).hexdigest() for x in before],
    'coverage_limit':'All-28 finite/guard audits; full-bit restoration and causal comparison at layer 27. Earlier pinned micro supplies independent FP32 accuracy.'}

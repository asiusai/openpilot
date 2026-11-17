"""Real JIT input storage and diagnostic output poisoning. No numerical kernel changes."""
import hashlib


def physical_info(tensor):
  return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),op=tensor.uop.op.name,
              device=tensor.uop.device,virtual=tensor.uop.is_virtual,realized=tensor.uop.is_realized)


def materialize(sample,device,records=None):
  import numpy as np
  from tinygrad import Tensor,dtypes
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.uop.ops import Ops
  values=[];facts=[]
  for key in ('a','b','weight','scale'):
    array=np.asarray(sample[key])
    value=Tensor(array,device=device)
    if key!='scale':value=value.bitcast(dtypes.bfloat16)
    before=physical_info(value)
    value=value.clone(device=device).realize()
    assert not value.uop.is_virtual and value.uop.is_realized and value.uop.base.op is Ops.BUFFER
    assert value.shape==array.shape and value.dtype==(dtypes.float32 if key=='scale' else dtypes.bfloat16)
    actual=value.bitcast(dtypes.uint16 if key!='scale' else dtypes.uint32).numpy()
    expected=array if key!='scale' else array.view(np.uint32)
    np.testing.assert_array_equal(actual,expected)
    facts.append(dict(argument=key,before=before,after=physical_info(value),
                      uploaded_bytes_sha256=hashlib.sha256(array.tobytes()).hexdigest()))
    values.append(value)
  prepared=_prepare_jit_inputs(tuple(values),{})
  assert len(prepared[0])==len(set(prepared[0]))==4
  if records is not None:records.append(dict(arguments=facts,distinct_real_input_buffers=4))
  return tuple(values)


def span(tensor):
  buffer=tensor._buffer()
  return buffer.base,buffer.offset,buffer.nbytes


def overlap(a,b):
  return a[0] is b[0] and max(a[1],b[1]) < min(a[1]+a[2],b[1]+b[2])


def overwrite_bytes(tensor,array,verify=True):
  import numpy as np
  from tinygrad.device import Buffer
  raw=memoryview(np.ascontiguousarray(array)).cast('B')
  target=tensor._buffer()
  assert target.nbytes==len(raw)
  original_uop=tensor.uop
  target.copy_from(Buffer('NPY',len(raw),opaque=raw))
  if verify:assert bytes(target.as_memoryview())==bytes(raw), 'Diagnostic buffer write/readback mismatch'
  assert tensor.uop is original_uop, 'Buffer poisoning must preserve Tensor/JIT views'
  return len(raw)


class PoisonedJit:
  def __init__(self,inner,label,state,expected_outputs=None):
    self.inner,self.label,self.state,self.expected_outputs=inner,label,state,expected_outputs

  def __getattr__(self,key):return getattr(self.inner,key)

  def poison(self,args):
    import numpy as np
    from tinygrad import Tensor,dtypes
    if self.inner.captured is None:return
    outputs=self.inner.captured.ret
    outputs=tuple(outputs) if isinstance(outputs,(tuple,list)) else (outputs,)
    if self.expected_outputs is not None:assert len(outputs)==self.expected_outputs
    assert all(isinstance(value,Tensor) for value in outputs)
    input_spans=[span(value) for value in args if isinstance(value,Tensor)]
    output_spans=[span(value) for value in outputs]
    assert not any(overlap(out,inp) for out in output_spans for inp in input_spans), 'Poison output aliases an input'
    assert not any(overlap(out,other) for i,out in enumerate(output_spans) for other in output_spans[i+1:]), 'Poison outputs overlap'
    total=0
    for value in outputs:
      if value.dtype==dtypes.bfloat16:array=np.full(value.shape,0x7fc1,np.uint16)
      elif value.dtype.itemsize==1:array=np.full(value.shape,0x7f,np.uint8)
      else:raise AssertionError('Unexpected poison output dtype')
      total+=overwrite_bytes(value,array)
    self.state['poisoned_replays']+=1
    self.state['verified_poison_bytes']+=total

  def __call__(self,*args):
    self.poison(args)
    return self.inner(*args)

  def run_without_poison(self,*args):
    assert self.inner.captured is not None and self.inner.cnt>=2
    assert self.state['poisoned_replays']>0, 'Timing requires a prior poisoned replay'
    self.state['unpoisoned_timing_calls']+=1
    return self.inner(*args)


def jit_factory(original,result):
  labels=('single-baseline','single-candidate','batch-baseline','batch-candidate')
  states=result.setdefault('poison_checks',[])
  def create(function):
    index=len(states);label=f'layer-group-{index//4}/'+labels[index%4]
    expected=2 if index%4<2 else 17
    state=dict(label=label,poisoned_replays=0,verified_poison_bytes=0,unpoisoned_timing_calls=0)
    states.append(state)
    return PoisonedJit(original(function),label,state,expected)
  return create

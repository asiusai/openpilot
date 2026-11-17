"""Optional exact content guard for one Consumer's repeated vision-position hash.

No model, adapter, sampler, input buffer, or selection policy is replaced.
An immutable byte shadow authorizes digest reuse only after a complete bit comparison.
"""
from collections import Counter
import hashlib
from pathlib import Path
from types import FunctionType, MethodType
import numpy as np

CONSUMER_SHA256='65db37fd614e5f378edc56afc34b947c2ac58151f2b9019c771b0a67cbfceccf'
POSITION_SHAPE=(7200,1152)


class PositionDigest:
  def __init__(self,original):
    self.original=original
    self.shadow=None
    self.saved_digest=None
    self.counts=Counter()

  def __call__(self,value):
    if (type(value) is not np.ndarray or value.shape!=POSITION_SHAPE or
        value.dtype!=np.dtype(np.float32) or not value.flags.c_contiguous):
      return self.original(value)
    # Compare integer representations: NaNs, signed zero, and subnormals retain every bit.
    bits=value.reshape(-1).view(np.uint64)
    if self.shadow is not None and np.array_equal(bits,np.frombuffer(self.shadow,dtype=np.uint64)):
      self.counts['full_content_hits']+=1
      self.counts['compared_bytes']+=value.nbytes
      return self.saved_digest
    self.counts['full_content_misses']+=1
    # Both the original digest and future comparisons refer to this exact snapshot.
    shadow=value.tobytes(order='C')
    snapshot=np.frombuffer(shadow,dtype=value.dtype).reshape(value.shape)
    digest=self.original(snapshot)
    self.shadow,self.saved_digest=shadow,digest
    return digest

  def state(self):
    return dict(counts=dict(self.counts),shadow_bytes=0 if self.shadow is None else len(self.shadow),
      maximum_shadow_bytes=int(np.prod(POSITION_SHAPE))*4,
      equality='complete uint64 representation comparison; no timestamp, identity, or version shortcut')


class Handle:
  def __init__(self,consumer):
    original=consumer._bind_inputs
    path=Path(original.__func__.__code__.co_filename)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CONSUMER_SHA256:
      raise ValueError('Unsupported Consumer source')
    if original.__func__.__name__!='_bind_inputs' or original.__self__ is not consumer:
      raise ValueError('Unexpected input binding method')
    self.consumer=consumer
    self.had_instance_slot='_bind_inputs' in vars(consumer)
    self.old_slot=vars(consumer).get('_bind_inputs')
    self.original=original
    self.guard=PositionDigest(original.__func__.__globals__['digest'])
    fn=original.__func__
    self.delegate=FunctionType(fn.__code__,dict(fn.__globals__,digest=self.guard),fn.__name__,fn.__defaults__,fn.__closure__)
    self.delegate.__kwdefaults__=fn.__kwdefaults__
    delegate=self.delegate
    def bind(owner,prepared,packed,upload):
      return delegate(owner,prepared,packed,upload)
    self.installed=MethodType(bind,consumer)
    consumer._bind_inputs=self.installed
    self.closed=False
    self.provenance=dict(consumer_source_sha256=CONSUMER_SHA256,
      overlay_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      original_method_code_retained=True,scope='one Consumer._bind_inputs vision-position digest only')

  def close(self):
    if self.closed:return
    if self.consumer._bind_inputs is not self.installed:
      raise ValueError('Input binding ownership changed before cleanup')
    if self.had_instance_slot:self.consumer._bind_inputs=self.old_slot
    else:del self.consumer._bind_inputs
    assert self.consumer._bind_inputs==self.original
    self.guard.shadow=None;self.closed=True

  def __enter__(self):return self
  def __exit__(self,*error):self.close()


def install(consumer):return Handle(consumer)

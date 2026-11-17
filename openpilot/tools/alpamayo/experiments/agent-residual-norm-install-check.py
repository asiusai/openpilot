"""CPU dispatch checks through the actual frozen model, norm installer, and quantizer."""
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import MethodType,SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
os.environ.update(DEV='NULL',PROFILE='0')
sys.path[:0]=['/tmp/opencode/alpamayo-asius-20261114/tinygrad',str(HERE.parent)]
from tinygrad import Device,dtypes
from tinygrad.uop.ops import Ops


def load(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_')+'_rn_check',path)
  m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Node:
  def __init__(self,tensor,op=Ops.BUFFER,src=()): self.tensor,self.op,self.dtype,self.src=tensor,op,tensor.dtype,src


class T:
  gemms=[]
  def __init__(self,data=None,*,shape=None,dtype=dtypes.bfloat16,device='mock',tag='input'):
    if isinstance(data,Node): self.__dict__.update(data.tensor.__dict__);return
    self.shape,self.dtype,self.device,self.tag=tuple(shape) if shape is not None else (len(data),),dtype,device,tag
    self.data=data;self.uop=Node(self)
  def numel(self): return math.prod(self.shape)
  def realize(self,*others): return self
  def reshape(self,*shape):
    if len(shape)==1 and isinstance(shape[0],tuple): shape=shape[0]
    shape=tuple(self.numel()//math.prod(n for n in shape if n!=-1) if n==-1 else n for n in shape)
    assert math.prod(shape)==self.numel()
    return T(shape=shape,dtype=self.dtype,tag=self.tag)
  def transpose(self,a,b):
    shape=list(self.shape);shape[a],shape[b]=shape[b],shape[a]
    return T(shape=shape,dtype=self.dtype,tag=self.tag)
  def __add__(self,other):
    assert self.shape==other.shape
    t=T(shape=self.shape,dtype=self.dtype,tag='residual');t.uop=Node(t,Ops.ADD,(self.uop,other.uop));return t
  def replace(self,other): self.uop,self.tag=other.uop,other.tag;return self
  def float(self): return T(shape=self.shape,dtype=dtypes.float32,tag=self.tag)
  def __truediv__(self,scale): return self
  def clamp(self,*args): return self
  def cast(self,dtype): return T(shape=self.shape,dtype=dtype,tag=self.tag)
  def relu(self): return self
  def square(self): return self
  @staticmethod
  def empty(*shape,device='mock',dtype=None):
    if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
    return T(shape=shape,dtype=dtype)
  @staticmethod
  def custom_kernel(out,*args,**kwargs): T.gemms.append(args[1].tag);return (out,)


def main():
  assert not Device._opened_devices
  from tinygrad import Tensor
  a=Tensor.empty(1,2048,2048,device='NULL',dtype=dtypes.bfloat16)
  b=Tensor.empty(a.shape,device='NULL',dtype=dtypes.bfloat16)
  residual=a+b
  assert residual.uop.op is Ops.ADD and residual.uop.src==(a.uop,b.uop)
  old_children=residual.uop.src
  residual.replace(Tensor.empty(a.shape,device='NULL',dtype=dtypes.bfloat16))
  assert old_children==(a.uop,b.uop) and residual.uop is not a.uop
  install=load(HERE/'agent-residual-norm-install.py')
  frozen=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/current-profile-dynamic-2048/report-sources')
  core,norm,pq=[load(frozen/name) for name in ('model.py','fused-norm.py','quantize-packed.py')]
  pq.Tensor=T;pq.pack=lambda value:value
  norm.norm=lambda x,*args,**kwargs:T(shape=x.shape,dtype=x.dtype,tag='normalized')
  original_quantize=pq.quantize
  fixed={f'{install.PREFIX}{i}/{slot}':.125+i/512+slot/1024 for i in range(28) for slot in range(4)}

  class Model:
    dtype,fp8_wmma,cuda_kernels=dtypes.bfloat16,True,False
    config={'vlm_config':{'text_config':{'num_hidden_layers':28}}}
    def __init__(self):
      self.weights,self.weight_scales,self.events={},{},[]
      self.prefill_jit=SimpleNamespace(cnt=0,captured=None,reset=lambda:self.events.append(('reset',)))
      for i in range(28):
        prefix=install.PREFIX+str(i)
        self.weights[prefix+'.post_attention_layernorm.weight']=T(shape=(2048,))
        self.weights[prefix+'.input_layernorm.weight']=T(shape=(2048,))
        for suffix,n,k in (('self_attn.q_proj',2048,2048),('self_attn.k_proj',1024,2048),('self_attn.v_proj',1024,2048),
                           ('self_attn.o_proj',2048,2048),('mlp.fc1',9216,2048),('mlp.fc2',2048,9216)):
          name=prefix+'.'+suffix
          self.weights[name+'.weight']=T(shape=(n,k),dtype=dtypes.fp8e4m3,tag=name)
          self.weight_scales[name+'.weight']=T([.5],dtype=dtypes.float32)
      self.text_layer=MethodType(core.EdgeModel.text_layer,self)
      norm.install(self);pq.install(self,fixed_scales=fixed)
    def norm(self,x,name,**options): self.events.append(('fallback-norm',name));return x
    def quantize(self,x):self.events.append(('dynamic',));return x,None
    def linear(self,x,name,quantized_input=None,heads=None): raise AssertionError('Unexpected fallback linear')
    def rope(self,q,k,*args):self.events.append(('rope',));return q,k
    def attention(self,q,k,v,mask,causal):self.events.append(('attention',causal));return q

  def invoke(model,rows=2048,index=0):
    return model.text_layer(T(shape=(1,rows,2048)),None,None,install.PREFIX+str(index),None,None,None)
  baseline=Model();T.gemms=[]
  for index in range(28):invoke(baseline,index=index)
  base_gemms=T.gemms[:]
  model=Model();old_norm,old_layer=model.norm,model.text_layer
  class Kernel:
    calls=0
    @staticmethod
    def fused(a,b,w,s):
      assert a.shape==b.shape==(1,2048,2048) and w.shape==(2048,) and s.dtype==dtypes.float32
      Kernel.calls+=1
      return T(shape=a.shape,tag='fused-residual'),T(shape=(2048,2048),dtype=dtypes.fp8e4m3,tag='fused-packed')
  T.gemms=[]
  with patch('tinygrad.Tensor',T), install.installed(model,Kernel) as session:
    for index in range(28):invoke(model,index=index)
    assert session.events==[dict(prefix=install.PREFIX+str(index),slot_before=2,slot_after=3,rows=2048) for index in range(28)]
    assert session.state==dict(prefill=False,prefix=None,index=0)
    assert model.events==baseline.events and T.gemms==base_gemms
    count=Kernel.calls;invoke(model,128);assert Kernel.calls==count
  assert model.norm==old_norm and model.text_layer==old_layer and pq.quantize is original_quantize
  try:
    with patch('tinygrad.Tensor',T), install.installed(model,SimpleNamespace(fused=lambda *args:(_ for _ in ()).throw(RuntimeError('injected')))):
      invoke(model)
  except RuntimeError as error:assert str(error)=='injected'
  else:raise AssertionError('Expected failure')
  assert model.norm==old_norm and model.text_layer==old_layer and pq.quantize is original_quantize
  assert not Device._opened_devices
  print('PASS: actual frozen text_layer and installers; same six GEMMs/RoPE/attention; slot 2 -> 3; non-target bypass; finally restoration')


if __name__=='__main__':main()

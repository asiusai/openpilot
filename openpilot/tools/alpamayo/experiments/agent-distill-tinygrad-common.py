"""Private CPU-first differentiable block plumbing; never an optimizer or route trainer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

EXP=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-distill-tinygrad-artifacts'
BASE=Path('/tmp/opencode/alpamayo-asius-20261114')
TINY=BASE/'tinygrad'
PY=BASE/'upstream/.venv/bin/python'
REVISION='89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
SELECTED_PCI='0000:03:00.0'
TARGET='KFD:0+AMD:LLVM:gfx1100'
# Declared before any tinygrad numerical result. No fitting or route-dependent limits.
TOLERANCES={
  'forward':{'atol':3e-5,'rtol':5e-4,'relative_l2':2e-4},
  'gradient':{'atol':3e-7,'rtol':3e-3,'relative_l2':2e-4},
  'fc2_derivative':{'atol':1e-8,'rtol':2e-5,'relative_l2':2e-5}}

def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):
  h=hashlib.sha256()
  with Path(path).open('rb') as stream:
    for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
  return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
  sys.modules[name]=m;spec.loader.exec_module(m);return m
def native():return load(EXP/'agent-distill-feasibility-block.py','agent_distill_tinygrad_native')
def no_gpu_fds():
  for p in Path('/proc/self/fd').iterdir():
    try:target=str(p.readlink())
    except FileNotFoundError:continue
    assert not target.startswith(('/dev/dri/','/dev/kfd','/dev/nvidia','/dev/bus/usb/')),target
def compare(actual,expected,kind):
  import numpy as np
  assert actual.shape==expected.shape
  assert np.isfinite(actual).all() and np.isfinite(expected).all()
  a,b=actual.astype(np.float64),expected.astype(np.float64);d=np.abs(a-b);tol=TOLERANCES[kind]
  allowed=tol['atol']+tol['rtol']*np.abs(b)
  relative=float(np.linalg.norm((a-b).ravel())/max(float(np.linalg.norm(b.ravel())),1e-30))
  result={'max_absolute':float(d.max()),'relative_l2':relative,'max_scaled_error':float((d/allowed).max()),
          'finite':True,'passed':bool((d<=allowed).all() and relative<=tol['relative_l2'])}
  return result
def arrays():
  import numpy as np
  manifest=read(ROOT/'oracle.json');assert manifest['complete'] and manifest['optimizer_updates']==0
  for path,digest in manifest['source_sha256'].items():assert file_sha(path)==digest,path
  result={}
  for name,item in manifest['arrays'].items():
    path=ROOT/item['file'];assert file_sha(path)==item['sha256'],name
    result[name]=np.load(path,allow_pickle=False)
    assert list(result[name].shape)==item['shape'] and str(result[name].dtype)==item['dtype']
  assert manifest['tolerances']==TOLERANCES
  return manifest,result
def passive_topology():
  import os
  rows=[]
  for node in sorted(Path('/sys/devices/virtual/kfd/kfd/topology/nodes').iterdir(),key=lambda p:int(p.name)):
    gpu=int((node/'gpu_id').read_text())
    if not gpu:continue
    props={k:int(v) for k,v in (line.split() for line in (node/'properties').read_text().splitlines())}
    dev=Path(f'/sys/class/drm/renderD{props["drm_render_minor"]}/device')
    rows.append({'node':node.name,'gpu_id':gpu,'gfx_target_version':props['gfx_target_version'],
      'render_minor':props['drm_render_minor'],'pci':dev.resolve().name,'vendor':(dev/'vendor').read_text().strip(),
      'device':(dev/'device').read_text().strip(),'driver':(dev/'driver').resolve().name,
      'vram_total':int((dev/'mem_info_vram_total').read_text()),
      'render_rw_access':os.access(f'/dev/dri/renderD{props["drm_render_minor"]}',os.R_OK|os.W_OK)})
  return {'nodes':rows,'kfd_rw_access':os.access('/dev/kfd',os.R_OK|os.W_OK),'no_device_opened':True}


class Block:
  """Original native FP32 math expressed solely through differentiable Tensor operations."""
  def __init__(self,values,config,device):
    from tinygrad import Tensor
    self.weights={name[2:]:Tensor(value.copy(),device=device) for name,value in values.items() if name.startswith('w:')}
    self.eps=config['rms_norm_eps']
    assert config['hidden_size']==2048 and config['intermediate_size']==9216
    assert config['num_attention_heads']==16 and config['num_key_value_heads']==8 and config['head_dim']==128
    assert config['hidden_act']=='relu2' and not config['attention_bias'] and not config['mlp_bias']
    assert config['attention_dropout']==0 and self.eps==1e-5
    assert sum(w.numel() for w in self.weights.values())==50335744
  def linear(self,x,name):return x@self.weights[name+'.weight'].transpose(-1,-2)
  def norm(self,x,name):return x*(x.square().mean(-1,keepdim=True)+self.eps).rsqrt()*self.weights[name+'.weight']
  def __call__(self,x,cos,sin,causal=True):
    h=self.norm(x,'input_layernorm');b,l,_=x.shape
    q,k,v=[self.linear(h,'self_attn.'+n+'_proj').reshape(b,l,heads,128).transpose(1,2)
           for n,heads in (('q',16),('k',8),('v',8))]
    c,s=cos.unsqueeze(1),sin.unsqueeze(1)
    def rotate(t):return (-t[...,64:]).cat(t[...,:64],dim=-1)
    q,k=q*c+rotate(q)*s,k*c+rotate(k)*s
    attention=q.scaled_dot_product_attention(k,v,dropout_p=0.0,is_causal=causal,enable_gqa=True)
    residual=x+self.linear(attention.transpose(1,2).reshape(b,l,2048),'self_attn.o_proj')
    activation=self.linear(self.norm(residual,'post_attention_layernorm'),'mlp.fc1').relu().square()
    return (residual+self.linear(activation,'mlp.fc2'),k,v),activation


def recovery_loss(outputs,targets,valid_rows=3):
  y,k,v=outputs;ty,tk,tv=targets
  return (y[:,:valid_rows]-ty[:,:valid_rows].detach()).square().mean()+.1*(
    (k[:,:,:valid_rows]-tk[:,:,:valid_rows].detach()).square().mean()+
    (v[:,:,:valid_rows]-tv[:,:,:valid_rows].detach()).square().mean())

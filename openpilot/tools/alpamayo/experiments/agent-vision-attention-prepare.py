"""Offline program enumeration, complete-shape FP32 references, and layout/resource audit."""
import ast, hashlib, importlib.util, json, math, os, re, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-vision-attention-artifacts')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
os.environ.update(DEV='CPU',CPU='1',AMD='0',ALLOW_DEVICE_USAGE='0',PARALLEL='0',PROFILE='0',
  CACHEDB=str(ROOT/'compiler-cache.db'),TMPDIR=str(ROOT),BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',
  LLVMOPT='1',GMMU='0',TC_OPT='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
sys.path.insert(0,str(TINY));sys.path.insert(0,str(EXP))
C=__import__('agent-vision-attention-common')

def bf16(x):
  import numpy as np
  x=np.asarray(x,dtype=np.float32);u=x.view(np.uint32)
  return ((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
def float32(x):
  import numpy as np
  return (x.astype(np.uint32)<<16).view(np.float32)

def oracle(q,k,v):
  import numpy as np
  out=np.empty(q.shape,np.float32);lse=np.empty(q.shape[:-1],np.float32)
  for b in range(q.shape[0]):
    for h in range(16):
      scores=(q[b,h]@k[b,h].T)*np.float32(72**-.5)
      m=scores.max(axis=-1,keepdims=True);p=np.exp(scores-m,dtype=np.float32)
      denom=p.sum(-1,keepdims=True,dtype=np.float32)
      out[b,h]=(p/denom)@v[b,h];lse[b,h]=m[:,0]+np.log(denom[:,0])
  return out,lse

def producer_check(renderer):
  """Run actual vision_layer Q/K/V producer functions without weights or device access.

  Norm/quantizer inputs are symbolic. Only producer layout, not model arithmetic, is proved here.
  """
  from tinygrad import Tensor,dtypes,Device
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp
  sys.path.insert(0,str(C.CURRENT))
  model=C.load(C.CURRENT/'model.py')
  bf=C.load(C.CURRENT/'experiments/bf16-linear.py')
  fp=C.load(C.CURRENT/'experiments/tiled-gemm.py')
  def tensor(s,dt=dtypes.bfloat16): return Tensor(UOp.from_buffer(Buffer('AMD',math.prod(s)*dt.itemsize),dt)).reshape(s)
  old=type(Device).__getitem__
  def symbolic(devices,key):
    assert key in ('CPU','AMD'),key
    return SimpleNamespace(renderer=renderer)
  type(Device).__getitem__=symbolic
  rows=[]
  class Stop(Exception): pass
  try:
    for b in (1,2):
      for layer in (0,2):
        captured=[];m=model.EdgeModel.__new__(model.EdgeModel)
        m.dtype=dtypes.bfloat16;m.weight_scales={}
        if layer==2: m.weight_scales[f'vlm.model.visual.encoder.layers.{layer}.self_attn.q_proj.weight']=1
        m.norm=lambda x,*a,**k:x
        m.quantize=lambda x:(tensor((b*720,1152),dtypes.fp8e4m3),tensor((b*720,),dtypes.float32))
        def linear(x,name,quantized=None,heads=None):
          assert heads==16 and name.endswith(('_proj'))
          bias=tensor((1152,))
          if layer==0: result=bf.linear(x.reshape(b*720,1152),tensor((1,1,72,72,4,16,4)),1152,bias)
          else:
            q,s=quantized
            result=fp.gemm(q,tensor((1152,1152),dtypes.fp8e4m3),(s,tensor((1,),dtypes.float32)),row_scale=True,bias=bias)
          return result.reshape(b,720,1152)
        m.linear=linear
        def attention(q,k,v):
          captured.extend((q,k,v));raise Stop
        m.attention=attention
        try:m.vision_layer(tensor((b,720,1152)),layer)
        except Stop:pass
        assert len(captured)==3 and all(x.shape==C.shape(b) for x in captured)
        assert all(x.uop.op.name=='PERMUTE' and x.uop.marg==(0,2,1,3) for x in captured)
        rows.append({'batch':b,'layer':layer,'producer':'packed_BF16_plus_bias' if layer==0 else 'row_major_FP8_GEMM',
          'logical_shape':list(captured[0].shape),'producer_shape':[b,720,16,72],
          'logical_element_strides':[829440,72,1152,1],
          'permute':list(captured[0].uop.marg),'qkv_uop_keys':[x.uop.key.hex() for x in captured],
          'boundary_B':'BF16 values in BSHD storage; first-two-layer bias may fuse with materialization in the live model',
          'gpu_execution':False})
  finally: type(Device).__getitem__=old
  return rows

def main():
  import numpy as np
  from tinygrad import Tensor,dtypes,Device
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.codegen import to_program
  I=C.load(EXP/'agent-ck-source-identity.py')
  assert not Device._opened_devices
  def blocked(*a,**k):raise AssertionError('Device open forbidden during offline preparation')
  type(Device).__getitem__=blocked
  target=ROOT/'prepared';target.mkdir(exist_ok=True);assert not (target/'protocol.json').exists()
  online=C.load(C.CURRENT/'experiments/online-attention.py')
  source,blob=online.compiled(16,16,720,720,False,waves=8,width=72,block_keys=32,stage_kv=True,
    key_padding=4,kv_load=1,value_layout=False,probability_mma=True)
  baseline=ROOT/'baseline';baseline.mkdir(exist_ok=True)
  (baseline/'source.cpp').write_text(source);(baseline/'kernel.hsaco').write_bytes(blob)
  compiler=C.load(EXP/'agent-vision-attention-compile.py')
  action=compiler.reviewed_function('agent-ck-attention-v2-inspect.py','comgr_action');g=compiler.comgr
  (baseline/'disassembly.txt').write_bytes(action(blob,g.AMD_COMGR_DATA_KIND_EXECUTABLE,g.AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE,g.AMD_COMGR_DATA_KIND_SOURCE,[]))
  (baseline/'readelf.txt').write_text(subprocess.check_output(['llvm-readelf-18','--all',str(baseline/'kernel.hsaco')],text=True))
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  producers=producer_check(renderer)
  def tensor(s,dt=dtypes.bfloat16):return Tensor(UOp.from_buffer(Buffer('AMD',math.prod(s)*dt.itemsize),dt)).reshape(s)
  associations,programs,graphs={},{},{}
  for b in (1,2):
    for boundary in ('A','B'):
      for arm in ('baseline','candidate'):
        s=C.shape(b) if boundary=='A' else (b,720,16,72)
        inputs=[tensor(s) for _ in range(3)]+[tensor((math.prod(C.shape(b))+512,)),tensor((b*16*720+512,),dtypes.float32)]
        with C.abi_scope(): result=C.outputs(arm,boundary,*inputs)
        raw,variables=result[0].linear_with_vars(*result[1:]);assert not variables
        calls=[]
        for call in raw.src:
          p=to_program(call.body,renderer);item=I.identity(p);key=I.key(item)
          associations[key]=item;digest=item['binary_sha256']
          programs.setdefault(digest,{'associations':[]})
          if key not in programs[digest]['associations']:programs[digest]['associations'].append(key)
          (target/(digest+'.elf')).write_bytes(p.src[3].arg)
          calls.append((call.replace(src=(p,*call.src[1:])),'AMD','COMPUTE:0'))
        rows=I.batch_rows(calls);native=[r for r in rows if associations[r['association']]['name'] in (C.NAME,'alpamayo_online')]
        assert len(native)==1 and native[0]['arguments'][0]['offset_bytes']==512
        if arm=='candidate': assert native[0]['arguments'][1]['offset_bytes']==1024
        graphs[f'{b}_{boundary}_{arm}']=rows
  cases={};rng=np.random.default_rng(172011)
  base={k:bf16(rng.normal(size=C.shape(2)).astype(np.float32)) for k in ('q','k','v')}
  variants={'random':base}
  for key in ('q','k','v'):
    values={k:x.copy() for k,x in base.items()}
    # Last true key/row/dimension and last head/batch exercise every tail boundary.
    a=float32(values[key]);a[1,15,719,64:72]+=np.float32(6 if key!='v' else 32)
    values[key]=bf16(a);variants[key+'_tail']=values
  variants['high_logits']={k:bf16(float32(v)*(3 if k in ('q','k') else 1)) for k,v in base.items()}
  variants['uniform']={k:(np.zeros_like(v) if k in ('q','k') else v) for k,v in base.items()}
  cpu=[]
  for name,values in variants.items():
    q,k,v=(float32(values[k]) for k in ('q','k','v'));o,l=oracle(q,k,v)
    case={**values,'o':o,'lse':l};files={}
    for key,x in case.items():
      p=target/f'{name}-{key}.npy';np.save(p,x,allow_pickle=False);files[key]={'path':str(p),'sha256':C.sha(p)}
    cases[name]=files
    assert np.isfinite(o).all() and np.isfinite(l).all()
    cpu.append({'fixture':name,'shape':list(q.shape),'reference':'independent full FP32 softmax, no probability BF16 rounding','finite':True})
  # Label-exact logical BSHD -> contiguous BHSD -> BSHD conversion at all elements.
  labels=np.arange(2*720*16*72,dtype=np.uint32).reshape(2,720,16,72)
  packed=np.ascontiguousarray(labels.transpose(0,2,1,3))
  assert np.array_equal(packed.transpose(0,2,1,3),labels)
  for b in (1,2):
    seen=set()
    for z in range(b):
      for y in range(12):
        for x in range(16):
          linear=x+16*(y+12*z);batch=linear//192;head=(linear%192)//12;tile=linear%12
          seen.update((batch,head,r) for r in range(tile*64,min((tile+1)*64,720)))
    assert len(seen)==b*16*720
  resources={}
  for arm in ('baseline','candidate'):
    text=(ROOT/arm/'readelf.txt').read_text();asm=(ROOT/arm/'disassembly.txt').read_text()
    resources[arm]={k:int(re.search(r'\.'+k+r':\s*(\d+)',text)[1]) for k in
      ('vgpr_count','sgpr_count','vgpr_spill_count','sgpr_spill_count','group_segment_fixed_size','private_segment_fixed_size','workgroup_processor_mode')}
    resources[arm].update(static_instructions=sum(bool(re.match(r'\s+[sv]_[a-z]',l)) for l in asm.splitlines()),
      scratch_instructions=sum('scratch_' in l for l in asm.splitlines()),binary_sha256=C.sha(ROOT/arm/'kernel.hsaco'))
  proof={'gpu_executed':False,'opened_devices':list(Device._opened_devices),'producers':producers,'graphs':graphs,
    'references':cpu,'label_layout_all_elements':True,'dispatch_all_rows_once':True,'resources':resources}
  (target/'cpu-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
  limits={'rmse':{'relative':1.1,'floor':2e-5},'max_abs':{'relative':1.25,'floor':.002},
    'per_head_rmse':{'relative':1.2,'floor':3e-5},'lse_rmse':.001,'lse_max_abs':.005,
    'mutation_min_delta':1e-4,'speed_median_ratio':.95,'speed_p95_ratio':1.05}
  paths=[EXP/(n+'.py') for n in ('agent-vision-attention-common','agent-vision-attention-prepare','agent-vision-attention-run')]
  paths+=[EXP/'agent-vision-attention-native.cpp',EXP/'agent-ck-source-identity.py',EXP/'agent-ck-attention-v3-run.py',
    C.CURRENT/'model.py',C.CURRENT/'experiments/online-attention.py',C.CURRENT/'experiments/batched-vision.py',
    C.CURRENT/'experiments/bf16-linear.py',C.CURRENT/'experiments/tiled-gemm.py',
    ROOT/'candidate/kernel.hsaco',ROOT/'candidate/source.cpp',ROOT/'baseline/kernel.hsaco',ROOT/'baseline/source.cpp',target/'cpu-proof.json']
  plan={'classification':'Numerical vision attention variant; no model or route qualification','gpu_executed':False,
    'timeout_seconds':300,'tinygrad_commit':'89846dcfb383a98ee01cfec3bba440ae7dcf9b1e',
    'source_sha256':{str(p):C.sha(p) for p in paths},'associations':associations,'programs':programs,'fixtures':cases,'limits':limits,
    'replay_order':['random','v_tail','k_tail','q_tail']*3,
    'boundaries':{'A':'resident contiguous BHSD -> BHSD output; candidate also writes explicit LSE',
      'B':'actual FP8 producer BSHD layout -> Q/K/V BHSD materialization -> attention -> contiguous BSHD output; candidate LSE included'},
    'timing':{'blocks':6,'replays_per_block':20,'fence':'one final device synchronize per block','diagnostics_in_timing':False},
    'resources':resources}
  (target/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
  print(json.dumps({'prepared':str(target),'programs':len(programs),'associations':len(associations),'resources':resources},indent=2))

if __name__=='__main__':main()

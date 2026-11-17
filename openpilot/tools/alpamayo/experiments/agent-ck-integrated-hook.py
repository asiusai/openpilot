"""All-28 instance hook. Original text_layer, RoPE, cache stores, and decoder remain authoritative."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import time

G,OC,LC=256,4194304,32768

def audit(store,returned):
  """Whole-output counts, including guarded and returned storage. Zero means success."""
  from tinygrad import Tensor,dtypes
  dt=dtypes.uint16 if store.dtype.itemsize==2 else dtypes.uint32
  mask,poison,guard=(0x7f80,0x7fc1,0x5aa5) if dt==dtypes.uint16 else (0x7f800000,0x7fc12345,0x5aa55aa5)
  a=store.bitcast(dt);b=returned.bitcast(dt).flatten();body=a[G:-G]
  return Tensor.stack(*[x.cast(dtypes.int32).sum() for x in (
    (body&mask)==mask,body==poison,a[:G]!=guard,a[-G:]!=guard,
    body!=b,(b&mask)==mask,b==poison)]).contiguous().realize()

def poison(common,store):
  # Existing qualified integer fill; repeats inside the language TinyJit on every replay.
  common.load('agent-ck-qualified-common').utility(store,mode='poison').realize()

@contextmanager
def native_scope(common,report):
  """Keep one adapter scope through capture, replay, and the final fence. No renderer replacement."""
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  from tinygrad.engine.realize import get_call_kernel_args,get_call_outs_ins
  from tinygrad.engine import realize
  from tinygrad.uop.ops import Ops
  helper=common.load('agent-ck-output-common');adapter=helper.load('agent-ck-attention-v2-adapter')
  identity=common.load('agent-ck-source-identity')
  expected=json.loads((common.PREPARED/'ck-identity.json').read_text())
  def check(p):
    if p.op is not Ops.PROGRAM: return False
    name=p.src[0].arg.name
    same_blob=hashlib.sha256(p.src[3].arg).hexdigest()==adapter.ELF_SHA
    if name!=adapter.NAME and not same_blob: return False
    assert identity.identity(p)==expected,'Changed CK source/AST/ELF/target/ABI association'
    return True
  with adapter.abi_scope():
    old_build,old_args,old_queues=ops_amd.amd_build_program,ops_amd.AMDComputeQueue.kernargs,hcq2._build_queues
    old_compile=realize.lower_and_compile
    def observe(linear,verbose=True):
      compiled=old_compile(linear,verbose=verbose)
      for call in compiled.toposort():
        if call.op is not Ops.CALL or not check(call.body): continue
        rows=[]
        for u in call.src[1:]:
          b=u.buffer.ensure_allocated();address=int(b.get_storage().buf);base=int(b.base.get_storage().buf)
          rows.append({'gpu_address':address,'base_gpu_address':base,'offset_bytes':b.offset,
            'bytes':b.nbytes,'base_bytes':b.base.nbytes})
        common.storage_gate(rows,[OC*2,LC*4,OC*2,OC,OC])
        assert [r['offset_bytes'] for r in rows[:2]]==[512,1024]
        report.setdefault('raw_native_pointers',[]).append(rows)
      return compiled
    def build(dev,p,devices):
      if check(p):
        assert dev.arch=='gfx1200' and dev.is_usb and tuple(dev.target)==(12,0,0)
        descriptor,_=ops_amd._amd_program_image(dev,p.src[3].arg)
        assert asdict(descriptor)==expected['descriptor']
      return old_build(dev,p,devices)
    def args(queue,call,p,data):
      if check(p):
        assert adapter.ACTIVE and asdict(data)==expected['descriptor']
        assert [x for x,_ in hcq2.layout_args(get_call_kernel_args(call,p,queue.devs))]==[0,8,16,24,32]
        views=[helper.view_info(u) for u in call.src[1:]]
        assert [x['bytes'] for x in views]==[OC*2,LC*4,OC*2,OC,OC]
        assert [x['offset_bytes'] for x in views[:2]]==[G*2,G*4]
        report.setdefault('native_argument_templates',{}).setdefault(identity.key(views),views)
        report['native_kernarg_builds']=report.get('native_kernarg_builds',0)+1
      return old_args(queue,call,p,data)
    def queues(ctx):
      tracker=hcq2.DepsTracker([(d,q) for _,d,q in ctx.batch]);native=[]
      for tag,(call,device,queue) in enumerate(ctx.batch):
        outs,ins=get_call_outs_ins(call)
        deps=tracker.access_resources([(u,i in outs) for i,u in enumerate(call.src[1:]) if not u.is_variable],tag)
        if check(call.body):
          assert tuple(outs)==(0,1) and tuple(ins)==(2,3,4)
          native.append({'tag':tag,'device':device,'queue':queue,'outs':list(outs),'ins':list(ins),
            'dependencies':deps,'arguments':[helper.view_info(u) for u in call.src[1:]]})
      if native: report.setdefault('native_dependency_batches',[]).append(native)
      return old_queues(ctx)
    ops_amd.amd_build_program,ops_amd.AMDComputeQueue.kernargs,hcq2._build_queues=build,args,queues
    realize.lower_and_compile=observe
    try: yield adapter
    finally:
      assert ops_amd.amd_build_program is build and ops_amd.AMDComputeQueue.kernargs is args and hcq2._build_queues is queues
      ops_amd.amd_build_program,ops_amd.AMDComputeQueue.kernargs,hcq2._build_queues=old_build,old_args,old_queues
      assert realize.lower_and_compile is observe
      realize.lower_and_compile=old_compile

class Integration:
  def __init__(self,common,model,decoder,arm,report):
    from tinygrad import Device
    self.common,self.model,self.decoder,self.arm,self.report=common,model,decoder,arm,report
    self.helper=common.load('agent-ck-output-common');self.io=common.load('agent-ck-attention-v3-run')
    self.device=Device['AMD'];self.renderer=self.device.renderer;self.compiler=self.renderer.compiler
    self.layers={};self.pending={};self.last_args=None;self.closed=False;self.control_done=False
    self.started=time.monotonic()
    self.hook=common.load('agent-ck-model-hook').Hook(model,self.attention)
    self.original_jit=model.padded_prefill.jit
    owner=self
    class ObservedJit:
      def __getattr__(self,name): return getattr(owner.original_jit,name)
      def reset(self): owner.original_jit.reset()
      def __call__(self,*args,**kwargs):
        assert not kwargs
        before=owner.original_jit.cnt;start=len(owner.hook.calls)
        ret=owner.original_jit(*args)
        if before<2: assert owner.hook.calls[start:]==list(range(28)),'Incomplete selected prefill coverage'
        else: assert len(owner.hook.calls)==start,'Python hook unexpectedly ran during replay'
        owner.last_args=args
        screen=sum(p['phase'].startswith('screen_') for p in owner.report.setdefault('prefills',[]))
        owner.after_prefill(f'screen_{screen}_original64_{common.INDICES[screen]}',before)
        if owner.original_jit.captured is not None and not owner.control_done:
          if arm=='candidate': common.load('agent-ck-integrated-controls').run(owner,ret)
          owner.control_done=True
        return ret
    self.proxy=ObservedJit();model.padded_prefill.jit=self.proxy

  def attention(self,hook,context,q,k,v,original):
    from tinygrad import Tensor,dtypes
    from tinygrad.uop.ops import Ops
    index=context['layer'];cache=context['cache']
    # Preserve the actual producer AFTER/STORE chain before materialization. Never reconstruct RoPE or the KV cache.
    ancestors=k.uop.toposort()|v.uop.toposort()
    assert cache.uop.base in ancestors and any(u.op is Ops.STORE for u in ancestors),'Lost live cache update dependency'
    logical={'layer':index,'q_shape':list(q.shape),'k_shape':list(k.shape),'v_shape':list(v.shape),
      'q_key':q.uop.key.hex(),'k_key':k.uop.key.hex(),'v_key':v.uop.key.hex(),'cache_key':cache.uop.key.hex(),
      'cache_shape':list(cache.shape),'cache_store_dependencies':[u.key.hex() for u in ancestors if u.op is Ops.STORE],
      'movement':[{ 'op':u.op.name,'shape':list(u.shape),'arg':str(u.arg)} for u in (q.uop,k.uop,v.uop)]}
    if self.arm=='candidate':
      packed=tuple(x.contiguous() for x in (q,k,v));packed[0].realize(*packed[1:])
      if index not in self.layers:
        self.layers[index]={'stores':(self.io.allocate('AMD',(OC+2*G,),dtypes.bfloat16),
          self.io.allocate('AMD',(LC+2*G,),dtypes.float32))}
      row=self.layers[index];stores=row['stores']
      for store in stores: poison(self.common,store)
      out,lse=self.helper.outputs('candidate',True,*packed,*stores)
      out.realize(lse)
      prefix=Tensor.stack(*[(p.bitcast(dtypes.uint16)!=cache[j,:,:,:2048].bitcast(dtypes.uint16)).cast(dtypes.int32).sum()
        for j,p in enumerate(packed[1:])]).realize()
      row.update(out=out,lse=lse,packed=packed,flags=(audit(stores[0],out),audit(stores[1],lse),prefix))
    else:
      out=original(q,k,v,None,True).realize()
      self.layers[index]={'out':out,'flags':((~out.isfinite()).cast(dtypes.int32).sum().reshape(1).realize(),)}
    self.pending[index]=logical
    return out

  def download(self,tensor):
    import numpy as np
    tensor=tensor.contiguous().realize()
    self.device.synchronize();data=self.io.read(tensor,self.device)
    self.report['readback_bytes']=self.report.get('readback_bytes',0)+len(data)
    return np.frombuffer(data,{2:np.uint16,4:np.uint32}[tensor.dtype.itemsize]).reshape(tensor.shape).copy()

  def after_prefill(self,phase,before):
    from tinygrad import Tensor
    self.report['phase']=phase
    self.device.synchronize()
    assert self.device.renderer is self.renderer and self.renderer.compiler is self.compiler
    assert set(self.layers)==set(range(28))
    flags=Tensor.cat(*[f.flatten() for i in range(28) for f in self.layers[i]['flags']]).contiguous().realize()
    counts=self.download(flags).reshape(28,-1)
    assert not counts.any(),('Per-layer finite/poison/redzone/binding audit',phase,counts.tolist())
    if self.pending:
      assert set(self.pending)==set(range(28))
      for i,row in sorted(self.pending.items()):
        if self.arm=='candidate':
          stores=self.layers[i]['stores']
          from tinygrad.uop.ops import UOp
          tensors=[Tensor(UOp.from_buffer(s.uop.buffer.view(n*s.dtype.itemsize,G*s.dtype.itemsize).ensure_allocated(),s.dtype)).reshape(shape)
            for s,n,shape in zip(stores,(OC,LC),(self.helper.QSHAPE,self.helper.LSHAPE),strict=True)]+list(self.layers[i]['packed'])
          physical=[self.helper.storage_info(x,True) for x in tensors]
          self.common.storage_gate(physical,[OC*2,LC*4,OC*2,OC,OC])
          for item,store,offset in zip(physical[:2],stores,(512,1024),strict=True):
            assert item['gpu_address']==self.helper.storage_info(store,True)['gpu_address']+offset
          row['actual_arguments']=physical
          row['returned_allocations']=[self.helper.storage_info(self.layers[i][x],True) for x in ('out','lse')]
          addresses=[r['gpu_address'] for r in physical]
          assert any([r['gpu_address'] for r in xs]==addresses for xs in self.report['raw_native_pointers']), 'Unobserved CK physical arguments'
        self.report.setdefault('live_layouts',[]).append(row)
      self.pending.clear()
    captured=self.original_jit.captured
    native=0
    if captured is not None:
      from tinygrad.runtime.support.hcq2 import HCQInfo
      from tinygrad.uop.ops import Ops
      for call in captured._linear.src:
        if isinstance(call.arg.aux,HCQInfo): native+=sum(k[1]=='alpamayo_ck_d128' for k in call.arg.aux.kernels)
        elif call.body.op is Ops.PROGRAM: native+=call.body.src[0].arg.name=='alpamayo_ck_d128'
      assert native==(28 if self.arm=='candidate' else 0),('Captured CK coverage',native)
    self.report.setdefault('prefills',[]).append({'phase':phase,'jit_count_before':before,'jit_count_after':self.original_jit.cnt,
      'captured_native_calls':native,'all28_audit_counts':counts.tolist(),'elapsed_seconds':time.monotonic()-self.started,
      'audit_columns':'O seven counts, LSE seven counts, packed K/V versus actual cache prefix two exact counts' if self.arm=='candidate' else 'O nonfinite count',
      'captured_key':captured._linear.key.hex() if captured else None})

  def close(self):
    assert not self.closed and self.model.padded_prefill.jit is self.proxy
    self.device.synchronize()
    self.model.padded_prefill.jit=self.original_jit
    # Invalidate the graph that contains optional native calls before releasing their owners and ABI scope.
    self.original_jit.reset();self.hook.close();self.layers.clear();self.pending.clear();self.last_args=None
    assert self.device.renderer is self.renderer and self.renderer.compiler is self.compiler
    self.closed=True

"""Parent-only vision microtest. Default mode verifies files without opening a device."""
import argparse, hashlib, importlib, json, math, os, subprocess, sys, time, traceback
from contextlib import contextmanager
from functools import partial
from pathlib import Path

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-vision-attention-artifacts')
PROTOCOL=ROOT/'protocol-final-v2.json'
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
sys.path.insert(0,str(EXP))
C=importlib.import_module('agent-vision-attention-common')

def verify():
  assert not sys.flags.optimize
  p=json.loads(PROTOCOL.read_text())
  for path,h in p['source_sha256'].items():assert C.sha(path)==h,('changed source',path)
  assert subprocess.check_output(['git','-C',str(TINY),'rev-parse','HEAD'],text=True).strip()==p['tinygrad_commit']
  for fixture in p['fixtures'].values():
    for item in fixture.values():assert C.sha(item['path'])==item['sha256']
  assert p['timeout_seconds']==300
  return p

@contextmanager
def bindings(plan,report):
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import UOp
  from tinygrad import dtypes
  I=C.load(EXP/'agent-ck-source-identity.py')
  build,queues,packet=ops_amd.amd_build_program,hcq2._build_queues,ops_amd.dispatch_packet
  def checked(dev,p,devices):
    key,item=I.authorize(p,plan)
    report['program_associations'][key]=item
    return build(dev,p,devices)
  def inspected(ctx):
    for call,_,_ in ctx.batch:I.authorize(call.body,plan)
    rows=I.batch_rows(ctx.batch)
    for row in rows:
      item=plan['associations'][row['association']]
      if item['name'] not in (C.NAME,'alpamayo_online'):continue
      assert row['arguments'][0]['offset_bytes']==512
      if item['name']==C.NAME:assert row['arguments'][1]['offset_bytes']==1024 and row['outs']==[0,1]
    report['hcq_batches'].append({'phase':report['phase'],'rows':rows})
    return queues(ctx)
  def typed(*a,**k):return [UOp.const(x,dtypes.uint32) if isinstance(x,int) else x for x in packet(*a,**k)]
  ops_amd.amd_build_program,hcq2._build_queues,ops_amd.dispatch_packet=checked,inspected,typed
  try:yield
  finally:
    assert ops_amd.amd_build_program is checked and hcq2._build_queues is inspected and ops_amd.dispatch_packet is typed
    ops_amd.amd_build_program,hcq2._build_queues,ops_amd.dispatch_packet=build,queues,packet

def storage(t):
  b=t.uop.buffer;ptr=int(b.get_storage().buf);base=int(b.base.get_storage().buf)
  assert ptr==base+b.offset and 0<=b.offset and b.offset+b.nbytes<=b.base.nbytes
  assert ptr%16==0
  return {'shape':list(t.shape),'dtype':str(t.dtype),'buffer':id(b),'base':id(b.base),
    'pointer':ptr,'base_pointer':base,'offset':b.offset,'bytes':b.nbytes,'base_bytes':b.base.nbytes}

def check_words(raw,returned,batch,kind,boundary):
  import numpy as np
  dt=np.uint16 if kind=='O' else np.uint32
  sentinel=0x7fc1 if kind=='O' else 0x7fc12345
  count=math.prod(C.shape(batch)) if kind=='O' else batch*16*720
  whole=np.frombuffer(raw,dt);got=np.frombuffer(returned,dt)
  assert whole.size==count+512 and got.size==count
  assert np.all(whole[:256]==sentinel) and np.all(whole[-256:]==sentinel),'redzone modified'
  interior=whole[256:-256]
  if kind=='O' and boundary=='B':got=got.reshape(batch,720,16,72).transpose(0,2,1,3).copy().reshape(-1)
  assert np.array_equal(interior,got),'returned output differs from guarded destination'
  assert not np.any(interior==sentinel),'poison survived'
  floats=((interior.astype(np.uint32)<<16).view(np.float32) if kind=='O' else interior.view(np.float32))
  assert np.isfinite(floats).all(),'nonfinite output'
  return interior.copy(),floats.reshape(C.shape(batch) if kind=='O' else (batch,16,720))

def run(plan,output):
  import numpy as np
  from tinygrad import Device,TinyJit,dtypes
  from tinygrad.device import Compiled
  io=C.load(EXP/'agent-ck-attention-v3-run.py')
  report={'completed':False,'gpu_executed':False,'model_integrated':False,'classification':plan['classification'],
    'protocol_sha256':C.sha(PROTOCOL),'phase':'opening','fixtures':[],'replays':[],'timings':{},
    'program_associations':{},'hcq_batches':[],'native_abi':[],'storage':[],
    'io':{'write_bytes':0,'write_calls':0,'read_bytes':0,'read_calls':0,'poison_bytes':0},'phase_seconds':{}}
  started=time.monotonic()
  def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  def phase(name):report['phase']=name;save();return time.monotonic()
  def write(t,data,poison=False):
    assert len(data)==t.uop.buffer.nbytes and len(data)%4==0
    io.write(t,data);report['io']['write_bytes']+=len(data);report['io']['write_calls']+=1
    if poison:report['io']['poison_bytes']+=len(data)
  def read(t):
    assert t.uop.buffer.offset%4==0 and t.numel()*t.dtype.itemsize%4==0
    data=io.read(t,dev);report['io']['read_bytes']+=len(data);report['io']['read_calls']+=1;return data
  cases={n:{k:np.load(v['path'],allow_pickle=False) for k,v in f.items()} for n,f in plan['fixtures'].items()}
  states={};first={}
  try:
    Compiled.wait_timeout_ms=10000
    dev=Device[Device.DEFAULT]
    assert dev.arch=='gfx1200' and dev.is_usb and not dev.prof_enabled
    report['gpu_executed']=True
    original_renderer=dev.renderer
    with bindings(plan,report),C.abi_scope(report['native_abi']):
      def state(b,boundary,arm):
        key=(b,boundary,arm)
        if key not in states:
          s=C.shape(b) if boundary=='A' else (b,720,16,72)
          tensors=[io.allocate(dev.device,s,dtypes.bfloat16) for _ in range(3)]
          tensors += [io.allocate(dev.device,(math.prod(C.shape(b))+512,),dtypes.bfloat16),
                      io.allocate(dev.device,(b*16*720+512,),dtypes.float32)]
          states[key]={'tensors':tensors,'jit':TinyJit(partial(C.execute,arm,boundary)),'uploaded':[None]*3}
          info=[storage(t) for t in tensors]
          for i,a in enumerate(info):
            for other in info[i+1:]:
              assert a['pointer']+a['bytes']<=other['pointer'] or other['pointer']+other['bytes']<=a['pointer'], 'overlapping native storage'
          report['storage'].append({'state':list(key),'tensors':info})
        return states[key]
      def upload(st,b,boundary,fixture):
        for i,k in enumerate(('q','k','v')):
          a=cases[fixture][k][:b]
          if boundary=='B':a=a.transpose(0,2,1,3).copy()
          payload=a.tobytes()
          # Exact content comparison; unchanged arrays avoid redundant host uploads.
          if st['uploaded'][i]!=payload:write(st['tensors'][i],payload);st['uploaded'][i]=payload
      def snapshot(b,boundary,arm,fixture,tag):
        tick=phase(tag);st=state(b,boundary,arm);dev.synchronize();upload(st,b,boundary,fixture)
        os_,ls=st['tensors'][3:]
        write(os_,np.full(os_.numel(),0x7fc1,np.uint16).tobytes(),True)
        if arm=='candidate':write(ls,np.full(ls.numel(),0x7fc12345,np.uint32).tobytes(),True)
        result=st['jit'](*st['tensors']);dev.synchronize()
        obits,o=check_words(read(os_),read(result[0]),b,'O',boundary)
        row={'batch':b,'boundary':boundary,'arm':arm,'fixture':fixture,'phase':tag,'output':C.metrics(o,cases[fixture]['o'][:b])}
        lbits=None
        if arm=='candidate':
          lbits,lse=check_words(read(ls),read(result[1]),b,'LSE',boundary)
          d=lse.astype(np.float64)-cases[fixture]['lse'][:b].astype(np.float64)
          row['lse']={'rmse':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.max(np.abs(d)))}
          assert row['lse']['rmse']<=plan['limits']['lse_rmse'] and row['lse']['max_abs']<=plan['limits']['lse_max_abs']
        if boundary=='B':
          resident=first[(b,'A',arm,fixture)]
          assert np.array_equal(obits,resident[0]) and (lbits is None or np.array_equal(lbits,resident[1])), 'layout boundary changed arithmetic or reused stale input'
        row.update(all_finite=True,poison_remaining=0,redzone_mismatches=0,guarded_returned_mismatches=0)
        fk=(b,boundary,arm,fixture)
        if fk in first:
          assert np.array_equal(obits,first[fk][0]) and (lbits is None or np.array_equal(lbits,first[fk][1])),('stale/nondeterministic replay',tag)
          row['exact_first_result']=True
        else:first[fk]=(obits,lbits,o.copy())
        report['phase_seconds'][tag]=time.monotonic()-tick
        return row
      for b,fixture in [(1,'random')]+[(2,n) for n in plan['fixtures']]:
        pair={}
        for arm in ('baseline','candidate'):pair[arm]=snapshot(b,'A',arm,fixture,f'fixture_{b}_{fixture}_{arm}')
        C.numeric_gate(pair['candidate']['output'],pair['baseline']['output'],plan['limits'])
        report['fixtures'].append(pair);save()
      # All mutation outputs use complete independent FP32 references before these structural checks.
      for arm in ('baseline','candidate'):
        base=first[(2,'A',arm,'random')]
        for field in ('q','k','v'):
          modified=first[(2,'A',arm,field+'_tail')];delta=np.abs(base[2]-modified[2])
          assert np.array_equal(base[0].reshape(C.shape(2))[:1],modified[0].reshape(C.shape(2))[:1])
          assert np.array_equal(base[0].reshape(C.shape(2))[1,:15],modified[0].reshape(C.shape(2))[1,:15])
          assert delta[1,15].max()>plan['limits']['mutation_min_delta']
          if field=='q':assert np.array_equal(base[0].reshape(C.shape(2))[1,15,:719],modified[0].reshape(C.shape(2))[1,15,:719])
          else:assert delta[1,15,:719].max()>plan['limits']['mutation_min_delta'],'last key must affect earlier rows: NONCAUSAL'
          if field=='v' and arm=='candidate':assert np.array_equal(base[1],modified[1]),'V must not alter LSE'
      report['mutations_noncausal_heads_batches_passed']=True
      for n,fixture in enumerate(plan['replay_order']):
        report['replays'].append(snapshot(2,'A','candidate',fixture,f'ordered_mutable_{n}_{fixture}'));save()
      for arm in ('baseline','candidate'):
        row=snapshot(2,'B',arm,'random',f'layout_B_{arm}_first')
        a=first[(2,'A',arm,'random')];b=first[(2,'B',arm,'random')]
        assert np.array_equal(a[0],b[0]) and (a[1] is None or np.array_equal(a[1],b[1]))
        snapshot(2,'B',arm,'random',f'layout_B_{arm}_capture')
        snapshot(2,'A',arm,'random',f'layout_A_{arm}_restored')
      report['numerical_micro_passed']=True
      # Replays are actual serialized TinyJit calls. All packing/conversion/LSE work remains in each graph.
      for boundary in ('A','B'):
        report['timings'][boundary]={arm:[] for arm in ('baseline','candidate')}
        for block in range(plan['timing']['blocks']):
          arms=('baseline','candidate') if block%2==0 else ('candidate','baseline')
          for arm in arms:
            phase(f'timing_{boundary}_{block}_{arm}');st=state(2,boundary,arm)
            dev.synchronize();tick=time.perf_counter()
            for _ in range(plan['timing']['replays_per_block']):st['jit'](*st['tensors'])
            dev.synchronize();elapsed=(time.perf_counter()-tick)*1000/plan['timing']['replays_per_block']
            report['timings'][boundary][arm].append(elapsed)
        for arm in ('baseline','candidate'):
          snapshot(2,boundary,arm,'v_tail',f'after_timing_{boundary}_{arm}_mutation')
          snapshot(2,boundary,arm,'random',f'after_timing_{boundary}_{arm}_restore')
      report['speed']={}
      for boundary,arms in report['timings'].items():
        med={a:float(np.median(v)) for a,v in arms.items()};p95={a:float(np.percentile(v,95)) for a,v in arms.items()}
        report['speed'][boundary]={'median_ms':med,'p95_block_ms':p95,'median_ratio':med['candidate']/med['baseline'],
          'p95_ratio':p95['candidate']/p95['baseline']}
      report['speed_screen_passed']=all(v['median_ratio']<=plan['limits']['speed_median_ratio'] and v['p95_ratio']<=plan['limits']['speed_p95_ratio'] for v in report['speed'].values())
      dev.synchronize();assert dev.renderer is original_renderer
    report['hooks_restored']=not C.ACTIVE;report['completed']=True
  except BaseException as e:
    report['failure']={'phase':report['phase'],'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()};raise
  finally:
    report['elapsed_seconds']=time.monotonic()-started;report['adapter_active_after']=C.ACTIVE;save()

def main():
  p=argparse.ArgumentParser();p.add_argument('--parent-run',action='store_true');p.add_argument('--output',type=Path)
  args=p.parse_args();plan=verify()
  if not args.parent_run:print(json.dumps({'gpu_executed':False,'preflight':'passed','protocol_sha256':C.sha(PROTOCOL)}));return
  assert args.output and args.output.parent.parent==ROOT.parent and args.output.parent.name.startswith('agent-vision-attention-parent-')
  assert not args.output.exists()
  os.environ.update(DEV='USB+AMD:LLVM',CPU='0',AMD='0',ALLOW_DEVICE_USAGE='1',PARALLEL='0',PROFILE='0',
    CACHEDB=str(ROOT/'compiler-cache.db'),TMPDIR=str(ROOT),BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',LLVMOPT='1',GMMU='0',TC_OPT='2',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XDG_CACHE_HOME=str(ROOT/'runtime-cache'))
  sys.path.insert(0,str(TINY));run(plan,args.output)

if __name__=='__main__':main()

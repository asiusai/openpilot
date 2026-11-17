"""Prepare fixed graphs, programs, input patches, and the parent protocol without device access."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-qualified-work')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main(target):
  ROOT.mkdir(exist_ok=True);target.mkdir(exist_ok=False)
  os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0',TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'),PARALLEL='0',
    PROFILE='0',BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',LLVMOPT='1',GMMU='0',TC_OPT='2',DEBUG='0')
  sys.path.insert(0,str(TINY))
  import numpy as np
  from tinygrad import Tensor,dtypes
  from tinygrad.device import Buffer,Device
  from tinygrad.uop.ops import UOp
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.codegen import to_program
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import DepsTracker,unwrap_view
  from tinygrad.runtime import ops_amd
  common=load('agent-ck-qualified-common');old=common.load('agent-ck-output-common')
  adapter=old.load('agent-ck-attention-v2-adapter')
  previous,oracle,bound,reuse=load('agent-ck-qualified-evidence').verify()
  cases={f['name']:{k:np.load(Path(previous['oracle'])/v['path'],allow_pickle=False) for k,v in f['files'].items()} for f in oracle['fixtures']}
  bank,mapping={},{}
  for fixture,data in cases.items():
    mapping[fixture]={}
    for key in ('q','k','v'):
      payload=data[key].tobytes();digest=hashlib.sha256(payload).hexdigest();mapping[fixture][key]=digest
      if digest in bank: continue
      alternatives=[]
      for other,item in bank.items():
        if item['bytes']==len(payload):
          parent=cases[item['fixture']][item['key']].tobytes();spans=common.ranges(parent,payload)
          alternatives.append((sum(size for _,size in spans),other,spans))
      best=min(alternatives,default=(len(payload),None,[[0,len(payload)]]))
      if best[0]>=len(payload): best=(len(payload),None,[[0,len(payload)]])
      bank[digest]={'fixture':fixture,'key':key,'shape':list(data[key].shape),'bytes':len(payload),
        'parent':best[1],'upload_ranges':best[2],'upload_bytes':best[0]}
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  programs,proofs={},{}
  def tensor(shape,dt=dtypes.bfloat16): return Tensor(UOp.from_buffer(Buffer('AMD',math.prod(shape)*dt.itemsize),dt)).reshape(shape)
  def compile_case(name,inputs,result):
    started=time.monotonic();raw,variables=result[0].linear_with_vars(*result[1:]);assert not variables
    tracker=DepsTracker([('AMD','COMPUTE:0')]*len(raw.src));rows=[]
    for tag,call in enumerate(raw.src):
      program=to_program(call.body,renderer);compiled=call.replace(src=(program,*call.src[1:]))
      blob,source=program.src[3].arg,program.src[2].arg;digest=hashlib.sha256(blob).hexdigest()
      if digest not in programs:
        descriptor,_=ops_amd._amd_program_image(SimpleNamespace(arch='gfx1200',target=(12,0,0),iface=SimpleNamespace(props={'lds_size_in_kb':64})),blob)
        assert ((descriptor.rsrc1>>29)&1)==0 and not descriptor.enable_dispatch_ptr
        (target/(digest+'.hsaco')).write_bytes(blob);(target/(digest+'.source')).write_text(source)
        programs[digest]={'name':program.src[0].arg.name,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
          'outs':list(program.arg.outs),'ins':list(program.arg.ins),'global':list(program.arg.global_size),'local':list(program.arg.local_size),
          'descriptor':asdict(descriptor),'actual_mode':'CU'}
      outs,ins=get_call_outs_ins(compiled)
      rows.append({'tag':tag,'name':program.src[0].arg.name,'binary_sha256':digest,'outs':outs,'ins':ins,
        'dependencies':tracker.access_resources([(u,i in outs) for i,u in enumerate(call.src[1:])],tag),
        'arguments':[old.view_info(u) for u in call.src[1:]],
        'root_slots':[next((i for i,t in enumerate(inputs) if unwrap_view(u)[0] is t.uop.base),None) for u in call.src[1:]]})
    proofs[name]={'calls':rows,'compile_seconds':time.monotonic()-started}
    if name.startswith('attention'):
      native=[r for r in rows if r['name'] in ('alpamayo_ck_d128','agent_attention_rowsoftmax')]
      assert len(native)==1 and native[0]['root_slots'][0]==3 and native[0]['arguments'][0]['offset_bytes']==512
      if 'candidate' in name:
        assert native[0]['root_slots'][1]==4 and native[0]['arguments'][1]['offset_bytes']==1024
        assert native[0]['binary_sha256']=='cf087e9fbd2b094ead1069c4b0f8183ceb10f6da5253bdc904502c9552b0248b'
      else: assert native[0]['binary_sha256']=='25eefe9f887548d55d03387d0dfba6b4f7ae7c4691d4527d4997975bcf4c4440'
    elif name.startswith('utility'):
      writes=[r for r in rows if r['name'].startswith('ckq_')]
      assert len(writes)==1 and writes[0]['root_slots'][0]==0 and writes[0]['arguments'][0]['offset_bytes']==0
    print(json.dumps({'case':name,'calls':len(rows),'seconds':proofs[name]['compile_seconds']}),flush=True)
  for boundary in ('A','B'):
    for arm in ('baseline','candidate'):
      shapes=[common.QSHAPE,common.KVSHAPE,common.KVSHAPE] if boundary=='A' else [(1,2048,16,128),(1,8,2304,128),(1,8,2304,128)]
      inputs=[tensor(s) for s in shapes]+[tensor((common.OC+512,)),tensor((common.LC+512,),dtypes.float32)]
      with adapter.abi_scope(): result=common.outputs(boundary,arm,*inputs)
      compile_case(f'attention_{boundary}_{arm}',inputs,result)
  for count,dt in [(common.OC,dtypes.bfloat16),(2097152,dtypes.bfloat16),(common.OC+512,dtypes.bfloat16),(common.LC+512,dtypes.float32)]:
    inputs=[tensor((count,),dt),tensor((count,),dt)]
    compile_case(f'utility_copy_{count}',inputs,(common.utility(*inputs),))
  for count,dt in [(common.OC+512,dtypes.bfloat16),(common.LC+512,dtypes.float32)]:
    inputs=[tensor((count,),dt)];compile_case(f'utility_poison_{count}',inputs,(common.utility(*inputs,mode='poison'),))
  for count in (len(common.CHANNELS)*(common.OC+512)//256,len(common.CHANNELS)*(common.LC+512)//256,2048):
    inputs=[tensor((count,),dtypes.uint32)];compile_case(f'utility_sentinel_{count}',inputs,(common.utility(*inputs,mode='sentinel'),))
  for mode,ds,ss in [('q_B',(1,2048,16,128),common.QSHAPE),('kv_B',(1,8,2304,128),common.KVSHAPE)]:
    inputs=[tensor(ds),tensor(ss)];compile_case('utility_'+mode,inputs,(common.utility(*inputs,mode=mode),))
  for boundary,kind in [('A','O'),('B','O'),('A','LSE')]:
    count,dt=(common.OC,dtypes.bfloat16) if kind=='O' else (common.LC,dtypes.float32)
    shape=(1,2048,2048) if boundary=='B' else (common.QSHAPE if kind=='O' else common.LSHAPE)
    inputs=[tensor((count+512,),dt),tensor(shape,dt),tensor((count+512,),dt)]
    compile_case(f'validator_{boundary}_{kind}',inputs,(common.validator(*inputs,boundary=boundary),))
  inputs=[tensor((1,8,2304,128)) for _ in range(2)]
  compile_case('tail_validator',inputs,(common.tail_validator(*inputs),))
  assert not Device._opened_devices
  (target/'cpu-proof.json').write_text(json.dumps({'gpu_executed':False,'opened_devices':[],'cases':proofs},indent=2)+'\n')
  own=[EXP/(name+'.py') for name in ('agent-ck-qualified-common','agent-ck-qualified-evidence','agent-ck-qualified-prepare',
                                   'agent-ck-qualified-run','agent-ck-qualified-check')]
  own += [EXP.parent/'model.py',TINY/'tinygrad/engine/realize.py',TINY/'tinygrad/runtime/support/memory.py',
          TINY/'tinygrad/runtime/support/usb.py',target/'cpu-proof.json',ROOT/'cpu-check-r2.json']
  own += list(target.glob('*.hsaco'))+list(target.glob('*.source'))
  bound.update({str(p):common.sha(p) for p in own})
  upload=sum(x['upload_bytes'] for x in bank.values())
  # Complete guarded and returned O for both arms, and LSE for CK, for seven A fixtures plus one B fixture.
  snapshot_bytes=2*((common.OC+512)*2+common.OC*2)+(common.LC+512)*4+common.LC*4
  flag_bytes=len(common.CHANNELS)*((common.OC+512)//256+(common.LC+512)//256)*4
  plan={'schema':1,'gpu_executed':False,'adoption_allowed':False,'timeout_seconds':300,'internal_timeout_seconds':290,
    'tinygrad':str(TINY),'tinygrad_commit':previous['tinygrad_commit'],'oracle':previous['oracle'],'reuse':reuse,
    'source_sha256':bound,'programs':programs,'limits':previous['limits'],'replay_order':previous['replay_order'],
    'fixtures':[f['name'] for f in oracle['fixtures']],'input_bank':bank,'fixture_inputs':mapping,
    'boundaries':previous['boundaries'],'boundary_B_limit':previous['boundary_B_limit'],
    'timing':{'alternating_samples':12,'calls_per_sample':5,'fence':'One final synchronize per five serial TinyJit calls; no validator inside timing.'},
    'budget':{'fixture_upload_bytes':upload,'full_initial_and_B_read_bytes':snapshot_bytes*8,
      'flags_per_candidate_check_bytes':flag_bytes,'forty_replay_flag_bytes':flag_bytes*40,
      'conservative_total_payload_bytes':upload+snapshot_bytes*8+flag_bytes*90+8*1024*1024,
      'prior_measured_payload_bytes':119054336,'prior_elapsed_seconds':reuse['original_elapsed_seconds'],
      'compilation_cpu_seconds':sum(v['compile_seconds'] for v in proofs.values()),
      'rationale':'Resident immutable fixture bank; byte-proven 8KiB upload patches; GPU poison; exact per-word replay comparisons. Full first fixture arrays remain mandatory.'},
    'classification':'Numerical attention variant. Micro accuracy and speed qualification only; no model or route adoption.',
    'validator':{'kind':'Seven integer counts per 256 words, every bit; no hash/tolerance substitution','channels':common.CHANNELS,
      'execution':'Normal TinyJit for copies, poison, and validators. Captured flag buffers receive 0xffffffff before every replay; each sentinel utility is fully read back once.',
      'negative_controls':'Independent CPU full flag arrays for all 16/32 bit positions, guarded and returned NaNs/infinities, both redzones, final words, and KV tails.'}}
  (target/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
  print(json.dumps({'protocol':str(target/'protocol.json'),'sha256':common.sha(target/'protocol.json'),'programs':len(programs),'budget':plan['budget']},indent=2))


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'prepared-final');args=parser.parse_args()
  assert args.out.parent.resolve()==ROOT.resolve();main(args.out)

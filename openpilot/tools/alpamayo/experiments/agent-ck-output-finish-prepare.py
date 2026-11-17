"""CPU-only preparation. Compiles the unchanged graph; opens no GPU or USB device."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-output-finish-work')
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')


def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def main():
  ROOT.mkdir(exist_ok=True)
  os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0',TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'),PARALLEL='0',
    PROFILE='0',BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',LLVMOPT='1',GMMU='0',TC_OPT='2',DEBUG='0')
  sys.path.insert(0,str(TINY))
  evidence=load('agent-ck-output-finish-evidence')
  previous,report,reuse=evidence.verify()
  _,patch=evidence.fixture_proof(previous['oracle'])
  from tinygrad import Tensor,dtypes
  from tinygrad.device import Buffer,Device
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.codegen import to_program
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_view,DepsTracker
  helper=load('agent-ck-output-common');adapter=helper.load('agent-ck-attention-v2-adapter')
  target=ROOT/'prepared';target.mkdir(exist_ok=False)
  shapes=[helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE,(helper.OC+512,),(helper.LC+512,)]
  inputs=[Tensor(UOp.from_buffer(Buffer('AMD',math.prod(s)*(4 if i==4 else 2)),
    dtypes.float32 if i==4 else dtypes.bfloat16)).reshape(s) for i,s in enumerate(shapes)]
  roots=[t.uop.base for t in inputs]
  with adapter.abi_scope(): outputs=helper.outputs('candidate',True,*inputs)
  raw,variables=outputs[0].linear_with_vars(*outputs[1:]);assert not variables
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  compiled,programs=[],{}
  for call in raw.src:
    program=to_program(call.body,renderer)
    compiled.append(call.replace(src=(program,*call.src[1:])))
    blob,source=program.src[3].arg,program.src[2].arg
    digest=hashlib.sha256(blob).hexdigest()
    assert digest in previous['programs'] and hashlib.sha256(source.encode()).hexdigest()==previous['programs'][digest]['source_sha256']
    (target/(digest+'.hsaco')).write_bytes(blob);(target/(digest+'.source')).write_text(source)
    programs[digest]=previous['programs'][digest]
  assert len(compiled)==len(programs)==3
  tracker=DepsTracker([('AMD','COMPUTE:0')]*len(compiled))
  rows=[]
  for tag,call in enumerate(compiled):
    outs,ins=get_call_outs_ins(call)
    rows.append({'tag':tag,'name':call.body.src[0].arg.name,'outs':outs,'ins':ins,
      'dependencies':tracker.access_resources([(u,i in outs) for i,u in enumerate(call.src[1:])],tag),
      'arguments':[helper.view_info(x) for x in call.src[1:]],'binary_sha256':hashlib.sha256(call.body.src[3].arg).hexdigest()})
  assert rows[0]['name']=='alpamayo_ck_d128' and rows[0]['outs']==(0,1) and rows[0]['ins']==(2,3,4)
  assert rows[1]['dependencies']==rows[2]['dependencies']==[0]
  assert unwrap_view(compiled[0].src[1])==(roots[3],512)
  assert unwrap_view(compiled[0].src[2])==(roots[4],1024)
  for index in (1,2): assert rows[index]['arguments'][1]==rows[0]['arguments'][index-1]
  linear=UOp(Ops.LINEAR,src=tuple(compiled))
  replay=linear.substitute({r:UOp.param(i,r.dtype,r.max_numel(),r.device) for i,r in enumerate(roots)},walk=True)
  replay_views=[helper.view_info(x) for x in replay.src[0].src[1:]]
  assert [v['base_slot'] for v in replay_views]==[3,4,0,1,2]
  assert [v['offset_bytes'] for v in replay_views]==[512,1024,0,0,0]
  assert all(v['base_op']=='PARAM' for v in replay_views)
  assert not Device._opened_devices
  proof={'gpu_executed':False,'opened_devices':list(Device._opened_devices),'scheduled_calls':rows,
    'replay_native_arguments':replay_views,'source_and_binaries_identical_to_completed_first_A':True,
    'storage':[helper.storage_info(t) for t in inputs[3:]],'returned':[helper.storage_info(t) for t in outputs],
    'input_patch':patch,'reuse':reuse,'actual_mode_required':'CU','compiler_workers':0}
  (target/'cpu-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
  files=[EXP/(name+'.py') for name in ('agent-ck-output-finish-evidence','agent-ck-output-finish-run',
    'agent-ck-output-finish-prepare','agent-ck-output-finish-check')]
  files += [evidence.REPORT,evidence.ORIGINAL,evidence.PREVIOUS,Path('/tmp/opencode/run-alpamayo-ck-output.py')]
  files += list(target.glob('*.hsaco'))+list(target.glob('*.source'))+[target/'cpu-proof.json']
  files += [TINY/'tinygrad'/p for p in ('runtime/support/memory.py','runtime/support/system.py','runtime/support/usb.py',
    'renderer/llvmir.py','runtime/support/compiler_amd.py','codegen/__init__.py','engine/worker.py','helpers.py')]
  bound=dict(previous['source_sha256']);bound.update({str(p):evidence.sha(p) for p in files})
  phases=[{'name':'candidate_normalized_reestablish_A','fixture':'a','kind':'first','jit_count_before':0},
    {'name':'candidate_normalized_capture_A','fixture':'a','kind':'capture','jit_count_before':1},
    {'name':'candidate_normalized_replay_mutate_v','fixture':'mutate_v','kind':'replay','jit_count_before':2},
    {'name':'candidate_normalized_replay_A','fixture':'a','kind':'replay','jit_count_before':3}]
  plan={'schema':1,'gpu_executed':False,'adoption_allowed':False,'timeout_seconds':120,'internal_timeout_seconds':110,
    'tinygrad':str(TINY),'tinygrad_commit':'89846dcfb383a98ee01cfec3bba440ae7dcf9b1e','oracle':previous['oracle'],
    'limits':previous['limits'],'programs':programs,'source_sha256':bound,'input_patch':patch,'reuse':reuse,'phases':phases,
    'io_budget':{'input_write_bytes':16793600,'poison_write_bytes':34091008,'readback_bytes':68169728},
    'source_binding':'Exact previous protocol/report/adapter/fixtures, full first-A output hashes, unchanged three candidate programs.',
    'fresh_process':'Normal TinyJit first/capture/replay/replay. No counter changes, prebuilt capture, or pickle reuse.',
    'deadline_rationale':'Four candidate phases only; 119054336 payload bytes. Previous nine-phase run stopped during a redundant upload.',
    'scope':'Guarded-output binding completion only. Candidate is a numerical variant. No timing, model integration, or performance claim.',
    'remaining_gates':['all seven full-shape fixtures including separate Q/K mutations and causal-future/GQA fixtures',
      '40 ordered mutable replays with poison and exact restoration',
      'both fair boundary timings including packing, return conversion, LSE and synchronization',
      'actual model integration and route quality','under-100ms complete pipeline']}
  (target/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
  print(json.dumps({'gpu_executed':False,'opened_devices':list(Device._opened_devices),'protocol':str(target/'protocol.json'),
    'protocol_sha256':evidence.sha(target/'protocol.json'),'program_count':len(programs),'source_count':len(bound)},indent=2))


if __name__=='__main__': main()

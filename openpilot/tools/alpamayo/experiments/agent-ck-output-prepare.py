"""CPU-only complete schedule, native pointer, HCQ dependency and replay-slot proof."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

EXP = Path(__file__).parent
ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-output')
ROOT.mkdir(exist_ok=True)
os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0',TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'),PARALLEL='0')
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
sys.path.insert(0,str(TINY))


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  result = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(result)
  return result


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  from tinygrad import Tensor,dtypes
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.codegen import to_program
  from tinygrad.engine.realize import get_call_outs_ins
  from tinygrad.runtime.support.hcq2 import unwrap_view,DepsTracker
  helper = load('agent-ck-output-common')
  adapter = helper.load('agent-ck-attention-v2-adapter')
  target = ROOT/'prepared-final'
  target.mkdir(exist_ok=False)
  renderer = AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
  records,programs = {},{}
  for arm in ('baseline','candidate'):
    for normalize in (False,True):
      key = arm+('-normalized' if normalize else '-original')
      shapes = [helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE,(helper.OC+512,),(helper.LC+512,)]
      inputs = [Tensor(UOp.from_buffer(Buffer('AMD',math.prod(s)*(4 if i==4 else 2)),
                    dtypes.float32 if i==4 else dtypes.bfloat16)).reshape(s) for i,s in enumerate(shapes)]
      roots = [t.uop.base for t in inputs]
      with adapter.abi_scope(): outputs = helper.outputs(arm,normalize,*inputs)
      raw, variables = outputs[0].linear_with_vars(*outputs[1:])
      assert not variables
      compiled = []
      for call in raw.src:
        program = to_program(call.body,renderer)
        compiled.append(call.replace(src=(program,*call.src[1:])))
        blob,source = program.src[3].arg,program.src[2].arg
        digest = hashlib.sha256(blob).hexdigest()
        if digest not in programs:
          (target/(digest+'.hsaco')).write_bytes(blob)
          (target/(digest+'.source')).write_text(source)
          programs[digest] = {'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'name':program.src[0].arg.name,'native':program.src[0].arg.name in ('agent_attention_rowsoftmax','alpamayo_ck_d128')}
      linear = UOp(Ops.LINEAR,src=tuple(compiled))
      tracker = DepsTracker([('AMD','COMPUTE:0')]*len(compiled))
      rows = []
      for tag,call in enumerate(compiled):
        outs,ins = get_call_outs_ins(call)
        deps = tracker.access_resources([(s,i in outs) for i,s in enumerate(call.src[1:])],tag)
        rows.append({'tag':tag,'name':call.body.src[0].arg.name,'outs':outs,'ins':ins,'dependencies':deps,
          'binary_sha256':hashlib.sha256(call.body.src[3].arg).hexdigest(),
          'arguments':[helper.view_info(x) for x in call.src[1:]]})
      native_tag = next(i for i,c in enumerate(compiled) if c.body.src[0].arg.name in ('agent_attention_rowsoftmax','alpamayo_ck_d128'))
      native = compiled[native_tag]
      obase,ooffset = unwrap_view(native.src[1])
      assert (obase is roots[3]) == normalize
      if normalize: assert ooffset == 512
      if arm == 'candidate':
        lbase,loffset = unwrap_view(native.src[2])
        assert (lbase is roots[4]) == normalize
        if normalize: assert loffset == 1024
      # Apply the same root-buffer to positional PARAM substitution as jit_lower.
      replay = linear.substitute({root:UOp.param(i,root.dtype,root.max_numel(),root.device)
                                 for i,root in enumerate(roots)},walk=True)
      replay_native = replay.src[native_tag]
      replay_views = [helper.view_info(x) for x in replay_native.src[1:]]
      if normalize:
        assert replay_views[0]['base_op']=='PARAM' and replay_views[0]['base_slot']==3 and replay_views[0]['offset_bytes']==512
        if arm=='candidate': assert replay_views[1]['base_slot']==4 and replay_views[1]['offset_bytes']==1024
        post_copies = [r for r in rows[native_tag+1:] if native_tag in r['dependencies']]
        assert len(post_copies)==(2 if arm=='candidate' else 1), rows
      writes_guard = [r['tag'] for r in rows for index in r['outs']
                      if unwrap_view(compiled[r['tag']].src[1+index])[0] is roots[3]]
      assert writes_guard == ([native_tag] if normalize else [])
      records[key] = {'native_tag':native_tag,'native_writes_guarded_output':normalize,
        'guarded_output_writing_tags':writes_guard,'scheduled_calls':rows,'replay_native_arguments':replay_views,
        'storage':[helper.storage_info(t) for t in inputs[3:]],
        'returned':[helper.storage_info(t) for t in outputs]}
  (target/'cpu-schedule-proof.json').write_text(json.dumps(records,indent=2)+'\n')
  old = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-attention-v3/prepared-final')
  oracle = EXP/'agent-ck-attention-v2/oracle-r2'
  files = [EXP/(name+'.py') for name in ('agent-ck-output-common','agent-ck-output-run','agent-ck-output-prepare',
    'agent-ck-attention-v2-adapter','agent-ck-attention-v3-run')]
  files += [old/'baseline.hsaco',old/'baseline.cpp',old/'protocol.json',oracle/'manifest.json',
    EXP/'agent-ck-attention-v2/native-overlay/kernel.hsaco',EXP/'agent-ck-attention-v2/native-overlay/source.cpp',
    Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-attention-parent/report.json')]
  files += [TINY/'tinygrad'/name for name in ('tensor.py','device.py','engine/jit.py','engine/realize.py',
    'schedule/__init__.py','schedule/rangeify.py','runtime/support/hcq2.py','runtime/ops_amd.py','uop/ops.py')]
  files += list(target.glob('*.hsaco'))+list(target.glob('*.source'))+[target/'cpu-schedule-proof.json']
  for fixture in ('a','mutate_v'): files += list((oracle/fixture).glob('*.npy'))
  manifest = {'schema':1,'gpu_executed':False,'timeout_seconds':120,'tinygrad':str(TINY),'oracle':str(oracle),
    'source_sha256':{str(p):sha(p) for p in files},'programs':programs,
    'limits':json.loads((old/'protocol.json').read_text())['limits'],
    'phases':['baseline_original_first_A','baseline_normalized_first_A','baseline_normalized_capture_A',
      'baseline_normalized_replay_mutate_v','baseline_normalized_replay_A',
      'candidate_normalized_first_A','candidate_normalized_capture_A','candidate_normalized_replay_mutate_v','candidate_normalized_replay_A'],
    'scope':'Binding diagnostic only. No timing or adoption. Full causal/GQA/seven-fixture/replay40 gates remain required after this test.'}
  (target/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n')
  print(json.dumps({'gpu_executed':False,'cases':list(records),'program_count':len(programs),
    'cause':'Logical output slice materialized into a temporary without copy-back; contiguous view registration preserves physical offset.',
    'protocol':str(target/'protocol.json')},indent=2))


if __name__=='__main__': main()

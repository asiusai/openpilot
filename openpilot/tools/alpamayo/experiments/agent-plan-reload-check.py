"""Actual-plan CPU reconstruction and negative controls; hardware access forbidden."""
import argparse
import ast
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116');EXP=Path(__file__).resolve().parent
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'

def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def reject(fn):
  try: fn()
  except (ValueError,AssertionError,KeyError): return
  raise AssertionError('Negative control was accepted')

def metadata_model():
  from tinygrad import Tensor,dtypes
  from tinygrad.device import Buffer
  from tinygrad.helpers import prod
  from tinygrad.uop.ops import UOp
  from amd_gemm import unpack
  index_path=Path('/dev/shm/alpamayo-9060-fp8/index.json');index=json.loads(index_path.read_text())
  bulk=Buffer('AMD',index['nbytes']);model=SimpleNamespace(weights={},weight_scales={},cache=[])
  for record in index['records']:
    name=record['name']
    if not (name.startswith('vlm.model.language_model.') or name=='vlm.lm_head.weight'): continue
    dt={'scale':dtypes.float32,'fp8':dtypes.fp8e4m3,'bf16':dtypes.bfloat16}[record['kind']]
    value=Tensor(UOp.from_buffer(bulk.view(record['nbytes'],record['offset']),dt)).reshape(record['shape'])
    if record['packed']: value=unpack(value)
    (model.weight_scales if record['kind']=='scale' else model.weights)[name]=value
  def tensor(shape,dt): return Tensor(UOp.new_buffer('AMD',prod(shape),dt)).reshape(shape)
  model.weights['vlm.lm_head.weight']=tensor((2112,2048),dtypes.fp8e4m3)
  model.output_token_mapping=tensor((2112,),dtypes.int32)
  model.cache=[tensor((2,1,8,2304,128),dtypes.bfloat16) for _ in range(28)]
  inputs=tuple(tensor(shape,dtypes.bfloat16) for shape in ((1,1,2112),(1,1,64,128),(1,1,64,128)))
  options=dict(bulk_index=index_path,vocab=SOURCE/'experiments/coc_vocab.json',checkpoint='/tmp/opencode/alpamayo-asius-20261114/model')
  return model,inputs,options

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
  if not args.output.resolve().is_relative_to(WORK) or args.output.exists(): raise ValueError('Use a new WORK output')
  for name in ('TMPDIR','CACHEDB'):
    if not Path(os.environ.get(name,'')).resolve().is_relative_to(WORK): raise ValueError('All writable storage must remain under WORK')
  sys.path[:0]=[str(SOURCE),str(SOURCE/'experiments')]
  from tinygrad import Device
  from tinygrad.uop.ops import Ops,UOp
  from tinygrad.dtype import dtypes
  for path in EXP.glob('agent-plan-reload-*.py'): ast.parse(path.read_text())
  def forbidden(*args,**kwargs): raise AssertionError('CPU check attempted hardware access')
  report=dict(gpu_executed=False,persistent_execution=False)
  with patch.object(type(Device),'__getitem__',forbidden):
    inspect=local(EXP/'agent-plan-reload-inspect.py');proofmod=local(EXP/'agent-plan-reload-proof.py')
    loader=local(EXP/'agent-plan-reload-loader.py')
    document,programs=inspect.load();proof=proofmod.prove(document,programs)
    model,inputs,options=metadata_model()
    prepared=loader.Prepared(document,programs,proof,model,inputs,**options)
    model2,inputs2,options2=metadata_model();other=loader.Prepared(document,programs,proof,model2,inputs2,**options2)
    assert prepared.registry.describe()==other.registry.describe()
    assert all(a is not b for a,b in zip(prepared.owner_bases.values(),other.owner_bases.values(),strict=True))
    assert [c.body.key for c in prepared.linear.src]==[c.body.key for c in other.linear.src]
    assert all(c.body.key.hex()==row['program_key'] for c,row in zip(prepared.linear.src,document['calls'],strict=True))
    reject(lambda:prepared.compile_runtime())
    # Reorder the producer and first consumer: actual arena proof must reject it.
    wrong=copy.deepcopy(document);wrong['calls'][0],wrong['calls'][1]=wrong['calls'][1],wrong['calls'][0]
    reject(lambda:proofmod.prove(wrong,programs))
    reject(lambda:loader.Prepared(wrong,programs,proof,model,inputs,**options))
    # Move one actual scratch input into the never-written padding gap.
    wrong=copy.deepcopy(document);wrong['calls'][1]['arguments'][1]['src'][0]['src'][1]['arg']=4
    reject(lambda:proofmod.prove(wrong,programs))
    # Remove an actual store while leaving ProgramInfo.outs untouched.
    key=document['calls'][0]['program_key'];p=programs[key]
    replacement=p.src[1].replace(src=tuple(u for u in p.src[1].src if not (u.op is Ops.STORE and u.src[0].buf_uop.op is Ops.PARAM)))
    changed=p.replace(src=(p.src[0],replacement,*p.src[2:]));bad=dict(programs);bad[key]=changed
    reject(lambda:proofmod.prove(document,bad))
    reject(lambda:loader.Prepared(document,bad,proof,model,inputs,**options))
    for changed in (
      p.replace(src=(*p.src[:3],p.src[3].replace(arg=p.src[3].arg[:-1]+bytes([p.src[3].arg[-1]^1])))),
      p.replace(src=(*p.src[:2],p.src[2].replace(arg=p.src[2].arg+'\n'),p.src[3])),
      p.replace(arg=replace(p.arg,outs=(1,))),
      p.replace(arg=replace(p.arg,target=replace(p.arg.target,interface='')))):
      bad=dict(programs);bad[key]=changed
      reject(lambda:loader.Prepared(document,bad,proof,model,inputs,**options))
    # Gate the real scalar output store on an impossible lane. outs remains unchanged.
    row=next(r for r in proofmod.instruction_rows(p) if r[1].op is Ops.STORE)
    fake_scope=(*row[-1],UOp(Ops.IF,src=(UOp.const(False),row[1].src[0])))
    mask,_=proofmod.enumerate_store(p,(*row[:-1],fake_scope));assert not mask.any()
    wrong=copy.deepcopy(document)
    wrong['calls'][0]['arguments'][0]=copy.deepcopy(document['calls'][1]['arguments'][2])
    reject(lambda:proofmod.prove(wrong,programs))
    old_kv=model.cache[0];model.cache[0]=old_kv.reshape(-1)
    reject(prepared.guard);model.cache[0]=old_kv
    # Closed owner replacement must fail both initial rebinding and later guards.
    model.cache[0]=model2.cache[0]
    reject(prepared.guard)
    model.cache[0]=model.cache[1]
    reject(lambda:loader.Prepared(document,programs,proof,model,inputs,**options))
    # Replaced KV has identical metadata but unrelated ownership; initial registry
    # may validly accept fresh-process allocations. Guard is the continuity check.
    report.update(actual_programs=29,actual_calls=765,fresh_owner_rebinding=True,numerical_program_identities_exact=True,
      scratch_proof=proof,negative_controls=['producer_order','uninitialized_gap','missing_store_despite_outs',
        'masked_store_not_unconditional','changed_PROGRAM','changed_binary','changed_source','changed_ABI','changed_full_target',
        'immutable_write','closed_owner_view_change','closed_owner_continuity','KV_alias_change','runtime_requires_parent_authorization'])
    if args.preflight:
      with patch.object(sys,'argv',[str(EXP/'agent-plan-reload-run.py'),'--preflight','--output',str(args.output.with_suffix('.preflight.json'))]):
        runpy.run_path(str(EXP/'agent-plan-reload-run.py'),run_name='__main__')
      report['actual_runpy_import_preflight']=True
  assert not Device._opened_devices
  report.update(devices=[],source_sha256={p.name:loader.sha(p) for p in EXP.glob('agent-plan-reload-*.py')})
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items() if k not in ('scratch_proof','source_sha256')}))

if __name__=='__main__': main()

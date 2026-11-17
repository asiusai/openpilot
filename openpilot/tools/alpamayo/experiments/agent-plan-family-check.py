"""CPU tests of actual family API/layout logic plus frozen real-plan proof negatives."""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

EXP=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
sys.dont_write_bytecode=True;sys.path[:0]=[str(TG),str(SOURCE),str(SOURCE/'experiments')]

def local(path):
  import importlib.util
  spec=importlib.util.spec_from_file_location('family_check_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def reject(fn):
  try: fn()
  except (ValueError,AssertionError,KeyError): return
  raise AssertionError('Negative control was accepted')

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  contract=local(EXP/'agent-plan-family-contract.py');contract.fresh(args.output)
  for name in ('TMPDIR','CACHEDB','XDG_CACHE_HOME'):
    if not Path(os.environ.get(name,'')).resolve().is_relative_to(WORK): raise ValueError('Caches must stay on WORK')
  for name in contract.SOURCES: ast.parse((EXP/name).read_text())
  from tinygrad import Tensor,TinyJit,Device,dtypes
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.helpers import Context
  import numpy as np
  getitem=type(Device).__getitem__
  def forbid(*a,**kw): raise AssertionError('Metadata inspection opened a device')
  def cpu(self,key):
    if key is None or key.split(':')[0] not in ('CPU','NPY','PYTHON'): raise AssertionError('Hardware device forbidden')
    return getitem(self,key)
  result=dict(gpu_executed=False,passed=False,actual_family_scratch_proved=False,actual_family_reloaded=False,
              cpu_fixture_scope='Original BlockDecoder API and alias/output layout; fixture arithmetic is not Alpamayo')
  with patch.object(type(Device),'__getitem__',forbid):
    contract.source_gate();signatures=contract.signatures()
    inspector=contract.local(EXP/'agent-plan-reload-inspect.py');doc,programs=inspector.load()
    proof_module=contract.local(EXP/'agent-plan-reload-proof.py');proof=proof_module.prove(doc,programs)
    metadata=contract.local(EXP/'agent-plan-reload-check.py');model,inputs,options=metadata.metadata_model()
    loader=contract.local(EXP/'agent-plan-reload-loader.py');prepared=loader.Prepared(doc,programs,proof,model,inputs,**options)
    views={name:contract.view_recipe(t) for name,t in zip(('return/logits','return/token'),prepared.ret,strict=True)}
    member=contract.validate_document(doc,1,signatures['1'],views)
    # Exercise the new production offline reader on the existing real capture.
    # This does not promote count1 evidence to any larger length.
    sidepath=args.output.with_suffix('.count1-sidecar.json');contract.fresh(sidepath)
    sidepath.write_text(json.dumps(dict(count=1,audit_sha256=inspector.CAPTURE_SHA,output_views=views,contract=member),indent=2)+'\n')
    summary=dict(audit=str(inspector.CAPTURE),audit_sha256=inspector.CAPTURE_SHA,sidecar=str(sidepath),
                 sidecar_sha256=contract.sha(sidepath),contract=member)
    offline=contract.local(EXP/'agent-plan-family-offline.py')
    _,_,production_proof,_=offline.inspect_member(contract,summary,1,signatures['1'])
    if production_proof!=proof: raise AssertionError('New offline path changed the actual count1 proof')
    wrong_summary=copy.deepcopy(summary);wrong_summary['audit_sha256']='0'*64
    reject(lambda:offline.inspect_member(contract,wrong_summary,1,signatures['1']))
    wrong=copy.deepcopy(doc);wrong['calls'][0],wrong['calls'][1]=wrong['calls'][1],wrong['calls'][0]
    reject(lambda:proof_module.prove(wrong,programs))
    wrong=copy.deepcopy(doc);wrong['calls'][1]['arguments'][1]['src'][0]['src'][1]['arg']=4
    reject(lambda:proof_module.prove(wrong,programs))
    firstkey=doc['calls'][0]['program_key'];prg=programs[firstkey]
    instructions=prg.src[1].replace(src=tuple(u for u in prg.src[1].src if not (u.op is Ops.STORE and u.src[0].buf_uop.op is Ops.PARAM)))
    bad=dict(programs);bad[firstkey]=prg.replace(src=(prg.src[0],instructions,*prg.src[2:]))
    reject(lambda:proof_module.prove(doc,bad))
    for field in ('aliases','input_contract'):
      wrong=copy.deepcopy(doc);wrong[field]=[];reject(lambda:contract.validate_document(wrong,1,signatures['1'],views))
    wrong=copy.deepcopy(views);wrong['return/token']=[{'op':'PYLITERAL','shape':[1]}]
    reject(lambda:contract.validate_document(doc,1,signatures['1'],wrong))
    frozen=contract.local(EXP/'agent-plan-bind-run.py');capture=contract.local(EXP/'agent-plan-family-capture.py')
    derived=capture.install(frozen,contract,{'counts':list(contract.COUNTS),'limits':contract.LIMITS},0)
    result.update(frozen_count1_proof_rechecked=True,setup_derivation=derived,
      signatures=[dict(count=n,calls=len(signatures[str(n)]['kernels']),unique_programs=len({x['program_key'] for x in signatures[str(n)]['kernels']}),
        programs_absent_from_count1=len({x['program_key'] for x in signatures[str(n)]['kernels']}-set(programs)),variables=contract.bounds(n)) for n in contract.COUNTS],
      production_offline_reader_actual_count1_passed=True,
      count1_success_sha256=contract.sha(contract.SUCCESS),negative_controls=['changed_capture_digest','scratch_producer_order','scratch_unwritten_gap',
        'missing_STORE_despite_ProgramInfo_outs','missing_alias','changed_input_ABI','unreviewed_output_view'])
  # Execute the original BlockDecoder.decode method, not an imitation of its
  # unrolling/concatenation/variable API. Only _greedy_decode arithmetic is a fixture.
  module_ast=ast.parse((SOURCE/'experiments/block-decode.py').read_text())
  cls=next(n for n in module_ast.body if isinstance(n,ast.ClassDef) and n.name=='BlockDecoder')
  cls=copy.deepcopy(cls);cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='decode']
  namespace=dict(Tensor=Tensor,TinyJit=TinyJit,UOp=UOp)
  exec(compile(ast.Module(body=[cls],type_ignores=[]),'<original-BlockDecoder.decode>','exec'),namespace)
  with Context(DEV='CPU:LLVM'),patch.object(type(Device),'__getitem__',cpu):
    x=Tensor(np.arange(2112,dtype=np.float32).reshape(1,1,2112)%17,device='CPU').cast(dtypes.bfloat16).realize()
    c=Tensor(np.arange(64,dtype=np.float32).reshape(1,1,64,1),device='CPU').cast(dtypes.bfloat16).realize();s=c.clone().realize()
    rows=[]
    for count in contract.COUNTS:
      kv=Tensor.zeros(2,1,1,2304,1,device='CPU',dtype=dtypes.bfloat16).contiguous().realize()
      model=SimpleNamespace(cache=[kv])
      def greedy(logits,cos,sin,position,step):
        part=cos[:,:,step:step+1,:].reshape(1,1,1)
        value=(logits+part+1).cast(dtypes.bfloat16).realize()
        cache=model.cache[0]
        store=cache[:,:,:,position:position+1,:].uop.store(value[:,:,:1].reshape(1,1,1,1,1).expand(2,1,1,1,1).uop)
        Tensor(cache.uop.after(store)).realize()
        token=logits.argmax(axis=-1).float().reshape(1,1).realize()
        return value,token
      model._greedy_decode=greedy;decoder=namespace['BlockDecoder']();decoder.model=model;decoder.jits={}
      def reset(): model.cache[0].assign(Tensor.zeros(*kv.shape,device='CPU',dtype=dtypes.bfloat16)).realize()
      reset();first=decoder.decode(x,c,s,2027,0,count);saved=[v.numpy().copy() for v in first];savedkv=kv.numpy().copy()
      for _ in range(2):
        reset();actual=decoder.decode(x,c,s,2027,0,count)
        for a,b in zip(actual,saved,strict=True): np.testing.assert_array_equal(a.numpy(),b)
        np.testing.assert_array_equal(kv.numpy(),savedkv)
      reset();actual[0].assign(x).realize();aliased=decoder.decode(actual[0],c,s,2027,0,count)
      for a,b in zip(aliased,saved,strict=True): np.testing.assert_array_equal(a.numpy(),b)
      np.testing.assert_array_equal(kv.numpy(),savedkv)
      jit=decoder.jits[(c.shape,count,2304)]
      if len(jit.captured._linears)!=2 or aliased[1].shape!=(count,): raise AssertionError('Original alias/token API differs')
      recipes=[contract.view_recipe(t) for t in jit.captured.ret]
      for tensor,recipe in zip(jit.captured.ret,recipes,strict=True):
        audit=contract.local(EXP/'agent-plan-bind-audit.py');registry=audit.Registry();registry.add('return',tensor,'test',{})
        reconstructed=contract.rebuild_view(registry.rows[0],recipe)
        if reconstructed.uop.base is tensor.uop.base: raise AssertionError('Output rebinding reused process storage')
      ranges=contract.bounds(count)
      reset();Tensor.realize(*decoder.decode(x,c,s,ranges['block_position']['max'],ranges['block_step']['max'],count))
      reject(lambda:decoder.decode(x,c,s,ranges['block_position']['max']+1,0,count))
      reject(lambda:decoder.decode(x,c,s,2027,ranges['block_step']['max']+1,count))
      rows.append(dict(count=count,token_shape=list(aliased[1].shape),alias_variants=2,output_views=recipes,
        variable_limits_executed=True,out_of_range_rejected=True,eager_capture_replay_alias_bits_equal=True))
    result['CPU_original_decode_cases']=rows
    # Reuse the qualified production transfer without replacing Buffer methods.
    from tinygrad.device import Buffer
    poison=contract.local(EXP/'agent-plan-poison-copy.py').poison_copy
    b=Buffer('CPU',2112*2)
    for pattern in (0xA5,0x5A):
      poison(b,pattern)
      if bytes(b.host[:b.nbytes])!=bytes([pattern])*b.nbytes: raise AssertionError('Poison bytes differ')
    values=Tensor.arange(2112).to(x.device).numpy();np.testing.assert_array_equal(values,np.arange(2112))
    if hasattr(Buffer,'copyin'): raise AssertionError('Compatibility Buffer alias appeared')
    result['qualified_production_poison_and_arange_rechecked']=True
  result.update(passed=True,devices=sorted(Device._opened_devices),sources={name:contract.sha(EXP/name) for name in contract.SOURCES})
  args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('sources','CPU_original_decode_cases')},indent=2))

if __name__=='__main__': main()

"""CPU compiler/ownership checks; optionally inspect a real parent metadata capture."""
import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SOURCE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
os.environ['DEV']='USB+AMD:LLVM'
sys.dont_write_bytecode=True


def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
  return value


def rejects(fn):
  try: fn()
  except (ValueError,AssertionError,KeyError): return
  raise AssertionError('Negative control was accepted')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--capture',type=Path)
  cli=parser.parse_args()
  if not cli.output.resolve().is_relative_to(WORK) or cli.output.exists(): raise ValueError('Use a new output under WORK')
  if not Path(os.environ.get('CACHEDB','')).resolve().is_relative_to(WORK): raise ValueError('CPU compiler cache must stay under WORK')
  if not Path(os.environ.get('TMPDIR','')).resolve().is_relative_to(WORK): raise ValueError('CPU temporary files must stay under WORK')
  sys.path[:0]=[str(SOURCE),str(SOURCE/'experiments')]
  for path in ROOT.glob('agent-plan-bind-*.py'): ast.parse(path.read_text())
  codec=local(ROOT/'agent-plan-bind-codec.py');audit=local(ROOT/'agent-plan-bind-audit.py');run=local(ROOT/'agent-plan-bind-run.py')
  for name,expected in run.PINS.items():
    if run.sha(run.TG/name)!=expected: raise ValueError('Pinned source changed: '+name)
  from tinygrad import Device,Tensor,dtypes
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp,Ops
  from tinygrad.codegen import do_to_program
  from tinygrad.helpers import Target,prod
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  report=dict(cpu_only=True,gpu_executed=False,actual_captured_jit_available=cli.capture is not None,
              persistent_plan_executed=False,qualified=False)
  def forbidden(*args,**kwargs): raise AssertionError('No device may open during offline inspection')
  with patch.object(type(Device),'__getitem__',forbidden):
    if cli.capture:
      document=json.loads(cli.capture.read_text())
      if document['kind']!='actual-count1-pre-hcq-diagnostic' or document['executable'] is not False: raise ValueError('Unexpected capture')
      if document['source_sha256']!={p.name:run.sha(p) for p in ROOT.glob('agent-plan-bind-*.py')}:
        raise ValueError('Capture helper sources changed')
      programs={key:codec.load_offline(cli.capture.parent/'programs',row) for key,row in document['programs'].items()}
      for call in document['calls']:
        program=programs[call['program_key']]
        if list(program.arg.outs)!=call['outputs'] or list(program.arg.ins)!=call['inputs']: raise ValueError('Call ABI differs')
      qualified=json.loads(run.QUALIFIED.read_text())['startup_cache']['serial_signatures']['1']
      actual=[dict(device='AMD',name=c['name'],program_key=c['program_key'],output_indices=c['outputs'],input_indices=c['inputs'])
              for c in document['calls']]
      identity=(actual==qualified['kernels'] and document['input_contract']==qualified['input_contract'] and
        document['input_names']==qualified['input_names'] and len(document['aliases'])==qualified['linked_alias_variants'])
      if identity!=document['qualified_numerical_identities_equal']: raise ValueError('Capture identity verdict was changed')
      rejects(lambda:audit.validate_binding_contract(document['registry'],document['registry']))
      report.update(actual_programs_reconstructed=len(programs),actual_calls=len(document['calls']),
                    blockers=document['blockers'],actual_numerical_identities_equal=document['qualified_numerical_identities_equal'],
                    actual_capture_sha256=run.sha(cli.capture))
    else:
      qualified=json.loads(run.QUALIFIED.read_text());known={k['program_key'] for k in qualified['startup_cache']['serial_signatures']['1']['kernels']}
      original=local(SOURCE/'amd_decode.py')
      # These three qualified count1 PROGRAMs were compiled in CU mode. Reproduce
      # their exact identities for serialization tests; change no runtime compiler.
      renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200','USB'))
      artifacts=cli.output.parent/(cli.output.stem+'-programs');artifacts.mkdir()
      rows=[]
      for n,k,packed in ((1024,2048,True),(2048,2048,True),(2112,2048,False)):
        from amd_gemm import unpack
        def tensor(shape,dt): return Tensor(UOp.new_buffer('AMD',prod(shape),dt)).reshape(shape)
        weight=tensor((n,k),dtypes.fp8e4m3)
        if packed: weight=unpack(weight)
        output=original.gemv(tensor((1,k),dtypes.fp8e4m3),weight,
          (tensor((),dtypes.float32),tensor((1,),dtypes.float32)),packed=packed)
        linear=output.schedule_linear()
        if len(linear.src)!=1: raise AssertionError('Unexpected additional GEMV scheduling work')
        program=do_to_program(linear.src[0].body,renderer)
        if program.key.hex() not in known: raise AssertionError('CPU GEMV does not match a qualified count1 program')
        record=codec.save(program,artifacts)
        restored=codec.load_offline(artifacts,record)
        assert restored.key==program.key and restored.arg==program.arg and restored.src[3].arg==program.src[3].arg
        document,blobs=codec.encode(program)
        bad=copy.deepcopy(document);bad['nodes'][bad['root']]['arg']['fields']['target']['fields']['interface']=''
        rejects(lambda:codec.Decoder(blobs).decode(bad))
        bad=copy.deepcopy(document);bad['nodes'][bad['root']]['arg']['fields']['outs']={'tuple':[1]}
        rejects(lambda:codec.Decoder(blobs).decode(bad))
        wrong=dict(blobs);binary_key=next(iter(wrong));wrong[binary_key]=bytes([wrong[binary_key][0]^1])+wrong[binary_key][1:]
        rejects(lambda:codec.Decoder(wrong).decode(document))
        bad=copy.deepcopy(document);bad['nodes'][0]['op']='GETADDR'
        rejects(lambda:codec.Decoder(blobs).decode(bad))
        bad=copy.deepcopy(document);next(row for row in bad['nodes'] if row['op']=='SOURCE')['arg']+='\n'
        rejects(lambda:codec.Decoder(blobs).decode(bad))
        rows.append(dict(n=n,k=k,packed=packed,program_key=program.key.hex(),record=record,negative_controls=5))
      rejects(lambda:codec.Encoder().node(UOp.new_buffer('AMD',16,dtypes.uint8)))
      rejects(lambda:codec.Decoder({}).value({'class':'__import__','fields':{}}))
      index_path=Path('/dev/shm/alpamayo-9060-fp8/index.json');index=json.loads(index_path.read_text())
      from amd_gemm import unpack
      def metadata_registry():
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
        inputs=(tensor((1,1,2112),dtypes.bfloat16),tensor((1,1,64,128),dtypes.bfloat16),tensor((1,1,64,128),dtypes.bfloat16))
        ret=(tensor((1,1,2112),dtypes.bfloat16),tensor((1,),dtypes.float32))
        registry=audit.model_registry(model,inputs,ret,bulk_index=index_path,
          vocab=SOURCE/'experiments/coc_vocab.json',checkpoint='/tmp/opencode/alpamayo-asius-20261114/model')
        return registry,model,inputs,ret
      left,model,inputs,ret=metadata_registry();right,*_=metadata_registry()
      contract=left.describe();audit.validate_binding_contract(contract,right.describe())
      for field,value in (('offset_bytes',1),('shape',[1]),('role','caller_mutable_kv'),('source',{'wrong_source':True}),
                          ('storage','storage.changed_alias')):
        bad=copy.deepcopy(contract);bad['owners'][0][field]=value
        rejects(lambda:audit.validate_binding_contract(contract,bad))
      bad=copy.deepcopy(contract);bad['scratch']=[dict(name='scratch.0',initialization_proved=False)]
      rejects(lambda:audit.validate_binding_contract(bad,bad))
      known_buffer=left.buffer(model.weights['vlm.model.language_model.layers.0.mlp.fc1.weight'].uop.base)
      assert known_buffer['owners']==['weight/vlm.model.language_model.layers.0.mlp.fc1.weight']
      unknown=UOp.new_buffer('AMD',512,dtypes.bfloat16);left.buffer(unknown)
      rejects(lambda:audit.validate_binding_contract(left.describe(),left.describe()))
      # Verify the exact frozen normalizer without loading any future image array.
      policy=local(SOURCE/'experiments/agent-paced-dynamic-policy.py');policy.verify_snapshot()
      with policy.imports(): replay=local(SOURCE/'experiments/agent-live-replay.py')
      pipeline=local(SOURCE/'experiments/benchmark-pipeline.py')
      dataset=run.first_dataset(replay,qualified)
      _,_,protocol=replay.normalize(SOURCE.parent/'pipeline-invocation.json',dataset,pipeline,cli.output.with_suffix('.unused.json'))
      if protocol!=qualified['protocol'] or len(dataset.rows)!=1: raise AssertionError('Diagnostic recipe or current-input selection changed')
      import tinygrad.engine.jit as jit_module
      spy=[]
      def unchanged(linear,*args,**kwargs): spy.append(len(linear.src));return linear
      inputs=[UOp.new_buffer('AMD',4,dtypes.float32)]
      linear=UOp(Ops.LINEAR,src=(program.call(*inputs),))
      with patch.object(jit_module,'compile_linear',unchanged),patch('tinygrad.engine.realize.lower_and_compile',lambda x:x):
        with audit.capture_boundary(inputs) as boundary:
          jit_module.compile_linear(linear,input_uops=inputs,beam=0,cache=False)
          jit_module.compile_linear(UOp(Ops.LINEAR))
        assert jit_module.compile_linear is unchanged and boundary['linear'] is linear and spy==[1,0]
      report.update(qualified_actual_gemv_programs=rows,actual_bulk_index_sha256=run.sha(index_path),
        actual_bulk_declared_bytes=index['nbytes'],metadata_owners=len(contract['owners']),
        metadata_storage_groups=len(contract['storages']),kv_capacity_bytes=sum(r['nbytes'] for r in contract['owners'] if r['role']=='caller_mutable_kv'),
        ownership_roles={role:sum(r['role']==role for r in contract['owners']) for role in {r['role'] for r in contract['owners']}},
        fresh_owner_metadata_equal=True,source_shape_offset_role_and_alias_mutations_rejected=True,
        unknown_owner_rejected=True,unproved_scratch_rejected=True,buffer_and_dynamic_class_payloads_rejected=True,
        frozen_recipe_preflight_equal=True,current_input_metadata_rows=1,future_image_arrays_loaded=0,
        scoped_capture_hook_and_empty_alias_compile_checked=True,
        note='Actual qualified GEMV compiler artifacts and actual FP8 index metadata; no CapturedJit was fabricated or executed')
  assert not Device._opened_devices
  report.update(devices=[],source_hashes={p.name:run.sha(p) for p in ROOT.glob('agent-plan-bind-*.py')})
  with cli.output.open('x') as stream: json.dump(report,stream,indent=2);stream.write('\n')
  print(json.dumps({k:v for k,v in report.items() if k not in ('qualified_actual_gemv_programs','source_hashes')},indent=2))


if __name__=='__main__': main()

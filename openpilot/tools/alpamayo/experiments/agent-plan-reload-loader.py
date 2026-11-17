"""Restricted reconstruction of the SHA-bound real count1 plan.

CPU construction is separate from explicit parent-only runtime compilation/linking.
No serialized process addresses, pickle, executable Python, or arbitrary plan inputs.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from tinygrad import Tensor,dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod
from tinygrad.uop.ops import UOp,Ops
from tinygrad.uop.movement import mop_cleanup

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
TG=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')

def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def runtime_sources():
  return {str(p.relative_to(TG)):sha(p) for p in sorted((TG/'tinygrad').rglob('*.py'))}

def source_gate(protocol):
  if runtime_sources()!=protocol['tinygrad_sources']: raise ValueError('Pinned runtime/compiler/ABI source changed')
  if {p.name:sha(p) for p in EXP.glob('agent-plan-reload-*.py')}!=protocol['reload_sources']:
    raise ValueError('Reviewed reload sources changed')
  if {p.name:sha(p) for p in EXP.glob('agent-plan-bind-*.py')}!=protocol['bind_sources']:
    raise ValueError('Frozen binding helpers changed')
  for path,digest in protocol['files'].items():
    if sha(path)!=digest: raise ValueError('Frozen factory/checkpoint/protocol identity changed: '+path)

def contract(inputs):
  info=[]
  for tensor in inputs:
    if tensor.uop.base.op is not Ops.BUFFER: raise ValueError('Input must have stable backing storage')
    view,bindings=tensor.uop.substitute({tensor.uop.base:UOp(Ops.NOOP)},extra_pm=mop_cleanup).unbind_all()
    if bindings: raise ValueError('Unexpected symbolic input layout')
    info.append((view,(),tensor.dtype,tensor.device))
  return info

def contract_json(info):
  return [dict(view_key=v.key.hex(),variable_keys=[u.key.hex() for u in vs],dtype=str(dt),device=dev) for v,vs,dt,dev in info]

def ownership(model,inputs,ret):
  owners={'weight/'+k:v for k,v in model.weights.items() if k.startswith('vlm.model.language_model.') or k=='vlm.lm_head.weight'}
  owners.update({'scale/'+k:v for k,v in model.weight_scales.items() if k.startswith('vlm.model.language_model.') or k=='vlm.lm_head.weight'})
  owners['mapping/output_token_mapping']=model.output_token_mapping
  owners.update({'kv/'+str(i):v for i,v in enumerate(model.cache)})
  owners.update(zip(('input/logits','input/cos','input/sin'),inputs,strict=True))
  owners.update(zip(('return/logits','return/token'),ret,strict=True))
  return owners

class Prepared:
  def __init__(self,document,programs,proof,model,inputs,*,bulk_index,vocab,checkpoint):
    inspect=local(EXP/'agent-plan-reload-inspect.py')
    if sha(inspect.CAPTURE)!=inspect.CAPTURE_SHA or document!=json.loads(inspect.CAPTURE.read_text()):
      raise ValueError('Only the exact reviewed actual plan is accepted')
    if set(programs)!=set(document['programs']) or any(p.key.hex()!=key for key,p in programs.items()):
      raise ValueError('PROGRAM identity changed')
    if not proof['proved'] or proof['calls']!=765: raise ValueError('No complete scratch proof')
    if len(document['calls'])!=765 or len(programs)!=29 or document['registry']['unresolved']:
      raise ValueError('Unexpected actual plan')
    self.document,self.programs,self.proof,self.model=document,programs,proof,model
    self.inputs=tuple(inputs);self.expected_info=contract(inputs)
    if contract_json(self.expected_info)!=document['input_contract'] or document['input_names']!=[0,1,2]:
      raise ValueError('Actual input contract differs')
    self.ret=tuple(Tensor(UOp.new_buffer('AMD',prod(row['shape']),getattr(dtypes,row['dtype']))).reshape(row['shape'])
                   for row in document['registry']['owners'] if row['name'] in ('return/logits','return/token'))
    # Original _greedy_decode returns token shape (1,1); BlockDecoder reshapes it
    # to (1,). Preserve that exact recorded view chain, including its layout key.
    self.ret=(self.ret[0],self.ret[1].reshape(1,1).reshape(1))
    bind=local(EXP/'agent-plan-bind-audit.py')
    self.registry=bind.model_registry(model,inputs,self.ret,bulk_index=bulk_index,vocab=vocab,checkpoint=checkpoint)
    actual=self.registry.describe();expected=document['registry']
    if actual['owners']!=expected['owners'] or actual['storages']!=expected['storages'] or actual['unresolved']:
      different=[dict(owner=a['name'],fields=[k for k in a if a[k]!=b.get(k)])
        for a,b in zip(actual['owners'],expected['owners']) if a!=b]
      raise ValueError('Closed owner, source, storage alias, layout, or dtype contract changed: '+str(different[:3]))
    self.owners=ownership(model,inputs,self.ret)
    self.owner_bases={name:t.uop.base for name,t in self.owners.items() if not name.startswith('input/')}
    self.owner_views={name:(tuple(t.shape),t.dtype,t.uop.substitute({t.uop.base:UOp(Ops.NOOP)}).key)
                      for name,t in self.owners.items() if name in self.owner_bases}
    self.rows={r['name']:r for r in expected['owners']}
    spec,=expected['scratch']
    if spec['nbytes']!=10499328 or spec['device']!='AMD' or spec['dtype']!='i8': raise ValueError('Scratch contract changed')
    self.scratch=UOp.alloc((spec['nbytes'],),dtypes.int8,device='AMD')
    self.variables={}
    for program in programs.values():
      for p in program.arg.params:
        if p.addrspace is not AddrSpace.ALU: continue
        value=dict(dtype=p.dtype.name,min=p.vmin,max=p.vmax)
        if not p.is_variable or p.is_bound_var: raise ValueError('Unreviewed runtime scalar ABI')
        if p.expr in self.variables and self.variables[p.expr]!=value: raise ValueError('Conflicting variable ranges')
        self.variables[p.expr]=value
    if self.variables!={'block_position':{'dtype':'i32','min':0,'max':2303},'block_step':{'dtype':'i32','min':0,'max':63}}:
      raise ValueError('Symbolic decoder variable contract changed')
    calls=[]
    for index,row in enumerate(document['calls']):
      program=programs[row['program_key']]
      if row['index']!=index or row['ordered_dependencies']: raise ValueError('Unreviewed outer ordering/dependency')
      if list(program.arg.outs)!=row['outputs'] or list(program.arg.ins)!=row['inputs']: raise ValueError('Program access ABI changed')
      args=tuple(self.argument(a) for a in row['arguments'])
      for p in program.arg.params:
        if p.addrspace is AddrSpace.ALU: continue
        if p.arg.slot>=len(args) or args[p.arg.slot].nbytes()!=p.max_numel()*p.dtype.itemsize or args[p.arg.slot].dtype!=p.dtype:
          raise ValueError('Argument extent/dtype differs from PROGRAM ABI')
      calls.append(program.call(*args))
    self.linear=UOp(Ops.LINEAR,src=tuple(calls));self.jit=None

  def argument(self,arg):
    if arg['kind']=='buffer':
      if len(arg['owners'])!=1: raise ValueError('Ambiguous owner')
      name,=arg['owners'];row=self.rows[name]
      if any(arg[k]!=row[k] for k in ('storage','offset_bytes','nbytes')) or arg['dtype']!=row['base_dtype']:
        raise ValueError('Unreviewed owner subview')
      return self.owners[name].uop.base
    if arg['kind']=='input':
      slot=arg['slot']
      if slot not in (0,1,2) or arg['device']!='AMD': raise ValueError('Unreviewed input slot/device')
      base=self.inputs[slot].uop.base
      if base.nbytes()!=arg['nbytes'] or base.dtype.name!=arg['dtype']: raise ValueError('Explicit input extent changed')
      return UOp.param(slot,base.dtype,base.max_numel(),'AMD')
    if arg=={'kind':'scratch','name':'scratch.0'}: return self.scratch
    if arg['kind']!='expression': raise ValueError('Unknown declarative argument')
    if arg['op']=='CONST' and not arg['src'] and type(arg['arg']) is int: return UOp.const(arg['arg'])
    if arg['op']=='SHRINK' and arg['arg'] is None and len(arg['src'])==3:
      base,start,size=(self.argument(a) for a in arg['src'])
      if start.op is not Ops.CONST or size.op is not Ops.CONST or start.arg<0 or start.arg+size.arg>base.max_numel():
        raise ValueError('Invalid scratch byte slice')
      return UOp(Ops.SHRINK,src=(base,start,size))
    if arg['op']=='BITCAST' and set(arg['arg'])=={'dtype'} and len(arg['src'])==1:
      dt=getattr(dtypes,arg['arg']['dtype'],None)
      if dt not in (dtypes.bfloat16,dtypes.float32,dtypes.int32,dtypes.fp8e4m3): raise ValueError('Unreviewed view dtype')
      return self.argument(arg['src'][0]).bitcast(dt)
    raise ValueError('Unreviewed outer expression')

  def guard(self):
    actual=ownership(self.model,self.inputs,self.ret)
    if any(actual[name].uop.base is not base for name,base in self.owner_bases.items()):
      raise ValueError('A closed weight, scale, KV, mapping, or return allocation was replaced')
    if any((tuple(actual[name].shape),actual[name].dtype,
            actual[name].uop.substitute({actual[name].uop.base:UOp(Ops.NOOP)}).key)!=view for name,view in self.owner_views.items()):
      raise ValueError('A closed owner view or dtype changed')

  def compile_runtime(self,*,parent_gpu_authorized=False):
    if parent_gpu_authorized is not True: raise ValueError('Explicit parent GPU execution is required')
    if self.jit is not None: raise ValueError('Runtime plan already compiled')
    from tinygrad.engine.jit import CapturedJit,_TinyJit
    from tinygrad.engine.realize import compile_linear
    self.guard()
    compiled=compile_linear(self.linear,beam=0,input_uops=[t.uop.base for t in self.inputs],cache=False)
    keys=[key.hex() for c in compiled.src for _,_,_,_,key,_,_ in c.without_after.arg.aux.kernels]
    if keys!=[c['program_key'] for c in self.document['calls']]: raise ValueError('HCQ changed the numerical PROGRAM identities')
    self.jit=_TinyJit(None,captured=CapturedJit(self.ret,compiled,[0,1,2],self.expected_info))
    return self

  def __call__(self,logits,cos,sin,position,step):
    if self.jit is None: raise ValueError('No parent-authorized runtime plan')
    self.guard()
    if type(position) is not int or not 0<=position<=2303 or type(step) is not int or not 0<=step<=63:
      raise ValueError('Decoder variable is outside its original range')
    return self.jit(logits,cos,sin,UOp.variable('block_position',0,2303).bind(position),UOp.variable('block_step',0,63).bind(step))

def build(model,inputs,*,bulk_index,vocab,checkpoint):
  inspect=local(EXP/'agent-plan-reload-inspect.py');proof_module=local(EXP/'agent-plan-reload-proof.py')
  document,programs=inspect.load();proof=proof_module.prove(document,programs)
  return Prepared(document,programs,proof,model,inputs,bulk_index=bulk_index,vocab=vocab,checkpoint=checkpoint)

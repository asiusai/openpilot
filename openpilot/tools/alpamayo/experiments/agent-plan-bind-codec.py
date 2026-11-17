"""Restricted, non-executing JSON codec for unlinked tinygrad PROGRAMs.

This reconstructs compiler artifacts for inspection only. It is not a plan loader.
No arbitrary classes, callbacks, pickle, Buffer objects, GETADDR, or device opens.
"""
import dataclasses
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import struct

from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.device import BufferSpec
from tinygrad.dtype import AddrSpace, ConstFloat, DType, Invalid, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer import Estimates
from tinygrad.uop.ops import AxisType, CallInfo, CustomFunction, KernelInfo, Ops, ParamArg, ProgramInfo, UOp

CLASSES={c.__name__:c for c in (Opt,BufferSpec,Target,Estimates,CallInfo,CustomFunction,KernelInfo,ParamArg,ProgramInfo)}
ENUMS={c.__name__:c for c in (Ops,AxisType,AddrSpace,OptOps)}
DTYPES={v.name:v for v in vars(type(dtypes)).values() if isinstance(v,DType)}
DTYPES.update({v.name:v for v in (*dtypes.all,*dtypes.weaks)})
TARGET=Target('AMD','LLVM','gfx1200','USB')
MAX_NODES=200000
MAX_BLOB=16*1024**2
MAX_JSON=64*1024**2


def sha(value): return hashlib.sha256(value).hexdigest()


def checked_dataclass(value):
  if type(value) is ParamArg and (value.buffer is not None or value.bind_on_realize):
    raise ValueError('Runtime storage is forbidden in a program artifact')
  if type(value) is BufferSpec and value.external_ptr is not None: raise ValueError('External address is forbidden')
  if type(value) is CallInfo and (value.grad_fxn is not None or value.aux is not None):
    raise ValueError('Callbacks and linked runtime context are forbidden')


class Encoder:
  def __init__(self): self.nodes=[];self.indices={};self.active=set();self.blobs={}

  def value(self,value):
    if isinstance(value,UOp): return {'node':self.node(value)}
    if value is Invalid: return {'invalid':True}
    if isinstance(value,Enum):
      if type(value).__name__ not in ENUMS or ENUMS[type(value).__name__] is not type(value): raise ValueError('Unknown enum')
      return {'enum':[type(value).__name__,value.name]}
    if isinstance(value,DType):
      if DTYPES.get(value.name) is not value: raise ValueError('Unknown dtype')
      return {'dtype':value.name}
    if isinstance(value,float): return {'float64':struct.pack('>d',value).hex(),'const':isinstance(value,ConstFloat)}
    if value is None or type(value) in (bool,int,str): return value
    if isinstance(value,tuple): return {'tuple':[self.value(v) for v in value]}
    if type(value).__name__ in CLASSES and CLASSES[type(value).__name__] is type(value):
      checked_dataclass(value)
      return {'class':type(value).__name__,'fields':{f.name:self.value(getattr(value,f.name)) for f in dataclasses.fields(value)}}
    raise ValueError('Unreviewed argument type: '+type(value).__name__)

  def node(self,u):
    if u in self.indices: return self.indices[u]
    if u in self.active: raise ValueError('Cyclic program')
    if u.op in (Ops.GETADDR,Ops.PYLITERAL) or (u.op is Ops.BUFFER and u.addrspace not in (AddrSpace.LOCAL,AddrSpace.REG)):
      raise ValueError('Runtime storage/address/Python node is forbidden')
    self.active.add(u)
    sources=[self.node(s) for s in u.src]
    if u.op is Ops.BINARY:
      if type(u.arg) is not bytes or len(u.arg)>MAX_BLOB: raise ValueError('Invalid binary')
      key=sha(u.arg);self.blobs[key]=u.arg;argument={'blob':key,'bytes':len(u.arg)}
    else: argument=self.value(u.arg)
    if len(self.nodes)>=MAX_NODES: raise ValueError('Program node limit')
    index=len(self.nodes);self.indices[u]=index
    self.nodes.append(dict(op=u.op.name,arg=argument,src=sources,key=u.key.hex(),dtype=u.dtype.name))
    self.active.remove(u)
    return index


class Decoder:
  def __init__(self,blobs): self.nodes=[];self.blobs=blobs

  def value(self,value):
    if value is None or type(value) in (bool,int,str): return value
    if type(value) is not dict: raise ValueError('Invalid declarative value')
    if set(value)=={'node'}:
      i=value['node']
      if type(i) is not int or not 0<=i<len(self.nodes): raise ValueError('Forward/cyclic node reference')
      return self.nodes[i]
    if set(value)=={'invalid'} and value['invalid'] is True: return Invalid
    if set(value)=={'dtype'}:
      if value['dtype'] not in DTYPES: raise ValueError('Unknown dtype')
      return DTYPES[value['dtype']]
    if set(value)=={'enum'}:
      cls,name=value['enum']
      if cls not in ENUMS or name not in ENUMS[cls].__members__: raise ValueError('Unknown enum')
      return ENUMS[cls][name]
    if set(value)=={'float64','const'}:
      if type(value['const']) is not bool or not re.fullmatch('[0-9a-f]{16}',value['float64']): raise ValueError('Bad float')
      result=struct.unpack('>d',bytes.fromhex(value['float64']))[0]
      return ConstFloat(result) if value['const'] else result
    if set(value)=={'tuple'}: return tuple(self.value(v) for v in value['tuple'])
    if set(value)=={'class','fields'}:
      if value['class'] not in CLASSES: raise ValueError('Unknown declarative class')
      cls=CLASSES[value['class']]
      if set(value['fields'])!={f.name for f in dataclasses.fields(cls)}: raise ValueError('Dataclass schema changed')
      result=cls(**{k:self.value(v) for k,v in value['fields'].items()})
      checked_dataclass(result)
      return result
    raise ValueError('Unknown declarative tag')

  def decode(self,document):
    if set(document)!={'schema','root','nodes','program_key'} or document['schema']!=1: raise ValueError('Unknown program schema')
    if not 0<len(document['nodes'])<=MAX_NODES: raise ValueError('Program node bound')
    for row in document['nodes']:
      if set(row)!={'op','arg','src','key','dtype'} or row['op'] not in Ops.__members__: raise ValueError('Unknown node schema')
      op=Ops[row['op']]
      if op in (Ops.GETADDR,Ops.PYLITERAL): raise ValueError('Storage/address/Python node is forbidden')
      src=tuple(self.value({'node':v}) for v in row['src'])
      if op is Ops.BINARY:
        value=row['arg']
        if set(value)!={'blob','bytes'} or not re.fullmatch('[0-9a-f]{64}',value['blob']): raise ValueError('Bad blob reference')
        argument=self.blobs[value['blob']]
        if len(argument)!=value['bytes'] or len(argument)>MAX_BLOB or sha(argument)!=value['blob']: raise ValueError('Binary identity changed')
      else: argument=self.value(row['arg'])
      if op is Ops.BUFFER and (not isinstance(argument,ParamArg) or argument.addrspace not in (AddrSpace.LOCAL,AddrSpace.REG)):
        raise ValueError('Runtime buffer is forbidden')
      node=UOp(op,src=src,arg=argument)
      if node.key.hex()!=row['key'] or node.dtype.name!=row['dtype']: raise ValueError('UOp or ABI identity changed')
      self.nodes.append(node)
    program=self.value({'node':document['root']})
    if program.op is not Ops.PROGRAM or program.key.hex()!=document['program_key']: raise ValueError('Program identity changed')
    if program.arg.target!=TARGET: raise ValueError('Full USB+AMD:LLVM:gfx1200 target is mandatory')
    return program


def encode(program):
  if program.op is not Ops.PROGRAM or not isinstance(program.arg,ProgramInfo) or program.arg.target!=TARGET:
    raise ValueError('Require an unlinked compiled gfx1200 PROGRAM with the full target')
  encoder=Encoder();root=encoder.node(program)
  document=dict(schema=1,root=root,nodes=encoder.nodes,program_key=program.key.hex())
  restored=Decoder(encoder.blobs).decode(document)
  if restored.key!=program.key or restored.arg!=program.arg: raise AssertionError('Offline artifact reconstruction differs')
  return document,encoder.blobs


def save(program,directory):
  directory=Path(directory);directory.mkdir(exist_ok=True)
  document,blobs=encode(program)
  raw=json.dumps(document,separators=(',',':'),allow_nan=False).encode()
  if len(raw)>MAX_JSON: raise ValueError('JSON artifact bound')
  name=program.key.hex()+'.json';path=directory/name
  if path.exists():
    if path.read_bytes()!=raw: raise ValueError('Existing program artifact differs')
  else:
    with path.open('xb') as stream: stream.write(raw)
  for key,value in blobs.items():
    path=directory/(key+'.elf')
    if path.exists():
      if path.read_bytes()!=value: raise ValueError('Existing binary artifact differs')
    else:
      with path.open('xb') as stream: stream.write(value)
  return dict(file=name,json_sha256=sha(raw),program_key=program.key.hex(),nodes=len(document['nodes']),
              binaries={key:len(value) for key,value in blobs.items()},abi=repr(program.arg))


def load_offline(directory,record):
  """Reconstruct only a PROGRAM for CPU inspection. Never allocate/link/execute it."""
  if record['file']!=record['program_key']+'.json' or not re.fullmatch('[0-9a-f]{64}',record['program_key']):
    raise ValueError('Invalid artifact filename')
  path=Path(directory)/record['file']
  if path.is_symlink() or path.stat().st_size>MAX_JSON: raise ValueError('Invalid artifact file')
  raw=path.read_bytes()
  if sha(raw)!=record['json_sha256']: raise ValueError('Source/UOp artifact digest changed')
  blobs={}
  for key,size in record['binaries'].items():
    if not re.fullmatch('[0-9a-f]{64}',key) or not 0<size<=MAX_BLOB: raise ValueError('Invalid binary declaration')
    path=Path(directory)/(key+'.elf')
    if path.is_symlink() or path.stat().st_size!=size: raise ValueError('Invalid binary file')
    blobs[key]=path.read_bytes()
  program=Decoder(blobs).decode(json.loads(raw))
  if program.key.hex()!=record['program_key'] or repr(program.arg)!=record['abi']: raise ValueError('ABI changed')
  return program

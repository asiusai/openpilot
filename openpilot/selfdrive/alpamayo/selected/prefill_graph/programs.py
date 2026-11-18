"""Read the checked numerical PROGRAM table; buffers and commands are bound separately."""
import gzip
import hashlib
import json
import struct
from functools import cache

from tinygrad.codegen import to_program_config
from tinygrad.dtype import AddrSpace, ConstFloat, Invalid
from tinygrad.uop.ops import CallInfo, CustomFunction, Ops, ParamArg, ProgramInfo, UOp

from ..frozen import ASSETS, TINYGRAD_REVISION, checked
from ..vision_pair.programs import CLASSES, DTYPES, ENUMS, TARGET, require


@cache
def selected_config():
  config = json.loads((ASSETS / 'environment.json').read_bytes())
  return tuple(config[v.key] for v in to_program_config)


def supported_compiler():
  return tuple(str(v.value) for v in to_program_config) == selected_config()


def load_programs(asset, digest, renderer):
  require(renderer.target == TARGET, 'Unsupported prefill renderer')
  plan = json.loads(gzip.decompress(checked(asset, digest)))
  require(plan['pin'] == TINYGRAD_REVISION and plan['target'] == str(TARGET), 'Prefill compiler pin or target differs')
  checked(ASSETS / 'manifest.json', plan['manifest_sha'])
  require(plan['config'] == {v.key: v.value for v in to_program_config}, 'Prefill compiler configuration differs')
  nodes, binaries = [], {key: bytes.fromhex(v) for key, v in plan['blobs'].items()}
  require(all(b.startswith(b'\x7fELF') and hashlib.sha256(b).hexdigest() == key for key, b in binaries.items()), 'Invalid program binary')

  def node(i):
    require(type(i) is int and 0 <= i < len(nodes), 'Invalid program node index')
    return nodes[i]

  def value(v):
    if not isinstance(v, dict):
      return v
    if 'node' in v:
      return node(v['node'])
    if 'dtype' in v:
      return DTYPES[v['dtype']]
    if 'enum' in v:
      name, member = v['enum']
      return ENUMS[name][member]
    if 'tuple' in v:
      return tuple(value(i) for i in v['tuple'])
    if 'float64' in v:
      f = struct.unpack('>d', bytes.fromhex(v['float64']))[0]
      return ConstFloat(f) if v['const'] else f
    if v == {'invalid': True}:
      return Invalid
    if 'class' in v:
      fields = {k: value(i) for k, i in v['fields'].items()}
      if v['class'] == 'CustomFunction':
        require(fields == {'name': 'llvm.amdgcn.ds.swizzle', 'dtype': DTYPES['i32']}, 'Unsupported program intrinsic')
        return CustomFunction(**fields)
      if v['class'] == 'CallInfo':
        require(fields == vars(CallInfo()), 'External callback in program')
        return CallInfo()
      obj = CLASSES[v['class']](**fields)
      require(not isinstance(obj, ParamArg) or (obj.buffer is None and not obj.bind_on_realize), 'Storage in program parameter')
      return obj
    raise ValueError('Unknown program field')

  for name, sources, argument in plan['nodes']:
    op = Ops[name]
    require(op not in (Ops.GETADDR, Ops.PYLITERAL), 'Address or Python literal in program')
    arg = binaries[argument['blob']] if op is Ops.BINARY else value(argument)
    require(op is not Ops.BUFFER or arg.addrspace in (AddrSpace.LOCAL, AddrSpace.REG), 'Runtime buffer in program')
    nodes.append(UOp(op, src=tuple(node(i) for i in sources), arg=arg))
  programs = {key: node(i) for key, i in plan['programs'].items()}
  require(all(p.op is Ops.PROGRAM and isinstance(p.arg, ProgramInfo) and p.arg.target == TARGET and p.key.hex() == key
              for key, p in programs.items()), 'Numerical program or ABI identity differs')
  return plan, programs

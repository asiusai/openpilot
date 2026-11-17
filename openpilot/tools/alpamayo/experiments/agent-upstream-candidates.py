"""Scoped, opt-in upstream backports for the pinned tinygrad process.

No device opens on import. Enter before model/JIT creation in a fresh process.
These changes have no GPU speed or route-quality qualification.
The linearizer scope is diagnostic: its upstream regression still fails here.
Source: https://github.com/tinygrad/tinygrad (MIT), commits recorded below.
The upstream notice is retained in agent-upstream-LICENSE.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import heapq
import inspect
from pathlib import Path
import runpy
import sys
from typing import Any

from tinygrad.dtype import AddrSpace
from tinygrad.helpers import TUPLE_ORDER, getenv
from tinygrad.uop.ops import Ops, UOp, multirange_str

PIN = '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
UPSTREAM = '5db89dca1e08012bcc6da699fe86898e24ee8027'
COMMITS = {
  'linearizer': 'cf335e26b7ab63aa5da9132f884e2b4bad7495b4',
  'recursive_properties': '802ca3abf9a0007e4a98ea7155488c59fc0fa146',
  'usb_typed_reads': '3f3718ce2c300705cdcb156284b36572384791e8',
}
PIN_FUNCTION_SHA256 = {
  'linearizer': 'd34e0ff4e34b3fac7f1cc2aa755a71b0111df42a0724c29d689de96efcb371da',
  'recursive_properties': '28887692fd6c941ce268de15ff52468d75720b3df2ddc9781d4459a4ece6d93e',
  'usb_typed_reads': '1582db882c06f902e4e876c2a09f644be9f0dc3b3960b2268124264d2d463963',
}


def upstream_linearize(sink:UOp) -> list[UOp]:
  # Body copied from cf335e26b7, unchanged at UPSTREAM.
  lst = list(sink.toposort())
  out_degree:defaultdict[UOp, int] = defaultdict(int)
  priorities:dict[UOp, tuple[int, int, Any]] = {}
  for u in reversed(lst):
    for s in u.src_without_body: out_degree[s] += 1
    extra = None
    match u.op:
      case Ops.PARAM: priority, extra = -20, u.arg.slot
      case Ops.BUFFER | Ops.ALLOC: priority = -17 if u.addrspace == AddrSpace.LOCAL else -18
      case Ops.LOAD: priority = -1
      case Ops.STORE: priority = 1
      case Ops.RANGE: priority = 5
      case Ops.END | Ops.BACKEDGE: priority = -5
      case _: priority = 0
    priorities[u] = (len(u.ranges), priority, extra)
  nkey = {u:i for i,u in enumerate(sorted(lst, key=lambda x: priorities[x]+(x.tuplize if TUPLE_ORDER else ())))}
  heap = [(-nkey[sink], sink)]
  newlst = []
  while heap:
    newlst.append(u:=heapq.heappop(heap)[1])
    for v in u.src_without_body:
      out_degree[v] -= 1
      if out_degree[v] == 0: heapq.heappush(heap, (-nkey[v],v))
  newlst = newlst[::-1]
  if getenv('DEBUG_LINEARIZE'):
    for i,u in enumerate(newlst):
      print(f"{i:4d} {str(u.op):20s} {multirange_str(u.ranges, color=True, pad=10)} {priorities[u]}")
  return newlst


def upstream_recursive_get(self, x:UOp|None, owner=None):
  # Body copied from 802ca3abf9, unchanged at UPSTREAM.
  if x is None: return self
  nm, stack = self.nm, [x]
  while stack:
    if nm in (d:=(node:=stack[-1]).__dict__): stack.pop()
    elif missing:=[s for s in node.src if nm not in s.__dict__]: stack.extend(reversed(missing))
    else:
      d[nm] = self.fxn(node)
      stack.pop()
  return x.__dict__[nm]


def upstream_usb_getitem(self, index):
  # Body copied from 3f3718ce2c, unchanged at UPSTREAM.
  off, sz = self._off_from_index(index)
  if self.pcimem:
    assert sz % 4 == 0 and off % 4 == 0, f'pcie_mem_read requires 4-byte aligned access, got off={off}, sz={sz}'
    data = self.usb.pcie_mem_read(self.addr + off, sz)
  else: data = self.usb.read(self.addr + off, sz)
  return (data if self.fmt == 'B' else memoryview(data).cast(self.fmt).tolist()) if isinstance(index, slice) else int.from_bytes(data, 'little')


@contextmanager
def scoped_backports(*, linearizer=False, recursive_properties=False, usb_typed_reads=False):
  """Restore functions and the in-memory program cache even after exceptions.

  Existing JIT/runner caches are not reusable across modes. Use a new process.
  This scope does not change USB submission order, arithmetic, or model inputs.
  """
  import tinygrad.codegen as codegen
  import tinygrad.codegen.late.linearizer as linearizer_module
  from tinygrad import Device
  from tinygrad.uop.ops import recursive_property
  from tinygrad.runtime.support.usb import USBMMIOInterface
  assert not [d for d in Device._opened_devices if d.split(':')[0] not in ('CPU', 'PYTHON', 'NPY')], 'Enter before opening a GPU'
  changes = []
  if linearizer:
    assert codegen.linearize is linearizer_module.linearize
    changes += [(linearizer_module, 'linearize', upstream_linearize, 'linearizer'),
                (codegen, 'linearize', upstream_linearize, 'linearizer')]
  if recursive_properties: changes.append((recursive_property, '__get__', upstream_recursive_get, 'recursive_properties'))
  if usb_typed_reads: changes.append((USBMMIOInterface, '__getitem__', upstream_usb_getitem, 'usb_typed_reads'))
  originals = [(obj, name, getattr(obj, name)) for obj, name, _, _ in changes]
  for (obj, name, old), (_, _, _, key) in zip(originals, changes):
    actual = hashlib.sha256(inspect.getsource(old).encode()).hexdigest()
    assert actual == PIN_FUNCTION_SHA256[key], f'Unexpected {key} source: {actual}'
  saved_cache = dict(codegen.to_program_cache)
  try:
    # The compiler cache key has no backport-mode field.
    codegen.to_program_cache.clear()
    for obj, name, replacement, _ in changes: setattr(obj, name, replacement)
    yield
  finally:
    for obj, name, old in reversed(originals): setattr(obj, name, old)
    codegen.to_program_cache.clear()
    codegen.to_program_cache.update(saved_cache)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--recursive-properties', action='store_true')
  parser.add_argument('--usb-typed-reads', action='store_true')
  parser.add_argument('script', type=Path)
  parser.add_argument('script_args', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  assert args.script.is_file(), args.script
  sys.argv = [str(args.script), *args.script_args]
  sys.path.insert(0, str(args.script.resolve().parent))
  with scoped_backports(recursive_properties=args.recursive_properties, usb_typed_reads=args.usb_typed_reads):
    runpy.run_path(str(args.script), run_name='__main__')


if __name__ == '__main__': main()

"""Offline inspection of the actual captured PROGRAM memory instructions."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
EXP=Path(__file__).resolve().parent
CAPTURE=WORK/'agent-plan-bind-capture1/agent-plan-bind-capture/audit.json'
CAPTURE_SHA='7fe7e733e3c53f5d7d67c7e5959d2f01896110c41263f75714d85f179cbaa275'

def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  return module

def load():
  if hashlib.sha256(CAPTURE.read_bytes()).hexdigest()!=CAPTURE_SHA: raise ValueError('Actual capture identity changed')
  document=json.loads(CAPTURE.read_text())
  codec=local(EXP/'agent-plan-bind-codec.py')
  if hashlib.sha256((EXP/'agent-plan-bind-codec.py').read_bytes()).hexdigest()!=document['source_sha256']['agent-plan-bind-codec.py']:
    raise ValueError('Frozen declarative decoder changed')
  programs={key:codec.load_offline(CAPTURE.parent/'programs',record) for key,record in document['programs'].items()}
  return document,programs

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  if not args.output.resolve().is_relative_to(WORK) or args.output.exists(): raise ValueError('Use a new WORK output')
  from tinygrad import Device
  from tinygrad.uop.ops import Ops
  def forbidden(*args,**kwargs): raise AssertionError('Device access forbidden')
  with patch.object(type(Device),'__getitem__',forbidden):
    document,programs=load();rows=[]
    for key,program in programs.items():
      ops=list(program.src[1].src);globals=[];scope=[]
      for i,u in enumerate(ops):
        if u.op in (Ops.IF,Ops.RANGE): scope.append(u)
        if u.op in (Ops.ENDIF,Ops.END): scope.pop()
        if u.op not in (Ops.LOAD,Ops.STORE): continue
        pointer=u.src[0]
        try: base=pointer.buf_uop
        except Exception: base=None
        if base is not None and base.op is Ops.PARAM:
          globals.append(dict(index=i,op=u.op.name,param=base.arg.slot,pointer=repr(pointer),shape=list(u.shape),
            dtype=u.dtype.name,value_dtype=u.src[1].dtype.name if u.op is Ops.STORE else None,
            scope=[repr(s) for s in scope]))
      rows.append(dict(key=key,name=program.src[0].arg.name,globals=globals,
        operations=dict(Counter(u.op.name for u in ops)),params=repr(program.arg),
        variables=[repr(p) for p in program.arg.params if p.is_variable]))
    report=dict(capture_sha256=CAPTURE_SHA,calls=len(document['calls']),programs=rows,
      outer_dependencies=[c for c in document['calls'] if c['ordered_dependencies']],
      argument_kinds=dict(Counter(a['kind'] for c in document['calls'] for a in c['arguments'])))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(programs=len(rows),global_memory_instructions=sum(len(r['globals']) for r in rows),
      output=str(args.output),devices=list(Device._opened_devices))))

if __name__=='__main__': main()

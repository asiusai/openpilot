"""Two private K64 configurations of the unchanged qualified packed adapter.

AMD source: ROCm/rocm-blogs d908d68d5741b0f449d7c31efa733fe8d4865b51.
No explicit upstream source license was found in the earlier inspection.
Private research only; no redistribution or publication is authorized here.
"""
from functools import lru_cache,partial
import importlib.util
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace

HERE=Path(__file__).resolve().parent


def load(name):
  spec=importlib.util.spec_from_file_location('agent_lds_small_'+name.replace('-','_'),HERE/(name+'.py'))
  value=importlib.util.module_from_spec(spec)
  sys.modules[spec.name]=value
  spec.loader.exec_module(value)
  return value


def private(fn,**replacements):
  result=FunctionType(fn.__code__,dict(fn.__globals__,**replacements),fn.__name__,fn.__defaults__,fn.__closure__)
  result.__kwdefaults__=fn.__kwdefaults__
  return result


H=load('agent-lds-gemm-check')
K=H.K
CONFIGS={'small128':(128,128,64,2,2),'small96':(128,96,64,2,2)}
source=private(K.source,CONFIGS=CONFIGS)
compiled=lru_cache(None)(private(K.compiled.__wrapped__,source=source))
program=private(K.program,CONFIGS=CONFIGS,compiled=compiled)
compiler_assembly=K.compiler_assembly
compile_one=private(H.compile_one,K=SimpleNamespace(CONFIGS=CONFIGS,compiler_assembly=compiler_assembly))


def functions(render,row_scale=False):
  return {'current':H.functions(render,row_scale)['current'],
    **{label:partial(program,mode=label,row_scale=row_scale) for label in CONFIGS}}

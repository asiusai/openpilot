"""One-line capture overlay: resolve actual bound position expressions with pinned tinygrad APIs."""
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ORIGINAL=HERE/'agent-distill-objective-capture.py'
ORIGINAL_SHA='7b045707be2ebec71ead5b4d6f881405e24939ed884532c38a8e36960eb213a6'


def resolved_position(value):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp,Ops
  if type(value) is int:result=value
  elif isinstance(value,UOp):
    if value.shape!=() or not dtypes.is_int(value.dtype):raise ValueError('Position must be a scalar integer expression')
    by_variable={};by_name={}
    for node in value.backward_slice_with_self:
      if node.op is not Ops.PARAM:continue
      if not node.is_bound_var:raise ValueError('Position contains an unbound parameter')
      variable,bound=node.unbind()
      if type(bound) is not int:raise ValueError('Noninteger position binding')
      if variable in by_variable and by_variable[variable]!=bound:raise ValueError('Conflicting bindings of one position variable')
      if variable.expr in by_name and by_name[variable.expr]!=variable:raise ValueError('Ambiguous position variable name')
      by_variable[variable]=bound;by_name[variable.expr]=variable
    expression,bindings=value.unbind_all()
    if bindings!=by_variable or set(expression.variables())!=set(bindings):raise ValueError('Unresolved position variables')
    result=expression.sym_infer({variable.expr:bound for variable,bound in bindings.items()})
  else:raise ValueError('Unsupported position type')
  if type(result) is not int or not 0<=result<2304:raise ValueError('Invalid resolved cache position')
  return result


def load():
  if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()!=ORIGINAL_SHA:raise ValueError('Original capture source changed')
  spec=importlib.util.spec_from_file_location('objective_position_original',ORIGINAL)
  original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
  source=inspect.getsource(original.aligned_targets)
  before='start=cache_pos if isinstance(cache_pos,int) else cache_pos.unbind()[1]'
  if source.count(before)!=1:raise ValueError('Capture position boundary changed')
  original.resolved_position=resolved_position
  exec(compile(source.replace(before,'start=resolved_position(cache_pos)'),str(ORIGINAL)+':bound-position-overlay','exec'),original.__dict__)
  return original


def main():load().main()


if __name__=='__main__':main()

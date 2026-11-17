"""Local lowering repair for a reshaped single-tile WMMA accumulator.

The pinned lowering pass handles WMMA and permuted WMMA additions. It misses
RESHAPE(WMMA) + running_accumulator in the 1x1 tile. Add that missing rule to
private function globals. Never replace tinygrad's global matchers or renderer.
"""
from functools import lru_cache
from types import FunctionType


@lru_cache(None)
def lowering():
  import tinygrad.codegen as codegen
  from tinygrad.uop.ops import Ops, PatternMatcher, UPat
  reshape_add = PatternMatcher([
    (UPat(Ops.RESHAPE, src=(UPat(Ops.WMMA, name='wmma'), UPat()), name='reshape') + UPat.var('add'),
      lambda wmma, reshape, add: (wmma+add.reshape(wmma.shape)).reshape(reshape.shape)),
  ])
  original = codegen.full_rewrite_to_sink
  private = FunctionType(original.__code__, dict(original.__globals__,
    pm_reduce_local=reshape_add+codegen.pm_reduce_local,
    pm_expand_broadcast=reshape_add+codegen.pm_expand_broadcast), original.__name__, original.__defaults__)
  private.__kwdefaults__ = original.__kwdefaults__
  return private


def kernel(*args, raw_kernel, renderer):
  from tinygrad.codegen import to_program
  from tinygrad.uop.ops import Ops, ProgramInfo, UOp
  ast = raw_kernel(*args)
  # The original scheduled AST and scale/bias epilogue stay intact.
  sink = lowering()(ast, renderer, optimize=not ast.arg.optimized)
  return to_program(UOp(Ops.PROGRAM, src=(sink,), arg=ProgramInfo.from_sink(sink, renderer.target)), renderer)

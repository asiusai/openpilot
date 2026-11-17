"""Bounded gather/scatter primitives for CPU cache-addressing checks only.

No installation hook. These do not make the FastV selection policy causal.
Scatter skips negative padding slots and preserves all other original slots.
"""
from tinygrad import Tensor
from tinygrad.uop.ops import KernelInfo, UOp


def _gather(out, source, slots):
  rows, capacity, width = out.shape[-2], source.shape[-2], source.shape[-1]
  count = out.max_numel()
  i = UOp.range(count, 0)
  plane, row, column = i//(rows*width), i//width%rows, i%width
  slot = slots.flatten()[row]
  value = source.flatten().index((plane*capacity*width+slot*width+column).valid((slot >= 0)&(slot < capacity)))
  return out.flatten()[i].store(value).end(i).sink(arg=KernelInfo(name='agent_token_prune_gather', opts_to_apply=()))


def _scatter(out, source, slots):
  rows, capacity, width = source.shape[-2], out.shape[-2], out.shape[-1]
  i = UOp.range(source.max_numel(), 0)
  plane, row, column = i//(rows*width), i//width%rows, i%width
  slot = slots.flatten()[row]
  target = out.flatten().index((plane*capacity*width+slot*width+column).valid((slot >= 0)&(slot < capacity)))
  return target.store(source.flatten()[i]).end(i).sink(arg=KernelInfo(name='agent_token_prune_scatter', opts_to_apply=()))


def _language_mask(out, prefix_valid, limits):
  capacity = out.shape[-1]
  i = UOp.range(out.max_numel(), 0)
  key = i%capacity
  valid = (prefix_valid.flatten()[i] | (key >= limits[0])) & (key <= limits[1])
  return out.flatten()[i].store(valid.where(0.0, -float('inf'))).end(i).sink(
    arg=KernelInfo(name='agent_token_prune_mask', opts_to_apply=()))


def gather(source, slots):
  assert slots.ndim == 1 and source.ndim >= 2
  out = Tensor.empty(*source.shape[:-2], slots.shape[0], source.shape[-1], device=source.device, dtype=source.dtype)
  return Tensor.custom_kernel(out, source, slots, fxn=_gather)[0]


def scatter(cache, values, slots):
  assert slots.ndim == 1 and values.shape[-2] == slots.shape[0]
  assert cache.shape[:-2] == values.shape[:-2] and cache.shape[-1] == values.shape[-1] and cache.dtype == values.dtype
  # The caller must validate uniqueness of all nonnegative slots before dispatch.
  return Tensor.custom_kernel(cache, values, slots, fxn=_scatter)[0]


def language_mask(prefix_valid, prompt_and_current):
  from tinygrad import dtypes
  assert prefix_valid.ndim == 2 and prefix_valid.shape[0] == 28 and prompt_and_current.shape == (2,)
  out = Tensor.empty(*prefix_valid.shape, device=prefix_valid.device, dtype=dtypes.float32)
  return Tensor.custom_kernel(out, prefix_valid, prompt_and_current, fxn=_language_mask)[0]

"""Exact poison transfer through the pinned normal Buffer.copy_from API."""
from tinygrad import Device
from tinygrad.device import Buffer


def poison_copy(buffer,pattern):
  if not isinstance(buffer,Buffer) or buffer.nbytes<=0: raise ValueError('Require a nonempty owned Buffer')
  if type(pattern) is not int or pattern not in (0xA5,0x5A): raise ValueError('Preserve the two reviewed poison patterns')
  target=buffer.ensure_allocated()
  # initial_value copies immutable bytes into writable host staging storage.
  # Retain this Buffer until the destination device has completed the transfer.
  staging=Buffer('CPU',target.nbytes,initial_value=bytes([pattern])*target.nbytes)
  try:
    target.copy_from(staging)
  finally:
    Device[target.device].synchronize()
  return target

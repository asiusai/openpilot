"""Narrow instance-only experiment: capture serial lengths 2..6 on first use.

Length one must already have normal capture, replay, and output-alias linking.
All existing warm shapes, replays, KV restoration, and layout gates remain.
No graph is deserialized. No tinygrad class or module function is patched.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

TARGETS = (2, 3, 4, 5, 6)


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_signature(jit):
  """Stable compiled-kernel sequence and input contract, excluding addresses.

HCQ kernel metadata deliberately has no argument slots for some owned buffers.
Do not infer storage dependencies from those empty slots. Route parity remains
mandatory; this signature is an additional no-new-kernels check, not its substitute.
"""
  from tinygrad.runtime.support.hcq2 import HCQInfo
  from tinygrad.uop.ops import Ops
  captured = jit.captured
  if captured is None:
    raise ValueError('Require an actual captured JIT')
  kernels = []
  for call in captured._linear.src:
    call = call.without_after
    if isinstance(call.arg.aux, HCQInfo):
      for dev, name, estimates, stamps, key, bufs, io in call.arg.aux.kernels:
        kernels.append(dict(device=dev, name=name, program_key=None if key is None else key.hex(),
                            output_indices=list(io[0]), input_indices=list(io[1])))
    elif call.body.op is Ops.PROGRAM:
      kernels.append(dict(device=str(call.device), name=call.body.src[0].arg.name,
        program_key=call.body.key.hex(), output_indices=list(call.body.arg.outs), input_indices=list(call.body.arg.ins)))
    elif call.body.op is Ops.STORE:
      kernels.append(dict(device=str(call.device), copy=True))
    else:
      raise ValueError(f'Unreviewed execution-plan call: {call.body.op}')
  contract = [dict(view_key=view.key.hex(), variable_keys=[v.key.hex() for v in variables], dtype=str(dtype), device=device)
              for view,variables,dtype,device in captured.expected_input_info]
  value = dict(kernels=kernels, input_names=captured.expected_names, input_contract=contract)
  return dict(**value, digest=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),
              linked_alias_variants=len(captured._linears))


class Observed:
  """Per-instance host-call observation. This is not kernel or exclusive phase time."""
  def __init__(self, inner, key, report):
    self.inner, self.key, self.report = inner, key, report

  @property
  def cnt(self): return self.inner.cnt

  @property
  def captured(self): return self.inner.captured

  def __call__(self, *args, **kwargs):
    phase = 'eager' if self.inner.cnt == 0 else 'capture' if self.inner.cnt == 1 else 'replay'
    started = time.perf_counter()
    row = dict(count=self.key[1], phase=phase, success=False)
    try:
      result = self.inner(*args, **kwargs)
      row['success'] = True
      return result
    finally:
      row['host_call_ms'] = (time.perf_counter()-started)*1000
      self.report['calls'].append(row)


class CaptureMap(dict):
  def __init__(self, values, enabled, report):
    super().__init__(values)
    self.enabled, self.report = enabled, report
    if values:
      raise ValueError('Install before any serial graph is created')

  def __setitem__(self, key, jit):
    if key in self or not isinstance(key,tuple) or len(key)!=3 or key[1] not in range(1,7):
      raise ValueError('Unexpected serial graph replacement or shape')
    if jit.cnt != 0 or jit.captured is not None or jit.prune:
      raise ValueError('Only a fresh ordinary TinyJit can enter direct capture')
    reference = self.get((key[0],1,key[2]))
    primed = reference is not None and reference.cnt >= 3 and reference.captured is not None
    primed = primed and len(reference.captured._linears) >= 2
    armed = bool(self.enabled and key[1] in TARGETS and primed)
    self.report['created'].append(dict(count=key[1], direct_capture=armed,
      reason='normal length-one and output-alias primed' if armed else 'ordinary first call retained'))
    if armed:
      # Pinned _TinyJit.__call__: cnt==1 traces AND executes the capture.
      # It is a phase selector, not proof that an earlier call happened.
      jit.cnt = 1
    super().__setitem__(key, Observed(jit,key,self.report))


@contextmanager
def install(decoder, enabled, closed_graphs):
  """Restore original objects on failure; retain only ordinary TinyJits on success."""
  original = decoder.jits
  report = dict(schema=1, variant='serial-first-capture' if enabled else 'ordinary', created=[], calls=[],
    targets=list(TARGETS), shapes_removed=[], persistent_cache=False,
    timing_kind='host call duration; eager/capture includes Python, compilation, linking, and internal waits; not exclusive GPU time')
  tracking = CaptureMap(original,enabled,report)
  decoder.jits = tracking
  success = False
  try:
    yield report
    values = dict(decoder.jits)
    if sorted(key[1] for key in values) != list(range(1,7)):
      raise AssertionError('Every original serial length 1..6 must remain warmed')
    signatures = {}
    for key, observed in values.items():
      if not isinstance(observed,Observed) or observed.cnt < 3:
        raise AssertionError('Unexpected or unwarmed serial execution plan')
      signatures[str(key[1])] = graph_signature(observed.inner)
      if signatures[str(key[1])]['linked_alias_variants'] < 2:
        raise AssertionError('Original output-alias link is missing')
    report['serial_signatures'] = signatures
    report['actual_successful_calls'] = len([r for r in report['calls'] if r['success']])
    report['skipped_eager_lengths'] = [r['count'] for r in report['created'] if r['direct_capture']]
    report['phase_host_ms'] = {p:sum(r['host_call_ms'] for r in report['calls'] if r['phase']==p)
                               for p in ('eager','capture','replay')}
    # No observer or alternative execution algorithm survives initialization.
    decoder.jits = closed_graphs({key:value.inner for key,value in values.items()})
    success = True
  finally:
    if not success:
      decoder.jits = original


def compare_signatures(before, after):
  if before['variant'] != 'ordinary' or after['variant'] != 'serial-first-capture':
    raise ValueError('Require the paired original and direct-capture variants')
  if after['skipped_eager_lengths'] != list(TARGETS):
    raise ValueError('This first input did not qualify every requested direct-capture length')
  if before['serial_signatures'] != after['serial_signatures']:
    raise AssertionError('Compiled serial kernel sequence, input contract, or alias coverage changed')
  for count in range(1,7):
    rows = [r for r in after['calls'] if r['count']==count]
    if not rows or not all(r['success'] for r in rows) or sum(r['phase']=='capture' for r in rows)!=1:
      raise AssertionError('Missing successful capture for a serial shape')
    if sum(r['phase']=='replay' for r in rows)<2:
      raise AssertionError('Both ordinary replay and output-alias replay remain mandatory')
  return dict(exact_kernel_sequences=True, exact_input_contracts=True, alias_links_preserved=True,
              shapes=list(range(1,7)), skipped_eager_lengths=list(TARGETS))

"""Source-bound paired route execution. GPU execution belongs to the parent."""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import statistics
import sys

HERE = Path(__file__).resolve().parent


def sha(raw): return hashlib.sha256(raw).hexdigest()


def require(value, message):
  if not value: raise ValueError(message)


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), HERE/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def option(command, flag, default=None):
  require(command.count(flag) <= 1, f'Duplicate {flag}')
  return command[command.index(flag)+1] if flag in command else default


def replace(command, flag, value):
  option(command, flag)
  if flag in command: command[command.index(flag)+1] = str(value)
  else: command.extend((flag, str(value)))


def inspect(path):
  """Read every nested invocation; validate effective settings before freezing."""
  chain, visited = [], set()
  for _ in range(8):
    path = Path(path).resolve()
    require(path not in visited, 'Invocation cycle')
    visited.add(path)
    raw = path.read_bytes()
    row = json.loads(raw)
    command = row['command'][1:]
    program = Path(command[0]).resolve()
    require(program.is_file(), 'Missing source program')
    for word in command:
      require(word.split('=')[0] not in ('--vision-calibration', '--capture-vision-scales', '--audit',
        '--profile-kernels', '--profile-host', '--row-quantize', '--online-expert', '--front-history-indices',
        '--camera-pixel-scales', '--phrase-drafts'), 'Rejected fixed vision, reduced input, or diagnostic configuration')
    for key, value in [('AM_POWER_LIMIT', '144'), ('AM_PERF_LEVEL', 'auto'), ('TC_OPT', '2')]:
      require(row['environment'].get(key) == value, f'Unexpected {key}')
    chain.append(dict(path=str(path), sha256=sha(raw), command=command, environment=row['environment']))
    if program.name == 'benchmark-pipeline.py': break
    require(program.name in ('agent-speculation-run.py', 'agent-vision-tiles-route.py'), 'Unsupported wrapper')
    if program.name == 'agent-speculation-run.py':
      require(option(command, '--mode', 'candidate') == 'candidate'
              and option(command, '--prefill-attention', 'baseline') == 'shuffle-fma', 'Keep nominal prefill and shuffle-FMA')
    path = option(command, '--invocation')
    require(path, 'Missing nested invocation')
  else: raise ValueError('Invocation nesting too deep')
  terminal = chain[-1]['command']
  require(any(Path(row['command'][0]).name == 'agent-speculation-run.py' for row in chain), 'Nominal prefill wrapper required')
  def effective(flag, default):
    for row in chain:
      c = row['command']
      if flag in c: return option(c, flag)
      if flag == '--pad-prompt-to' and Path(c[0]).name == 'agent-speculation-run.py': return '2112'
    return default
  require(effective('--pad-prompt-to', None) == '2048', 'Require effective padding 2048')
  for flag, default, expected in [('--steps', '10', '3'), ('--trajectory-solver', 'euler', 'euler'),
      ('--pixel-scale', '1', '1'), ('--history-pixel-scale', '1', '1'), ('--history-feature-scale', '1', '1'),
      ('--bf16-vision-layers', '0', '2'), ('--vision-group-limit', '0', '2'), ('--lookup-fallback', '6', '6')]:
    require(option(terminal, flag, default) == expected, f'Unexpected {flag}')
  require(terminal[terminal.index('--draft-tile')+1:terminal.index('--draft-tile')+3] == ['1', '4'], 'Require baseline decoder 1x4')
  for flag in ('--prefill-calibration', '--packed-quantize', '--quantize-reduction', '--grouped-quantize-reduction',
      '--batched-vision', '--online-attention', '--expert-kernels', '--llvm-wgp', '--lookup-draft', '--lookup-carry',
      '--compact-draft-io', '--direct-readback', '--initial-noise'):
    require(flag in terminal, f'Missing {flag}')
  return chain


def verify_protocol(protocol):
  for filename, expected in protocol['files'].items():
    require(sha(Path(filename).read_bytes()) == expected, f'Frozen file changed: {filename}')
  chain = inspect(protocol['invocation'])
  require(chain == protocol['chain'], 'Frozen invocation chain changed')
  require(protocol['warmup']+protocol['runs'] == len(protocol['samples']) and protocol['samples'], 'Incomplete input schedule')
  return chain


class AuditJit:
  """Observe outputs after real captured calls. Audit timings are never accepted."""
  def __init__(self, jit, decoder, records): self.jit, self.decoder, self.records = jit, decoder, records
  def __getattr__(self, name): return getattr(self.jit, name)
  def __call__(self, tokens, cos, sin, position, step):
    result = self.jit(tokens, cos, sin, position, step)
    import numpy as np
    def fingerprint(value):
      array = self.decoder.readback(value.float().contiguous().realize())
      require(np.isfinite(array).all(), 'Nonfinite verifier audit value')
      return dict(shape=list(array.shape), float32_bytes_sha256=sha(array.astype('<f4', copy=False).tobytes()))
    pos, offset, count = position.unbind()[1], step.unbind()[1], tokens.shape[0]
    self.records.append(dict(rows=count, position=pos, step=offset, input_tokens=fingerprint(tokens),
      cos=fingerprint(cos[:, :, offset:offset+count]), sin=fingerprint(sin[:, :, offset:offset+count]),
      logits=fingerprint(result[0]), predicted_tokens=fingerprint(result[1]),
      written_kv=[fingerprint(cache[:, :, :, pos:pos+count]) for cache in self.decoder.model.cache]))
    return result


class AuditVerifiers(dict):
  def __init__(self, decoder, records): super().__init__(); self.decoder, self.records = decoder, records
  def __setitem__(self, key, jit): super().__setitem__(key, AuditJit(jit, self.decoder, self.records))


@contextmanager
def intercept(expected, hook, permitted, variant, audit, records):
  original_spec = importlib.util.spec_from_file_location
  with ExitStack() as stack:
    def spec(name, location, *args, **kwargs):
      result = original_spec(name, location, *args, **kwargs)
      if Path(location).resolve() != expected.resolve(): return result
      loader = result.loader
      class Loader:
        def create_module(self, value): return loader.create_module(value)
        def exec_module(self, module):
          loader.exec_module(module)
          original = module.LookupDecoder
          class Decoder(original):
            # Pipeline sets tile and readback after construction. Install at the
            # first run, before prefill speculation can reach any verifier.
            def run(self, *values, **options):
              if not hasattr(self, '_agent_decode_fast_record'):
                require(not self.verifiers, 'Verifier already captured')
                record = self._agent_decode_fast_record = {'audit': [], 'variant': variant}
                records.append(record)
                if variant == 'candidate':
                  from tinygrad import Device
                  device = Device[Device.DEFAULT]
                  require(device.arch == 'gfx1200' and device.is_usb, 'Wrong GPU target')
                  require(type(device.renderer.compiler).__name__ == 'WGPCompiler', 'WGP compiler required')
                  record['hook'] = stack.enter_context(hook.scoped(self, permitted, renderer=device.renderer))
                if audit:
                  previous = self.verifiers
                  self.verifiers = AuditVerifiers(self, record['audit'])
                  def restore():
                    for jit in self.verifiers.values(): jit.reset()
                    self.verifiers.clear()
                    self.verifiers = previous
                  stack.callback(restore)
              return super().run(*values, **options)
          module.LookupDecoder = Decoder
      result.loader = Loader()
      return result
    importlib.util.spec_from_file_location = spec
    try: yield
    finally: importlib.util.spec_from_file_location = original_spec


def compare(before, after, protocol):
  import numpy as np
  old, new = before['warmup']+before['runs'], after['warmup']+after['runs']
  for rows in (old, new): require([row['sample'] for row in rows] == protocol['samples'], 'Sample schedule differs')
  for a, b in zip(old, new, strict=True):
    x, y = a['prediction'], b['prediction']
    require(x['generated_token_ids'] == y['generated_token_ids'] and x['reasoning'] == y['reasoning'], 'Reasoning/tokens differ')
    require(a.get('draft') == b.get('draft'), 'Acceptance, serial fallback, final cache length, or rotary offset differs')
    for field in ('xyz', 'rot'):
      u, v = np.asarray(x[field], np.float64), np.asarray(y[field], np.float64)
      require(u.shape == v.shape and u.size and np.isfinite(u).all() and np.isfinite(v).all(), 'Invalid trajectory')
      require(np.array_equal(u.view(np.uint64), v.view(np.uint64)), 'Trajectory/rotation bits differ')
  if protocol['audit']:
    a, b = before['agent_decode_fast']['decoders'][0]['audit'], after['agent_decode_fast']['decoders'][0]['audit']
    require(a and a == b, 'Verifier logits, tokens, mRoPE, or written K/V differ; empty audit is insufficient')
  measured = []
  if not protocol['audit']:
    for a, b in zip(before['runs'], after['runs'], strict=True):
      measured.append(dict(sample=a['sample'], lookup_rounds=b.get('draft', {}).get('lookup_rounds', 0),
        pipeline_delta_ms=a['pipeline_ms']-b['pipeline_ms'], service_delta_ms=a['service_ms']-b['service_ms']))
  return dict(all_exact=True, samples=len(old), acceptance_fallback_exact=True, verifier_audit_exact=protocol['audit'],
              measured_pairs=measured, median_pipeline_saving_ms=statistics.median(r['pipeline_delta_ms'] for r in measured) if measured else None)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--protocol', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--variant', choices=('current', 'candidate'), required=True)
  p.add_argument('--reference', type=Path)
  p.add_argument('--quality-report', type=Path, help='Completed exact 64-sample audit candidate, required before measured runs')
  p.add_argument('--plan', action='store_true')
  args = p.parse_args()
  raw = args.protocol.read_bytes(); protocol = json.loads(raw)
  chain = verify_protocol(protocol)
  environment = chain[-1]['environment']
  hook = load('agent-decode-fast-hook')
  permitted, admitted = hook.evidence(protocol['evidence'], protocol['choices'], environment)
  # Recheck the selected cache records on CPU immediately before any GPU path.
  check = hook.load('agent-decode-gemm')
  terminal = chain[-1]['command']
  for row in json.loads(Path(protocol['evidence']).read_text())['cases']:
    if tuple(row['shape_mnk']) not in permitted: continue
    identity = check.model_fixture(row['case'], Path(environment['ALPAMAYO_FP8_CACHE']),
      Path(option(terminal, '--checkpoint')), Path(option(terminal, '--reduced-vocab')))['identity']
    require(identity == row['model_identity'], 'Selected model cache record changed')
  info = dict(protocol_sha256=sha(raw), family_sha256=protocol['family_sha256'], samples=len(protocol['samples']),
    variant=args.variant, plan_only=args.plan, matrix_evidence=admitted,
    audit=protocol['audit'], timings_valid_for_throughput=not protocol['audit'], route_completed=False, exactness_validated=False, decoders=[])
  reference = json.loads(args.reference.read_text()) if args.reference else None
  if reference is not None:
    old = reference['agent_decode_fast']
    require(old['variant'] == 'current' and old['route_completed'] and 'failure' not in old
      and old['protocol_sha256'] == info['protocol_sha256'], 'Reference protocol differs or incomplete')
  if args.plan:
    args.output.write_text(json.dumps(dict(info, gpu_executed=False), indent=2)+'\n')
    print(json.dumps(dict(plan_only=True, samples=len(protocol['samples']), selected=protocol['choices'])))
    return
  if not protocol['audit']:
    require(args.quality_report is not None, 'Measured runs require a completed exact quality audit')
    quality_raw = args.quality_report.read_bytes()
    quality = json.loads(quality_raw)['agent_decode_fast']
    require(quality['variant'] == 'candidate' and quality['audit'] and quality['route_completed']
      and quality['exactness_validated'] and quality['samples'] == 64 and 'failure' not in quality
      and quality['family_sha256'] == protocol['family_sha256'], 'Quality audit failed or sources/choices changed')
    info['quality_report_sha256'] = sha(quality_raw)
  require(args.variant == 'current' or reference is not None, 'Candidate needs matched current reference')
  command = chain[0]['command'].copy(); replace(command, '--output', args.output)
  old_environment, old_argv, old_path = os.environ.copy(), sys.argv[:], sys.path[:]
  args.output.parent.mkdir(parents=True, exist_ok=True)
  try:
    for key, value in environment.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    # Source-bound protocol compilation defaults cannot inherit a hidden tuning override.
    os.environ.update(BEAM='0', JITBEAM='0', PARALLEL='0', PROFILE='0', DEBUG='0')
    os.environ.pop('LLVMOPT', None)
    sys.path.insert(0, str(Path(chain[-1]['command'][0]).parent.parent))
    sys.argv = command
    with intercept(Path(chain[-1]['command'][0]).with_name('lookup-decode.py'), hook, permitted,
                   args.variant, protocol['audit'], info['decoders']):
      runpy.run_path(command[0], run_name='__main__')
    require(len(info['decoders']) == 1, 'Expected one actual lookup decoder')
    if args.variant == 'candidate': require(info['decoders'][0]['hook']['python_trace_calls'], 'Selected GEMMs never traced')
    result = json.loads(args.output.read_text())
    require(len(result['warmup']) == protocol['warmup'] and len(result['runs']) == protocol['runs'], 'Incomplete route')
    require([r['sample'] for r in result['warmup']+result['runs']] == protocol['samples'], 'Wrong input schedule')
    result['agent_decode_fast'] = info
    if reference is not None:
      info['comparison'] = compare(reference, result, protocol)
      info['exactness_validated'] = True
    info['route_completed'] = True
    if protocol['audit']: result['summary']['profiling_enabled'] = True
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
  except BaseException as exc:
    info['failure'] = f'{type(exc).__name__}: {exc}'
    raise
  finally:
    os.environ.clear(); os.environ.update(old_environment)
    sys.argv, sys.path = old_argv, old_path
    args.output.with_suffix('.agent-decode-fast.json').write_text(json.dumps(info, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()

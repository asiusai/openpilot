"""Strict external adaptation of the frozen offline normalizer for dynamic vision.

No initializer, decoder, padding implementation, or inference hook is replaced.
The original normalization function is compiled with three exact, hash-pinned
text substitutions. Its original sample, flag, runtime, and verification checks
remain active. Both reference/Consumer parity and paced replay use this adapter.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
REPLAY_SHA = '4aac2b9b092c2c1b90a881b17dc1ab2f8e2be39fe0e5eb4d044eaa26438d22ec'
POLICY = 'dynamic-vision-pad2048-nominal-verified-prefill-shuffle-fma-v1'
SUBSTITUTIONS = (
  ('not args.lookup_draft or args.vision_calibration is None or args.reduced_vocab is None',
   'not args.lookup_draft or args.vision_calibration is not None or args.reduced_vocab is None'),
  ('Preserve the selected lookup runner, reduced vocabulary, and fixed vision calibration',
   'Preserve the selected lookup runner, reduced vocabulary, and dynamic vision'),
  ("('initial_noise', 'prefill_calibration', 'vision_calibration', 'reduced_vocab')",
   "('initial_noise', 'prefill_calibration', 'reduced_vocab')"),
)


def sha(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT/(name+'.py'))
  value = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


def manifest():
  path = ROOT.parent.parent/'manifest.json'
  value = json.loads(path.read_text())
  if value['policy'] != POLICY or Path(value['source_root']).resolve() != ROOT.parent:
    raise ValueError('Run the prepared immutable dynamic source snapshot')
  return path, value


def verify_snapshot():
  path, value = manifest()
  base = path.parent
  actual = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and p != path}
  if actual != set(value['files']):
    raise ValueError('Immutable snapshot file inventory changed')
  for name, expected in value['files'].items():
    entry = base/name
    if entry.is_symlink() or sha(entry) != expected or entry.stat().st_mode & 0o222:
      raise ValueError(f'Immutable snapshot changed or writable: {name}')
  if path.stat().st_mode & 0o222:
    raise ValueError('Snapshot manifest must be read-only')
  return value


def selected_args(args):
  if args.vision_calibration is not None or args.pad_prompt_to != 2048:
    raise ValueError('Selected dynamic vision and padding 2048 are mandatory')
  if (args.steps != 3 or args.prefill_calibration is None or args.verify_vision_warmup or
      args.front_history_indices is not None or args.pixel_scale != 1 or args.history_pixel_scale != 1 or
      args.history_feature_scale != 1):
    raise ValueError('Preserve nominal fixed prefill, three steps, complete history, and continuous verification policy')


def adapt(replay):
  if sha(replay.__file__) != REPLAY_SHA:
    raise ValueError('Frozen normalizer source differs; review the adapter explicitly')
  if getattr(replay, '_dynamic_policy', None) is not None:
    raise ValueError('Dynamic adapter cannot be installed twice')
  source = inspect.getsource(replay.normalize)
  for before, after in SUBSTITUTIONS:
    if source.count(before) != 1:
      raise ValueError('Frozen normalizer substitution is ambiguous')
    source = source.replace(before, after, 1)
  namespace = dict(replay.__dict__)
  exec(compile(source, str(ROOT/'agent-paced-dynamic-policy.py')+':adapted-normalize', 'exec'), namespace)
  original = namespace['normalize']
  transformation = dict(policy=POLICY, frozen_replay_sha256=REPLAY_SHA, adapter_sha256=sha(__file__),
                        adapted_normalize_sha256=hashlib.sha256(source.encode()).hexdigest(),
                        substitutions=[dict(before=a, after=b) for a, b in SUBSTITUTIONS])

  def normalize(invocation, dataset, pipeline, output):
    manifest_path, declared = manifest()
    expected = manifest_path.parent/'pipeline-invocation.json'
    if Path(invocation).resolve() != expected or sha(invocation) != declared['files']['pipeline-invocation.json']:
      raise ValueError('Use the exact prepared pipeline invocation')
    normalized, args, protocol = original(invocation, dataset, pipeline, output)
    selected_args(args)
    protocol['files']['vision_calibration'] = None  # Explicit absence; never hash a placeholder calibration.
    protocol['dynamic_policy'] = dict(transformation, snapshot_manifest_sha256=sha(manifest_path),
      pipeline_invocation_sha256=sha(invocation), decoder='speculative', prefill_attention='shuffle-fma')
    return normalized, args, protocol

  replay.normalize = normalize
  replay._dynamic_policy = transformation
  return replay


@contextmanager
def imports():
  """Compose with, and preserve, the actual speculative launcher's import hooks."""
  original = importlib.util.spec_from_file_location
  def spec(name, location, *args, **kwargs):
    result = original(name, location, *args, **kwargs)
    if Path(location).resolve() != ROOT/'agent-live-replay.py':
      return result
    loader = result.loader
    class Loader:
      def create_module(self, value): return loader.create_module(value)
      def exec_module(self, value):
        loader.exec_module(value)
        adapt(value)
    result.loader = Loader()
    return result
  with patch.object(importlib.util, 'spec_from_file_location', spec):
    yield


def qualified_gate(report, *, reference=None):
  """Additional strict policy check; the frozen paced protocol/source gate still runs."""
  path, declared = manifest()
  if (report.get('mode') != 'consumer' or report.get('gpu_executed') is not True or
      report.get('predictions_qualified') is not True or report.get('comparison', {}).get('exact') is not True or
      report['comparison'].get('queries') != 64 or report['comparison'].get('differences') != []):
    raise ValueError('New exact real-GPU 64-query Consumer parity is required')
  if report.get('decoder') != 'speculative' or report.get('prefill_attention') != 'shuffle-fma':
    raise ValueError('Parity must use target-verified speculative prefill and shuffle-FMA')
  flags = report['protocol']['flags']
  if flags.get('vision_calibration') is not None or flags.get('pad_prompt_to') != 2048:
    raise ValueError('Old fixed-vision or padding policy cannot qualify this replay')
  if report['protocol'].get('dynamic_policy', {}).get('snapshot_manifest_sha256') != sha(path):
    raise ValueError('Parity belongs to another immutable dynamic snapshot')
  expected = {p.name: declared['files'][str(p.relative_to(path.parent))]
              for p in [*ROOT.glob('*.py'), *ROOT.glob('*.c'), ROOT.parent/'model.py']}
  if report.get('initialized', {}).get('source_sha256') != expected:
    raise ValueError('Parity must qualify the exact complete current source set')
  if len(report.get('rows', [])) != 64 or not all(r.get('completed') is True for r in report['rows']):
    raise ValueError('Every original query needs a successful completion barrier')
  if reference is not None:
    with imports():
      replay = local('agent-live-replay')
      comparison = replay.compare(reference, report)
    if comparison != report['comparison'] or not comparison['exact']:
      raise ValueError('Independent exact parity recomputation failed')
  return report

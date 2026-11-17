"""Source-bound composition protocol; original numeric gates remain independent."""
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import CodeType

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
EXP = Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments')
ROOT = WORK/'agent-expert-combined-source'
PROTOCOL = WORK/'agent-expert-combined-protocol.json'
BASE = EXP/'agent-exact-combined-protocol-v1/manifest.json'
BASE_SHA = '9993139d4f60efcf2b6dac28dcc946140c5c2cc0ee0bd3d0625de94ae7717f0f'
PV_ROOT = WORK/'agent-expert-pv-qualified-source'
PV_PROTOCOL = WORK/'agent-expert-pv-qualified-protocol.json'
PV_SHA = '658bcb7d828a975e69f55cdf221a3308d30ce6f2a499a4ecc5ed53c1bee59a42'
MLP = WORK/'agent-fused-mlp-2048-protocol-v2/alpamayo'
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PY = '/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'
COUNTS = {'quality': (2, 2), 'native': (12, 24)}
INDICES = [0, 16, 32, 48]
CORE = {'model.py':'fce24a07fdd48bd24c05e67518c986bcb462d09dfb75b47daf4140dc4e67f5fb',
 'amd_attention.py':'fa9d2f01c6557c97541f8a62e4e02ee7cf2608584787cad7d42b9922a00866f4',
 'experiments/expert-kernels.py':'4676128ef21bd42f851311b3b5dba0563ab6cd1e54d367a3bca2e377f5a53b8b',
 'experiments/wgp_compiler.py':'ccd7427848f13a8032e8af3c305fc8b2db3150242b3ff67a63f25acd52da8890'}
PRIOR = {
 'agent-exact-combined-quality-reference':'eb6288437049ab426959735b607dcaeacee8862ef63325f131b31125019bbf02',
 'agent-exact-combined-quality-candidate':'6ad4fadda536ddf6d9a1bb6a61229057d27e75f9de9fc90e7a249fc27c479aed',
 'agent-exact-combined-native-reference':'b5adce4c80faf0c41dba7b007f8f7c1de43cb1edf098f8c163ba88d0b0c12ab6',
 'agent-exact-combined-native-candidate':'554f4a8d73bb30782f8c8efe0987651f62acc72e859e9276c49e7b23fd3ab8d0',
 'agent-expert-pv-qualified-quality-reference':'069e11d02b7424f1504b80dc80d887775c3ce5c9895d447402d1d5892751e842',
 'agent-expert-pv-qualified-quality-candidate':'5937bc33cdcf5e0ab5f7c650d502a5a412b6a4a727149c17cd9036463ffc9dbc',
 'agent-expert-pv-qualified-native-reference':'1ed7d3830fb586cc2754e18d3b5784386a5cfe05fed9596e908c96b6a18f3ada',
 'agent-expert-pv-qualified-native-candidate':'97fb538138635107af0d2b1d285b8378887585e78b1106743487c507d21697b1'}


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value, *, new=False):
  path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('x' if new else 'w') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec); sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


@lru_cache(None)
def base():
  assert sha(BASE) == BASE_SHA
  manifest = read(BASE)
  for name in ('common', 'install', 'run'):
    path = EXP/f'agent-exact-combined-{name}.py'
    assert sha(path) == manifest['files'][str(path)]
  module = load(EXP/'agent-exact-combined-common.py', 'expert_combined_frozen_common')
  module.verify(manifest)
  # Only the private module's artifact destination changes. Source and sampler ELF remain pinned.
  module.HERE = WORK/'agent-expert-combined-artifacts'
  module.HERE.mkdir(exist_ok=True)
  return module, manifest


def paths():
  old = WORK/'agent-expert-pv-route-source/alpamayo'
  for rel, expected in CORE.items(): assert sha(MLP/rel) == sha(old/rel) == expected, rel
  sys.path[:0] = [str(TINY), str(MLP/'experiments'), str(MLP)]
  module = importlib.import_module('wgp_compiler')
  assert Path(module.__file__).resolve() == MLP/'experiments/wgp_compiler.py'
  assert sha(module.__file__) == CORE['experiments/wgp_compiler.py']
  def codes(code):
    yield code
    for child in code.co_consts:
      if isinstance(child,CodeType): yield from codes(child)
  for name in ('__init__','__reduce__'):
    fn = getattr(module.WGPCompiler,name)
    expected = next(c for c in codes(compile(Path(module.__file__).read_bytes(),fn.__code__.co_filename,'exec'))
                    if c.co_qualname == fn.__code__.co_qualname)
    assert fn.__code__ == expected, 'Runtime WGP compiler code changed'
  return module


@lru_cache(None)
def pv():
  assert sha(PV_PROTOCOL) == PV_SHA
  protocol = read(PV_PROTOCOL)
  for path, digest in protocol['files'].items(): assert sha(path) == digest, path
  value = load(PV_ROOT/'agent-expert-pv-qualified-common.py', 'expert_combined_frozen_pv_common')
  _, evidence = value.protocol(PV_PROTOCOL, PV_SHA)
  # Explicit byte-identical relocation, not a new numeric/source qualification.
  paths()
  value.SNAP = MLP
  value.paths = paths
  return value, evidence


def prior():
  c, _ = base()
  result = {}
  for name, digest in PRIOR.items():
    path = WORK/name/'report.json'
    assert sha(path) == digest
    report = read(path)
    side = path.with_suffix('.agent-exact-combined.json' if name.startswith('agent-exact-combined') else '.agent-expert-pv-qualified.json')
    key = 'agent_exact_combined' if name.startswith('agent-exact-combined') else 'agent_expert_pv'
    c.equal(read(side), report[key], 'prior.sidecar')
    meta = report[key]
    assert not meta.get('failure')
    if key == 'agent_exact_combined':
      assert meta['passed'] and meta['hooks_restored']
      if name.endswith('candidate'): assert meta['paired_exact']['exact'] and meta['combined_gpu_qualified']
    else:
      assert meta['route_completed'] and meta['coverage_validated']
      if name.endswith('candidate'): assert meta['exactness_validated'] and meta['comparison']['all_exact']
    result[name] = {'report_sha256':digest, 'sidecar_sha256':sha(side),
      'speculation_sha256':sha(path.with_suffix('.speculation.json')), 'queries':len(report['warmup'])+len(report['runs'])}
  for stage in COUNTS:
    a, b = [read(WORK/f'agent-exact-combined-{stage}-{mode}/report.json') for mode in ('reference','candidate')]
    c.compare(a, b, stage)
  return result


def protocol(path, digest):
  assert sha(path) == digest, 'Composition protocol changed'
  data = read(path)
  assert data['counts'] == {k:list(v) for k,v in COUNTS.items()} and data['quality_indices'] == INDICES
  assert data['base_sha256'] == BASE_SHA and data['pv_protocol_sha256'] == PV_SHA
  for source, expected in data['files'].items(): assert sha(source) == expected, source
  c, manifest = base(); c.verify(manifest)
  p, evidence = pv()
  c.equal(evidence, data['pv_evidence'], 'PV.independent_evidence')
  c.equal(prior(), data['prior'], 'prior_64_and_36_qualifications')
  assert not data['automatic_paced'] and data['parent_child_timeout_seconds'] == 900
  paths()
  return data, evidence


def option(command, flag):
  assert command.count(flag) == 1, flag
  return command[command.index(flag)+1]


def plan(data, stage, output):
  invocation = read(data['stages'][stage]['invocation'])
  outer = invocation['command'][1:].copy()
  terminal = read(option(outer, '--invocation'))['command'][1:]
  assert Path(outer[0]) == MLP/'experiments/agent-speculation-run.py'
  assert Path(terminal[0]) == MLP/'experiments/benchmark-pipeline.py'
  assert option(outer,'--mode') == 'candidate' and option(outer,'--prefill-attention') == 'shuffle-fma'
  assert option(outer,'--pad-prompt-to') == option(terminal,'--pad-prompt-to') == '2048'
  for cmd in (outer, terminal):
    assert tuple(int(option(cmd, '--'+k)) for k in ('warmup','runs')) == COUNTS[stage]
    assert not any(f in cmd for f in ('--vision-calibration','--capture-vision-scales','--audit','--online-expert','--warm-current-shapes'))
  assert option(terminal,'--steps') == '3' and option(terminal,'--pixel-scale') == '1'
  samples = terminal[terminal.index('--samples')+1:terminal.index('--initial-noise')][:sum(COUNTS[stage])]
  assert samples == data['stages'][stage]['scheduled_samples']
  outer[outer.index('--output')+1] = str(output)
  return {'effective_command':outer, 'terminal_command':terminal, 'warmup':COUNTS[stage][0], 'runs':COUNTS[stage][1],
    'scheduled_samples':samples, 'empty_initial_lookup_and_vision_cache':True,
    'query_indices':INDICES if stage == 'quality' else list(range(36)),
    'pair_identity_sha256':sha(data['stages'][stage]['invocation'])}

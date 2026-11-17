"""Storage-only adaptation of the exact-qualified nearest paced lifecycle."""
import copy, hashlib, importlib.util, inspect, json, os, shutil, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
CPU=WORK/'agent-nearest-storage-cpu'
WEIGHTS=WORK/'agent-nearest-storage-weights'
DEFAULT=WORK/'nearest-paced-storage-run/report.json'
PROTOCOL=WORK/'agent-nearest-storage-protocol.json'
CHECKS=WORK/'agent-nearest-storage-checks.json'
PINS={
  EXP/'agent-nearest-paced-run.py':'7c72221d18171e60d37eb9d6c92e712f7b3e7b64af6d18ca8d5346c940ced508',
  EXP/'agent-nearest-paced-protocol.json':'4e6b8b499fcf634ee73407140300ec4c2ccd25c94d2968c9e853a02758ed4787',
  EXP/'agent-nearest-paced-fix-protocol.json':'507d30a87c3423a5579eae7343b67addd92127e0c512606d689b38ac214653f1',
  EXP/'agent-nearest-paced-fix-source/agent-nearest-paced-fix-run.py':'71f0ec991154fcccbefac2bacedba21dcb7fa88feb7cb9dba26bbe267fb17446',
  EXP/'agent-nearest-paced-fix-parent-run/agent-nearest-paced-fix-admission.json':'a104f5013f82a479249ff619686a449579a0d9b31ba2324601a737e337f68061',
  EXP/'agent-nearest-paced-fix-parent-run/agent-nearest-paced-fix-completion.json':'ad65e6a81ddb37ece141973a203296b0236c9531f3a9b90fb85fb958ab46e2bd',
  EXP/'agent-nearest-paced-fix-parent-run/report.startup-spec.json':'0d23d95fa4af0eaf2e88a8244e14977006cf5dc8e47e4766472f33954716184f',
  WORK/'agent-nearest-paced-fix-parent/run.log':'67438b5488d0bc0f43b5c3a582411f579c38c04458d0ed0f6258bf8ea6352f26',
}
WEIGHT_PINS={'index.json':'c4366c5b7e92b4cd5e2d826881b99b9a80361ec171ebd4a50d51b0f718dffa3e',
  'weights.bin':'dd6b716a2ade3646eff71d874bef676926422d9aca39358fd3f8b4fe75755052'}
sys.dont_write_bytecode=True

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
  with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(path):
  spec=importlib.util.spec_from_file_location('nearest_storage_'+Path(path).stem.replace('-','_'),path)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module
def sources():return {p.name:sha(p) for p in sorted(EXP.glob('agent-nearest-storage-*.py'))}
def check_pins(pins):
  for path,expected in pins.items():
    if sha(path)!=expected:raise ValueError('Frozen source/evidence changed: '+str(path))
def inherited():
  check_pins(PINS)
  fix=load(EXP/'agent-nearest-paced-fix-source/agent-nearest-paced-fix-run.py');fix.verify_frozen()
  old=fix.original();return old

def mounted(path):
  mount=Path('/mnt/nvme-500')
  if not mount.is_mount() or not Path(path).resolve().is_relative_to(mount):raise ValueError('Require the mounted500GB filesystem')
  existing=Path(path)
  while not existing.exists():existing=existing.parent
  if existing.stat().st_dev!=mount.stat().st_dev or existing.stat().st_dev==Path('/').stat().st_dev:
    raise ValueError('Output resolved onto the wrong filesystem')
  return existing

def reserve(path,upcoming=64*1024**2):
  existing=mounted(path)
  sys.path[:0]=[str(BASE),str(BASE/'experiments')]
  bridge=load(BASE/'experiments/agent-live-bridge.py')
  usage=shutil.disk_usage(existing)
  if not bridge.storage_ok(usage.total,usage.free,upcoming):raise RuntimeError('Original10%/5GiB output reserve would be crossed')
  return dict(path=str(path),existing_parent=str(existing),device=existing.stat().st_dev,total_bytes=usage.total,
    free_bytes=usage.free,upcoming_bytes=upcoming,original_storage_ok=True,root_device=Path('/').stat().st_dev)

def cache_environment(directory):
  return dict(CACHEDB=str(directory/'compiler-cache.db'),XDG_CACHE_HOME=str(directory/'cache'),
    TMPDIR=str(directory/'tmp'),HF_HOME=str(directory/'cache/huggingface'),TORCH_HOME=str(directory/'cache/torch'),
    ALPAMAYO_FP8_CACHE=str(WEIGHTS))

def configure(directory):
  proof=reserve(directory)
  environment=cache_environment(directory)
  for key in ('XDG_CACHE_HOME','TMPDIR','HF_HOME','TORCH_HOME'):Path(environment[key]).mkdir(parents=True,exist_ok=True)
  os.environ.update(environment);tempfile.tempdir=None
  return proof

def setup_cpu(old):
  old.a.setup_cpu();configure(CPU/'environment')

def validate_output(output):
  if output!=DEFAULT or output.name!='report.json' or output.parent.is_symlink():raise ValueError('Use the single new500GB canonical output directory')
  reserve(output.parent)

def weights():
  mounted(WEIGHTS)
  if set(p.name for p in WEIGHTS.iterdir())!={'index.json','weights.bin'}:raise ValueError('Unexpected private weight-cache inventory')
  check_pins({WEIGHTS/name:digest for name,digest in WEIGHT_PINS.items()})
  index=read(WEIGHTS/'index.json')
  if index['sha256']!=WEIGHT_PINS['weights.bin'] or index['nbytes']!=(WEIGHTS/'weights.bin').stat().st_size:
    raise ValueError('Private copied weights differ from the original cache')
  return dict(path=str(WEIGHTS),files=WEIGHT_PINS,nbytes=index['nbytes'],unchanged_bytes=True)

@contextmanager
def observer(old,c,plan,saved,output,records):
  """Preserve the old observer; change only cache pathnames after its exact normalizer."""
  original=importlib.util.spec_from_file_location;restored=[]
  def spec(name,location,*args,**kwargs):
    result=original(name,location,*args,**kwargs)
    if Path(location).resolve()!=BASE/'experiments/agent-paced-replay.py':return result
    loader=result.loader
    class Loader:
      def create_module(self,value):return loader.create_module(value)
      def exec_module(self,module):
        loader.exec_module(module);local=module.local
        def relocated(key):
          value=local(key)
          if key=='agent-live-replay':
            normalize=value.normalize
            def storage(*args,**kwargs):
              normalized,parsed,protocol=normalize(*args,**kwargs)
              if protocol!=saved['qualified_factory_protocol']:raise ValueError('Original exact factory gate changed')
              before=copy.deepcopy(normalized);normalized=copy.deepcopy(normalized)
              paths=cache_environment(output.parent)
              normalized['environment'].update(paths)
              changes={key:dict(before=before['environment'].get(key),after=normalized['environment'][key]) for key in paths
                if before['environment'].get(key)!=normalized['environment'][key]}
              if normalized['command']!=before['command']:raise ValueError('Storage adaptation changed model arguments')
              for key,value in before['environment'].items():
                if key not in paths and normalized['environment'][key]!=value:raise ValueError('Numerical environment changed')
              records.setdefault('storage_path_changes',[]).append(changes)
              if len(records['storage_path_changes'])!=1:raise ValueError('Only one original normalization is admitted')
              return normalized,parsed,protocol
            value.normalize=storage
          return value
        module.local=relocated;restored.append((module,local))
    result.loader=Loader();return result
  # The new loader must run AFTER the old observer, before startup's private transform.
  with old.observer(c,plan,saved,output,records):
    original=importlib.util.spec_from_file_location
    try:
      with patch.object(importlib.util,'spec_from_file_location',spec):yield
    finally:
      for module,local in restored:module.local=local

def transform(old,verifier):
  source=inspect.getsource(old.main)
  changes=[
    ('  args = parser.parse_args(); a.setup_cpu()',
     '  args = parser.parse_args(); _storage_setup_cpu(); _storage_validate_output(args.output)'),
    ("  if args.output.parent.parent.resolve() != HERE or not args.output.parent.name.startswith('agent-nearest-paced-'):\n"
     "    raise ValueError('Parent output must use a new owned agent-nearest-paced-* directory')",
     '  _storage_validate_output(args.output)'),
    ('  args.output.parent.mkdir()',
     '  args.output.parent.mkdir()\n  storage_filesystem = _storage_configure(args.output.parent)'),
    ('    source_sha256=a.sources(),protocol_sha256=a.sha(a.PROTOCOL),admission=admission,',
     '    source_sha256=_storage_sources(),protocol_sha256=a.sha(_storage_protocol),admission=admission,\n'
     '    original_source_sha256=a.sources(),original_protocol_sha256=a.sha(a.PROTOCOL),storage_filesystem=storage_filesystem,'),
    ("  a.save(args.output.with_suffix('.admission.json'),record)",
     "  a.save(args.output.parent/'agent-nearest-storage-admission.json',record)"),
    ("  finally: a.save(args.output.with_suffix('.nearest-admission-result.json'),record)",
     "  finally: a.save(args.output.parent/'agent-nearest-storage-completion.json',record)"),
  ]
  adapted=source
  for before,after in changes:
    if adapted.count(before)!=1:raise ValueError('Ambiguous storage-only source substitution')
    adapted=adapted.replace(before,after,1)
  namespace=dict(old.__dict__,DEFAULT=DEFAULT,verify_frozen=verifier,_storage_protocol=PROTOCOL,_storage_sources=sources,
    _storage_setup_cpu=lambda:setup_cpu(old),_storage_validate_output=validate_output,_storage_configure=configure,
    observer=lambda *args:observer(old,*args))
  exec(compile(adapted,str(EXP/'agent-nearest-storage-run.py')+':storage-paths','exec'),namespace)
  return namespace['main'],dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),adapted_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
    substitutions=[dict(before=x,after=y) for x,y in changes],default_output=str(DEFAULT))

def verify_frozen():
  old=inherited();protocol=read(PROTOCOL)
  if protocol['source_sha256']!=sources():raise ValueError('Storage helper source changed')
  check_pins(protocol['artifact_sha256']);weights()
  checks=read(CHECKS)
  if not checks['passed'] or checks['gpu_executed'] or checks['source_sha256']!=sources():raise ValueError('Current full-call-path CPU proof required')
  _,transformation=transform(old,lambda:protocol)
  if transformation!=protocol['transformation'] or protocol['admission']!=old.verify_frozen()['admission']:
    raise ValueError('Frozen paths or original admission changed')
  return protocol

def main():
  old=inherited();function,_=transform(old,verify_frozen);function()

if __name__=='__main__':main()

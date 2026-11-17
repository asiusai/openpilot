"""CPU-only staging and sealing. All snapshots, caches, and evidence use nvme-500."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import shlex
import shutil
import sqlite3

C=importlib.import_module('agent-compiler-speed-common')
HERE=Path(__file__).resolve().parent
OLD=C.WORK/'agent-expert-pv-route-source'
MODE_NAMES=('agent-compiler-mode-renderer.py','agent-compiler-mode-proof.py','agent-compiler-mode-inventory.py',
            'agent-compiler-mode-LICENSE.tinygrad.txt')
SPEED_NAMES=('agent-compiler-speed-common.py','agent-compiler-speed-run.py','agent-compiler-speed-check.py',
             'agent-compiler-speed-queue.py','agent-compiler-speed-prepare.py')
OLD_MANIFEST_SHA='2c3806e48c943b452799da09098f1d3ea725a65f300a746cfc41d13e4758623c'
EVIDENCE_SHA={'original':'bfb9cfff39ed3a7d682de55a8c1c9043cf8370afa36b6efaeaf749b1bafe355e',
              'consistent-wgp':'b5ce6db8feb9f01cdf12bee350296cf70538c5349a71cc3ab093104f27fbe611'}


def keep(source,dest,files,kind,digest=None):
  data=Path(source).read_bytes()
  if digest:assert C.sha(data)==digest,source
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
  files[str(dest.relative_to(C.ROOT))]={'sha256':C.sha(data),'origin':str(source),'kind':kind,'size':len(data)}


def private(files):
  for name in SPEED_NAMES:keep(HERE/name,C.PRIVATE/name,files,'new-private-wrapper')
  for name in MODE_NAMES:keep(HERE/name,C.PRIVATE/name,files,'unchanged-qualified-compiler-support')
  wrapper=Path('/tmp/opencode/run-alpamayo-speed-trial.py')
  keep(wrapper,C.PRIVATE/wrapper.name,files,'unchanged-serial-parent-wrapper',
       'c931aa3d0f6fa789999927e873c5eb6f3dcd273f8d666339f681510bc4a092d3')


def stage():
  assert not C.ROOT.exists() and not C.CACHE.parent.exists()
  assert C.ROOT.is_relative_to('/mnt/nvme-500') and shutil.disk_usage(C.WORK).free>10*1024**3
  assert C.file_sha(OLD/'source-manifest.json')==OLD_MANIFEST_SHA
  old=C.read(OLD/'source-manifest.json');files={}
  C.ROOT.mkdir();C.PRIVATE.mkdir()
  for name,item in old['files'].items():
    keep(OLD/name,C.ROOT/name,files,'baseline-byte-copy',item['sha256'])
  private(files)
  keep(OLD/'source-manifest.json',C.ROOT/'evidence/original-source-manifest.json',files,'independent-baseline-anchor',OLD_MANIFEST_SHA)
  for arm in C.ARMS:
    source=C.WORK/f'agent-compiler-mode-{arm}/report.json'
    keep(source,C.ROOT/'evidence'/f'mode-{arm}.json',files,'completed-two-query-evidence',EVIDENCE_SHA[arm])
    record=C.read(source)
    assert record['complete'] and record['gpu_executed'] and record['self_validation']['all_exact']
    if arm=='consistent-wgp':assert record['strict_two_query_comparison']['all_exact']
    for suffix,kind in (('', 'model'),('.speculation','speculation'),('.jsonl','progress')):
      if suffix=='':p=Path(record['model_report'])
      elif suffix=='.jsonl':p=Path(record['model_report']).with_suffix('.jsonl')
      else:p=Path(record['model_report']).with_suffix('.speculation.json')
      keep(p,C.ROOT/'evidence'/f'mode-{arm}-{kind}.json',files,'completed-two-query-model',record['model_artifact_sha256'][str(p)])
  # Independently retain the actual tested renderer/support source hashes.
  tested=C.read(C.WORK/'agent-compiler-mode-consistent-wgp/report.json')['source_sha256']
  for name in MODE_NAMES:
    if name.endswith('.py'):assert C.file_sha(C.PRIVATE/name)==tested[str(HERE/name)]
  replacements=[]
  for label in C.COUNTS:
    for depth in (0,1):
      path=C.ROOT/'invocations'/f'{label}-{depth}.json'
      before=C.read(path);after=copy.deepcopy(before)
      for i,value in enumerate(after['command']):
        if value.startswith(str(OLD)+'/'):
          after['command'][i]=str(C.ROOT)+value[len(str(OLD)):]
      after['environment'].update(CACHEDB=str(C.CACHE),PYTHONPATH=':'.join(map(str,(C.PRIVATE,C.SNAP/'experiments',C.TINY,C.SNAP))),
        PARALLEL='32',LLVMOPT='1',TMPDIR=str(C.SCRATCH),PYTHONDONTWRITEBYTECODE='1')
      C.write(path,after)
      replacements.append({'path':str(path.relative_to(C.ROOT)),'before':before,'after':after})
      files[str(path.relative_to(C.ROOT))].update(sha256=C.file_sha(path),kind='path-cache-environment-relocation',size=path.stat().st_size)
  C.CACHE.parent.mkdir();C.SCRATCH.mkdir()
  seed=C.CACHE.with_name('seed.db')
  source=Path('/tmp/opencode/alpamayo-asius-20261114/cache.db')
  with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as src, sqlite3.connect(seed) as dst:src.backup(dst)
  shutil.copyfile(seed,C.CACHE)
  seed.chmod(0o444)
  cache={'source':str(source),'read_only_sqlite_backup':True,'seed':str(seed),'seed_sha256':C.file_sha(seed),
    'working_cache':str(C.CACHE),'bytes':seed.stat().st_size,'shared_between_fixed_arms':True}
  C.write(C.ROOT/'cache-provenance.json',cache)
  files['cache-provenance.json']={'sha256':C.file_sha(C.ROOT/'cache-provenance.json'),'kind':'cache-provenance'}
  external=copy.deepcopy(old['external_pins'])
  template=Path('/mnt/nvme-500/tmp/alpamayo-calibrated-two-six-20261116/matched-warm/2cam-invocation.json')
  external[str(template)]={'sha256':C.file_sha(template),'size':template.stat().st_size}
  for group in (external,old['data_input_pins']):
    for path,item in group.items():assert C.file_sha(path)==item['sha256'],path
  manifest={'files':files,'external_pins':external,'data_input_pins':old['data_input_pins'],
    'original_manifest_sha256':OLD_MANIFEST_SHA,'invocation_relocations':replacements,
    'frozen':False,'gpu_executed':False,'original_files_modified':False}
  C.write(C.ROOT/'source-manifest.json',manifest)
  print(json.dumps({'staged':True,'root':str(C.ROOT),'cache_bytes':cache['bytes'],'gpu_executed':False}))


def refresh():
  assert not (C.ROOT/'protocol.json').exists()
  m=C.read(C.ROOT/'source-manifest.json');assert not m['frozen']
  private(m['files']);C.write(C.ROOT/'source-manifest.json',m)
  print(json.dumps({'refreshed_new_private_sources':True,'gpu_executed':False}))


def seal(check_path):
  assert not (C.ROOT/'protocol.json').exists()
  check=C.read(check_path);assert check['complete'] and check['cpu_only'] and not check['gpu_executed']
  for name,digest in check['source_sha256'].items():assert C.file_sha(name)==digest
  m=C.read(C.ROOT/'source-manifest.json')
  for name,item in m['files'].items():assert C.file_sha(C.ROOT/name)==item['sha256'],name
  for node in m['invocation_relocations']:
    before,after=node['before'],node['after'];restored=copy.deepcopy(after)
    restored['command']=[str(OLD)+x[len(str(C.ROOT)):] if x.startswith(str(C.ROOT)+'/') else x for x in restored['command']]
    restored['environment']=before['environment']
    assert restored==before and C.read(C.ROOT/node['path'])==after
  m['frozen']=True;C.write(C.ROOT/'source-manifest.json',m)
  plans={stage:C.route().plan(C.ROOT/'invocations'/f'{stage}-0.json',C.out(stage,'original')/'agent-compiler-speed-model.json') for stage in C.COUNTS}
  protocol={'schema':1,'timeout_seconds':900,'counts':{k:list(v) for k,v in C.COUNTS.items()},'plans':plans,
    'manifest_sha256':C.file_sha(C.ROOT/'source-manifest.json'),'cpu_evidence_sha256':{str(check_path):C.file_sha(check_path)},
    'run_order':[['quality','original'],['quality','consistent-wgp'],['native','original'],['native','consistent-wgp']],
    'candidate_change':'Only the byte-identical qualified renderer subclass installation',
    'reference_reuse':'Fresh reference: prior 64-query reports have different recipes/wrappers; exact original identity is unproven',
    'policy':'dynamic vision; fixed language; pad2048; target-verified NOMINAL six-token draft; shuffle-FMA; full images; 3 Euler',
    'observation':'Warmup-only program/capture audit; zero measured observers; full host compiler-cache audit after all queries',
    'storage':'All new snapshots, artifacts, compiler caches, protocols and outputs are under /mnt/nvme-500',
    'native_limit':'One unpaced 12-warmup/24-measured pair; no live cadence or statistical-significance claim',
    'quality_required_before_native':True,'gpu_executed_during_preparation':False}
  path=C.ROOT/'protocol.json';C.write(path,protocol);digest=C.file_sha(path)
  commands=C.commands(digest)
  C.write(C.ROOT/'commands.json',{'protocol_sha256':digest,'commands':commands,'shell':{k:shlex.join(v) for k,v in commands.items()},'executed':False})
  for name in m['files']:(C.ROOT/name).chmod(0o444)
  for name in ('protocol.json','source-manifest.json','commands.json'):(C.ROOT/name).chmod(0o444)
  print(json.dumps({'sealed':True,'protocol_sha256':digest,'commands':str(C.ROOT/'commands.json'),'gpu_executed':False}))


def main():
  p=argparse.ArgumentParser(description=__doc__)
  g=p.add_mutually_exclusive_group(required=True);g.add_argument('--stage',action='store_true');g.add_argument('--refresh',action='store_true');g.add_argument('--seal',type=Path)
  a=p.parse_args()
  if a.stage:stage()
  elif a.refresh:refresh()
  else:seal(a.seal)


if __name__=='__main__':main()

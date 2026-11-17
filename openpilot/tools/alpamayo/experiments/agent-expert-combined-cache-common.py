"""Exact cache admission and storage-only retry of the frozen composition protocol."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
OLD = WORK/'agent-expert-combined-source'
OLD_PROTOCOL = WORK/'agent-expert-combined-protocol.json'
OLD_SHA = '4e11ba366c177e61220f71703981834ebaa23fea8f2375bb22cd52d46ef4820e'
ROOT = WORK/'agent-expert-combined-cache-source'
PROTOCOL = WORK/'agent-expert-combined-cache-protocol.json'
ARTIFACTS = WORK/'agent-expert-combined-cache-artifacts'
CACHE = ARTIFACTS/'fp8-cache'
ORIGIN = Path('/dev/shm/alpamayo-9060-fp8')
MANIFEST = WORK/'agent-expert-pv-route-source/source-manifest.json'
MANIFEST_SHA = '2c3806e48c943b452799da09098f1d3ea725a65f300a746cfc41d13e4758623c'
INDEX_SHA = 'c4366c5b7e92b4cd5e2d826881b99b9a80361ec171ebd4a50d51b0f718dffa3e'
WEIGHTS_SHA = 'dd6b716a2ade3646eff71d874bef676926422d9aca39358fd3f8b4fe75755052'
WEIGHTS_BYTES = 2929010944
LOADER = WORK/'agent-fused-mlp-2048-protocol-v2/alpamayo/experiments/benchmark-fast.py'
FAILURE = WORK/'agent-expert-combined-quality-reference'
PY = '/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
WRAPPER = Path('/tmp/opencode/run-alpamayo-speed-trial.py')
SAMPLER = WORK/'agent-expert-combined-artifacts/agent-exact-combined-artifacts/12460b8ec1eae4609835e65cab03ec855bff660cc872123cf1494f652cee8dd7.so'
SAMPLER_SHA = '7852db0138732648621b06711cac067e2f23ec364bf3859b59d45a2780f29f7b'


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path,value,*,new=False):
  path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('x' if new else 'w') as stream: stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(path,name):
  spec = importlib.util.spec_from_file_location(name,path)
  module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
  return module


def original():
  assert sha(OLD_PROTOCOL) == OLD_SHA
  data = read(OLD_PROTOCOL)
  for name in ('common','run','install','queue'):
    path = OLD/f'agent-expert-combined-{name}.py'
    assert sha(path) == data['files'][str(path)], path
  return load(OLD/'agent-expert-combined-common.py','expert_combined_cache_original_common')


def qualification():
  # The independent frozen manifest pins the index. Its bytes pin the complete weights.
  assert sha(MANIFEST) == MANIFEST_SHA
  pin = read(MANIFEST)['external_pins'][str(ORIGIN/'index.json')]
  assert pin == {'sha256':INDEX_SHA,'size':308917}
  assert sha(ORIGIN/'index.json') == INDEX_SHA and (ORIGIN/'index.json').stat().st_size == pin['size']
  index = read(ORIGIN/'index.json')
  assert index['sha256'] == WEIGHTS_SHA and index['nbytes'] == WEIGHTS_BYTES
  return {'manifest':str(MANIFEST),'manifest_sha256':MANIFEST_SHA,'index_sha256':INDEX_SHA,
    'index_bytes':308917,'weights_sha256':WEIGHTS_SHA,'weights_bytes':WEIGHTS_BYTES}


def validate_index(index,actual_bytes):
  assert set(index) == {'format','sha256','nbytes','records'} and index['format'] == 1
  assert index['sha256'] == WEIGHTS_SHA and index['nbytes'] == actual_bytes == WEIGHTS_BYTES
  from tinygrad import dtypes
  types = {'scale':dtypes.float32,'fp8':dtypes.fp8e4m3,'bf16':dtypes.bfloat16}
  names,previous_end,counts = set(),0,dict.fromkeys(types,0)
  for record in index['records']:
    assert set(record) == {'name','kind','packed','shape','offset','nbytes'}
    name,kind,shape = record['name'],record['kind'],record['shape']
    assert isinstance(name,str) and name and '/' not in name and '\\' not in name
    assert kind in types and type(record['packed']) is bool
    assert type(shape) is list and shape and all(type(n) is int and n > 0 for n in shape)
    assert type(record['offset']) is type(record['nbytes']) is int
    assert record['nbytes'] == math.prod(shape)*types[kind].itemsize
    assert record['offset'] % 256 == 0 and record['offset'] >= previous_end
    previous_end = record['offset']+record['nbytes']; assert previous_end <= actual_bytes
    assert (name,kind) not in names; names.add((name,kind)); counts[kind] += 1
    if record['packed']: assert kind == 'fp8' and len(shape) == 2 and all(n % 16 == 0 for n in shape)
    if kind == 'scale': assert shape == [1] and not record['packed']
    if kind == 'bf16': assert not record['packed']
  assert previous_end == actual_bytes and len(index['records']) == 1411
  assert counts == {'scale':504,'fp8':504,'bf16':403}
  assert {n for n,k in names if k == 'scale'} == {n for n,k in names if k == 'fp8'}
  assert not {n for n,k in names if k == 'bf16'} & {n for n,k in names if k == 'fp8'}
  return {'records':len(names),'kinds':counts,'all_extents_in_bounds':True,'nonoverlapping_aligned_records':True,
    'dtype_itemsize_shape_bytes_exact':True,'final_extent':previous_end}


def cache_check():
  evidence = qualification()
  assert CACHE.resolve().is_relative_to(WORK) and CACHE.stat().st_dev == Path('/mnt/nvme-500').stat().st_dev
  assert CACHE.stat().st_dev != Path('/').stat().st_dev
  for name,expected,size in (('index.json',INDEX_SHA,308917),('weights.bin',WEIGHTS_SHA,WEIGHTS_BYTES)):
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and len(relative.parts) == 1 and '..' not in relative.parts
    path = CACHE/name
    assert path.is_file() and not path.is_symlink() and path.resolve().parent == CACHE.resolve()
    assert path.stat().st_size == size and sha(path) == expected, name
  return {'cache':str(CACHE),'independent_qualification':evidence,
    'index':validate_index(read(CACHE/'index.json'),(CACHE/'weights.bin').stat().st_size)}


def materialize():
  qualification()
  assert not CACHE.exists(), 'Use an exclusive new cache destination'
  assert WORK.stat().st_dev == Path('/mnt/nvme-500').stat().st_dev != Path('/').stat().st_dev
  assert shutil.disk_usage(WORK).free > WEIGHTS_BYTES+64*1024*1024
  CACHE.mkdir(parents=True)
  copied = {}
  for name,expected,size in (('weights.bin',WEIGHTS_SHA,WEIGHTS_BYTES),('index.json',INDEX_SHA,308917)):
    source,target = ORIGIN/name,CACHE/name
    before = source.stat(); assert before.st_size == size
    digest = hashlib.sha256(); count = 0
    with source.open('rb') as src,target.open('xb') as dest:
      while chunk := src.read(16*1024*1024):
        dest.write(chunk); digest.update(chunk); count += len(chunk)
      dest.flush(); os.fsync(dest.fileno())
    after = source.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) == (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    assert count == size and digest.hexdigest() == expected
    assert target.stat().st_size == size and sha(target) == expected
    target.chmod(0o444)
    copied[name] = {'source':str(source),'destination':str(target),'bytes':count,
      'source_stream_sha256':digest.hexdigest(),'destination_readback_sha256':expected}
  return {'copied':copied,'storage_device':CACHE.stat().st_dev,'root_device':Path('/').stat().st_dev,
    'qualification':qualification(),'index':validate_index(read(CACHE/'index.json'),WEIGHTS_BYTES)}


def verify(path,digest):
  assert sha(path) == digest
  data = read(path)
  assert data['cache_retry']['original_protocol_sha256'] == OLD_SHA
  C = original()
  checked,evidence = C.protocol(path,digest)  # All original gates, including constructor/source/USB micro evidence.
  assert checked == data
  assert sha(SAMPLER) == SAMPLER_SHA, 'Existing qualified sampler artifact must remain read-only'
  # Check allowed changes against the immutable old recipe, independently of new file pins.
  old = read(OLD_PROTOCOL)
  assert data['stages'].keys() == old['stages'].keys()
  for stage in old['stages']:
    assert data['stages'][stage]['scheduled_samples'] == old['stages'][stage]['scheduled_samples']
    for suffix in ('invocation','pipeline'):
      before = read(old['stages'][stage][suffix]); after = read(data['stages'][stage][suffix])
      normalized = copy.deepcopy(after)
      for flag in ('--output','--invocation'):
        if flag in before['command']:
          normalized['command'][normalized['command'].index(flag)+1] = before['command'][before['command'].index(flag)+1]
      for key in ('ALPAMAYO_FP8_CACHE','CACHEDB','TMPDIR'): normalized['environment'][key] = before['environment'][key]
      assert normalized == before, 'Non-storage invocation change rejected'
      assert after['environment']['ALPAMAYO_FP8_CACHE'] == str(CACHE)
      for key in ('CACHEDB','TMPDIR'): assert Path(after['environment'][key]).is_relative_to(ARTIFACTS)
  return C,data,evidence

"""CPU-only execution of the actual canonical cache loader up to conversion admission."""
import argparse
import ast
import copy
import importlib.util
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('expert_combined_cache_preflight_common',HERE/'agent-expert-combined-cache-common.py')
N = importlib.util.module_from_spec(spec); spec.loader.exec_module(N)


def probe(checkpoint):
  C = N.original(); C.paths()
  from tinygrad import Device
  assert not Device._opened_devices
  base = C.read(C.BASE)
  assert C.sha(N.LOADER) == base['files'][str(N.LOADER)]
  tree = ast.parse(N.LOADER.read_text())
  paths = {n.right.value for n in ast.walk(tree) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Div)
           and isinstance(n.left,ast.Name) and n.left.id == 'directory' and isinstance(n.right,ast.Constant)}
  assert paths == {'index.json','weights.bin'}, 'Canonical loader cache paths changed'
  for path in paths: assert not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts
  index = N.read(checkpoint/'model.safetensors.index.json')
  weight = 'vlm.model.language_model.layers.0.mlp.fc1.weight'
  relative = PurePosixPath(index['weight_map'][weight])
  assert not relative.is_absolute() and '..' not in relative.parts and len(relative.parts) == 1
  shard = checkpoint/relative
  assert shard.is_file() and shard.resolve().parent == checkpoint.resolve()
  import model
  constructor = model.EdgeModel.__init__
  opened,facades,reached = [],[],[]
  def blocked(device):
    opened.append(device); raise AssertionError('Device opening forbidden during cache preflight')
  class MetadataOnlyDevice:
    DEFAULT = 'USB+AMD:LLVM'
    def __class_getitem__(cls,key):
      assert key == cls.DEFAULT; facades.append(key); return SimpleNamespace(arch='gfx1200')
  class ConversionBoundary(Exception): pass
  def stop_before_conversion(path):
    assert Path(path) == shard
    reached.append(str(path)); raise ConversionBoundary
  try:
    with patch.object(type(Device),'__getitem__',lambda self,key:blocked(key)):
      fast = C.load(N.LOADER,'expert_combined_cache_actual_factory')
      assert model.EdgeModel.__init__ is fast.load
      with patch.object(fast,'Device',MetadataOnlyDevice),patch.object(fast,'safe_load',stop_before_conversion),\
           patch.object(fast,'upload',lambda *a,**k:blocked('upload')),patch.dict(os.environ,{'ALPAMAYO_FP8_CACHE':str(N.CACHE)}):
        try: model.EdgeModel.__init__(object(),str(checkpoint),precision='fp8')
        except ConversionBoundary: pass
        else: raise AssertionError('Actual factory did not reach the conversion boundary')
        assert reached == [str(shard)] and facades == ['USB+AMD:LLVM'] and opened == []
        # Reproduce the original missing-cache failure without creating or changing files.
        missing = N.ARTIFACTS/'preflight-nonexistent-cache'
        assert not missing.exists()
        with patch.dict(os.environ,{'ALPAMAYO_FP8_CACHE':str(missing)}):
          try: model.EdgeModel.__init__(object(),str(checkpoint),precision='fp8')
          except FileNotFoundError as exc: assert Path(exc.filename) == missing/'index.json'
          else: raise AssertionError('Missing cache was not rejected')
        assert reached == [str(shard)]
  finally: model.EdgeModel.__init__ = constructor
  assert not opened and not Device._opened_devices and model.EdgeModel.__init__ is constructor
  return {'actual_factory_load_executed':True,'loader':str(N.LOADER),'loader_sha256':C.sha(N.LOADER),
    'cache_path':str(N.CACHE),'relative_cache_files':sorted(paths),'checkpoint_shard_relative':str(relative),
    'stopped_before_conversion_or_upload':True,'device_metadata_facade_only':True,'real_device_open_attempts':opened,
    'missing_cache_reproduced_before_conversion':True,'constructor_restored':True,'gpu_executed':False}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--checkpoint',type=Path,required=True)
  p.add_argument('--controls',action='store_true')
  args = p.parse_args(); assert not args.output.exists()
  cache = N.cache_check()
  result = probe(args.checkpoint)
  rejected = []
  if args.controls:
    index = N.read(N.CACHE/'index.json')
    for name in ('extent','overlap','dtype','size','shape','duplicate','path','digest'):
      bad = copy.deepcopy(index)
      if name == 'extent': bad['records'][-1]['offset'] += 256
      elif name == 'overlap': bad['records'][1]['offset'] = 0
      elif name == 'dtype': bad['records'][0]['kind'] = 'fp16'
      elif name == 'size': bad['records'][0]['nbytes'] += 2
      elif name == 'shape': bad['records'][0]['shape'][0] = -1
      elif name == 'duplicate': bad['records'][1]['name'] = bad['records'][0]['name']
      elif name == 'path': bad['records'][0]['name'] = '../foreign/weights.bin'
      else: bad['sha256'] = '0'*64
      try: N.validate_index(bad,N.WEIGHTS_BYTES)
      except AssertionError: rejected.append(name)
      else: raise AssertionError('Invalid cache index passed: '+name)
  N.write(args.output,{'complete':True,'gpu_executed':False,'cache':cache,'factory':result,'rejected':rejected,
    'source_sha256':{str(p):N.sha(p) for p in (Path(__file__),HERE/'agent-expert-combined-cache-common.py')}},new=True)
  print({'complete':True,'gpu_executed':False,'actual_factory_cache_path_checked':True})


if __name__ == '__main__': main()

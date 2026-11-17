"""Copy the existing prepared FP8 cache exactly to a new private500GB directory."""
import importlib.util,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_storage_prepare_run',HERE/'agent-nearest-storage-run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)

def main():
  old=run.inherited();run.setup_cpu(old)
  source=Path('/dev/shm/alpamayo-9060-fp8')
  if run.WEIGHTS.exists():raise FileExistsError('Retain previous cache evidence')
  index=run.read(source/'index.json')
  if run.sha(source/'index.json')!=run.WEIGHT_PINS['index.json'] or index['sha256']!=run.WEIGHT_PINS['weights.bin']:
    raise ValueError('Original prepared weight index changed')
  filesystem=run.reserve(run.WEIGHTS,index['nbytes']+64*1024**2)
  run.WEIGHTS.mkdir()
  for name in ('index.json','weights.bin'):
    with (source/name).open('rb') as src,(run.WEIGHTS/name).open('xb') as dst:shutil.copyfileobj(src,dst,16*1024**2)
  proof=run.weights()
  if run.sha(source/'index.json')!=run.WEIGHT_PINS['index.json']:raise ValueError('Original index changed during copy')
  result=dict(cpu_only=True,gpu_executed=False,source=str(source),filesystem=filesystem,copied=proof,
    original_cache_unchanged=True,numerical_conversion=False,downloaded=False)
  run.save(run.WORK/'agent-nearest-storage-weights.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

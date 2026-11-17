"""Two fixed twelve-prefill captures. CPU preflight by default; GPU execution is parent-only."""
import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_cohort_common_capture',HERE/'agent-distill-cohort-common.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
ORIGINAL=Path('/tmp/opencode/openpilot-alpamayo/openpilot/tools/alpamayo/experiments/agent-distill-command-capture.py')
original=c.module('agent_distill_cohort_original_capture',ORIGINAL)
Captured=original.Captured
capture_boundary=original.capture_boundary
find_quantizer=original.find_quantizer
DEFAULT=c.ROOT/'capture-v1/agent-distill-cohort-job-0.json'

def preflight(path):
  plan=c.read(path)
  assert plan['protocol']==c.protocol()['protocol'] and plan['parent_timeout_seconds']==300
  assert plan['capture_layers']==[12,15] and plan['expected_examples']==plan['max_model_prefills']==12
  assert plan['job'] in (0,1) and plan['coverage_sha256']==c.COVERAGE_SHA
  for filename,digest in plan['source_sha256'].items():assert c.sha(filename)==digest,filename
  assert c.sha(c.OLD_PLAN)==plan['original_capture_plan']['sha256']
  previous=c.read(c.OLD_PLAN)
  for filename,digest in previous['source_sha256'].items():assert c.sha(filename)==digest,filename
  pin=plan['inputs_manifest'];assert Path(pin['path'])==c.INPUTS and c.sha(pin['path'])==pin['sha256']
  manifest=c.validate_rows(c.read(pin['path']))
  for filename,digest in manifest['source_sha256'].items():assert c.sha(filename)==digest,filename
  assert manifest['admissible_for_internal_layer_capture'] and not manifest['admissible_for_driving_quality_validation']
  ids=c.coverage()['capture_plan']['job_'+str(plan['job'])+'_ids'];assert plan['scene_ids']==ids
  by_id={row['id']:row for row in manifest['samples']};selected=[by_id[value] for value in ids]
  assert len(selected)==12 and sum(row['split']=='train' for row in selected)==8 and sum(row['split']=='validation' for row in selected)==4
  for row in selected:
    assert not any(route in json.dumps(row) for route in c.EXCLUDED)
    assert row['conditioning']['basis']=='experimental command' and row['conditioning']['recorded_available_ns'] is None
  for key in ('runtime_root','runtime_files_sha256','environment','checkpoint_shards_sha256','fp8_cache','compiler_cache_seed','target_semantics'):
    assert plan[key]==previous[key],key
  recorded=c.read(plan['invocation_path']);old_inv=c.read(previous['invocation_path'])
  assert recorded['environment']==old_inv['environment']
  command=recorded['command'];old_command=old_inv['command'][:]
  a,b=old_command.index('--samples')+1,old_command.index('--initial-noise');old_command[a:b]=[row['sample'] for row in selected]
  old_command[old_command.index('--runs')+1]='12';old_command[old_command.index('--output')+1]=command[command.index('--output')+1]
  assert command==old_command and command[command.index('--warmup')+1]=='0'
  cache=plan['fp8_cache'];assert c.sha(Path(cache['path'])/'index.json')==cache['index_sha256']
  assert c.sha(Path(cache['path'])/'weights.bin')==cache['weights_sha256'] and (Path(cache['path'])/'weights.bin').stat().st_size==cache['nbytes']
  for filename,digest in plan['checkpoint_shards_sha256'].items():assert c.sha(filename)==digest,filename
  c.assert_no_devices()
  return plan,dict(manifest,samples=selected)

def prepared_inputs(plan,manifest):
  # Same original CPU adapter and full prepared/packed fingerprint checks.
  return original.prepared_inputs(plan,manifest)

def replace_once(text,before,after):
  assert text.count(before)==1,(before[:120],text.count(before));return text.replace(before,after)

def capture_source():
  text=inspect.getsource(original.execute)
  text=replace_once(text,"output.stem+'.agent-distill-command-targets'","output.stem+'.agent-distill-cohort-targets'")
  text=replace_once(text,"for split in ('train', 'validation', 'sealed_test')","for split in ('train', 'validation')")
  text=replace_once(text,'targets_sealed=True','sealed_drive_excluded=True, sealed_targets_opened=False')
  text=replace_once(text,"      prepared, packed = components.pad_inputs(prepared, packed, 2048)",
    "      prepared, packed = components.pad_inputs(prepared, packed, 2048)\n      assert c.fingerprint(packed) == row['padded_packed'], 'Changed full2048 padded IDs/mRoPE'")
  text=replace_once(text,"      np.savez(destination, **payload)","      c.storage(destination.parent, sum(value.nbytes for value in payload.values())+4096)\n      np.savez(destination, **payload)")
  text=replace_once(text,"    assert outcome['prefills'] == 9","    assert outcome['prefills'] == 12")
  text=replace_once(text,"    assert len(training) == 6 and len(sealed) == 3","    assert len(training) == 12 and len(sealed) == 0")
  text=replace_once(text,"c.write(targets/'agent-distill-command-train-validation.json'","c.write(targets/'agent-distill-cohort-train-validation.json'")
  text=replace_once(text,"    c.write(targets/'sealed_test/agent-distill-command-sealed-test.json', dict(common, samples=sealed, forbidden_for_fit_and_early_stopping=True))\n",'')
  text=replace_once(text,"str(targets/'agent-distill-command-train-validation.json')","str(targets/'agent-distill-cohort-train-validation.json')")
  text=replace_once(text,"      sealed_manifest=str(targets/'sealed_test/agent-distill-command-sealed-test.json'))","      sealed_manifest=None)")
  text=replace_once(text,"'--warmup', '0', '--runs', '9'","'--warmup', '0', '--runs', '12'")
  return text

exec(compile(capture_source(),str(Path(__file__).resolve())+':qualified-count-adapter','exec'),globals())

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,default=DEFAULT)
  parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-gpu',action='store_true');parser.add_argument('--check-imports',action='store_true')
  args=parser.parse_args();output=args.output.resolve()
  assert output.is_relative_to(c.ROOT) or output.is_relative_to(c.WORK/'agent-distill-cohort-gpu-job-0') or output.is_relative_to(c.WORK/'agent-distill-cohort-gpu-job-1')
  assert not output.exists() and not(args.execute_gpu and args.check_imports)
  c.storage(output.parent);output.parent.mkdir(parents=True,exist_ok=True)
  if 'tinygrad' in sys.modules:raise RuntimeError('Fresh process required before tinygrad imports')
  plan,manifest=preflight(args.plan);ready=prepared_inputs(plan,manifest)
  if not args.execute_gpu:
    imports=[]
    if args.check_imports:
      os.environ['DEV']='CPU';sys.path.insert(0,str(c.BASE/'tinygrad'));root=Path(plan['runtime_root'])
      for name in ('benchmark-pipeline','benchmark-fast','compact-inputs','quantize-packed','padded-prefill','agent-speculation-prefill'):
        value=c.module('distill_cohort_actual_import_'+name.replace('-','_'),root/'experiments'/(name+'.py'));imports.append(str(Path(value.__file__).resolve()))
      from tinygrad import Device
      assert not Device._opened_devices
    c.assert_no_devices();c.write(output,dict(preflight_passed=True,prepared_inputs=12,gpu_executed=False,training_executed=False,
      job=plan['job'],manifest_sha256=plan['inputs_manifest']['sha256'],imported_actual_modules=imports))
    print(json.dumps(dict(output=str(output),preflight_passed=True,job=plan['job'],gpu_executed=False),indent=2));return
  seed=plan['compiler_cache_seed'];assert c.sha(seed['path'])==seed['sha256']
  owned=output.with_suffix('.agent-distill-cohort-kernel-cache.db');assert not owned.exists()
  c.storage(output.parent,seed['bytes']+12*26215834);shutil.copyfile(seed['path'],owned)
  temporary=output.parent/(output.stem+'.agent-distill-cohort-tmp');temporary.mkdir(exist_ok=False)
  previous_env=dict(os.environ)
  try:
    for key,value in plan['environment'].items():
      if value is None:os.environ.pop(key,None)
      else:os.environ[key]=value
    os.environ['CACHEDB']=str(owned)
    os.environ['TMPDIR']=str(temporary)
    sys.path[:0]=[str(Path(plan['runtime_root'])/'experiments'),plan['runtime_root'],str(c.BASE/'tinygrad')]
    execute(plan,manifest,ready,output,args.plan)
  finally:
    os.environ.clear();os.environ.update(previous_env)

if __name__=='__main__':main()

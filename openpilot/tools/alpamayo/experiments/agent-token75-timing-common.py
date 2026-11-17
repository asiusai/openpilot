"""Shared CPU-only protocol utilities. One fixed75% numerical candidate only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-token75-timing-protocol'
SNAP = ROOT/'alpamayo'
FEXP = SNAP/'experiments'
OLD_ROOT = WORK/'token-current-source'
OLD_SNAP = OLD_ROOT/'alpamayo'
OLD_EXP = OLD_SNAP/'experiments'
PROTOCOL = ROOT/'agent-token75-timing-plan.json'
PY = Path('/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python')
POLICY = 'latest-full-history-75'
sys.dont_write_bytecode = True


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write_new(path,value):
  with Path(path).open('x') as stream: stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def require(value,message):
  if not value: raise ValueError(message)


def module(path,name):
  spec = importlib.util.spec_from_file_location(name,path)
  value = importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
  return value


def relocate(value,old,new):
  if isinstance(value,str): return value.replace(str(old),str(new))
  if isinstance(value,list): return [relocate(item,old,new) for item in value]
  if isinstance(value,dict): return {relocate(key,old,new):relocate(item,old,new) for key,item in value.items()}
  return value


def paths(command): return command[command.index('--samples')+1:command.index('--initial-noise')]


def set_option(command,name,value):
  flag='--'+name
  if flag in command: command[command.index(flag)+1]=str(value)
  else: command.extend((flag,str(value)))


def entries(report): return report['warmup']+report['runs']


def verify(plan,require_cpu=True):
  require(plan['policy']==POLICY and plan['warmup']==12 and plan['measured']==24,'Unexpected policy or query count')
  require(plan['recipe']['pad_prompt_to']==2112 and plan['recipe']['nominal_prefill'] is False,'Decoder/padding recipe changed')
  for path,digest in plan['frozen_sha256'].items(): require(sha(path)==digest,'Frozen protocol artifact changed: '+path)
  repository=plan['tinygrad']['path']
  require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()==plan['tinygrad']['revision'],'Pinned tinygrad revision changed')
  require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repository,text=True).strip(),'Pinned tinygrad has tracked changes')
  require(plan['native_continuous_Asius_validated'] is False and not plan['automatic_adoption_or_fullroute'],'Unexpected adoption claim')
  if require_cpu:
    report=read(plan['cpu_qualification'])
    require(report['cpu_only'] and report['actual_native_inputs']==36 and report['cpu_qualified'],'Missing actual-native CPU qualification')
    require(report['protocol_sha256']==sha(PROTOCOL),'CPU qualification belongs to another frozen protocol')


def validate_output(report,plan,arm):
  import numpy as np
  require(len(report['warmup'])==12 and len(report['runs'])==24,'Expected12warm24measured')
  rows=entries(report)
  require([row['sample'] for row in rows]==plan['samples'],'Native input order differs')
  require(report['native_preprocessing']['manifest_sha256']==plan['native_manifest_sha256'],'Native frame choices differ')
  require(report['native_preprocessing']['calibrations_sha256']==plan['native_calibrations_sha256'],'Calibration differs')
  require(report['pad_prompt_to']==2112 and report['vision_calibration_sha256'] is None,'Vision/padding recipe changed')
  require(report['prefill_calibration_sha256']==plan['fixed_language_scale_sha256'],'Fixed language scale changed')
  require(report['settings']['max_tokens']==64 and report['settings']['diffusion_steps']==3,'Reasoning or expert budget changed')
  require(report['settings'].get('coarse_timing',False) is False and 'prefill_speculation' not in report,'Decoder/timing semantics changed')
  require(report['trajectory_solver']=='euler' and report['vision_group_limit']==1,'Expert or vision scheduling changed')
  require(report['settings'].get('cache_rotary_inputs',False) is False and report['late_inputs'] is False,'Input scheduling changed')
  require(report['pixel_scale']==1 and report['history_pixel_scale']==1 and report['history_feature_scale']==1,'Image history was reduced')
  require(report['camera_profile']=='nvidia' and report['allow_missing_cameras'] and report['direct_byte_inputs'],'Native input recipe changed')
  require(report['lookup_draft'] and report['lookup_carry'] and report['lookup_fallback']==6 and report['decode_block']==6,'Ordinary lookup decoder changed')
  require(not report['summary']['profiling_enabled'],'Profiled run cannot qualify native timing')
  for index,row in enumerate(rows):
    prediction=row['prediction'];xyz=np.asarray(prediction['xyz']);rot=np.asarray(prediction['rot'])
    require(xyz.shape==(1,64,3) and rot.shape==(1,64,3,3),'Full trajectory shape changed')
    require(np.isfinite(xyz).all() and np.isfinite(rot).all(),'Nonfinite full trajectory')
    require(len(prediction['generated_token_ids'])==prediction['generated_tokens'],'Incomplete reasoning outputs')
    require(isinstance(prediction['reasoning'],str),'Missing genuine reasoning')
    if arm=='candidate':
      event=prediction['token_prune_policy']
      require(event['policy']==POLICY and event['query_ns']==plan['queries_ns'][index],'Wrong per-input policy or query')
      require(event['availability_source']=='native current-row capture_ns <= query_ns','Snapshot metadata replaced the actual native path')
      require(event['prompt_prefix_invariance_pass'] is False and event['older_image_retention']==.75 and event['latest_image_retention']==1,'Numerical policy changed')
  return rows

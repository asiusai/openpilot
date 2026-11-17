"""Parent-only frozen paired64 quality launcher. Default preflight is CPU-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
DEFAULT_PROTOCOL = WORK/'agent-history-nearest-data/agent-history-nearest-protocol.json'
sys.dont_write_bytecode = True


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(protocol):
  if protocol['classification'] != 'NUMERICAL INPUT POLICY' or protocol['queries'] != 64:
    raise ValueError('Unexpected input policy protocol')
  for path,digest in protocol['frozen_sha256'].items():
    if sha(path) != digest: raise ValueError('Frozen source/input changed: '+path)
  if protocol['automatic_speed_or_native_benchmark'] is not False: raise ValueError('Only quality trials are authorized')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
  parser.add_argument('--condition',choices=('floor','nearest'),required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--execute-gpu',action='store_true',help='PARENT ONLY; otherwise verify and print the command')
  args = parser.parse_args()
  protocol = json.loads(args.protocol.read_text())
  verify(protocol)
  command = protocol['quality_commands'][args.condition].copy()
  command[command.index('--output')+1] = str(args.output)
  if not args.execute_gpu:
    print(json.dumps(dict(cpu_only=True,gpu_executed=False,condition=args.condition,command=command,
      quality_qualified=False,protocol_sha256=sha(args.protocol)),indent=2))
    return
  if args.output.exists(): raise FileExistsError('Never overwrite a completed trial')
  invocation = json.loads(Path(command[command.index('--invocation')+1]).read_text())
  for key,value in invocation['environment'].items():
    if value is None: os.environ.pop(key,None)
    else: os.environ[key] = value
  for path in reversed(os.environ['PYTHONPATH'].split(os.pathsep)):
    if path: sys.path.insert(0,path)
  sys.path.insert(0,str(Path(command[1]).parent))
  original = sys.argv
  try:
    sys.argv = command[1:]
    runpy.run_path(command[1],run_name='__main__')
  finally: sys.argv = original
  report = json.loads(args.output.read_text())
  entries = report['warmup']+report['runs']
  if len(report['warmup']) != 2 or len(report['runs']) != 62: raise ValueError('Expected exactly64 quality queries')
  if [entry['sample'] for entry in entries] != protocol['sample_paths'][args.condition]: raise ValueError('Quality input order changed')
  if report['vision_calibration_sha256'] is not None: raise ValueError('Vision must retain dynamic scales')
  if report['pad_prompt_to'] != 2048 or report['settings']['diffusion_steps'] != 3: raise ValueError('Recipe changed')
  speculation = report['prefill_speculation']
  if speculation['mode'] != 'candidate' or speculation['prefill_attention'] != 'shuffle-fma': raise ValueError('Target-verified recipe changed')
  report['nearest_history_input_policy'] = dict(classification='NUMERICAL INPUT POLICY',condition=args.condition,
    protocol_sha256=sha(args.protocol),query_count=64,quality_passed=None,
    exact_original_nonimage_arrays=True,no_capture_after_query=True,calibration_asof_each_capture=True,
    policy='nearest native historical capture; original offsets; past tie; latest causal frame unchanged',
    measurement='Prepared calibrated64 input quality only; native load/calibration and CPU frame selection excluded from timing',
    required_gates='Unchanged5% ADE/FDE in all four windows, plus parent trajectory/speed/slowing/reasoning review')
  args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__': main()

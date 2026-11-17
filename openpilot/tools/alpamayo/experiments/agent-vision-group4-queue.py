"""Print the frozen parity pair. Only the parent may pass --execute-gpu."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('agent_vision_group4_queue_common',HERE/'agent-vision-group4-common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--execute-gpu',action='store_true')
  args=parser.parse_args()
  plan=c.read(c.PLAN)
  for relative,digest in plan['source_sha256'].items():c.require(c.sha(c.SNAP/relative)==digest,'Frozen source changed: '+relative)
  for path,digest in plan['input_files_sha256'].items():c.require(c.sha(path)==digest,'Frozen input/protocol changed: '+path)
  commands=c.read(c.ROOT/'agent-vision-group4-parent-commands.json')
  comparison=[str(c.PY),str(c.EXP/'agent-vision-group4-compare.py'),
    '--reference',str(c.WORK/'agent-vision-group4-group2-parity4/report.group4-audit.json'),
    '--candidate',str(c.WORK/'agent-vision-group4-group4-parity4/report.group4-audit.json'),
    '--output',str(c.ROOT/'agent-vision-group4-parity-comparison.json')]
  for command in [*commands,comparison]:print(shlex.join(command),flush=True)
  if not args.execute_gpu:return
  c.require(Path(__file__).resolve()==c.EXP/Path(__file__).name,'Execute frozen queue only')
  for command in commands:subprocess.run(command,check=True)
  subprocess.run(comparison,check=True)
  result=c.read(c.ROOT/'agent-vision-group4-parity-comparison.json')
  print(json.dumps(dict(exact=result['exact'],failed=result['failed'],timing_or_adoption_started=False),indent=2))
  if not result['exact']:sys.exit(2)


if __name__=='__main__':main()

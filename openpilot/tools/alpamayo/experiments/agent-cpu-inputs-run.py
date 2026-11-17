"""Optional adapter hook for a recorded pipeline; core sources stay unchanged."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--invocation',type=Path,required=True)
  parser.add_argument('--mode',choices=('reference','candidate'),default='candidate')
  parser.add_argument('--warmup',type=int,default=12)
  parser.add_argument('--runs',type=int,default=24)
  parser.add_argument('--compare-reference',type=Path)
  args=parser.parse_args()
  recorded=json.loads(args.invocation.read_text());command=recorded['command'][1:]
  for key,value in [('output',str(args.output)),('warmup',str(args.warmup)),('runs',str(args.runs))]:
    flag='--'+key
    if flag in command:command[command.index(flag)+1]=value
    else:command.extend((flag,value))
  if args.mode=='candidate':
    for flag in ('--cache-resize','--direct-byte-inputs'):
      if flag not in command:raise ValueError(f'Candidate requires the existing {flag} option')
  for key,value in recorded['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  instances=[]
  if args.mode=='candidate':
    original_spec=importlib.util.spec_from_file_location
    directory=Path(__file__).resolve().parent
    spec=original_spec('agent_cpu_inputs_adapter',directory/'agent-cpu-inputs-adapter.py')
    implementation=importlib.util.module_from_spec(spec);spec.loader.exec_module(implementation)
    source_digest=hashlib.sha256((directory/'fast-inputs.py').read_bytes()).digest()
    class Adapter(implementation.FastInputs):
      def __init__(self,*values,**options):
        super().__init__(*values,**options);instances.append(self)
    def hooked_spec(name,location,*values,**options):
      spec=original_spec(name,location,*values,**options);path=Path(location)
      if path.name!='fast-inputs.py' or path.resolve().parent!=Path(command[0]).resolve().parent:return spec
      if hashlib.sha256(path.read_bytes()).digest()!=source_digest:
        raise ValueError('Recorded pipeline FastInputs source differs from the validated adapter baseline')
      original_loader=spec.loader
      class Loader:
        def create_module(self,value):return original_loader.create_module(value)
        def exec_module(self,value):
          original_loader.exec_module(value);value.FastInputs=Adapter
      spec.loader=Loader();return spec
    importlib.util.spec_from_file_location=hooked_spec
  sys.argv=command
  try:runpy.run_path(command[0],run_name='__main__')
  finally:
    args.output.with_suffix('.cpu-inputs.json').write_text(json.dumps(dict(mode=args.mode,
      adapters=[instance.cache_state() for instance in instances]),indent=2)+'\n')
  if args.mode=='candidate' and not instances:raise AssertionError('CPU adapter hook was not used')
  if args.compare_reference:
    import numpy as np
    old,new=(json.loads(path.read_text()) for path in (args.compare_reference,args.output))
    expected={entry['sample']:entry['prediction'] for entry in old['warmup']+old['runs']}
    count=0
    for entry in new['warmup']+new['runs']:
      before,after=expected[entry['sample']],entry['prediction']
      assert before['generated_token_ids']==after['generated_token_ids']
      assert before['reasoning']==after['reasoning']
      for key in ('xyz','rot'):np.testing.assert_array_equal(after[key],before[key],err_msg=f'{entry["sample"]} {key}')
      count+=1
    if not count:raise AssertionError('No compared route predictions')
    args.output.with_suffix('.cpu-inputs-parity.json').write_text(json.dumps(dict(exact=True,predictions=count),indent=2)+'\n')


if __name__=='__main__':main()

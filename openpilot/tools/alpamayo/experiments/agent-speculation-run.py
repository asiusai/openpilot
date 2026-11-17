"""Run a recorded pipeline with optional complete prefill draft speculation.

This is a numerical candidate. Diagnostic audits and route comparisons are
required before accepting its accuracy or performance.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--invocation',type=Path,required=True)
  parser.add_argument('--mode',choices=('reference','candidate'),default='candidate')
  parser.add_argument('--audit',action='store_true')
  parser.add_argument('--warmup',type=int,default=12)
  parser.add_argument('--runs',type=int,default=24)
  parser.add_argument('--samples-json',type=Path)
  parser.add_argument('--compare-reference',type=Path)
  parser.add_argument('--prefill-attention',choices=('baseline','shuffle-fma'),default='baseline')
  parser.add_argument('--pad-prompt-to',type=int,default=2112)
  args = parser.parse_args()
  if args.pad_prompt_to <= 0: raise ValueError('Positive padded prompt bound required')
  if args.audit and args.mode != 'candidate': raise ValueError('Audit requires candidate mode')
  recorded = json.loads(args.invocation.read_text())
  command = recorded['command'][1:]
  for key,value in [('output',str(args.output)),('warmup',str(args.warmup)),('runs',str(args.runs)),('pad-prompt-to',str(args.pad_prompt_to))]:
    flag = '--'+key
    if flag in command: command[command.index(flag)+1] = value
    else: command.extend((flag,value))
  if args.samples_json:
    samples = json.loads(args.samples_json.read_text())
    begin,end = command.index('--samples')+1,command.index('--initial-noise')
    command[begin:end] = samples
  for flag in ('--prefill-calibration','--packed-quantize','--compact-inputs','--lookup-draft','--reduced-vocab'):
    if flag not in command: raise ValueError(f'Recorded invocation must include {flag}')
  for key,value in recorded['environment'].items():
    if value is None: os.environ.pop(key,None)
    else: os.environ[key] = value
  os.environ['PROFILE'] = '0'
  original_spec = importlib.util.spec_from_file_location
  experiment_dir = Path(__file__).resolve().parent
  def local(name):
    spec = original_spec(name.replace('-','_'),experiment_dir/(name+'.py'))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
  policy = local('agent-speculation-policy')
  checkpoint = Path(command[command.index('--checkpoint')+1])
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(str(checkpoint),local_files_only=True,padding_side='left')
  config = json.loads((checkpoint/'config.json').read_text())
  anchor = config['traj_ids']['future_start']
  draft = policy.validate_draft(tokenizer.encode(policy.DRAFT_TEXT,add_special_tokens=False),anchor)
  if draft != policy.CHECKPOINT_DRAFT_IDS: raise ValueError('Unexpected complete draft tokenization')
  if config['vlm_config']['image_token_id'] in draft: raise ValueError('Draft must contain no image tokens')
  vocab = set(json.loads(Path(command[command.index('--reduced-vocab')+1]).read_text()))
  if not set(draft)<=vocab: raise ValueError('Draft is outside the existing allowed reduced vocabulary')
  instances = []
  if args.mode == 'candidate' or args.prefill_attention != 'baseline':
    implementation = local('agent-speculation-prefill')
    def patch(module,path):
      if path.name == 'online-attention.py':
        original_install = module.install
        def install_attention(model,*values,**options):
          original_install(model,*values,**options)
          if args.prefill_attention != 'baseline':
            local('agent-attention-rowsoftmax').install(model,'prefill',args.prefill_attention,8)
        module.install = install_attention
      elif path.name == 'padded-prefill.py' and args.mode == 'candidate':
        module.pad_inputs = lambda prepared,packed,length: policy.padded_inputs(prepared,packed,length,draft,anchor)
        module.install = lambda model: implementation.install_prefill(model,draft,anchor,args.audit)
      elif path.name == 'lookup-decode.py' and args.mode == 'candidate':
        original = module.LookupDecoder
        class Decoder(original):
          def __init__(self,*values,**options):
            super().__init__(*values,**options)
            implementation.install_decoder(self,draft,anchor,args.audit)
            instances.append(self)
        module.LookupDecoder = Decoder
    def hooked_spec(name,location,*values,**options):
      spec = original_spec(name,location,*values,**options)
      path = Path(location)
      if path.resolve().parent != experiment_dir or path.name not in ('padded-prefill.py','lookup-decode.py','online-attention.py'):
        return spec
      original_loader = spec.loader
      class Loader:
        def create_module(self,value): return original_loader.create_module(value)
        def exec_module(self,value):
          original_loader.exec_module(value)
          patch(value,path)
      spec.loader = Loader()
      return spec
    importlib.util.spec_from_file_location = hooked_spec
  sys.argv = command
  try: runpy.run_path(command[0],run_name='__main__')
  finally:
    records = [record for decoder in instances for record in decoder.prefill_draft_records]
    result = dict(mode=args.mode,numerical_candidate=args.mode=='candidate',audit=args.audit,prefill_attention=args.prefill_attention,
                  padded_prompt_bound=args.pad_prompt_to,
                  timings_valid_for_throughput=not args.audit,draft=list(draft),records=records,
                  accepted=sum(record.get('accepted',False) for record in records),queries=len(records))
    if args.output.exists():
      report = json.loads(args.output.read_text())
      summary = report.setdefault('summary',{})
      summary['profiling_enabled'] = bool(summary.get('profiling_enabled') or args.audit)
      result['timings_valid_for_throughput'] = not summary['profiling_enabled']
      report['prefill_speculation'] = {key:value for key,value in result.items() if key != 'records'}
      args.output.write_text(json.dumps(report,indent=2)+'\n')
    args.output.with_suffix('.speculation.json').write_text(json.dumps(result,indent=2)+'\n')
  if args.compare_reference:
    import numpy as np
    old,new = (json.loads(path.read_text()) for path in (args.compare_reference,args.output))
    references = {entry['sample']:entry['prediction'] for entry in old['warmup']+old['runs']}
    rows = []
    for entry in new['warmup']+new['runs']:
      before,after = references[entry['sample']],entry['prediction']
      a,b = np.asarray(before['xyz']),np.asarray(after['xyz'])
      if not np.isfinite(b).all() or not np.isfinite(after['rot']).all(): raise AssertionError('Nonfinite trajectory')
      rows.append(dict(sample=entry['sample'],tokens_equal=before['generated_token_ids']==after['generated_token_ids'],
                       max_xyz_delta_m=float(np.abs(a-b).max()),mean_xyz_delta_m=float(np.abs(a-b).mean())))
    if not rows: raise AssertionError('No compared queries')
    comparison = dict(numerical_candidate=True,rows=rows,matched=len(rows),identical_tokens=sum(row['tokens_equal'] for row in rows),
                      route_accuracy_gate='Required separately with recorded motion and every reserved turn window')
    args.output.with_suffix('.speculation-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')


if __name__ == '__main__': main()

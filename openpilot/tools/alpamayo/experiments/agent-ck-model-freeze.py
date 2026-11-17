"""Build an exclusive protocol from completed CPU evidence. Never starts a device."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_model_common',EXP/'agent-ck-model-common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def draft(cpu):
  prepared=C.PREPARED;prepared.mkdir(exist_ok=False)
  for path in cpu.iterdir():
    if path.is_file(): shutil.copyfile(path,prepared/path.name)
  source=C.WORK/'padding-source/padding2048-quality.json'
  invocation=json.loads(source.read_text());cmd=invocation['command'];start=cmd.index('--samples')+1;end=cmd.index('--initial-noise')
  samples=cmd[start:end];assert len(samples)==64 and len(set(samples))==64
  selected=[samples[i] for i in C.INDICES]
  records=[]
  for index,sample in zip(C.INDICES,selected,strict=True):
    files={str(p):C.sha(p) for p in sorted(Path(sample).glob('*.npy'))}
    assert any(Path(p).name=='image_frames.npy' for p in files)
    records.append({'original64_index':index,'sample':sample,'file_sha256':files})
  cmd[start:end]=[selected[0]]
  for flag,value in [('--output',str(C.ROOT/'benchmark-unused.json')),('--warmup','0'),('--runs','1'),('--pad-prompt-to','2048')]:
    cmd[cmd.index(flag)+1]=value
  index=cmd.index('--vision-calibration');del cmd[index:index+2]
  assert cmd[1]==str(C.BASE/'experiments/benchmark-pipeline.py') and '--online-expert' not in cmd
  assert cmd[cmd.index('--steps')+1]=='3' and cmd[cmd.index('--decode-block')+1]=='6'
  env=invocation['environment'];env.update(ALPAMAYO_FP8_CACHE=str(C.CACHE),PYTHONPATH=f'{C.TINY}:{C.BASE}',
    CACHEDB=str(C.ROOT/'cache.db'),TMPDIR=str(C.ROOT/'tmp'),PARALLEL='1',BEAM='0',JITBEAM='0',LLVMOPT='1',DEBUG='0',
    PYTHONDONTWRITEBYTECODE='1',PROFILE='0',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
  invocation.pop('usb_node',None)
  C.write_json(prepared/'invocation.json',invocation)
  C.write_json(prepared/'screen8-inputs.json',records)
  C.write_json(prepared/'screen8-samples.json',selected)
  old=json.loads(C.OLD.read_text())
  plan={'schema':1,'scope':'first_actual_language_layer_layout_and_paired_micro_only','gpu_executed':False,
    'adoption_allowed':False,'timeout_seconds':180,'internal_timeout_seconds':170,
    'tinygrad':str(C.TINY),'tinygrad_commit':'89846dcfb383a98ee01cfec3bba440ae7dcf9b1e',
    'micro_evidence':{'path':str(C.MICRO),'sha256':C.sha(C.MICRO),'reused_without_restamping':True,
      'limits':'Passed seven synthetic full fixtures, 40 replay, mutation, tail, and validator gates qualify this binary only.'},
    'limits':old['limits'],'source_sha256':{},
    'launcher_command':[str(C.BASE/'experiments/agent-speculation-run.py'),'--output',str(C.ROOT/'benchmark-unused.json'),
      '--invocation',str(prepared/'invocation.json'),'--mode','candidate','--prefill-attention','shuffle-fma',
      '--pad-prompt-to','2048','--warmup','0','--runs','1'],
    'compiler_context':{'model':'Original WGP renderer instance and original worker serialization behavior; no compiler fix.',
      'standalone_micro':'Explicit private CU renderer inside exact-program guard; no model renderer assignment.',
      'producer':'Independently compiled exact CU and WGP associations; actual descriptor recorded for every call.'},
    'budget':{'historical_full_first_prediction_seconds':105.01143064000644,
      'scope':'Stops in layer 0 attention before O projection, remaining 27 layers, decoder, or trajectory.',
      'read_bytes':68423680,'poison_write_bytes':17045504,'reference':'One full actual-input CPU FP32 oracle.',
      'allowance_seconds':180,'limitation':'Estimate; exact IO and phase times are recorded. No throughput claim.'},
    'screen8_predeclared':{'runnable_now':False,'original64_source':str(source),'original64_source_sha256':C.sha(source),
      'indices':C.INDICES,'order':selected,'initial_decoder_history':[],
      'history_policy':'Fresh process per arm; empty initial history, same eight inputs/order. Each arm evolves its own verified decoder history.',
      'arms':['fresh selected shuffle-FMA original','fresh exact-shape CK numerical candidate'],
      'per_arm_timeout_seconds':600,'aggregate_ADE_ratio_max':1.05,'aggregate_FDE_ratio_max':1.05,
      'four_windows':[{'original64_indices':[lo,lo+15],'screen8_original_indices':[lo,lo+8],
        'ADE_ratio_max':1.05,'FDE_ratio_max':1.05} for lo in (0,16,32,48)],
      'inspect':['all generated text and token changes','speed profile','slowing','turn trajectory','all 28 selected calls and finite outputs'],
      'forbidden_comparator':'Candidate8 against historical64 predictions or historical decoder history.',
      'full64_requires':'Paired screen8 aggregate and each window pass, plus text/speed/slowing inspection; separate parent review.'},
    'remaining_gates':['Live all-28 integration, capture/replay/cleanup coverage and per-layer finite audits.',
      'Live Q/K/V mutation, causal/GQA, cache-tail poison, stale replay, and full output guard qualification after actual layout binding.',
      'Fresh same-order paired screen8 with unchanged 5% ADE/FDE aggregate and all four windows.',
      'Full64 route quality after screen review; actual full-pipeline timing and camera age; under100ms goal remains unmet.']}
  C.write_json(C.ROOT/'plan-prepared-final.json',plan)
  print(str(C.ROOT/'plan-prepared-final.json'))

def finalize(check):
  plan=json.loads((C.ROOT/'plan-prepared-final.json').read_text());previous=json.loads(C.OLD.read_text())
  plan['compiler_context']['cache_mode_omission']='to_program_key omits renderer.compiler features. Scoped compilation uses do_to_program with its private CU renderer.'
  for path,expected in previous['source_sha256'].items():
    assert C.sha(path)==expected, f'Previous frozen evidence changed: {path}'
  paths=set(previous['source_sha256'])
  paths.update(str(p) for p in C.BASE.rglob('*') if p.is_file() and p.suffix in ('.py','.c','.h','.json'))
  paths.update(str(p) for p in C.PREPARED.iterdir() if p.is_file())
  paths.update(str(p) for p in EXP.glob('agent-ck-model-*.py') if p.name!='agent-ck-model-parent.py')
  paths.update(map(str,[C.OLD,C.MICRO,check,C.CACHE/'weights.bin',C.CACHE/'index.json',C.CHECKPOINT/'config.json',
    C.CHECKPOINT/'model.safetensors.index.json',C.WORK/'padding-source/padding2048-quality.json',
    C.WORK/'prefill-calibration/report.prefill-scales.json',C.WORK/'quality-samples.json',EXP/'agent-expert-combined-cache-common.py']))
  for row in json.loads((C.PREPARED/'screen8-inputs.json').read_text()): paths.update(row['file_sha256'])
  invocation=json.loads((C.PREPARED/'invocation.json').read_text())
  for flag in ('--initial-noise','--reduced-vocab'): paths.add(invocation['command'][invocation['command'].index(flag)+1])
  # BF16 restoration reads the original checkpoint shards; bind those bytes too.
  index=json.loads((C.CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map']
  paths.update(str(C.CHECKPOINT/name) for name in set(index.values()))
  paths.update(str(p) for p in C.CHECKPOINT.iterdir() if p.is_file() and p.suffix in ('.json','.model'))
  plan['source_sha256']={str(Path(p)):C.sha(p) for p in sorted(paths)}
  assert subprocess.check_output(['git','-C',str(C.TINY),'rev-parse','HEAD'],text=True).strip()==plan['tinygrad_commit']
  proof=json.loads(check.read_text());assert proof['gpu_executed'] is False and proof['count']>=20
  plan['cpu_checks']={'path':str(check),'sha256':C.sha(check),'count':proof['count']}
  C.write_json(C.PROTOCOL,plan)
  print(json.dumps({'protocol':str(C.PROTOCOL),'sha256':C.sha(C.PROTOCOL),'bound_files':len(paths)}))

if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--cpu',type=Path);parser.add_argument('--check',type=Path)
  args=parser.parse_args();assert bool(args.cpu)!=bool(args.check)
  draft(args.cpu) if args.cpu else finalize(args.check)

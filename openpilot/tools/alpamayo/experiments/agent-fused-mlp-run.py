"""Prepare immutable replay manifests; parent-only reference/candidate route runs.

No subprocess or device operations occur in prepare mode. GPU modes must run
through the parent's serialized wrapper. Recorded controls remain unchanged;
only the output path and this optional fusion hook differ between paired runs.
"""
import argparse
from contextlib import ExitStack
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import struct
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
MICRO = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-packed-epilogue-kernels/report.json')
ENTRIES = {'benchmark-pipeline.py','agent-speculation-run.py','agent-cpu-inputs-run.py'}


def sha(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value): Path(path).write_text(json.dumps(value,indent=2)+'\n')


def load(path,name):
  spec = importlib.util.spec_from_file_location(name,path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def option(command,name):
  flag = '--'+name
  assert command.count(flag) == 1, f'Expected one {flag}'
  return command[command.index(flag)+1]


def micro_gate(path,required_rows=(2027,2112)):
  report = json.loads(Path(path).read_text())
  assert report['mode'] == 'gpu' and report['gpu_executed'] and report['complete']
  assert report['tinygrad_revision'] == '89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
  assert required_rows and set(required_rows) <= {2027,2048,2112}
  required = {(17,48,32),(65,80,64),*((r,9216,2048) for r in required_rows)}
  rows = {tuple(row['shape_mnk']):row for row in report['cases']}
  assert required <= set(rows)
  for shape in required:
    row = rows[shape]
    selected = row['variants']['transpose4x4']
    assert selected['status'] in ('exact','exact-single') and row['public_gemm_relu_square_quantize_packed_exact']
    assert {c['state'] for c in selected['checks']} == {0,1,2,3}
    assert all(c['exact_bytes'] and (not row['integer_control'] or c['independent_integer']) for c in selected['checks'])
    if not row['integer_control']: assert selected['changed_inputs_and_scales_after_capture_exact']
  probes = [row for row in report['epilogue_probes'] if row['variant'] == 'transpose4x4']
  assert probes and all(row['exact_bytes'] for row in probes)
  for source,digest in report['source_sha256'].items(): assert sha(source) == digest, f'Microcheck source changed: {source}'
  kernel = next(p for p in report['source_sha256'] if Path(p).name == 'agent-packed-epilogue.py')
  return {'variant':'transpose4x4','kernel':kernel,'report':str(Path(path).resolve()),'report_sha256':sha(path),
          'qualified_rows':list(required_rows),
          'tested_shapes':[list(s) for s in sorted(required)],'old_fused_enabled':False}


def make_manifest(invocation,micro_report=MICRO):
  """Pin recorded commands, nested wrappers, source trees, and small control files."""
  invocation = Path(invocation).resolve()
  files,chain,seen = {},[],set()
  def add(path):
    path = Path(path).resolve()
    assert path.is_file(), f'Missing replay input: {path}'
    files[str(path)] = sha(path)
  def record(path):
    path = Path(path).resolve()
    assert path not in seen and len(chain) < 4, 'Unsupported invocation cycle/depth'
    seen.add(path); add(path)
    saved = json.loads(path.read_text())
    command = saved['command']
    assert len(command) > 2 and Path(command[0]).is_file()
    program = Path(command[1]).resolve()
    assert program.name in ENTRIES, f'Unsupported recorded entry: {program.name}'
    assert '--output' in command
    assert not any(flag in command for flag in ('--capture-prefill-scales','--profile-kernels','--profile-host'))
    chain.append({'invocation':str(path),'program':str(program),'recorded':saved})
    for root in (program.parent,program.parent.parent):
      for pattern in ('*.py','*.c'):
        for source in root.glob(pattern): add(source)
    for name in ('prefill-calibration','initial-noise','reduced-vocab','native-manifest','native-calibrations',
                 'samples-json','vision-calibration','validation-reference','compare-reference'):
      if '--'+name in command: add(option(command,name))
    if '--checkpoint' in command:
      root = Path(option(command,'checkpoint'))
      for name in ('config.json','model.safetensors.index.json'): add(root/name)
    if program.name != 'benchmark-pipeline.py': record(option(command,'invocation'))
    else:
      for flag in ('--packed-quantize','--prefill-calibration','--llvm-wgp'): assert flag in command, f'Missing {flag}'
      assert '--row-quantize' not in command and '--fused-qkv' not in command
      assert '--initial-noise' in command, 'Paired diffusion requires recorded exact initial noise'
      scales = json.loads(Path(option(command,'prefill-calibration')).read_text())['scales']
      assert scales and all(math.isfinite(v) and v > 0 for v in scales.values())
      hook = load(HERE/'agent-fused-mlp.py','agent_fused_mlp_manifest_hook')
      hook.require_source(program.parent/'quantize-packed.py','quantize-packed.py')
      hook.require_source(program.parent.parent/'model.py','model.py')
      hook.require_source(program.parent.parent/'amd_gemm.py','amd_gemm.py')
  record(invocation)
  # Outermost launcher wins when it explicitly overrides the padding bound.
  padded_rows = next((int(option(row['recorded']['command'],'pad-prompt-to')) for row in chain
                     if '--pad-prompt-to' in row['recorded']['command']),2112)
  micro = micro_gate(micro_report,(padded_rows,))
  if padded_rows == 2048:
    assert all('--vision-calibration' not in row['recorded']['command'] and '--capture-vision-scales' not in row['recorded']['command'] for row in chain), '2048 qualification requires dynamic vision'
    outer_command = chain[0]['recorded']['command']
    assert Path(chain[0]['program']).name == 'agent-speculation-run.py'
    assert option(outer_command,'mode') == 'candidate' and option(outer_command,'prefill-attention') == 'shuffle-fma'
  add(micro_report); add(micro['kernel'])
  for source in HERE.glob('agent-fused-mlp*.py'): add(source)
  # PYTHONPATH identifies the pinned tinygrad checkout without importing it.
  tg_candidates = [Path(p) for row in chain for p in (row['recorded'].get('environment',{}).get('PYTHONPATH') or '').split(os.pathsep)
                   if p and (Path(p)/'tinygrad/engine/jit.py').is_file()]
  assert tg_candidates and len(set(p.resolve() for p in tg_candidates)) == 1
  tg = tg_candidates[0].resolve()
  for relative in ('tinygrad/engine/jit.py','tinygrad/engine/realize.py','tinygrad/runtime/support/hcq2.py',
                   'tinygrad/runtime/ops_amd.py','tinygrad/runtime/support/compiler_amd.py','tinygrad/uop/ops.py'):
    add(tg/relative)
  for row in chain:
    environment = row['recorded'].get('environment',{})
    assert str(environment.get('PROFILE') or '0') == '0'
  return {'format':1,'invocation':str(invocation),'chain':chain,'files':dict(sorted(files.items())),
          'helper_sha256':{p.name:sha(p) for p in HERE.glob('agent-fused-mlp*.py')},
          'micro':micro,'tinygrad_root':str(tg),'variant':'transpose4x4',
          'mutable_override_policy':'Only final output path differs; no warmup/run/sample/precision overrides',
          'fixture_policy':'Recorded immutable samples and native frame cache are reused by path; their bulk contents are not rehashed here.',
          'checkpoint_policy':'Checkpoint config and weight index are pinned; bulk checkpoint/cache weights remain the parent frozen fixtures.'}


def verify_manifest(manifest):
  assert manifest['format'] == 1 and manifest['variant'] == 'transpose4x4'
  for name,digest in manifest['helper_sha256'].items(): assert sha(HERE/name) == digest, f'Loaded helper differs from frozen manifest: {name}'
  for path,digest in manifest['files'].items(): assert sha(path) == digest, f'Frozen replay input changed: {path}'
  for row in manifest['chain']:
    assert json.loads(Path(row['invocation']).read_text()) == row['recorded']


def exact(a,b,path='prediction'):
  assert type(a) is type(b), f'Type changed: {path}'
  if isinstance(a,dict):
    assert a.keys() == b.keys(), f'Fields changed: {path}'
    for key in a: exact(a[key],b[key],path+'.'+key)
  elif isinstance(a,list):
    assert len(a) == len(b), f'Length changed: {path}'
    for i,(x,y) in enumerate(zip(a,b,strict=True)): exact(x,y,f'{path}[{i}]')
  elif isinstance(a,float):
    assert math.isfinite(a) and math.isfinite(b), f'Nonfinite value: {path}'
    assert struct.pack('<d',a) == struct.pack('<d',b), f'Float bits changed: {path}'
  else: assert a == b, f'Value changed: {path}'


def compare(actual,reference):
  count = 0
  for section in ('warmup','runs'):
    assert len(actual[section]) == len(reference[section]), 'Paired sequence lengths differ'
    for a,b in zip(actual[section],reference[section],strict=True):
      for key in ('sample','input_latest_timestamp_us'): exact(a[key],b[key],section+'.'+key)
      # Compare every prediction field except measured durations, including
      # prompt lengths, tokens, reasoning, xyz, rot, and future added fields.
      aa,bb = ({k:v for k,v in entry['prediction'].items() if k != 'timings_ms'} for entry in (a,b))
      for key in ('generated_token_ids','reasoning','xyz','rot'): assert key in aa
      exact(aa,bb)
      count += 1
  assert count and actual['runs'], 'No paired measured predictions'
  return {'predictions':count,'all_prediction_fields_except_timings_exact':True,'float_bits_exact':True,
          'route_quality_acceptance':'Parent must also run the reserved route quality gates'}


def captured_counts(model):
  from tinygrad.uop.ops import Ops
  objects = [('prefill_jit',getattr(model,'prefill_jit',None))]
  padded = getattr(model,'padded_prefill',None)
  if padded is not None:
    objects.append(('padded_prefill.jit',getattr(padded,'jit',None)))
    base = getattr(padded,'baseline',None)
    if base is not None: objects.append(('padded_prefill.baseline.jit',getattr(base,'jit',None)))
  result = []
  for name,jit in objects:
    captured = getattr(jit,'captured',None)
    if captured is None: continue
    names = []
    for call in captured._linear.src:
      call = call.without_after
      aux = getattr(call.arg,'aux',None)
      if hasattr(aux,'kernels'): names.extend(str(k[1]) for k in aux.kernels)
      elif call.op is Ops.CALL and call.body.op is Ops.PROGRAM: names.append(call.body.src[0].arg.name)
    result.append({'jit':name,'captured_fused_fc1_kernels':sum(n.startswith('agent_packed_epilogue') for n in names),
                   'captured_kernel_count':len(names)})
  return result


def run(args,manifest):
  verify_manifest(manifest)
  assert not args.output.exists(), 'Use a fresh report path'
  reference = json.loads(args.reference.read_text()) if args.reference else None
  if args.mode == 'candidate':
    assert reference is not None and reference['agent_fused_mlp']['mode'] == 'reference'
    assert reference['agent_fused_mlp']['passed'] and reference['agent_fused_mlp']['hooks_restored']
    assert reference['agent_fused_mlp']['manifest_sha256'] == sha(args.manifest)
  outer = manifest['chain'][0]['recorded']
  assert Path(sys.executable).resolve() == Path(outer['command'][0]).resolve(), 'Use the recorded Python executable'
  command = list(outer['command'][1:])
  command[command.index('--output')+1] = str(args.output)
  args.output.parent.mkdir(parents=True,exist_ok=True)
  status_path = args.output.with_suffix('.agent-fused-mlp.json')
  status = {'mode':args.mode,'manifest_sha256':sha(args.manifest),'variant':'transpose4x4','resolved_command':command,
            'installations':[],'captured_graphs':[],'paired_exact':None,'passed':False,'hooks_restored':False,
            'kernel_evidence':manifest['micro'],'pipeline_speed_claim':False}
  models,packed_installations,sessions = {},{},[]
  original_spec = importlib.util.spec_from_file_location
  hook = load(HERE/'agent-fused-mlp.py','agent_fused_mlp_runtime_hook')
  source_paths = set(manifest['files'])
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ,dict(os.environ),clear=True))
      for key,value in outer.get('environment',{}).items():
        if value is None: os.environ.pop(key,None)
        else: os.environ[key] = str(value)
      assert os.environ.get('PROFILE','0') == '0'
      sys_path = list(sys.path)
      stack.callback(lambda:sys.path.__setitem__(slice(None),sys_path))
      # Existing wrappers may set environment later. Ensure imports use the
      # frozen leaf model and the same pinned tinygrad checkout in every mode.
      leaf = Path(manifest['chain'][-1]['program'])
      sys.path[:0] = [manifest['tinygrad_root'],str(leaf.parent.parent),str(leaf.parent)]
      def arm(model):
        if id(model) in models: return
        models[id(model)] = model
        assert id(model) in packed_installations, 'Packed fixed quantizer was not installed before decoder initialization'
        module,scales = packed_installations[id(model)]
        if args.mode == 'candidate':
          from tinygrad import Device
          # Import only after the recorded wrappers establish their own runtime.
          kernel = load(manifest['micro']['kernel'],'agent_fused_mlp_passed_kernel')
          session = stack.enter_context(hook.install(model,module,kernel,Device[Device.DEFAULT],scales,
            qualified_rows=manifest['micro'].get('qualified_rows',(2027,2112))))
          sessions.append(session); status['installations'].append(session.stats)
      def after_load(module,path):
        if path.name == 'quantize-packed.py':
          original = module.install
          @functools.wraps(original)
          def install(model,*values,**kwargs):
            bound = __import__('inspect').signature(original).bind(model,*values,**kwargs); bound.apply_defaults()
            assert not bound.arguments['capture_scales'] and bound.arguments['fixed_scales']
            result = original(model,*values,**kwargs)
            assert id(model) not in packed_installations
            packed_installations[id(model)] = (module,dict(bound.arguments['fixed_scales']))
            return result
          stack.enter_context(patch.object(module,'install',install))
        elif path.name in ('draft-decode.py','block-decode.py'):
          cls = module.DraftDecoder if path.name == 'draft-decode.py' else module.BlockDecoder
          original = cls.__init__
          @functools.wraps(original)
          def initialize(decoder,model,*values,**kwargs):
            original(decoder,model,*values,**kwargs)
            arm(model)
          stack.enter_context(patch.object(cls,'__init__',initialize))
      def spec_hook(name,location,*values,**kwargs):
        spec = original_spec(name,location,*values,**kwargs)
        if spec is None or location is None: return spec
        path = Path(location).resolve()
        if path.name not in ('quantize-packed.py','draft-decode.py','block-decode.py'): return spec
        assert str(path) in source_paths and sha(path) == manifest['files'][str(path)], 'Unpinned installer import'
        original_loader = spec.loader
        class Loader:
          def create_module(self,value): return original_loader.create_module(value)
          def exec_module(self,value): original_loader.exec_module(value); after_load(value,path)
        spec.loader = Loader()
        return spec
      stack.enter_context(patch.object(importlib.util,'spec_from_file_location',spec_hook))
      stack.enter_context(patch.object(sys,'argv',command))
      write(status_path,status)
      runpy.run_path(command[0],run_name='__main__')
      assert len(models) == len(packed_installations) == 1
      model = next(iter(models.values()))
      status['captured_graphs'] = captured_counts(model)
      if manifest['micro'].get('qualified_rows') == [2048]:
        actual = json.loads(args.output.read_text())
        assert actual['pad_prompt_to'] == 2048 and actual['vision_calibration_sha256'] is None and not actual['capture_vision_scales']
        assert actual['prefill_speculation']['mode'] == 'candidate' and actual['prefill_speculation']['prefill_attention'] == 'shuffle-fma'
      if args.mode == 'candidate':
        assert len(sessions) == 1
        stats = sessions[0].stats
        assert len(stats['layer_hits']) == 28 and len(set(stats['layer_hits'].values())) == 1
        assert stats['python_trace_fc1'] == stats['python_trace_fc2'] == stats['python_trace_fused_quantize'] == stats['layer_completions'] > 0
        assert not stats['failed_text_layers']
        assert set(map(int,stats['shape_hits'])) <= set(manifest['micro'].get('qualified_rows',(2027,2112)))
        assert any(g['captured_fused_fc1_kernels'] == 28 for g in status['captured_graphs']), 'No complete fused prefill graph was captured'
        status['paired_exact'] = compare(json.loads(args.output.read_text()),reference)
      else: assert all(g['captured_fused_fc1_kernels'] == 0 for g in status['captured_graphs'])
    status['hooks_restored'] = all(session.stats['methods_restored'] for session in sessions)
    assert importlib.util.spec_from_file_location is original_spec and status['hooks_restored']
    verify_manifest(manifest)
    status['passed'] = True
    report = json.loads(args.output.read_text())
    report['agent_fused_mlp'] = status
    write(args.output,report)
  except BaseException as exc:
    status['failure'] = f'{type(exc).__name__}: {exc}'
    status['hooks_restored'] = importlib.util.spec_from_file_location is original_spec and all(s.stats['methods_restored'] for s in sessions)
    raise
  finally: write(status_path,status)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode',choices=('prepare','reference','candidate'),required=True)
  parser.add_argument('--output',required=True,type=Path)
  parser.add_argument('--invocation',type=Path)
  parser.add_argument('--manifest',type=Path)
  parser.add_argument('--micro-report',type=Path,default=MICRO)
  parser.add_argument('--reference',type=Path)
  args = parser.parse_args()
  if args.mode == 'prepare':
    assert args.invocation is not None and not args.output.exists()
    manifest = make_manifest(args.invocation,args.micro_report)
    write(args.output,manifest)
    print(json.dumps({'manifest':str(args.output),'pinned_files':len(manifest['files']),'gpu_executed':False}))
  else:
    assert args.manifest is not None and args.invocation is None
    assert args.mode == 'reference' or args.reference is not None
    run(args,json.loads(args.manifest.read_text()))


if __name__ == '__main__': main()

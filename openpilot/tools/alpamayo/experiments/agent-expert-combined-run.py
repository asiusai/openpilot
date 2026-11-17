"""Parent-only fresh four-query composition gate, then matched native36."""
import argparse
from contextlib import ExitStack
import importlib.util
import inspect
import os
from pathlib import Path
import pickle
import runpy
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def common():
  spec = importlib.util.spec_from_file_location('expert_combined_run_common', HERE/'agent-expert-combined-common.py')
  value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
  return value


def original_compare(C):
  p,_ = C.pv()
  return p.usb().load('agent-expert-pv-route-run')


def compare(C, before, after, planned, old_spec, new_spec):
  # Remove only this experiment's arm-specific metadata. The frozen canonical
  # comparator retains every other nonmeasurement field, including unknown fields.
  reports = [{k:v for k,v in r.items() if k != 'agent_expert_combined'} for r in (before,after)]
  result = original_compare(C).compare_reports(*reports, planned, old_spec, new_spec)
  if 'agent_expert_combined' in before and 'agent_expert_combined' in after:
    for name in ('alpamayo_attention_scores','alpamayo_attention_softmax'):
      hashes = [{v for cap in r['agent_expert_combined']['pv']['captures'] for v in cap['binaries'][name]} for r in (before,after)]
      assert hashes[0] == hashes[1], 'Non-PV program changed: '+name
  return result


def validate_pv(info, candidate, evidence, queries):
  assert info['candidate'] is candidate and info['evidence'] == evidence
  assert info['cleanup_exact'] and info['observer_cleanup_exact'] and not info['active']
  assert info['trajectory_calls'] >= queries and info['unqualified_trajectory_calls'] == info['outside_scope_selected'] == 0
  assert info['traces'] and all(t == {'pv':84,'qk':84} for t in info['traces'])
  assert info['bound_program_calls'] == (84*len(info['traces']) if candidate else 0)
  expected = evidence['candidate' if candidate else 'current']
  assert info['captures']
  for cap in info['captures']:
    assert cap['pv_calls'] == cap['qk_calls'] == cap['softmax_calls'] == 84
    assert cap['target'] == evidence['target'] and cap['program_keys'] == [expected['program_key']]
    assert cap['pv_full_description'] == expected
    assert cap['binaries'][expected['name']] == [expected['binary_sha256']]
    assert cap['binaries']['alpamayo_attention_scores'] and cap['binaries']['alpamayo_attention_softmax']


def validate(C, report, spec, planned, stage, candidate, evidence, digest):
  meta = report['agent_expert_combined']
  assert meta['protocol_sha256'] == digest and meta['stage'] == stage and meta['candidate'] is candidate
  assert meta['passed'] and not meta.get('failure') and meta['hooks_restored'] and meta['coverage_validated']
  assert meta['plan'] == planned
  validate_pv(meta['pv'],candidate,evidence,planned['warmup']+planned['runs'])
  assert meta['coverage']['all_28_mlp_and_50_vision_projections'] and meta['coverage']['quantizer_frame_empty']
  assert meta['coverage']['vision_batches_exercised'] == [2]
  assert meta['coverage']['lookup_initial'] == [{'history':[], 'vision_cache_empty':True,
    'vision_encoders_empty':True,'vision_stitchers_empty':True}]
  assert all(row['restored'] for row in meta['restorations'])
  assert report['arch'] == 'gfx1200' and report['settings']['diffusion_steps'] == 3 and report['trajectory_solver'] == 'euler'
  assert report['expert_kernels'] and report['llvm_wgp'] and not report['online_expert']
  assert report['pixel_scale'] == report['history_pixel_scale'] == report['history_feature_scale'] == 1
  assert report['vision_calibration_sha256'] is None and not report['capture_vision_scales']
  assert report['runtime_environment']['AM_POWER_LIMIT'] == '144' and report['runtime_environment']['AM_PERF_LEVEL'] == 'auto'
  assert report['pad_prompt_to'] == 2048 and report['batched_vision'] and report['group_vision'] and report['vision_group_limit'] == 2
  assert not report['profiling'] and not report['summary']['profiling_enabled']
  expected_sources = {p.name:C.sha(p) for p in [*(C.MLP/'experiments').glob('*.py'), *(C.MLP/'experiments').glob('*.c'), C.MLP/'model.py']}
  assert report['source_sha256'] == expected_sources
  for name,digest_source in expected_sources.items(): assert C.sha(Path(report['source_snapshot'])/name) == digest_source
  assert report['prefill_calibration_sha256'] == C.sha(C.option(planned['terminal_command'],'--prefill-calibration'))
  if stage == 'native':
    assert meta['coverage']['sampler_calls'] > 0 and meta['coverage']['sampler']['misses'] > 0
    for field,flag in (('manifest','--native-manifest'),('calibrations','--native-calibrations')):
      assert report['native_preprocessing'][field+'_sha256'] == C.sha(C.option(planned['terminal_command'],flag))
    stamps = [r['input_latest_timestamp_us'] for r in report['warmup']+report['runs']]
    assert all(b > a for a,b in zip(stamps,stamps[1:]))
  else:
    assert meta['coverage']['sampler_calls'] == 0
    records = spec['records']
    assert any(r['accepted'] for r in records) and any(not r['accepted'] for r in records), 'Require genuine target acceptance and fallback'
  return compare(C,report,report,planned,spec,spec)


def import_probe(C):
  C.paths()
  from tinygrad import Device
  assert not Device._opened_devices
  namespace = runpy.run_path(str(C.MLP/'experiments/benchmark-pipeline.py'),run_name='expert_combined_cpu_import_probe')
  exec('import wgp_compiler',namespace)
  module = namespace['wgp_compiler']
  assert Path(module.__file__).resolve() == C.MLP/'experiments/wgp_compiler.py'
  compiler = module.WGPCompiler('gfx1200')
  assert type(pickle.loads(pickle.dumps(compiler))) is type(compiler)
  assert not Device._opened_devices
  return {'actual_runpy_wgp_import_and_pickle':True,'opened_devices':[]}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--stage',choices=('quality','native'),required=True)
  p.add_argument('--variant',choices=('reference','candidate'),required=True)
  p.add_argument('--protocol',type=Path,required=True)
  p.add_argument('--protocol-sha',required=True)
  p.add_argument('--reference',type=Path)
  p.add_argument('--plan',action='store_true')
  args = p.parse_args(); assert sys.flags.optimize == 0
  C = common(); data,evidence = C.protocol(args.protocol,args.protocol_sha)
  planned = C.plan(data,args.stage,args.output)
  assert not args.output.exists()
  if args.plan:
    C.write(args.output,{'complete':True,'plan_only':True,'gpu_executed':False,'plan':planned,
      'import_path':import_probe(C),'protocol_sha256':args.protocol_sha},new=True)
    return
  candidate = args.variant == 'candidate'
  reference = C.read(args.reference) if args.reference else None
  if candidate:
    assert reference is not None
    validate(C,reference,C.read(args.reference.with_suffix('.speculation.json')),
      C.plan(data,args.stage,args.reference),args.stage,False,evidence,args.protocol_sha)
    assert reference['agent_expert_combined']['plan']['pair_identity_sha256'] == planned['pair_identity_sha256']
  else: assert reference is None
  invocation = C.read(data['stages'][args.stage]['invocation'])
  assert Path(sys.executable).resolve() == Path(invocation['command'][0]).resolve()
  status_path = args.output.with_suffix('.agent-expert-combined.json')
  status = {'stage':args.stage,'candidate':candidate,'protocol_sha256':args.protocol_sha,'plan':planned,
    'passed':False,'coverage_validated':False,'hooks_restored':False,'paired_exact':None,
    'prior_evidence':data['prior'],'pv_evidence':evidence,'new_scope':'4 fresh composition queries then native36; no paced adoption'}
  original_spec = importlib.util.spec_from_file_location
  composition = None
  C.write(status_path,status,new=True)
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ,dict(os.environ),clear=True))
      for key,value in invocation['environment'].items():
        if value is None: os.environ.pop(key,None)
        else: os.environ[key] = str(value)
      search = sys.path[:]; stack.callback(lambda:sys.path.__setitem__(slice(None),search))
      C.paths()
      installer = C.load(HERE/'agent-expert-combined-install.py','expert_combined_runtime_install')
      composition = installer.factory(C)(args.stage,candidate,evidence)
      with composition:
        stack.enter_context(patch.object(sys,'argv',planned['effective_command']))
        runpy.run_path(sys.argv[0],run_name='__main__')
        chain = C.load(C.EXP/'agent-exact-combined-run.py','expert_combined_chain_audit')
        chain.hook_chain(composition)
        status['coverage'] = composition.coverage()
      status['pv'] = composition.pv_handle.info
      status['restorations'] = composition.restorations
      status['hooks_restored'] = (importlib.util.spec_from_file_location is original_spec and
        all(r['restored'] for r in composition.restorations) and composition.session.stats['methods_restored'])
      assert status['hooks_restored']
      status['coverage_validated'] = status['passed'] = True
      report = C.read(args.output); report['agent_expert_combined'] = status
      spec = C.read(args.output.with_suffix('.speculation.json'))
      validate(C,report,spec,planned,args.stage,candidate,evidence,args.protocol_sha)
      if candidate:
        status['paired_exact'] = compare(C,reference,report,planned,C.read(args.reference.with_suffix('.speculation.json')),spec)
    C.protocol(args.protocol,args.protocol_sha)
    C.write(args.output,report)
  except BaseException as exc:
    status.update(passed=False,coverage_validated=False,failure={'type':type(exc).__name__,'message':str(exc)})
    raise
  finally:
    status['hooks_restored'] = importlib.util.spec_from_file_location is original_spec and (composition is None or
      all(r['restored'] for r in composition.restorations) and (composition.session is None or composition.session.stats['methods_restored']))
    if composition is not None:
      status['restorations'] = composition.restorations
      if composition.pv_handle is not None: status['pv'] = composition.pv_handle.info
    C.write(status_path,status)
    if args.output.exists():
      report = C.read(args.output); report['agent_expert_combined'] = status; C.write(args.output,report)


if __name__ == '__main__': main()

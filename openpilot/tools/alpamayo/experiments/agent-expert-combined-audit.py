"""CPU audit of frozen report validators using explicitly synthetic report metadata."""
import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch

ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-expert-combined-source')
DIGEST = '4e11ba366c177e61220f71703981834ebaa23fea8f2375bb22cd52d46ef4820e'
spec = importlib.util.spec_from_file_location('expert_combined_frozen_audit_common',ROOT/'agent-expert-combined-common.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


def main():
  data,evidence = C.protocol(C.PROTOCOL,DIGEST)
  R = C.load(ROOT/'agent-expert-combined-run.py','expert_combined_frozen_audit_runner')
  rejections,valid = [],[]
  def reject(label,fn):
    try: fn()
    except (AssertionError,ValueError,KeyError): rejections.append(label)
    else: raise AssertionError('Invalid report passed: '+label)
  for stage in C.COUNTS:
    path = C.WORK/f'agent-exact-combined-{stage}-candidate/report.json'
    report = C.read(path); old = report.pop('agent_exact_combined')
    side = C.read(path.with_suffix('.speculation.json'))
    if stage == 'quality':
      rows = report['warmup']+report['runs']; rows = [rows[i] for i in C.INDICES]
      report['warmup'],report['runs'] = rows[:2],rows[2:]
      side['records'] = [side['records'][i] for i in C.INDICES]
      side['queries'] = 4; side['accepted'] = sum(r['accepted'] for r in side['records'])
    planned = C.plan(data,stage,path)
    coverage = old['coverage']
    coverage.update(all_28_mlp_and_50_vision_projections=True,quantizer_frame_empty=True,vision_batches_exercised=[2],
      lookup_initial=[{'history':[],'vision_cache_empty':True,'vision_encoders_empty':True,'vision_stitchers_empty':True}])
    paired = []
    for candidate in (False,True):
      current = copy.deepcopy(report)
      pv_path = C.WORK/f'agent-expert-pv-qualified-{stage}-{"candidate" if candidate else "reference"}/report.json'
      info = C.read(pv_path)['agent_expert_pv']['hooks'][0]
      current['agent_expert_combined'] = {'protocol_sha256':DIGEST,'stage':stage,'candidate':candidate,'passed':True,
        'hooks_restored':True,'coverage_validated':True,'plan':planned,'pv':info,'coverage':coverage,'restorations':old['restorations']}
      R.validate(C,current,side,planned,stage,candidate,evidence,DIGEST)
      paired.append(current)
    R.compare(C,*paired,planned,side,side)
    valid.append({'stage':stage,'synthetic_queries':sum(C.COUNTS[stage]),'original_report_source_bytes_verified':True})
    for field in ('pv83','qk83','softmax83','pvtarget','binary','program','cleanup','observer','outside','trace','boundcount',
                  'fixedvision','source','lookup','coverage','compiler_power','qk_pair'):
      before,after = copy.deepcopy(paired)
      info = after['agent_expert_combined']['pv']; cap = info['captures'][0]
      if field == 'pv83': cap['pv_calls'] = 83
      elif field == 'qk83': cap['qk_calls'] = 83
      elif field == 'softmax83': cap['softmax_calls'] = 83
      elif field == 'pvtarget': cap['pv_full_description']['target'] = 'AMD:LLVM:gfx1200'
      elif field == 'binary': cap['binaries'][evidence['candidate']['name']] = ['changed']
      elif field == 'program': cap['program_keys'] = ['changed']
      elif field == 'cleanup': info['cleanup_exact'] = False
      elif field == 'observer': info['observer_cleanup_exact'] = False
      elif field == 'outside': info['outside_scope_selected'] = 1
      elif field == 'trace': info['traces'][0]['pv'] = 83
      elif field == 'boundcount': info['bound_program_calls'] -= 1
      elif field == 'fixedvision': after['vision_calibration_sha256'] = 'changed'
      elif field == 'source': after['source_sha256']['model.py'] = 'changed'
      elif field == 'lookup': after['agent_expert_combined']['coverage']['lookup_initial'][0]['history'] = [1]
      elif field == 'coverage': after['agent_expert_combined']['coverage']['all_28_mlp_and_50_vision_projections'] = False
      elif field == 'compiler_power': after['runtime_environment']['AM_POWER_LIMIT'] = '200'
      else: cap['binaries']['alpamayo_attention_scores'] = ['changed']
      def gate():
        R.validate(C,after,side,planned,stage,True,evidence,DIGEST)
        R.compare(C,before,after,planned,side,side)
      reject(stage+'/'+field,gate)
  raw = C.sha
  target = C.PV_ROOT/'agent-expert-pv-qualified-hook.py'
  # The independently pinned PV protocol verifies this file on a fresh process.
  protocol = C.read(C.PV_PROTOCOL)
  assert protocol['files'][str(target)] == raw(target)
  for target in (C.PROTOCOL,ROOT/'agent-expert-combined-install.py',C.BASE):
    with patch.object(C,'sha',lambda p: '0'*64 if Path(p)==target else raw(p)):
      reject('source/'+target.name,lambda:C.protocol(C.PROTOCOL,DIGEST))
  C.write(C.WORK/'agent-expert-combined-audit.json',{'complete':True,'gpu_executed':False,
    'synthetic_only':True,'fresh_composition_route_equality_established':False,'valid':valid,'rejections':rejections,
    'protocol_sha256':DIGEST,'audit_source_sha256':C.sha(__file__),
    'runner_sha256':C.sha(ROOT/'agent-expert-combined-run.py')},new=True)
  print({'complete':True,'gpu_executed':False,'rejections':len(rejections)})


if __name__ == '__main__': main()

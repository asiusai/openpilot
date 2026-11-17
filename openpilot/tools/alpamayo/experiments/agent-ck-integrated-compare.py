"""CPU paired quality screen. No old-history predictions serve as the arithmetic reference."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('ck_integrated_common',Path(__file__).with_name('agent-ck-integrated-common.py'))
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def compare(baseline,candidate,output):
  C.verify();reports=[json.loads(p.read_text()) for p in (baseline,candidate)]
  for arm,r in zip(('baseline','candidate'),reports,strict=True):
    assert r['completed'] and r['gpu_executed'] and r['arm']==arm and r['hooks_restored'] and r['initial_decoder_history']==[]
    assert r['source_protocol_sha256']==C.sha(C.PREPARED/'protocol.json') and C.sha(r['pipeline'])==r['pipeline_sha256']
  assert reports[1]['model_controls']['passed']
  a,b=[json.loads(Path(r['pipeline']).read_text()) for r in reports]
  samples=json.loads((C.PREPARED/'samples.json').read_text());rows=[]
  inputs={r['sample']:r for r in json.loads((C.PREPARED/'inputs.json').read_text())}
  assert [x['sample'] for x in a['runs']]==[x['sample'] for x in b['runs']]==samples
  for index,path,ar,br in zip(C.INDICES,samples,a['runs'],b['runs'],strict=True):
    truth=np.asarray(inputs[path]['recorded_motion_future_xyz'],np.float64).reshape(64,3)
    pa,pb=[np.asarray(r['prediction']['xyz'],np.float64).reshape(64,3) for r in (ar,br)]
    assert all(np.isfinite(x).all() for x in (truth,pa,pb))
    errors=[np.linalg.norm((p-truth)[:,:2],axis=1) for p in (pa,pb)]
    row={'original64_index':index,'sample':path,'baseline_ADE':float(errors[0].mean()),'candidate_ADE':float(errors[1].mean()),
      'baseline_FDE':float(errors[0][-1]),'candidate_FDE':float(errors[1][-1]),
      'tokens_equal':ar['prediction']['generated_token_ids']==br['prediction']['generated_token_ids'],
      'baseline_text':ar['prediction']['reasoning'],'candidate_text':br['prediction']['reasoning']}
    # The path uses 0.1-second future samples. Keep the full speed and turn series for manual inspection.
    for name,p in (('baseline',pa),('candidate',pb),('recorded',truth)):
      points=np.vstack((np.zeros((1,3)),p));velocity=np.diff(points,axis=0)/0.1
      speed=np.linalg.norm(velocity[:,:2],axis=1);heading=np.unwrap(np.arctan2(velocity[:,1],velocity[:,0]))
      row[name+'_speed_mps']=speed.tolist();row[name+'_acceleration_mps2']=(np.diff(speed)/0.1).tolist()
      row[name+'_heading_change_rad']=(heading-heading[0]).tolist()
    rows.append(row)
  def metrics(selected):
    mean={key:float(np.mean([r[key] for r in selected])) for key in ('baseline_ADE','candidate_ADE','baseline_FDE','candidate_FDE')}
    return {**mean,'count':len(selected),'passed':mean['candidate_ADE']<=1.05*mean['baseline_ADE'] and mean['candidate_FDE']<=1.05*mean['baseline_FDE']}
  aggregate=metrics(rows);windows={f'{lo}-{lo+15}':metrics(rows[lo//8:lo//8+2]) for lo in (0,16,32,48)}
  passed=aggregate['passed'] and all(w['passed'] for w in windows.values())
  C.write(output,{'paired_screen8_passed':passed,'aggregate':aggregate,'windows':windows,'rows':rows,
    'baseline_report_sha256':C.sha(baseline),'candidate_report_sha256':C.sha(candidate),'manual_review_required':True,
    'full64_allowed':False,'adoption_allowed':False,'throughput_claim_allowed':False,
    'limit':'Eight overlapping route samples. Review text, speed, slowing, and turns before a separately authorized full64 run.'})
  print(json.dumps({'paired_screen8_passed':passed,'aggregate':aggregate,'windows':windows,'manual_review_required':True}))
  return passed

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  if not compare(args.baseline,args.candidate,args.output): raise SystemExit(1)

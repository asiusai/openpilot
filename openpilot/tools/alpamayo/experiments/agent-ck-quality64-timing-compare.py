"""Paired full-pipeline and prefill timing after review and exact same-arm quality parity."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

s=importlib.util.spec_from_file_location('ck_timing_common',Path(__file__).with_name('agent-ck-quality64-timing-common.py'))
T=importlib.util.module_from_spec(s);s.loader.exec_module(T)

def compare(baseline,candidate,review,output):
  reviewed,quality=T.review_gate(review);reports=[];pipes=[]
  for arm,path in (('baseline',baseline),('candidate',candidate)):
    r=json.loads(path.read_text());T.validate_arm(r)
    assert r['arm']==arm and r['timing_identity_passed'] and r['source_protocol_sha256']==T.Q.sha(T.PREPARED/'protocol.json')
    assert r['timing_review_sha256']==T.Q.sha(review) and r['quality_comparison_sha256']==reviewed['comparison_sha256']
    assert T.Q.sha(r['pipeline'])==r['pipeline_sha256'];T.exact_quality_parity(quality[arm],r)
    reports.append(r);pipes.append(json.loads(Path(r['pipeline']).read_text()))
  indices=[i for i in range(64) if all(r['timing_query_state'][i]['warm_eligible'] for r in reports)]
  assert len(indices)>=16,('Insufficient common warm queries; no qualified timing comparison',indices)
  rows=[]
  for i in indices:
    a,b=[p['runs'][i] for p in pipes];assert a['sample']==b['sample']
    row={'original64_index':i,'sample':a['sample']}
    for arm,x in (('baseline',a),('candidate',b)):
      row[arm+'_service_ms']=x['service_ms'];row[arm+'_pipeline_ms']=x['pipeline_ms']
      row[arm+'_prefill_ms']=x['prediction']['timings_ms']['prefill']
    rows.append(row)
  summary={}
  for field in ('service_ms','pipeline_ms','prefill_ms'):
    values={arm:np.asarray([r[arm+'_'+field] for r in rows],np.float64) for arm in ('baseline','candidate')}
    assert all(np.isfinite(v).all() and (v>0).all() for v in values.values())
    summary[field]={arm:{'median':float(np.median(v)),'p95':float(np.percentile(v,95)),'mean':float(v.mean()),'max':float(v.max())}
      for arm,v in values.items()}
    summary[field]['candidate_to_baseline_median_ratio']=float(np.median(values['candidate'])/np.median(values['baseline']))
  T.Q.write(output,{'paired_lean_timing_qualified':True,'common_warm_indices':indices,'rows':rows,'summary':summary,
    'complete64_observed_summary':{arm:p['summary'] for arm,p in zip(('baseline','candidate'),pipes,strict=True)},
    'baseline_report_sha256':T.Q.sha(baseline),'candidate_report_sha256':T.Q.sha(candidate),'review_sha256':T.Q.sha(review),
    'adoption_allowed':False,'real_camera_cadence_verified':False,'v0_host_validated':False,
    'boundary':'Original loading/preprocessing/transfers/model/trajectory pipeline and a final fence. CK packing, LSE, and conversions remain included.',
    'limit':'Matched recorded-query inference measurements. No isolated CK micro ratio is a pipeline result. Full64 cold totals and warm metrics are separate.'})

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
  p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  compare(a.baseline,a.candidate,a.review,a.output)

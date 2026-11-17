"""Original 5% arithmetic, now all 64 queries and all four full windows."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

s=importlib.util.spec_from_file_location('ck_quality64_common',Path(__file__).with_name('agent-ck-quality64-common.py'))
C=importlib.util.module_from_spec(s);s.loader.exec_module(C)

def source():
  text=(C.EXP/'agent-ck-integrated-compare.py').read_text()
  for old,new in (("assert reports[1]['model_controls']['passed']","assert all(C.validate_arm(r) for r in reports)"),
    ('rows[lo//8:lo//8+2]','rows[lo:lo+16]'),('Eight overlapping route samples. Review text, speed, slowing, and turns before a separately authorized full64 run.',
     '64 overlapping route samples. Review every query; relative gates do not establish driving safety. No adoption.')):
    text=C.replace_once(text,old,new)
  return text.replace('paired_screen8_passed','paired_quality64_passed')

def compare(baseline,candidate,output):
  m=ModuleType('quality64_metrics');m.__file__=str(C.EXP/'agent-ck-integrated-compare.py')
  exec(compile(source(),m.__file__,'exec'),m.__dict__);m.C=C
  passed=m.compare(baseline,candidate,output)
  r=json.loads(output.read_text());lines=['# Full64 per-query review','',
    'Review both texts, speeds, slowing, and turns. These relative gates do not establish driving safety.','']
  for row in r['rows']:
    for field in ('speed_mps','acceleration_mps2','heading_change_rad'):
      row['max_abs_'+field+'_delta']=max(abs(a-b) for a,b in zip(row['baseline_'+field],row['candidate_'+field],strict=True))
    i=row['original64_index'];lines += [f'## Query {i}', '',f"Baseline: {row['baseline_text']}",f"Candidate: {row['candidate_text']}",'',
      f"ADE {row['baseline_ADE']:.6f} -> {row['candidate_ADE']:.6f} m; FDE {row['baseline_FDE']:.6f} -> {row['candidate_FDE']:.6f} m.",
      f"Maximum speed/acceleration/heading deltas: {row['max_abs_speed_mps_delta']:.6f} m/s, {row['max_abs_acceleration_mps2_delta']:.6f} m/s2, {row['max_abs_heading_change_rad_delta']:.6f} rad.",
      f"Final heading baseline/candidate/recorded: {row['baseline_heading_change_rad'][-1]:.6f}, {row['candidate_heading_change_rad'][-1]:.6f}, {row['recorded_heading_change_rad'][-1]:.6f} rad.",'']
  r['review_required_query_indices']=C.INDICES;r['known_screen8_limit']='At query24 both earlier eight-query arms severely missed the recorded right turn. Reinspect under the fresh full64 histories.'
  output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
  with output.with_suffix('.md').open('x') as f: f.write('\n'.join(lines)+'\n')
  return passed

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True);a=p.parse_args()
  if not compare(a.baseline,a.candidate,a.output): raise SystemExit(1)

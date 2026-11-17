"""Exercise the actual comparison's aggregate and complete 16-query window expressions."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np

s=importlib.util.spec_from_file_location('ck_quality_metrics',Path(__file__).with_name('agent-ck-quality64-compare.py'))
M=importlib.util.module_from_spec(s);s.loader.exec_module(M)

def main(output):
  tree=ast.parse(M.source());fx=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compare')
  nodes=[n for n in fx.body if isinstance(n,ast.FunctionDef) and n.name=='metrics' or isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id in ('aggregate','windows','passed') for t in n.targets)]
  code=compile(ast.Module(body=nodes,type_ignores=[]),'<actual_quality64_metrics>','exec')
  def evaluate(rows):
    ns={'np':np,'rows':rows};exec(code,ns);return ns
  rows=[{'baseline_ADE':1.,'candidate_ADE':1.049,'baseline_FDE':1.,'candidate_FDE':1.049} for _ in range(64)]
  good=evaluate(rows);assert good['passed'] and good['aggregate']['count']==64 and all(x['count']==16 for x in good['windows'].values())
  rejected=[]
  for lo in (0,16,32,48):
    for key in ('candidate_ADE','candidate_FDE'):
      bad=[{**x,'candidate_ADE':1.,'candidate_FDE':1.} for x in rows]
      for x in bad[lo:lo+16]: x[key]=1.06
      got=evaluate(bad);assert got['aggregate']['passed'] and not got['passed'] and not got['windows'][f'{lo}-{lo+15}']['passed']
      rejected.append([lo,key])
  M.C.write(output,{'gpu_executed':False,'all_four_full16_windows_checked':True,'within_5pct_passed':True,'window_failures_rejected':rejected})

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)

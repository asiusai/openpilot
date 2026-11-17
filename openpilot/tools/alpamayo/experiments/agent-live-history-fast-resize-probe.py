"""Check one exact batching hypothesis using the original torchvision resize."""
from contextlib import ExitStack
import json
from pathlib import Path
import time
import importlib.util
import sys

def main():
  path=Path(__file__).with_name('agent-live-history-fast-profile.py')
  spec=importlib.util.spec_from_file_location('history_probe_common',path);P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
  P.configure()
  import torch
  from torchvision.transforms.functional import resize,InterpolationMode
  torch.set_num_threads(4)
  with ExitStack() as stack:
    P.cpu_guards(stack)
    schedule=json.loads((P.WORK/'stream-10hz/floor/manifest.json').read_text())['schedule']
    images=P.sample(schedule[20])['image_frames'][:7,0]
    rows=[]
    for i in range(24):
      times={};outputs={}
      for arm in ([0,1] if i%2==0 else [1,0]):
        start=time.perf_counter_ns()
        if arm==0:result=[resize(image,[320,576],interpolation=InterpolationMode.BICUBIC) for image in images]
        else:
          # Include gathering independent cache-miss images, rather than assuming
          # that the selected misses are contiguous or materialized off-clock.
          result=list(resize(torch.stack(list(images)),[320,576],interpolation=InterpolationMode.BICUBIC).unbind(0))
        times[str(arm)]=(time.perf_counter_ns()-start)/1e6;outputs[arm]=result
      P.exact(outputs[0],outputs[1]);rows.append(dict(measured=i>=4,timing=times,exact=True))
    report=dict(complete=True,cpu_only=True,rows=rows,timing={str(a):P.stats([r['timing'][str(a)] for r in rows if r['measured']]) for a in (0,1)})
    output=P.WORK/'agent-live-history-fast-resize-probe.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['timing']),flush=True)

if __name__=='__main__':main()

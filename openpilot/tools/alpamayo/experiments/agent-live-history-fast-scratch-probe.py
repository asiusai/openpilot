"""Test private reusable float scratch with the identical ATen bicubic operation."""
from contextlib import ExitStack
import json
from pathlib import Path
import time
import importlib.util

def main():
  path=Path(__file__).with_name('agent-live-history-fast-profile.py')
  spec=importlib.util.spec_from_file_location('history_scratch_common',path);P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
  P.configure()
  import torch
  from torchvision.transforms.functional import resize,InterpolationMode
  torch.set_num_threads(4)
  with ExitStack() as stack:
    P.cpu_guards(stack)
    schedule=json.loads((P.WORK/'stream-10hz/floor/manifest.json').read_text())['schedule']
    images=P.sample(schedule[20])['image_frames'][:7,0]
    scratch=None;rows=[]
    for i in range(24):
      times={};outputs={}
      for arm in ([0,1] if i%2==0 else [1,0]):
        start=time.perf_counter_ns()
        if arm==0:result=[resize(image,[320,576],interpolation=InterpolationMode.BICUBIC) for image in images]
        else:
          if scratch is None:scratch=(torch.empty((1,3,432,768),dtype=torch.float32),torch.empty((1,3,320,576),dtype=torch.float32))
          result=[]
          for image in images:
            scratch[0].copy_(image)
            torch.ops.aten._upsample_bicubic2d_aa.out(scratch[0],[320,576],False,out=scratch[1])
            result.append(scratch[1].clamp_(0,255).round_().to(torch.uint8)[0])
        times[str(arm)]=(time.perf_counter_ns()-start)/1e6;outputs[arm]=result
      P.exact(outputs[0],outputs[1]);rows.append(dict(measured=i>=4,timing=times,exact=True))
    report=dict(complete=True,cpu_only=True,rows=rows,timing={str(a):P.stats([r['timing'][str(a)] for r in rows if r['measured']]) for a in (0,1)})
    output=P.WORK/'agent-live-history-fast-scratch-probe.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['timing']),flush=True)

if __name__=='__main__':main()

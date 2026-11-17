"""Bounded pure-byte SIMD packing check against the original image processor."""
from contextlib import ExitStack
import ctypes
import json
from pathlib import Path
import subprocess
import time
import importlib.util

def main():
  path=Path(__file__).with_name('agent-live-history-fast-profile.py')
  spec=importlib.util.spec_from_file_location('history_native_common',path);P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
  P.configure()
  import torch
  import numpy as np
  from torchvision.transforms.functional import resize,InterpolationMode
  torch.set_num_threads(4)
  source=Path(__file__).with_name('agent-live-history-fast-pack.c')
  binary=P.ARTIFACTS/'pack.so'
  command=['clang','-O3','-shared','-fPIC','-mssse3','-fno-fast-math','-o',str(binary),str(source)]
  subprocess.run(command,check=True,capture_output=True)
  lib=ctypes.CDLL(str(binary));lib.pack_320_576.argtypes=[ctypes.c_void_p,ctypes.c_void_p];lib.pack_320_576.restype=None
  with ExitStack() as stack:
    P.cpu_guards(stack)
    A=P.load(P.EXP/'agent-cpu-inputs-adapter.py','history_native_adapter')
    adapter=A.FastInputs(P.CHECKPOINT,camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True,cache_resize=True)
    schedule=json.loads((P.WORK/'stream-10hz/floor/manifest.json').read_text())['schedule']
    images=[resize(x,[320,576],interpolation=InterpolationMode.BICUBIC) for x in P.sample(schedule[20])['image_frames'][:7,0]]
    rows=[]
    for i in range(24):
      times={};outputs={}
      for arm in ([0,1] if i%2==0 else [1,0]):
        start=time.perf_counter_ns()
        if arm==0:result=[adapter.processor.image_processor.patchify(x.unsqueeze(0),16,2,1)[0].reshape(-1,768).contiguous() for x in images]
        else:
          result=[]
          for x in images:
            output=np.empty((720,768),dtype=np.uint8);lib.pack_320_576(output.ctypes.data,x.data_ptr())
            result.append(torch.from_numpy(output))
        times[str(arm)]=(time.perf_counter_ns()-start)/1e6;outputs[arm]=result
      P.exact(outputs[0],outputs[1]);rows.append(dict(measured=i>=4,timing=times,exact=True))
    report=dict(complete=True,cpu_only=True,rows=rows,command=command,source_sha256=P.sha(source),binary_sha256=P.sha(binary),
      timing={str(a):P.stats([r['timing'][str(a)] for r in rows if r['measured']]) for a in (0,1)})
    output=P.WORK/'agent-live-history-fast-pack-native-probe.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['timing']),flush=True)

if __name__=='__main__':main()

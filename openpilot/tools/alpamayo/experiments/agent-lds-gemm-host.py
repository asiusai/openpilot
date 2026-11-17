"""Execute independent CPU controls and the existing real 40-call CPU JIT proof."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='1',BEAM='0',JITBEAM='0',PARALLEL='0')
import numpy as np
from tinygrad import Device,Tensor,dtypes
from amd_gemm import pack

spec=importlib.util.spec_from_file_location('agent_lds_check',Path(__file__).with_name('agent-lds-gemm-check.py'))
H=importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',required=True,type=Path)
  args=parser.parse_args()
  records=[]
  shape=(17,48,256)
  for kind in H.CONTROLS:
    previous=None
    for state in range(7):
      row_scale=kind in ('row-bias','finite')
      bias=kind in ('row-bias','finite')
      data=H.fixture(shape,state,kind,row_scale)
      for name in ('a','w'):
        np.testing.assert_array_equal(pack(Tensor(data[name+'_unpacked'],device='CPU')).numpy(),data[name])
      if kind!='finite':
        a,w=(Tensor(H.E.decode_fp8(data[key+'_unpacked']),device='CPU') for key in ('a','w'))
        scales=Tensor(data['sx'][:,None] if row_scale else data['sx'],device='CPU')*Tensor(data['sw'],device='CPU')
        output=a.dot(w.T)*scales
        if bias: output=output+Tensor(H.E.unpack_bf16(data['bias']),device='CPU')
        expected=H.independent(data,bias,row_scale)
        np.testing.assert_array_equal(output.cast(dtypes.bfloat16).bitcast(dtypes.uint16).numpy(),expected)
        if previous is not None and kind!='signed-zero': assert not np.array_equal(expected,previous)
        previous=expected
      else:
        for name in ('a','w'):
          assert np.isfinite(H.E.decode_fp8(data[name])).all()
          assert set((0,128,1,129,126,254,56,184)) <= set(np.unique(data[name]))
      records.append(dict(kind=kind,state=state,packing_exact=True,independent_arithmetic=kind!='finite',finite_inputs=True))
  proof=H.E.cpu_checks()
  # Distinguish a running-C K16 recurrence from zero-C WMMA followed by an add.
  a,b=np.zeros(32,np.float32),np.zeros(32,np.float32)
  a[0],b[0]=128,1
  a[16:19],b[16:19]=[-64,-7,2.**-9],[1,9,2.**-9]
  table=H.E.decode_fp8(np.arange(256,dtype=np.uint8))
  assert all(np.any(table==value) for vector in (a,b) for value in vector)
  first=np.float32(a[:16].astype(np.float64)@b[:16].astype(np.float64))
  dot=a[16:].astype(np.float64)@b[16:].astype(np.float64)
  running=np.float32(np.float64(first)+dot)
  detached=np.float32(first+np.float32(dot))
  scale=np.nextafter(np.float32(1.00390625),np.float32(-np.inf))
  witness=H.E.bf16(np.array([running,detached],np.float32)*scale)
  np.testing.assert_array_equal(witness,np.array([0x3f81,0x3f80],np.uint16))
  assert not any(name.startswith('AMD') for name in Device._opened_devices)
  result=dict(gpu_executed=False,controls=records,real_cpu_jit_queue_checks=proof,
    exact_e4m3_rounding_witness=dict(bf16=[int(v) for v in witness],hardware_internal_tree_not_emulated=True),
    failure_isolation=H.failure_mocks(),geometry=H.geometry(),provenance=H.provenance())
  H.E.write_json(args.output,result)
  print(json.dumps(dict(gpu_executed=False,controls=len(records),cpu_queue_checks=len(proof))))


if __name__=='__main__': main()

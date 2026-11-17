"""One parent-only current versus packed 128x96 K64 FC2 trial.

This file prepares CPU evidence by default. GPU execution requires the exact source digest.
All six small families and all six post-timing actual mutations are fresh for the new binary.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_lds_small_audit_run',HERE/'agent-lds-small-audit.py')
A=importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
S,H,E=A.S,A.H,A.E
C=S.load('agent-lds-cold-run')
LABELS=('current','small96')


def compile_selected(render):
  rows={}
  for key,shape,bias,row_scale in (('actual',A.SHAPE,False,False),('scalar',(193,208,256),False,False),('row-bias',(193,208,256),True,True)):
    rows[key]={label:S.compile_one(shape,bias,row_scale,label,fn,render)
      for label,fn in S.functions(render,row_scale).items() if label in LABELS}
    assert all(v['status']=='compiled' for v in rows[key].values())
  assert rows['actual']['current']['binary_sha256']=='163bb2701324b94246f66dcb5e5b1104e279a13e4407c20133d514f95633d68e'
  assert rows['actual']['small96']['binary_sha256']=='54c8a0ab3f2100eb6729b7c229e6a7f115c083a9f71ab0d85e44b73b6a24f7cf'
  return rows


def upload(io,data):
  from tinygrad import dtypes
  return tuple(io.upload(data[key],dtypes.fp8e4m3 if key in ('a','w') else dtypes.bfloat16 if key=='bias' else dtypes.float32)
    for key in (*C.KEYS,'bias'))


def change(io,tensors,data,known,bias=False):
  changed=C.update_changed(io,tensors[:4],data,known)
  if bias:
    digest=E.sha(data['bias'].tobytes())
    if digest!=known['bias']: io.write(tensors[4],data['bias']); known['bias']=digest; changed.append('bias')
  return changed


def captures(shape,bias,fns,inputs,binaries,report):
  singles={label:E.make_jit(shape,bias,fns[label],1) for label in LABELS}
  for label,jit in singles.items():
    observed=[]
    with C.observe_numeric_binary(observed,binaries[label]):
      for _ in range(3): jit(*inputs)
    assert observed and len(E.captured_kernels(jit))==1
    report[label]=dict(binary_sha256=binaries[label],observed_calls=len(observed),captured_calls=1)
  return singles


def controls(io,report,save,compiled,render):
  from tinygrad import dtypes
  shape=(193,208,256)
  verifier=C.Verifier(io)
  report['fresh_small_controls']=[]
  report['control_validation']=verifier.records
  for kind in H.CONTROLS:
    bias=row_scale=kind=='row-bias'
    key='row-bias' if bias else 'scalar'
    fns={label:fn for label,fn in S.functions(render,row_scale).items() if label in LABELS}
    data=H.fixture(shape,0,kind,row_scale)
    tensors=upload(io,data)
    known={key:E.sha(data[key].tobytes()) for key in (*C.KEYS,'bias')}
    record=dict(kind=kind,shape_mnk=shape,bias=bias,row_scale=row_scale,captures={},states=[])
    report['fresh_small_controls'].append(record)
    binaries={label:compiled[key][label]['binary_sha256'] for label in LABELS}
    singles=captures(shape,bias,fns,tensors,binaries,record['captures'])
    expected_tensor=io.upload(np.zeros(shape[:2],np.uint16),dtypes.bfloat16) if kind!='finite' else None
    previous=None
    for state in range(7):
      data=H.fixture(shape,state,kind,row_scale)
      changed=change(io,tensors,data,known,bias)
      reference=verifier.run(singles['current'],tensors)
      verifier.check(reference,reference,f'{kind}-{state}-baseline-finite')
      if expected_tensor is not None:
        expected=H.independent(data,bias,row_scale)
        io.write(expected_tensor,expected)
        verifier.check(expected_tensor,reference,f'{kind}-{state}-independent-baseline')
      if previous is not None and kind!='signed-zero': verifier.check(previous,reference,f'{kind}-{state}-changed',False)
      verifier.check(reference,verifier.run(singles['small96'],tensors),f'{kind}-{state}-candidate')
      previous=reference.clone().realize()
      record['states'].append(dict(state=state,changed_tensors=changed,independent_reference=kind!='finite',
        finite_exact_bf16=True,poisoned_replays=True,input_sha256=dict(known)))
    report['io']=io.snapshot(); save()


def actual(io,report,save,compiled,render):
  from tinygrad import Tensor,dtypes
  shape=A.SHAPE
  verifier=C.Verifier(io)
  report['actual_validation']=verifier.records
  # Recheck the exact compact validator on hardware before using it for new GEMM acceptance.
  left=Tensor.empty(*shape[:2],device=io.device.device,dtype=dtypes.bfloat16).realize()
  right=Tensor.empty(*shape[:2],device=io.device.device,dtype=dtypes.bfloat16).realize()
  verifier.fill(left,0x3f80); verifier.fill(right,0x3f80)
  verifier.check(left,right,'validator-finite-equal')
  for index in (0,1023,1024,shape[0]*shape[1]//2,shape[0]*shape[1]-1):
    verifier.poke(right,index,0x3f81); verifier.check(left,right,f'validator-bit-{index}',False); verifier.poke(right,index,0x3f80)
  for word in (0x7f80,0x7fc1):
    verifier.poke(left,shape[0]*shape[1]-1,word); verifier.poke(right,shape[0]*shape[1]-1,word)
    verifier.check(left,right,f'validator-identical-nonfinite-{word}',False)
  verifier.fill(left); verifier.fill(right); verifier.check(left,right,'validator-identical-poison',False)

  base=H.fixture(shape,0)
  tensors=upload(io,base)
  known={key:E.sha(base[key].tobytes()) for key in (*C.KEYS,'bias')}
  fns={label:fn for label,fn in S.functions(render).items() if label in LABELS}
  binaries={label:compiled['actual'][label]['binary_sha256'] for label in LABELS}
  report['actual_single_captures']={}
  singles=captures(shape,False,fns,tensors,binaries,report['actual_single_captures'])
  arrays=C.weight_data(base)
  weights=(tensors[1],*(io.upload(w,dtypes.fp8e4m3) for w in arrays[1:]))
  report['weights']=dict(count=4,bytes_each=arrays[0].nbytes,total_bytes=sum(w.nbytes for w in arrays),
    sha256=[E.sha(w.tobytes()) for w in arrays],origin='Synthetic finite raw E4M3, not learned model weights')
  assert len({id(w.uop.buffer.base) for w in weights})==4 and len(set(report['weights']['sha256']))==4
  assert report['weights']['total_bytes']>64*1024**2
  a,_,sx,sw,dummy=tensors
  rotating_inputs=(a,sx,sw,*weights)
  references=[]
  for index,w in enumerate(weights):
    args=(a,w,sx,sw,dummy)
    reference=verifier.run(singles['current'],args).clone().realize()
    verifier.check(reference,reference,f'weight-{index}-baseline-finite')
    candidate=verifier.run(singles['small96'],args)
    verifier.check(reference,candidate,f'weight-{index}-candidate')
    if index==0:
      verifier.poke(candidate,shape[0]*shape[1]-1,0x7fc1)
      verifier.check(reference,candidate,'deliberately-corrupted-candidate-rejected',False)
      verifier.check(reference,verifier.run(singles['small96'],args),'candidate-replay-after-corruption')
    for prior_index,prior in enumerate(references): verifier.check(prior,reference,f'weight-{prior_index}-versus-{index}',False)
    references.append(reference)
  batches={workload:{label:C.RotatingJit(fns[label],order,binaries[label]) for label in LABELS} for workload,order in C.ORDERS.items()}
  report['queue_proofs']={}
  for workload,variants in batches.items():
    report['queue_proofs'][workload]={}
    for label,jit in variants.items():
      for _ in range(3): jit(*rotating_inputs)
      report['queue_proofs'][workload][label]=jit.audit_capture()
      verifier.check(references[C.ORDERS[workload][-1]],verifier.run(jit,rotating_inputs),f'pre-{workload}-{label}')
  # Change an actual captured input before timing, validate both 40-call graphs, then restore state zero.
  changed=change(io,tensors,H.fixture(shape,1),known)
  reference=verifier.run(singles['current'],tensors)
  verifier.check(reference,reference,'pre-changed-baseline-finite')
  verifier.check(references[0],reference,'pre-changed-reference',False)
  for label in LABELS: verifier.check(reference,verifier.run(batches['warm'][label],rotating_inputs),f'pre-changed-batch-{label}')
  report['pre_timing_changed_batch']=dict(state=1,changed_tensors=changed,both_variants_exact=True)
  change(io,tensors,base,known)
  for label in LABELS:
    verifier.check(references[0],verifier.run(batches['warm'][label],rotating_inputs),f'pre-restored-batch-{label}')
  report['io']=io.snapshot(); save()

  before=io.snapshot()
  report['timing']={workload:dict(provisional=True,rounds=[]) for workload in C.ORDERS}
  for iteration in range(3):
    for workload in (('warm','rotating') if iteration%2==0 else ('rotating','warm')):
      order=LABELS if iteration%2==0 else LABELS[::-1]
      pair=dict(iteration=iteration,order=order)
      for label in order:
        jit=batches[workload][label]
        io.device.synchronize(); jit(*rotating_inputs); io.device.synchronize()
        start=time.perf_counter(); jit(*rotating_inputs); io.device.synchronize()
        pair[label]=(time.perf_counter()-start)*1000
      report['timing'][workload]['rounds'].append(pair)
  assert io.snapshot()==before
  report['timing_data_transfer_bytes']=0
  save()
  for phase in range(4):
    args=(a,sx,sw,*(weights[phase:]+weights[:phase]))
    for label in LABELS:
      verifier.check(references[(phase+3)%4],verifier.run(batches['rotating'][label],args),f'post-rotation-{label}-phase-{phase}')
  # All six actual mutations are mandatory for the new binary after timing.
  previous=references[0]
  report['post_timing_mutations']=[]
  for state in range(7):
    changed=change(io,tensors,H.fixture(shape,state),known)
    reference=verifier.run(singles['current'],tensors)
    verifier.check(reference,reference,f'post-{state}-baseline-finite')
    if state: verifier.check(previous,reference,f'post-{state}-reference-changed',False)
    else: verifier.check(references[0],reference,'post-0-original-reference')
    verifier.check(reference,verifier.run(singles['small96'],tensors),f'post-{state}-candidate-single')
    for label in LABELS:
      verifier.check(reference,verifier.run(batches['warm'][label],rotating_inputs),f'post-{state}-{label}-batch')
    report['post_timing_mutations'].append(dict(state=state,changed_tensors=changed,input_sha256=dict(known),
      both_single_and_serial_40_exact=True,finite_all_bits=True,poison_before_each_run=True))
    previous=reference.clone().realize()
    report['io']=io.snapshot(); save()
  for workload,value in report['timing'].items():
    baseline,candidate=(statistics.median(row[label] for row in value['rounds'])/40 for label in LABELS)
    value.update(provisional=False,current_ms=baseline,small96_ms=candidate,small96_over_current=candidate/baseline,
      calls_per_timed_batch=40,conditioning_calls_per_sample=40,host_inclusive=True,actual_full_model=False,
      no_operand_or_output_transfers_during_timing=True,measured_dram_bytes=None,measured_cache_hit_rate=None)


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode',choices=('cpu','gpu'),default='cpu')
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--source-sha')
  args=parser.parse_args()
  os.environ.update(BEAM='0',JITBEAM='0',PARALLEL='0',PROFILE='0',DEBUG='0')
  if args.mode=='cpu': os.environ.update(DEV='CPU',ALLOW_DEVICE_USAGE='0')
  from tinygrad import Device
  report=dict(complete=False,gpu_executed=False,selected='small96',shape_mnk=A.SHAPE,provenance=A.provenance())
  assert args.mode!='gpu' or args.source_sha==report['provenance']['source_sha256'], 'Missing or changed source binding'
  def save(): E.write_json(args.output,report)
  try:
    render=H.renderer()
    report['compiled']=compile_selected(render)
    if args.mode=='cpu':
      report['geometry']=A.geometry()
      report['queue_checks']=S.load('agent-lds-small-check').cpu_checks(render,report['compiled'])
      assert not Device._opened_devices
    else:
      assert os.environ.get('TC_OPT')=='2' and os.environ.get('LLVMOPT')=='1'
      assert os.environ.get('AM_POWER_LIMIT')=='144' and os.environ.get('AM_PERF_LEVEL')=='auto'
      from tinygrad.device import Compiled
      Compiled.wait_timeout_ms=10000
      H.load('prefill-kernels').fix_dispatch_packet()
      device=Device[Device.DEFAULT]
      assert device.arch=='gfx1200' and device.is_usb
      H.load('wgp_compiler').install(device)
      report['gpu_executed']=True
      io=C.IO(device)
      controls(io,report,save,report['compiled'],render)
      actual(io,report,save,report['compiled'],render)
      report['io']=io.snapshot()
    assert A.provenance()==report['provenance']
    report['complete']=True; save()
    print(json.dumps(dict(complete=True,gpu_executed=report['gpu_executed'],source_sha256=report['provenance']['source_sha256'])))
  except BaseException as exc:
    report['failure']=f'{type(exc).__name__}: {exc}'
    report['complete']=False
    for timing in report.get('timing',{}).values(): timing['provisional']=True
    save(); raise


if __name__=='__main__': main()

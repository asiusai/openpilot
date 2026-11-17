"""CPU-only tiny synthetic optimizer/loss/recovery controls. Never fits actual checkpoint targets."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('agent_distill_fit_common_check',HERE/'agent-distill-fit-common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)

def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir',type=Path,default=C.ROOT/'cpu-controls-v2')
  args=parser.parse_args();root=args.output_dir.resolve()
  assert root.is_relative_to(C.ROOT) and root.name.startswith('cpu-controls-')
  C.ROOT.mkdir(parents=True,exist_ok=True);root.mkdir(exist_ok=False)
  def guard(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes)):return
    path=os.fsdecode(args[0])
    assert not path.startswith(('/dev/dri','/dev/kfd','/dev/nvidia','/dev/bus/usb'))
    assert not ('sealed_test' in path and path.endswith('.npz')),'Sealed target value access blocked'
    assert not (C.PROTECTED in path),'Protected route access blocked'
  sys.addaudithook(guard)
  import numpy as np
  os.environ.update(DEV='CPU:LLVM',TC='0',ALLOW_TF32='0',TRAINING='0',PROFILE='0',PMC='',SQTT='',VIZ='0',
    OPTIM_DTYPE='float32',FUSE_OPTIM='0',CONST_LR='0',PYTHONDONTWRITEBYTECODE='1',
    CACHEDB=str(root/'cpu-cache.db'),TMPDIR=str(root),CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
  sys.path.insert(0,str(C.BASE/'tinygrad'))
  from tinygrad import Tensor,Device
  from tinygrad.helpers import Context
  from tinygrad.nn.optim import AdamW
  B=C.graph();S=C.module('agent_distill_fit_state_check',HERE/'agent-distill-fit-state.py')
  failures=[]
  def reject(name,fn):
    try:fn()
    except (AssertionError,ValueError,KeyError,FileNotFoundError,TypeError):failures.append(name)
    else:raise AssertionError('Negative control accepted: '+name)
  # Exact bit interpretation, including negative zero and finite BF16 subnormals.
  codes=np.array([0,0x8000,1,0x8001,0x3f80,0xbf80,0x7f7f],np.uint16)
  actual=C.bf16(codes);np.testing.assert_array_equal(actual.view(np.uint32),codes.astype(np.uint32)<<16)
  reject('BF16 infinity',lambda:C.bf16(np.array([0x7f80],np.uint16)))
  reject('BF16 NaN',lambda:C.bf16(np.array([0x7fc1],np.uint16)))
  reject('BF16 wrong dtype',lambda:C.bf16(np.array([1],np.float32)))
  rows=C.records();canonical=C.read(C.MANIFEST)
  for name,mutate in [
    ('sealed record',lambda r:r.update(split='sealed_test')),
    ('protected source',lambda r:r.update(route_id=C.PROTECTED)),
    ('wrong command',lambda r:r['conditioning'].update(text='Take an inferred turn')),
    ('forged recorded timestamp',lambda r:r['conditioning'].update(recorded_available_ns=0)),
    ('padding in fit',lambda r:r.update(actual_prompt_tokens=2048)),
    ('modified capture hash',lambda r:r.update(capture_sha256='0'*64)),
    ('wrong drive split',lambda r:r.update(drive_id=rows[-1]['drive_id'])),
    ('renamed target path',lambda r:r.update(capture_path=str(root/'renamed.npz'))),
    ('wrong target dtype declaration',lambda r:r.update(dtype='float32'))]:
    value=copy.deepcopy(canonical);mutate(value['samples'][0]);reject(name,lambda v=value:C.validate_manifest(v,canonical))
  bad=copy.deepcopy(rows[0]);bad.update(split='sealed_test',capture_path=str(C.TARGETS/'sealed_test/sealed_test-command-0.npz'))
  reject('sealed target denied before file open',lambda:C.load_target(bad,rows))
  # Real target loading is limited to dtype/finite/shape validation, with no model call or loss.
  for row in rows:
    data=C.load_target(row,rows);assert set(data)==set(C.SHAPES)
  mask_results=[]
  for valid in (2027,2030):
    length=2048
    y=np.linspace(-.5,.5,length*2,dtype=np.float32).reshape(1,length,2)
    kv=np.arange(8*length,dtype=np.float32).reshape(1,8,length,1)/np.float32(8192)
    outputs=[Tensor(a.copy(),device='CPU') for a in (y,kv,kv*2)]
    targets=[Tensor(np.zeros_like(a),device='CPU').is_param_(False) for a in (y,kv,kv)]
    loss=C.loss(B,outputs,targets,valid)
    expected=np.mean(y[:,:valid]**2,dtype=np.float64)+.1*(np.mean(kv[:,:,:valid]**2,dtype=np.float64)+np.mean((kv*2)[:,:,:valid]**2,dtype=np.float64))
    assert np.isclose(float(loss.item()),expected,rtol=2e-6)
    changed=[a.numpy().copy() for a in targets]
    changed[0][:,valid:]=10000;changed[1][:,:,valid:]=-30000;changed[2][:,:,valid:]=15000
    altered=C.loss(B,outputs,[Tensor(a,device='CPU').is_param_(False) for a in changed],valid)
    np.testing.assert_array_equal(loss.numpy(),altered.numpy())
    gradients=loss.gradient(*outputs)
    for i,g in enumerate(gradients):
      a=g.numpy();assert np.count_nonzero(a[:,valid:] if i==0 else a[:,:,valid:])==0
    assert all(t.grad is None for t in targets)
    # Independent per-head/position K derivative catches a flattened/swrong axis mask.
    derivative=2*.1*kv[:,:,:valid]/(8*valid)
    np.testing.assert_allclose(gradients[1].numpy()[:,:,:valid],derivative,rtol=3e-6,atol=1e-10)
    reject('zero valid rows '+str(valid),lambda:C.loss(B,outputs,targets,0))
    reject('too many valid rows '+str(valid),lambda:C.loss(B,outputs,targets,2049))
    mask_results.append(dict(actual_prompt_rows=valid,loss_matches_numpy=True,END_draft_rows_excluded_exactly=True,padding_gradients_zero=True,KV_head_position_derivative=True))
  # Two-by-two synthetic regression, with an independent NumPy AdamW oracle.
  x=np.array([[1.,.5],[-.5,2.],[.25,-1.]],np.float32)
  start=np.array([[.25,-.375],[.125,.75]],np.float32)
  def make(initial):
    weights={'toy':Tensor(initial.copy(),device='CPU').is_param_(True).realize()}
    optimizer=AdamW(list(weights.values()),lr=3.5e-6,b1=.9,b2=.999,eps=1e-8,weight_decay=.01,fused=False)
    return weights,optimizer
  weights,opt=make(start);reference=start.copy();m=np.zeros_like(start);v=np.zeros_like(start);b1=b2=np.float32(1)
  def step(w,o):
    with Context(TRAINING=1):
      o.zero_grad();prediction=(Tensor(x,device='CPU').is_param_(False)@w['toy']).reshape(1,3,2)
      kv=prediction.reshape(1,3,2,1).transpose(1,2)
      targets=[Tensor.zeros(*a.shape,device='CPU').is_param_(False) for a in (prediction,kv,kv)]
      objective=C.loss(B,(prediction,kv,kv*2),targets,3)
      g=objective.gradient(w['toy'])[0];w['toy'].grad=g.realize();o.step()
      assert all(t.grad is None for t in targets)
    return g.numpy()
  state_dir=root/'state';state_dir.mkdir();context=dict(synthetic_only=True,hyperparameters=C.HYPER)
  shapes={'toy':(2,2)};S.save(state_dir,S.snapshot(weights,opt),dict(step=0,best_step=0),context,shapes)
  snapshots=[]
  for index in (1,2):
    old=reference.copy();gradient=np.float32(3/(3*2))*(x.T@(x@old))
    m=np.float32(.9)*m+np.float32(.1)*gradient;v=np.float32(.999)*v+np.float32(.001)*(gradient*gradient)
    b1*=np.float32(.9);b2*=np.float32(.999)
    reference=old-np.float32(3.5e-6)*((m/(np.float32(1)-b1))/(np.sqrt(v/(np.float32(1)-b2))+np.float32(1e-8))+np.float32(.01)*old)
    S.intent(state_dir,index-1);actual_gradient=step(weights,opt)
    np.testing.assert_allclose(actual_gradient,gradient,rtol=2e-6,atol=1e-7)
    np.testing.assert_allclose(weights['toy'].numpy(),reference,rtol=0,atol=1e-7)
    S.save(state_dir,S.snapshot(weights,opt),dict(step=index,best_step=0),context,shapes)
    snapshots.append(S.snapshot(weights,opt))
    if index==1:
      loaded,metadata=S.load(state_dir,context,shapes);resumed,resumed_opt=make(np.zeros_like(start));S.restore(resumed,resumed_opt,loaded)
  step(resumed,resumed_opt);recovered=S.snapshot(resumed,resumed_opt)
  for name,value in snapshots[1].items():np.testing.assert_array_equal(value,recovered[name])
  reject('wrong recovery context',lambda:S.load(state_dir,dict(wrong=True),shapes))
  corrupt=root/'corrupt-state';shutil.copytree(state_dir,corrupt)
  pointer=json.loads((corrupt/'latest.json').read_text())
  with (corrupt/pointer['file']).open('ab') as stream:stream.write(b'modified')
  reject('modified optimizer checkpoint',lambda:S.load(corrupt,context,shapes))
  ambiguous=root/'ambiguous-state';shutil.copytree(state_dir,ambiguous);S.intent(ambiguous,2)
  reject('uncommitted update replay',lambda:S.load(ambiguous,context,shapes))
  bad_arrays=copy.deepcopy(snapshots[1]);bad_arrays['learning_rate']*=2
  reject('changed checkpoint learning rate',lambda:S.validate_arrays(bad_arrays,shapes))
  assert set(Device._opened_devices)<= {'CPU','NPY','PYTHON'};C.no_devices()
  result=dict(complete=True,all_passed=True,gpu_executed=False,actual_data_optimizer_updates=0,
    actual_student_forward_executed=False,synthetic_optimizer_updates=3,synthetic_parameter_count=4,
    actual_train_validation_targets_verified=6,sealed_target_arrays_opened=0,mask_controls=mask_results,
    independent_adamw_numpy_match=True,optimizer_state_resume_bit_exact=True,negative_controls_rejected=failures,
    opened_devices=sorted(Device._opened_devices),source_sha256={str(p):C.sha(p) for p in (Path(__file__),HERE/'agent-distill-fit-state.py',HERE/'agent-distill-fit-common.py',C.GRAPH)})
  C.write(root/'agent-distill-fit-cpu-check.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

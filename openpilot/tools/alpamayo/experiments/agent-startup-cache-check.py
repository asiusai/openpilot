"""Actual pinned CPU TinyJit and BlockDecoder controls; no GPU backend allowed."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.update(DEV='CPU', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode = True
import numpy as np
from tinygrad import Device, Tensor, dtypes

ROOT = Path(__file__).resolve().parent


def local(name):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'),ROOT/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def bf16(value):
  bits = np.asarray(value,dtype=np.float32).view(np.uint32)
  return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def raw(value): return value.bitcast(dtypes.uint16).numpy().copy()


class Model:
  def __init__(self):
    self.python_greedy_calls = 0
    self.matrix = np.roll(np.eye(8,dtype=np.float32),1,axis=1)
    self.weight = Tensor(self.matrix,device='CPU').cast(dtypes.bfloat16).realize()
    self.cache = [Tensor.zeros(2,1,1,80,8,device='CPU',dtype=dtypes.bfloat16).realize()]

  def _greedy_decode(self, logits, cos, sin, position, step):
    self.python_greedy_calls += 1
    token = logits.reshape(-1).argmax().realize()
    value = (logits@self.weight+cos[:,:,step:step+1].reshape(1,1,8)-sin[:,:,step:step+1].reshape(1,1,8))
    value = value.cast(dtypes.bfloat16).realize()
    self.cache[0][:,:,:,position:position+1].assign(value.reshape(1,1,1,1,8).expand(2,1,1,1,8)).realize()
    return value,token.float().realize()


def warm_serial(decoder, logits, cos, sin):
  for count in range(1,7):
    key = (cos.shape,count,80)
    while key not in decoder.jits or decoder.jits[key].cnt<2:
      Tensor.realize(*decoder.decode(logits,cos,sin,10,0,count))
    output,_ = decoder.decode(logits,cos,sin,10,0,count)
    Tensor.realize(*decoder.decode(output,cos,sin,10,0,count))


def exercise(decoder, model, cos, sin):
  checked = 0
  for mutation in range(3):
    # Immutable model weight allocation is updated only as a CPU mutation control.
    model.matrix = np.roll(np.eye(8,dtype=np.float32),1+mutation,axis=1)
    model.weight.assign(Tensor(model.matrix,device='CPU').cast(dtypes.bfloat16)).realize()
    c = np.full((1,1,64,8),mutation+1,dtype=np.float32)
    s = np.zeros_like(c)
    s[:,:,:,::2] = 1
    cos.assign(Tensor(c,device='CPU').cast(dtypes.bfloat16)).realize()
    sin.assign(Tensor(s,device='CPU').cast(dtypes.bfloat16)).realize()
    for count in range(1,7):
      position, step = 3+mutation*17, mutation*9
      values = np.array([[[0,1,4,7,3,2,5,6]]],dtype=np.float32)+mutation*2
      expected = values.copy()
      expected_cache = model.cache[0].float().numpy().copy()
      tokens = []
      for i in range(count):
        tokens.append(int(np.argmax(expected.reshape(-1))))
        # Tinygrad BF16 arithmetic rounds each binary operator, including matmul.
        expected = bf16(bf16(bf16(expected@model.matrix)+c[:,:,step+i:step+i+1].reshape(1,1,8))-
                        s[:,:,step+i:step+i+1].reshape(1,1,8))
        expected_cache[:,:,:,position+i:position+i+1] = expected.reshape(1,1,1,1,8)
      got, emitted = decoder.decode(Tensor(values,device='CPU').cast(dtypes.bfloat16).realize(),cos,sin,position,step,count)
      np.testing.assert_array_equal(got.float().numpy(),expected)
      np.testing.assert_array_equal(emitted.numpy(),tokens)
      np.testing.assert_array_equal(model.cache[0].float().numpy(),expected_cache)
      checked += 1
  return checked


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path)
  args = parser.parse_args()
  prototype, block, old = local('agent-startup-cache-capture'),local('block-decode'),local('agent-startup-spec-warm')
  local('agent-startup-cache-run').validate_pins(prototype)
  original_get = type(Device).__getitem__
  def only_cpu(owner,key):
    if key not in ('CPU','NPY'): raise AssertionError('GPU backend access is forbidden')
    return original_get(owner,key)
  def no_network(*args,**kwargs): raise AssertionError('Network access is forbidden')
  reports, checks = [], 0
  with patch.object(type(Device),'__getitem__',only_cpu),patch.object(socket.socket,'connect',no_network):
    for enabled in (False,True):
      model = Model()
      decoder = block.BlockDecoder(model)
      logits = Tensor(np.arange(8,dtype=np.float32).reshape(1,1,8),device='CPU').cast(dtypes.bfloat16).realize()
      cos = Tensor.ones(1,1,64,8,device='CPU',dtype=dtypes.bfloat16).contiguous().realize()
      sin = Tensor.zeros(1,1,64,8,device='CPU',dtype=dtypes.bfloat16).contiguous().realize()
      with prototype.install(decoder,enabled,old.ClosedGraphs) as report:
        warm_serial(decoder,logits,cos,sin)
      report['cpu_python_greedy_body_calls_at_ready'] = model.python_greedy_calls
      assert model.python_greedy_calls==(22 if enabled else 42)
      reports.append(report)
      assert all(not isinstance(jit,prototype.Observed) for jit in decoder.jits.values())
      links = {key:tuple(jit.captured._linears) for key,jit in decoder.jits.items()}
      checks += exercise(decoder,model,cos,sin)
      assert model.python_greedy_calls==(22 if enabled else 42)
      assert links=={key:tuple(jit.captured._linears) for key,jit in decoder.jits.items()}
      before = model.cache[0].clone().realize()
      identity = tuple(id(t.uop.base) for t in model.cache)
      model.cache[0].assign(Tensor.zeros(model.cache[0].shape,device='CPU',dtype=dtypes.bfloat16)).realize()
      old.restore_prefill(model.cache,[before],lambda:None,identity=lambda:tuple(id(t.uop.base) for t in model.cache),
        equal=lambda a,b:np.testing.assert_array_equal(raw(a),raw(b)),copy_into=lambda a,b:a.assign(b).realize())
      assert identity==tuple(id(t.uop.base) for t in model.cache)
      try: decoder.decode(logits,cos,sin,10,0,7)
      except RuntimeError: pass
      else: raise AssertionError('Unwarmed length admitted')
    comparison = prototype.compare_signatures(*reports)
    for target in ('kernel','alias','replay'):
      bad = copy.deepcopy(reports[1])
      if target=='kernel': bad['serial_signatures']['2']['kernels'].pop()
      elif target=='alias': bad['serial_signatures']['2']['linked_alias_variants']=1
      else: bad['calls']=[r for r in bad['calls'] if not(r['count']==2 and r['phase']=='replay')]
      try: prototype.compare_signatures(reports[0],bad)
      except AssertionError: pass
      else: raise AssertionError('Corrupt qualification evidence accepted: '+target)
    # Failed construction must restore the caller's original map.
    decoder = SimpleNamespace(jits={})
    original = decoder.jits
    try:
      with prototype.install(decoder,True,old.ClosedGraphs): raise ValueError('injected')
    except ValueError: pass
    assert decoder.jits is original
    # Priming is mandatory; the candidate never skips a first-ever length-six call.
    from tinygrad import TinyJit
    ledger = {'created':[],'calls':[]}
    mapping = prototype.CaptureMap({},True,ledger)
    mapping[(cos.shape,6,80)] = TinyJit(lambda x:x+1)
    assert mapping[(cos.shape,6,80)].cnt==0 and not ledger['created'][0]['direct_capture']
  if any(name not in ('CPU','NPY') for name in Device._opened_devices): raise AssertionError('Unexpected backend')
  result = dict(cpu_only=True,gpu_executed=False,passed=True,independent_mutated_cases=checks,
    actual_block_decoder=True,actual_pinned_tinyjit=True,comparison=comparison,models=reports,
    restored_same_storage=True,unseen_length_rejected=True,priming_required=True,failure_restores_original_map=True,
    mutations_create_no_new_jit_links=True,corrupt_kernel_alias_and_replay_evidence_rejected=True,
    python_greedy_body_calls_at_ready=[r['cpu_python_greedy_body_calls_at_ready'] for r in reports],
    devices=list(Device._opened_devices),sources={name:prototype.sha(ROOT/name) for name in
      ('agent-startup-cache-check.py','agent-startup-cache-capture.py','agent-startup-cache-run.py')})
  if args.output:
    with args.output.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
  print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))


if __name__=='__main__': main()

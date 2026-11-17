"""Candidate binding completion. CPU verification by default; parent owns GPU execution."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-output-finish-work')
PROTOCOL=ROOT/'prepared/protocol.json'


def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def verify():
  evidence=load('agent-ck-output-finish-evidence')
  previous,_,reuse=evidence.verify()
  plan=json.loads(PROTOCOL.read_text())
  assert plan['schema']==1 and plan['timeout_seconds']==120 and plan['internal_timeout_seconds']==110
  assert plan['limits']==previous['limits'] and plan['reuse']==reuse and plan['adoption_allowed'] is False
  for path,digest in plan['source_sha256'].items(): assert evidence.sha(path)==digest, f'Changed frozen source: {path}'
  revision=subprocess.check_output(['git','-C',plan['tinygrad'],'rev-parse','HEAD'],text=True).strip()
  assert revision==plan['tinygrad_commit']=='89846dcfb383a98ee01cfec3bba440ae7dcf9b1e'
  arrays,proof=evidence.fixture_proof(plan['oracle'])
  assert proof==plan['input_patch']
  return plan,arrays


def statistics(bits,dtype,poison):
  import numpy as np
  values=((bits.astype(np.uint32)<<16).view(np.float32) if dtype=='bf16' else bits.view(np.float32)).reshape(-1)
  indices=np.unique(np.clip(np.array([0,1,63,64,127,128,255,256,8191,8192,131071,131072,len(bits)-2,len(bits)-1]),0,len(bits)-1))
  finite=np.isfinite(values)
  return {'elements':int(bits.size),'finite_count':int(finite.sum()),'nonfinite_count':int((~finite).sum()),
    'poison_count':int(np.count_nonzero(bits==poison)),'sha256':hashlib.sha256(bits.tobytes()).hexdigest(),
    'bounded_dump':[{'index':int(i),'bits':hex(int(bits[i])),'value':float(values[i]) if finite[i] else str(values[i])} for i in indices]}


@contextmanager
def observe_replays(helper,report,inputs):
  """Check the actual input UOps passed to normal captured HCQ execution."""
  import importlib
  jit=importlib.import_module('tinygrad.engine.jit')
  original=jit.run_linear
  def observed(linear,*args,**kwargs):
    if kwargs.get('jit',False):
      supplied=kwargs['input_uops']
      assert tuple(supplied)==tuple(t.uop.base for t in inputs)
      report['replay_dispatches'].append({'phase':report['active_phase'],'linked_linear_key':linear.key.hex(),
        'input_keys':[u.key.hex() for u in supplied],
        'input_storage':[helper.storage_info(t,True) for t in inputs]})
    return original(linear,*args,**kwargs)
  jit.run_linear=observed
  try: yield
  finally:
    assert jit.run_linear is observed
    jit.run_linear=original


def run(plan,arrays,output):
  import numpy as np
  from tinygrad import TinyJit,dtypes
  from tinygrad.device import Device,Compiled
  helper,old,observer=(load(name) for name in ('agent-ck-output-common','agent-ck-attention-v3-run','agent-ck-output-run'))
  evidence=load('agent-ck-output-finish-evidence')
  adapter=helper.load('agent-ck-attention-v2-adapter')
  report={'schema':1,'completed':False,'gpu_executed':False,'binding_proven':False,'adoption_allowed':False,
    'scope':plan['scope'],'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'reuse':plan['reuse'],'active_phase':'initialization','phases':[],'hcq_batches':[],
    'native_argument_records':[],'replay_dispatches':[],'io_completed':{'input_write_bytes':0,'poison_write_bytes':0,'readback_bytes':0}}
  def save(): output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  started=time.monotonic()
  def deadline(*_): raise TimeoutError('110-second candidate binding completion deadline')
  signal.signal(signal.SIGALRM,deadline);signal.alarm(plan['internal_timeout_seconds'])
  Compiled.wait_timeout_ms=10000
  save()
  try:
    device=Device[Device.DEFAULT]
    assert device.is_usb and device.arch=='gfx1200' and not device.prof_enabled
    report['gpu_executed']=True
    inputs=[old.allocate(device.device,s,dtypes.bfloat16) for s in (helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE)]
    storage=[old.allocate(device.device,(helper.OC+512,),dtypes.bfloat16),old.allocate(device.device,(helper.LC+512,),dtypes.float32)]
    all_inputs=(*inputs,*storage)
    identity=[helper.storage_info(t,True) for t in all_inputs]
    runner=TinyJit(lambda *xs:helper.execute('candidate',True,*xs))
    o_poison=np.full(helper.OC+512,0x7fc1,np.uint16);o_poison[:256]=o_poison[-256:]=0x5aa5
    l_poison=np.full(helper.LC+512,0x7fc12345,np.uint32);l_poison[:256]=l_poison[-256:]=0x5aa55aa5
    poisons=(o_poison.tobytes(),l_poison.tobytes())
    first=None
    captured_key=None
    def phase(spec):
      nonlocal first,captured_key
      name,fixture,kind=spec['name'],spec['fixture'],spec['kind']
      report['active_phase']=name
      row={'name':name,'arm':'candidate','fixture':fixture,'kind':kind,'normalize':True,'stage':'synchronize_before_input',
        'before':{'inputs':[helper.storage_info(t,True) for t in inputs],
                  'guarded_storage':[helper.storage_info(t,True) for t in storage]}}
      report['phases'].append(row);save()
      device.synchronize()
      assert [helper.storage_info(t,True) for t in all_inputs]==identity
      row['stage']='input_upload';save()
      row['input_writes']=[]
      if kind=='first':
        for key,tensor in zip(('q','k','v'),inputs,strict=True):
          payload=arrays['a'][key].tobytes();old.write(tensor,payload)
          row['input_writes'].append({'buffer':key,'byte_offset':0,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
          report['io_completed']['input_write_bytes']+=len(payload)
      elif kind=='replay':
        offset,length=plan['input_patch']['byte_offset'],plan['input_patch']['byte_length']
        payload=arrays[fixture]['v'].tobytes()[offset:offset+length]
        assert hashlib.sha256(payload).hexdigest()==plan['input_patch']['patch_sha256'][fixture]
        buffer=inputs[2].uop.buffer
        assert buffer.nbytes==4194304 and buffer.offset==0 and buffer.host is not None
        buffer.host.view(offset=offset,size=length,fmt='B')[:]=payload
        row['input_writes'].append({'buffer':'v','byte_offset':offset,'bytes':length,'sha256':hashlib.sha256(payload).hexdigest()})
        report['io_completed']['input_write_bytes']+=length
      else: assert kind=='capture'
      row['stage']='poison_full_guarded_O_LSE';save()
      for tensor,payload in zip(storage,poisons,strict=True):
        old.write(tensor,payload);report['io_completed']['poison_write_bytes']+=len(payload)
      row['stage']='execute';row['jit_count_before']=runner.cnt;save()
      assert runner.cnt==spec['jit_count_before']
      returned=runner(*all_inputs)
      row['jit_count_after']=runner.cnt
      assert runner.cnt==spec['jit_count_before']+1
      if runner.captured is not None:
        captured=runner.captured
        row['capture']={'expected_names':captured.expected_names,'input_buffer_keys':[t.uop.base.key.hex() for t in all_inputs],
          'staging_maps':[{u.key.hex():v.key.hex() for u,v in mapping.items()} for _,mapping in captured._linears.values()],
          'compiled_linear_key':captured._linear.key.hex()}
        assert captured.expected_names==[0,1,2,3,4] and len(captured._linears)==1
        assert all(not m for m in row['capture']['staging_maps'])
        if captured_key is None: captured_key=captured._linear.key.hex()
        assert captured_key==captured._linear.key.hex()
      row['stage']='synchronize_after_execute';save();device.synchronize()
      assert [helper.storage_info(t,True) for t in all_inputs]==identity
      row['after']={'guarded_storage':[helper.storage_info(t,True) for t in storage],
                    'returned':[helper.storage_info(t,True) for t in returned]}
      row['stage']='full_guarded_and_returned_readback';save()
      bits=[]
      for index,(key,count,dtype,kindstr,poison,guard) in enumerate([
          ('O',helper.OC,np.uint16,'bf16',0x7fc1,0x5aa5),('LSE',helper.LC,np.uint32,'f32',0x7fc12345,0x5aa55aa5)]):
        raw=old.read(storage[index],device);report['io_completed']['readback_bytes']+=len(raw)
        whole=np.frombuffer(raw,dtype);guarded=whole[256:256+count].copy()
        raw_return=old.read(returned[index],device);report['io_completed']['readback_bytes']+=len(raw_return)
        ret=np.frombuffer(raw_return,dtype)
        assert guarded.size==ret.size==count and whole.size==count+512
        row[key]={'guarded':statistics(guarded,kindstr,poison),'returned':statistics(ret,kindstr,poison),
          'guard_mismatch_count':int(np.count_nonzero(whole[:256]!=guard)+np.count_nonzero(whole[-256:]!=guard)),
          'guarded_returned_mismatch_count':int(np.count_nonzero(guarded!=ret)),
          'expected_kernel_gpu_address':identity[3+index]['base_gpu_address']+256*np.dtype(dtype).itemsize}
        save();evidence.component(row,key,count);bits.append(guarded)
      obits,lbits=bits
      value=(obits.astype(np.uint32)<<16).view(np.float32).reshape(helper.QSHAPE)
      row['stage']='independent_full_FP32_gates'
      row['fp32_error']=old.metrics(value,arrays[fixture]['output_fp32'])
      delta=lbits.view(np.float32).reshape(helper.LSHAPE).astype(np.float64)-arrays[fixture]['lse_fp32']
      row['lse_error']={'rmse':float(np.sqrt(np.mean(delta*delta))),'max_abs':float(np.max(np.abs(delta)))}
      save()
      evidence.error_gate(row['fp32_error'],plan['reuse']['baseline_errors'][fixture],plan['limits'])
      assert row['lse_error']['rmse']<=plan['limits']['lse_rmse_abs'] and row['lse_error']['max_abs']<=plan['limits']['lse_max_abs']
      if kind=='first': first=(obits.copy(),lbits.copy())
      assert first is not None
      if fixture=='a':
        row['same_A_output_mismatch_count']=int(np.count_nonzero(obits!=first[0]))
        row['same_A_lse_mismatch_count']=int(np.count_nonzero(lbits!=first[1]))
        assert row['same_A_output_mismatch_count']==row['same_A_lse_mismatch_count']==0
        assert row['O']['guarded']['sha256']==plan['reuse']['candidate_A_output_sha256']
        assert row['LSE']['guarded']['sha256']==plan['reuse']['candidate_A_lse_sha256']
        row['matches_frozen_candidate_first_A']=True
      else:
        original=(first[0].astype(np.uint32)<<16).view(np.float32).reshape(helper.QSHAPE)
        changed=value!=original
        row['mutation_max_abs_change']=float(np.max(np.abs(value-original)))
        row['mutation_changed_elements']=int(changed.sum())
        # V head 2 can affect only Q heads 4/5, and rows before 128 remain causal-independent.
        allowed=np.zeros(helper.QSHAPE,bool);allowed[:,4:6,128:,:]=True
        row['mutation_changed_outside_causal_GQA_support']=int(np.count_nonzero(changed & ~allowed))
        row['mutation_LSE_mismatch_count']=int(np.count_nonzero(lbits!=first[1]))
        assert row['mutation_max_abs_change']>plan['limits']['mutation_min_output_max_abs_change']
        assert row['mutation_changed_outside_causal_GQA_support']==row['mutation_LSE_mismatch_count']==0
      row['stage']='complete';row['phase_gates_passed']=True;save()
    programs=ROOT/'parent-programs';programs.mkdir(exist_ok=False)
    with adapter.abi_scope(),observer.bindings(helper,plan,report),old.record_programs(report,programs),observe_replays(helper,report,all_inputs):
      for spec in plan['phases']: phase(spec)
    report['active_phase']='binding_evidence_validation';save()
    dispatches=report['replay_dispatches']
    assert [r['phase'] for r in dispatches]==[s['name'] for s in plan['phases'][1:]]
    assert len({r['linked_linear_key'] for r in dispatches})==1
    for record in report['native_argument_records']:
      assert record['name']=='alpamayo_ck_d128' and record['kernarg_bytes']==296
      assert record['pointer_offsets']==[0,8,16,24,32] and record['dispatch_pointer']==record['CU_mode_bit']==0
      assert record['CK_implicit_hex']==adapter.implicit_args().hex()
      assert [v['offset_bytes'] for v in record['arguments']]==[512,1024,0,0,0]
    captured=[r for r in report['native_argument_records'] if [a['base_slot'] for a in r['arguments']]==[3,4,0,1,2]]
    assert captured and all(r['phase']==plan['phases'][1]['name'] for r in captured)
    for batch in report['hcq_batches']:
      rows=batch['calls'];assert len(rows)==3 and rows[0]['name']=='alpamayo_ck_d128'
      assert rows[0]['outs']==(0,1) and rows[0]['ins']==(2,3,4)
      assert rows[1]['dependencies']==rows[2]['dependencies']==[0]
      for index in (1,2):
        assert rows[index]['arguments'][1]==rows[0]['arguments'][index-1]
    assert report['hcq_batches'] and set(report['actual_programs'])==set(plan['programs'])
    assert report['io_completed']==plan['io_budget']
    report['binding_proven']=True;report['completed']=True;report['active_phase']='complete'
    report['remaining_gates']=plan['remaining_gates'];device.synchronize()
  except BaseException as error:
    report['failure']={'phase':report['active_phase'],'type':type(error).__name__,'message':str(error)}
    raise
  finally:
    signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--parent-run',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'parent-result.json')
  args=parser.parse_args();plan,arrays=verify()
  if not args.parent_run: print(json.dumps({'cpu_verification':'passed','gpu_executed':False,'protocol':str(PROTOCOL),'phases':plan['phases']}))
  else:
    assert not sys.flags.optimize and args.output.parent.resolve()==ROOT.resolve() and not args.output.exists()
    os.environ.update(DEV='USB+AMD:LLVM',CPU='0',AMD='0',ALLOW_DEVICE_USAGE='1',PARALLEL='0',PROFILE='0',
      BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',LLVMOPT='1',GMMU='0',TC_OPT='2',DEBUG='0',
      TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'),PYTHONDONTWRITEBYTECODE='1',
      OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    sys.path.insert(0,plan['tinygrad']);run(plan,arrays,args.output)

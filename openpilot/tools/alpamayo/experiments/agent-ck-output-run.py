"""One <=120-second parent-only guarded-output binding diagnostic.

Default: CPU hash verification only. No numerical kernel or old gate is changed.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

EXP = Path(__file__).parent
ROOT = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-output')
PROTOCOL = ROOT/'prepared-final/protocol.json'


def load(name):
  spec = importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def verify():
  plan = json.loads(PROTOCOL.read_text())
  assert plan['schema']==1 and plan['timeout_seconds']==120
  for path,expected in plan['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, f'Changed frozen file: {path}'
  return plan


@contextmanager
def bindings(helper,plan,report):
  from tinygrad.runtime import ops_amd
  from tinygrad.runtime.support import hcq2
  from tinygrad.engine.realize import get_call_outs_ins,get_call_kernel_args
  from tinygrad.uop.ops import UOp
  from tinygrad import dtypes
  old_build,old_queues,old_args,old_packet = ops_amd.amd_build_program,hcq2._build_queues,ops_amd.AMDComputeQueue.kernargs,ops_amd.dispatch_packet
  def build(dev,program,devices):
    blob = program.src[3].arg
    digest = hashlib.sha256(blob).hexdigest()
    assert digest in plan['programs'], f'Unprepared program {program.src[0].arg.name}: {digest}'
    assert hashlib.sha256(program.src[2].arg.encode()).hexdigest()==plan['programs'][digest]['source_sha256']
    data,_ = ops_amd._amd_program_image(dev,blob)
    assert ((data.rsrc1>>29)&1)==0
    return old_build(dev,program,devices)
  def packet(*args,**kwargs):
    return [UOp.const(x,dtypes.uint32) if isinstance(x,int) else x for x in old_packet(*args,**kwargs)]
  def queues(ctx):
    qnames = [(d,q) for _,d,q in ctx.batch]
    tracker = hcq2.DepsTracker(qnames)
    rows = []
    for tag,(call,device,queue) in enumerate(ctx.batch):
      outs,ins = get_call_outs_ins(call)
      deps = tracker.access_resources([(u,i in outs) for i,u in enumerate(call.src[1:]) if not u.is_variable],tag)
      rows.append({'tag':tag,'name':call.body.src[0].arg.name,'device':device,'queue':queue,
        'outs':outs,'ins':ins,'dependencies':deps,'arguments':[helper.view_info(x) for x in call.src[1:] if not x.is_variable]})
    report['hcq_batches'].append({'phase':report['active_phase'],'calls':rows})
    return old_queues(ctx)
  def kernargs(queue,call,program,data):
    name = program.src[0].arg.name
    if name in ('agent_attention_rowsoftmax','alpamayo_ck_d128'):
      values = get_call_kernel_args(call,program,queue.devs)
      from tinygrad.runtime.support.hcq2 import layout_args
      views = [helper.view_info(x) for x in call.src[1:]]
      if '_normalized_' in report['active_phase']:
        assert views[0]['offset_bytes']==512 and views[0]['bytes']==helper.OC*2
        if name=='alpamayo_ck_d128': assert views[1]['offset_bytes']==1024 and views[1]['bytes']==helper.LC*4
      report['native_argument_records'].append({'phase':report['active_phase'],'name':name,
        'pointer_offsets':[offset for offset,_ in layout_args(values)],'kernarg_bytes':data.kernargs_segment_size,
        'dispatch_pointer':data.enable_dispatch_ptr,'CU_mode_bit':(data.rsrc1>>29)&1,
        'arguments':views,
        'CK_implicit_hex':helper.load('agent-ck-attention-v2-adapter').implicit_args().hex() if name=='alpamayo_ck_d128' else None})
    return old_args(queue,call,program,data)
  ops_amd.amd_build_program,hcq2._build_queues,ops_amd.AMDComputeQueue.kernargs,ops_amd.dispatch_packet = build,queues,kernargs,packet
  try: yield
  finally:
    assert ops_amd.amd_build_program is build and hcq2._build_queues is queues and ops_amd.AMDComputeQueue.kernargs is kernargs
    ops_amd.amd_build_program,hcq2._build_queues,ops_amd.AMDComputeQueue.kernargs,ops_amd.dispatch_packet = old_build,old_queues,old_args,old_packet


def run(plan,output):
  import numpy as np
  from tinygrad import TinyJit,dtypes
  from tinygrad.device import Device,Compiled
  helper,old = load('agent-ck-output-common'),load('agent-ck-attention-v3-run')
  adapter = helper.load('agent-ck-attention-v2-adapter')
  report = {'schema':1,'completed':False,'gpu_executed':False,'binding_proven':False,'adoption_allowed':False,
    'scope':plan['scope'],'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
    'active_phase':'initialization','phases':[],'hcq_batches':[],'native_argument_records':[],
    'fix':'Explicit contiguous registration of O and LSE offset views before custom_kernel. Guards and native binaries retained.'}
  def save(): output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  started = time.monotonic()
  def deadline(*_): raise TimeoutError('110-second binding diagnostic deadline')
  signal.signal(signal.SIGALRM,deadline)
  signal.alarm(110)
  Compiled.wait_timeout_ms=10000
  save()
  try:
    device = Device[Device.DEFAULT]
    assert device.is_usb and device.arch=='gfx1200' and not device.prof_enabled
    report['gpu_executed']=True
    arrays = {name:{key:np.load(Path(plan['oracle'])/name/(key+'.npy'),allow_pickle=False)
              for key in ('q','k','v','output_fp32','lse_fp32')} for name in ('a','mutate_v')}
    inputs = [old.allocate(device.device,shape,dtypes.bfloat16) for shape in (helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE)]
    stores = {key:[old.allocate(device.device,(helper.OC+512,),dtypes.bfloat16),
                  old.allocate(device.device,(helper.LC+512,),dtypes.float32)]
              for key in ('original','baseline','candidate')}
    o_poison = np.full(helper.OC+512,0x7fc1,np.uint16)
    l_poison = np.full(helper.LC+512,0x7fc12345,np.uint32)
    o_poison[:256]=o_poison[-256:]=0x5aa5
    l_poison[:256]=l_poison[-256:]=0x5aa55aa5
    first,errors = {},{}
    runners = {arm:TinyJit(lambda *xs, arm=arm:helper.execute(arm,True,*xs)) for arm in ('baseline','candidate')}
    def statistics(bits,dtype,poison):
      values = ((bits.astype(np.uint32)<<16).view(np.float32) if dtype=='bf16' else bits.view(np.float32)).reshape(-1)
      indices = np.unique(np.clip(np.array([0,1,63,64,127,128,255,256,8191,8192,131071,131072,len(bits)-2,len(bits)-1]),0,len(bits)-1))
      finite = np.isfinite(values)
      return {'elements':int(bits.size),'finite_count':int(finite.sum()),'nonfinite_count':int((~finite).sum()),
        'poison_count':int(np.count_nonzero(bits==poison)), 'sha256':hashlib.sha256(bits.tobytes()).hexdigest(),
        'bounded_dump':[{'index':int(i),'bits':hex(int(bits[i])),'value':float(values[i]) if finite[i] else str(values[i])} for i in indices]}
    def phase(name,arm,fixture,normalize,kind):
      report['active_phase']=name
      device.synchronize()
      for key,t in zip(('q','k','v'),inputs): old.write(t,arrays[fixture][key].tobytes())
      storage = stores[arm if normalize else 'original']
      old.write(storage[0],o_poison.tobytes());old.write(storage[1],l_poison.tobytes())
      row = {'name':name,'arm':arm,'fixture':fixture,'normalize':normalize,'kind':kind,
        'before':{'inputs':[helper.storage_info(t,True) for t in inputs],
          'guarded_storage':[helper.storage_info(t,True) for t in storage]}}
      report['phases'].append(row);save()
      if normalize:
        count = runners[arm].cnt
        row['jit_count_before']=count
        returned = runners[arm](*inputs,*storage)
        row['jit_count_after']=runners[arm].cnt
        assert runners[arm].cnt==count+1
        if kind=='replay': assert count>=2
        if runners[arm].captured is not None:
          captured=runners[arm].captured
          row['capture']={'expected_names':captured.expected_names,
            'input_buffer_keys':[t.uop.base.key.hex() for t in (*inputs,*storage)],
            'staging_maps':[{u.key.hex():v.key.hex() for u,v in staging.items()} for _,staging in captured._linears.values()]}
          assert captured.expected_names==[0,1,2,3,4]
          assert all(not mapping for mapping in row['capture']['staging_maps']), 'Unexpected mutable-input staging'
      else: returned=helper.execute(arm,False,*inputs,*storage)
      device.synchronize()
      row['after']={'guarded_storage':[helper.storage_info(t,True) for t in storage],
        'returned':[helper.storage_info(t,True) for t in returned]}
      all_o=np.frombuffer(old.read(storage[0],device),np.uint16)
      obits=all_o[256:256+helper.OC].copy()
      ret_o=np.frombuffer(old.read(returned[0],device),np.uint16).copy()
      row['O']={'guarded':statistics(obits,'bf16',0x7fc1),'returned':statistics(ret_o,'bf16',0x7fc1),
        'guard_mismatch_count':int(np.count_nonzero(all_o[:256]!=o_poison[:256])+np.count_nonzero(all_o[-256:]!=o_poison[-256:])),
        'guarded_returned_mismatch_count':int(np.count_nonzero(obits!=ret_o)),
        'expected_kernel_gpu_address':row['before']['guarded_storage'][0]['base_gpu_address']+512}
      save()
      assert row['O']['guard_mismatch_count']==0
      if not normalize:
        row['original_guard_binding_failed']=row['O']['guarded']['nonfinite_count']>0
        row['classification']='Reproduction phase; original failure remains recorded and is not a numerical pass.'
        save();return
      assert row['O']['guarded']['nonfinite_count']==row['O']['returned']['nonfinite_count']==0, (name,'O finite gate')
      assert row['O']['guarded']['poison_count']==0 and row['O']['guarded_returned_mismatch_count']==0
      value=(obits.astype(np.uint32)<<16).view(np.float32).reshape(helper.QSHAPE)
      row['fp32_error']=old.metrics(value,arrays[fixture]['output_fp32'])
      if arm=='baseline': errors[fixture]=row['fp32_error']
      else:
        all_l=np.frombuffer(old.read(storage[1],device),np.uint32)
        lbits=all_l[256:256+helper.LC].copy()
        ret_l=np.frombuffer(old.read(returned[1],device),np.uint32).copy()
        row['LSE']={'guarded':statistics(lbits,'f32',0x7fc12345),'returned':statistics(ret_l,'f32',0x7fc12345),
          'guard_mismatch_count':int(np.count_nonzero(all_l[:256]!=l_poison[:256])+np.count_nonzero(all_l[-256:]!=l_poison[-256:])),
          'guarded_returned_mismatch_count':int(np.count_nonzero(lbits!=ret_l)),
          'expected_kernel_gpu_address':row['before']['guarded_storage'][1]['base_gpu_address']+1024}
        delta=lbits.view(np.float32).reshape(helper.LSHAPE).astype(np.float64)-arrays[fixture]['lse_fp32']
        row['lse_error']={'rmse':float(np.sqrt(np.mean(delta*delta))) if np.isfinite(delta).all() else None,
                          'max_abs':float(np.max(np.abs(delta))) if np.isfinite(delta).all() else None}
        save()
        assert row['LSE']['guard_mismatch_count']==row['LSE']['guarded_returned_mismatch_count']==0
        assert row['LSE']['guarded']['nonfinite_count']==row['LSE']['returned']['nonfinite_count']==0
        assert row['LSE']['guarded']['poison_count']==0
        old.error_gate(row['fp32_error'],errors[fixture],plan['limits'])
        assert row['lse_error']['rmse']<=plan['limits']['lse_rmse_abs'] and row['lse_error']['max_abs']<=plan['limits']['lse_max_abs']
      if kind=='first': first[arm]=(obits.copy(), lbits.copy() if arm=='candidate' else None)
      elif fixture=='a':
        row['same_A_output_mismatch_count']=int(np.count_nonzero(obits!=first[arm][0]));assert row['same_A_output_mismatch_count']==0
        if arm=='candidate': assert np.array_equal(lbits,first[arm][1])
      elif fixture=='mutate_v':
        original=(first[arm][0].astype(np.uint32)<<16).view(np.float32).reshape(helper.QSHAPE)
        row['mutation_max_abs_change']=float(np.max(np.abs(value-original)))
        assert row['mutation_max_abs_change']>plan['limits']['mutation_min_output_max_abs_change']
      row['phase_gates_passed']=True;save()
    with adapter.abi_scope(),bindings(helper,plan,report):
      phase('baseline_original_first_A','baseline','a',False,'first')
      for arm in ('baseline','candidate'):
        phase(arm+'_normalized_first_A',arm,'a',True,'first')
        phase(arm+'_normalized_capture_A',arm,'a',True,'capture')
        phase(arm+'_normalized_replay_mutate_v',arm,'mutate_v',True,'replay')
        phase(arm+'_normalized_replay_A',arm,'a',True,'replay')
      report['binding_proven']=True
      report['completed']=True
      device.synchronize()
  except BaseException as error:
    report['failure']={'phase':report['active_phase'],'type':type(error).__name__,'message':str(error)}
    raise
  finally:
    signal.alarm(0)
    report['elapsed_seconds']=time.monotonic()-started
    save()


if __name__=='__main__':
  parser=argparse.ArgumentParser()
  parser.add_argument('--parent-run',action='store_true')
  parser.add_argument('--output',type=Path,default=ROOT/'parent-result.json')
  args=parser.parse_args();plan=verify()
  if not args.parent_run: print(json.dumps({'cpu_verification':'passed','gpu_executed':False,'protocol':str(PROTOCOL)}))
  else:
    assert args.output.parent.resolve()==ROOT.resolve() and not args.output.exists()
    os.environ.update(DEV='USB+AMD:LLVM',CPU='0',AMD='0',ALLOW_DEVICE_USAGE='1',PARALLEL='0',PROFILE='0',
      BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',LLVMOPT='1',GMMU='0',TC_OPT='2',DEBUG='0',
      TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'),PYTHONDONTWRITEBYTECODE='1',
      OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    sys.path.insert(0,plan['tinygrad']);run(plan,args.output)

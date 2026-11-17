"""Parent-only complete numerical, ordered replay, and fair-boundary CK qualification."""
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
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-ck-qualified-work')
PROTOCOL=ROOT/'prepared-final/protocol.json'


def load(name):
  spec=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  return module


def verify():
  assert not sys.flags.optimize
  common=load('agent-ck-qualified-common');previous,oracle,_,reuse=load('agent-ck-qualified-evidence').verify()
  plan=json.loads(PROTOCOL.read_text())
  assert plan['schema']==1 and plan['timeout_seconds']==300 and plan['internal_timeout_seconds']==290
  assert plan['reuse']==reuse and plan['limits']==previous['limits'] and plan['replay_order']==oracle['replay_order']
  for path,digest in plan['source_sha256'].items(): assert common.sha(path)==digest,f'Changed frozen file: {path}'
  assert subprocess.check_output(['git','-C',plan['tinygrad'],'rev-parse','HEAD'],text=True).strip()==plan['tinygrad_commit']
  import numpy as np
  cases={f['name']:{key:np.load(Path(plan['oracle'])/item['path'],allow_pickle=False) for key,item in f['files'].items()}
    for f in oracle['fixtures']}
  for digest,node in plan['input_bank'].items():
    payload=cases[node['fixture']][node['key']].tobytes()
    assert hashlib.sha256(payload).hexdigest()==digest and len(payload)==node['bytes']
    if node['parent']:
      parent=plan['input_bank'][node['parent']];before=cases[parent['fixture']][parent['key']].tobytes()
      assert common.ranges(before,payload)==node['upload_ranges']
    else: assert node['upload_ranges']==[[0,len(payload)]]
  return plan,cases


@contextmanager
def compile_observer(report):
  from tinygrad.engine import realize
  original=realize.to_program
  def observed(*args,**kwargs):
    started=time.monotonic()
    result=original(*args,**kwargs)
    report['compilation'].append({'phase':report['active_phase'],'seconds':time.monotonic()-started,
      'name':result.src[0].arg.name,'binary_sha256':hashlib.sha256(result.src[3].arg).hexdigest()})
    return result
  realize.to_program=observed
  try: yield
  finally:
    assert realize.to_program is observed;realize.to_program=original


@contextmanager
def replay_observer(report,states):
  import importlib
  jit=importlib.import_module('tinygrad.engine.jit');original=jit.run_linear
  def observed(linear,*args,**kwargs):
    if kwargs.get('jit',False) and not report.get('in_timing',False):
      supplied=tuple(kwargs['input_uops'])
      matches=[key for key,xs in states.items() if supplied==tuple(t.uop.base for t in xs)]
      if len(supplied)==5:
        assert len(matches)==1
        report.setdefault('actual_replay_bindings',[]).append({'phase':report['active_phase'],'boundary':matches[0][0],
          'arm':matches[0][1],'linked_linear_key':linear.key.hex(),'input_keys':[u.key.hex() for u in supplied]})
      else:
        assert not matches and len(supplied) in (1,2,3)
        report['helper_captured_dispatches']=report.get('helper_captured_dispatches',0)+1
    return original(linear,*args,**kwargs)
  jit.run_linear=observed
  try: yield
  finally:
    assert jit.run_linear is observed;jit.run_linear=original


def run(plan,cases,output):
  import numpy as np
  from tinygrad import TinyJit,dtypes
  from tinygrad.device import Device,Compiled
  common=load('agent-ck-qualified-common');helper=common.load('agent-ck-output-common')
  adapter=helper.load('agent-ck-attention-v2-adapter')
  old=load('agent-ck-attention-v3-run');observations=load('agent-ck-output-run')
  finish=load('agent-ck-output-finish-run');gate=load('agent-ck-output-finish-evidence')
  report={'schema':1,'completed':False,'gpu_executed':False,'adoption_allowed':False,'model_integrated':False,
    'numerical_micro_gates_passed':False,'classification':plan['classification'],'limits':plan['limits'],
    'protocol_sha256':common.sha(PROTOCOL),'reused_binding_evidence':plan['reuse'],'active_phase':'initialization',
    'phases':[],'hcq_batches':[],'native_argument_records':[],'compilation':[],
    'io':{'host_write_bytes':0,'host_read_bytes':0,'host_write_calls':0,'host_read_calls':0},
    'boundaries':plan['boundaries'],'boundary_B_limit':plan['boundary_B_limit'],
    'timing_kind':'Synchronized wall time of complete serialized boundaries, including submit. No isolated device-time claim.'}
  def save():
    temporary=output.with_name(output.name+'.inprogress')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');os.replace(temporary,output)
  @contextmanager
  def phase(name):
    report['active_phase']=name
    row={'name':name,'complete':False,'io_before':dict(report['io'])};report['phases'].append(row);save()
    started=time.monotonic()
    try: yield row;row['complete']=True
    finally:
      row['seconds']=time.monotonic()-started
      row['io_delta']={k:report['io'][k]-v for k,v in row['io_before'].items()};save()
  started=time.monotonic()
  def expired(*_): raise TimeoutError('290-second qualified CK protocol deadline')
  signal.signal(signal.SIGALRM,expired);signal.alarm(290);Compiled.wait_timeout_ms=10000;save()
  try:
    with phase('device_initialization') as row:
      device=Device[Device.DEFAULT]
      assert device.is_usb and device.arch=='gfx1200' and not device.prof_enabled
      report['gpu_executed']=True
      row['runtime']={'device':device.device,'arch':device.arch,'renderer':type(device.renderer).__name__,'mode_required':'CU','parallel':0}
    def allocate(shape,dt=dtypes.bfloat16): return old.allocate(device.device,shape,dt)
    def read(tensor):
      data=old.read(tensor,device);report['io']['host_read_bytes']+=len(data);report['io']['host_read_calls']+=1;return data
    def write(tensor,data,offset=0):
      b=tensor.uop.buffer
      assert offset>=0 and offset+len(data)<=b.nbytes and offset%4==len(data)%4==0 and b.host is not None
      b.host.view(offset=offset,size=len(data),fmt='B')[:]=data
      report['io']['host_write_bytes']+=len(data);report['io']['host_write_calls']+=1
    helpers={};helper_keys={};sentinels_checked=set()
    def call_helper(tag,function,args):
      key=(tag,tuple((tuple(t.shape),str(t.dtype)) for t in args))
      if key not in helpers: helpers[key]=TinyJit(function)
      runner=helpers[key];before=runner.cnt
      result=runner(*args);assert runner.cnt==before+1
      if runner.captured is not None:
        assert runner.captured.expected_names==list(range(len(args))) and len(runner.captured._linears)==1
        assert all(not staging for _,staging in runner.captured._linears.values())
        graph=next(iter(runner.captured._linears.values()))[0].key.hex()
        if key not in helper_keys: helper_keys[key]=graph
        assert helper_keys[key]==graph
      return result,key
    def utility(dst,src=None,mode='copy'):
      args=(dst,) if src is None else (dst,src)
      return call_helper('utility_'+mode,lambda *ts:common.utility(*ts,mode=mode).realize(),args)[0]
    def checked_validator(tag,function,args):
      key=(tag,tuple((tuple(t.shape),str(t.dtype)) for t in args))
      if key in helpers and helpers[key].captured is not None:
        target=helpers[key].captured.ret
        utility(target,mode='sentinel')
        if key not in sentinels_checked:
          device.synchronize();raw=np.frombuffer(read(target),np.uint32)
          assert (raw==0xffffffff).all()
          report.setdefault('sentinel_full_readbacks',[]).append({'tag':tag,'words':int(raw.size),'all_sentinel':True})
          sentinels_checked.add(key)
      return call_helper(tag,function,args)[0]
    bank={}
    inputs={b:[allocate(s) for s in ([common.QSHAPE,common.KVSHAPE,common.KVSHAPE] if b=='A' else
      [(1,2048,16,128),(1,8,2304,128),(1,8,2304,128)])] for b in ('A','B')}
    stores={(b,arm):[allocate((common.OC+512,)),allocate((common.LC+512,),dtypes.float32)]
      for b in ('A','B') for arm in ('baseline','candidate')}
    runners={(b,arm):TinyJit(lambda *xs,b=b,arm=arm:common.execute(b,arm,*xs)) for b,arm in stores}
    identities={(b,arm):[helper.storage_info(t,True) for t in (*inputs[b],*stores[b,arm])] for b,arm in stores}
    state={'A':None,'B':None};capture_keys={};calls={key:0 for key in runners};references={};gpu_references={}
    report['invocations']=[];report['flags']=[]
    def select(boundary,fixture):
      wanted=plan['fixture_inputs'][fixture]
      for index,key in enumerate(('q','k','v')):
        if state[boundary] is None or wanted[key]!=plan['fixture_inputs'][state[boundary]][key]:
          utility(inputs[boundary][index],bank[wanted[key]],'copy' if boundary=='A' else ('q_B' if key=='q' else 'kv_B'))
      state[boundary]=fixture
    def poison(boundary,arm):
      for tensor in stores[boundary,arm][:(2 if arm=='candidate' else 1)]: utility(tensor,mode='poison')
    def invoke(boundary,arm):
      key=boundary,arm;runner=runners[key];xs=(*inputs[boundary],*stores[key]);count=runner.cnt
      report['active_arm']=arm;report['active_boundary']=boundary
      assert [helper.storage_info(t,True) for t in xs]==identities[key]
      returned=runner(*xs);assert runner.cnt==count+1;calls[key]+=1
      record={'phase':report['active_phase'],'boundary':boundary,'arm':arm,'before':count,'after':runner.cnt}
      if runner.captured is not None:
        capture=runner.captured
        assert capture.expected_names==[0,1,2,3,4] and len(capture._linears)==1
        assert all(not staging for _,staging in capture._linears.values())
        keyhash=next(iter(capture._linears.values()))[0].key.hex()
        if key not in capture_keys: capture_keys[key]=keyhash
        assert keyhash==capture_keys[key]
        record.update(linked_linear_key=keyhash,input_keys=[t.uop.base.key.hex() for t in xs],
                      input_gpu_addresses=[helper.storage_info(t,True)['gpu_address'] for t in xs])
      report['invocations'].append(record)
      return returned
    def snapshot(boundary,arm,returned):
      device.synchronize();result={};row={}
      report.setdefault('snapshots',[]).append({'phase':report['active_phase'],'boundary':boundary,'arm':arm,'components':row})
      for index,key in enumerate(('O','LSE') if arm=='candidate' else ('O',)):
        count,dt,kind,poison_word,guard_word=(common.OC,np.uint16,'bf16',0x7fc1,0x5aa5) if key=='O' else (common.LC,np.uint32,'f32',0x7fc12345,0x5aa55aa5)
        whole=np.frombuffer(read(stores[boundary,arm][index]),dt).copy()
        ret=np.frombuffer(read(returned[index]),dt).copy()
        if key=='O' and boundary=='B': ret=ret.reshape(1,2048,16,128).transpose(0,2,1,3).copy().reshape(-1)
        bits=whole[256:-256]
        assert whole.size==count+512 and ret.size==count
        row[key]={'guarded':finish.statistics(bits,kind,poison_word),'returned':finish.statistics(ret,kind,poison_word),
          'guard_mismatch_count':int(np.count_nonzero(whole[:256]!=guard_word)+np.count_nonzero(whole[-256:]!=guard_word)),
          'guarded_returned_mismatch_count':int(np.count_nonzero(bits!=ret)),
          'guarded_storage':helper.storage_info(stores[boundary,arm][index],True),'returned_storage':helper.storage_info(returned[index],True)}
        save();gate.component(row,key,count)
        result[key]=whole
      return result,row
    def check_flags(boundary,arm,returned,fixture,expected=None,label='exact',save_arrays=False):
      # The immutable references were copied only after complete CPU snapshot validation.
      flags=[]
      for index,key in enumerate(('O','LSE') if arm=='candidate' else ('O',)):
        args=(stores[boundary,arm][index],returned[index],gpu_references[fixture,arm][index])
        tag='validate_'+('LSE' if key=='LSE' else boundary+'_O')
        result=checked_validator(tag,lambda *xs,b=boundary:common.validator(*xs,boundary=b).realize(),args)
        flags.append(result)
      device.synchronize()
      record={'phase':report['active_phase'],'label':label,'boundary':boundary,'arm':arm,'fixture':fixture,'components':{}}
      for index,(key,tensor) in enumerate(zip(('O','LSE'),flags)):
        values=np.frombuffer(read(tensor),np.uint32).reshape(len(common.CHANNELS),-1).copy()
        target=np.zeros_like(values) if expected is None else expected[key]
        assert values.shape==target.shape
        record['components'][key]={'channel_counts':dict(zip(common.CHANNELS,values.sum(axis=1,dtype=np.uint64).tolist())),
          'mismatched_flag_words':int(np.count_nonzero(values!=target)),
          'full_flag_sha256':hashlib.sha256(values.tobytes()).hexdigest()}
        if save_arrays:
          filename=ROOT/(label+'-'+key+'-flags.npy');assert not filename.exists();np.save(filename,values,allow_pickle=False)
          record['components'][key]['saved_flags']=str(filename)
        report['flags'].append(record) if index==0 else None
        save();assert np.array_equal(values,target),(label,key,'complete flag array mismatch')
      return record
    def tails(expected=None,label='tail_zero'):
      flags=checked_validator('validate_tails',lambda *xs:common.tail_validator(*xs).realize(),tuple(inputs['B'][1:]));device.synchronize()
      values=np.frombuffer(read(flags),np.uint32).reshape(2,-1)
      target=np.zeros_like(values) if expected is None else expected
      record={'phase':report['active_phase'],'label':label,'counts':values.sum(axis=1,dtype=np.uint64).tolist(),
        'mismatched_flag_words':int(np.count_nonzero(values!=target))}
      report.setdefault('tail_checks',[]).append(record);save();assert np.array_equal(values,target)
    def patch_words(tensor,before,after):
      # Byte-granular negative controls use aligned dwords; unchanged neighbours are retained.
      before,after=before.tobytes(),after.tobytes()
      for offset,length in common.ranges(before,after,block=4): write(tensor,after[offset:offset+length],offset)
    programs=ROOT/'parent-programs';programs.mkdir(exist_ok=False)
    replay_states={key:(*inputs[key[0]],*stores[key]) for key in runners}
    with adapter.abi_scope(),observations.bindings(helper,plan,report),old.record_programs(report,programs),compile_observer(report),replay_observer(report,replay_states):
      with phase('resident_fixture_bank') as row:
        before=report['io']['host_write_bytes']
        for digest,node in plan['input_bank'].items():
          tensor=allocate(tuple(node['shape']));payload=cases[node['fixture']][node['key']].tobytes()
          if node['parent']: utility(tensor,bank[node['parent']]);device.synchronize()
          for offset,length in node['upload_ranges']: write(tensor,payload[offset:offset+length],offset)
          bank[digest]=tensor
        assert report['io']['host_write_bytes']-before==plan['budget']['fixture_upload_bytes']
        row['distinct_buffers']=len(bank);row['resident_bytes']=sum(n['bytes'] for n in plan['input_bank'].values())
      report['full_actual_shape_checks']=[]
      for fixture in plan['fixtures']:
        with phase('fixture_'+fixture+'_A_normalized_both') as row:
          select('A',fixture);measured={};checks={}
          for arm in ('baseline','candidate'):
            poison('A',arm);returned=invoke('A',arm)
            measured[arm],checks[arm]=snapshot('A',arm,returned)
            gpu_references[fixture,arm]=[]
            for index,key in enumerate(('O','LSE') if arm=='candidate' else ('O',)):
              target=allocate((common.OC+512,)) if key=='O' else allocate((common.LC+512,),dtypes.float32)
              utility(target,stores['A',arm][index]);gpu_references[fixture,arm].append(target)
            check_flags('A',arm,returned,fixture,label='initial_full_CPU_crosscheck_'+fixture)
          errors={arm:old.metrics((r['O'][256:-256].astype(np.uint32)<<16).view(np.float32).reshape(common.QSHAPE),
                                 cases[fixture]['output_fp32']) for arm,r in measured.items()}
          delta=measured['candidate']['LSE'][256:-256].view(np.float32).reshape(common.LSHAPE).astype(np.float64)-cases[fixture]['lse_fp32']
          lse={'rmse':float(np.sqrt(np.mean(delta*delta))),'max_abs':float(np.max(np.abs(delta)))}
          row.update(fixture=fixture,checks=checks,errors=errors,lse=lse);report['full_actual_shape_checks'].append(row);save()
          gate.error_gate(errors['candidate'],errors['baseline'],plan['limits'])
          assert lse['rmse']<=plan['limits']['lse_rmse_abs'] and lse['max_abs']<=plan['limits']['lse_max_abs']
          if fixture=='a':
            assert checks['candidate']['O']['guarded']['sha256']==plan['reuse']['candidate_A_output_sha256']
            assert checks['candidate']['LSE']['guarded']['sha256']==plan['reuse']['candidate_A_lse_sha256']
            assert checks['baseline']['O']['guarded']['sha256']==plan['reuse']['baseline_A_output_sha256']
          references[fixture]=measured
      with phase('full_causal_GQA_Q_K_V_mutation_gates') as row:
        def bits(fixture,arm): return references[fixture][arm]['O'][256:-256].reshape(common.QSHAPE)
        for arm in ('baseline','candidate'):
          a=(bits('a',arm).astype(np.uint32)<<16).view(np.float32)
          for fixture in ('mutate_q','mutate_k','mutate_v'):
            value=(bits(fixture,arm).astype(np.uint32)<<16).view(np.float32)
            difference=float(np.max(np.abs(value-a)));row[arm+'_'+fixture]=difference
            assert difference>plan['limits']['mutation_min_output_max_abs_change']
          assert np.array_equal(bits('a',arm)[:,:,:1024],bits('causal_future',arm)[:,:,:1024])
          assert not np.array_equal(bits('a',arm)[:,:,1024:],bits('causal_future',arm)[:,:,1024:])
          assert all(np.array_equal(bits('gqa',arm)[:,h],bits('gqa',arm)[:,h+1]) for h in range(0,16,2))
        la=references['a']['candidate']['LSE'][256:-256].reshape(common.LSHAPE)
        lf=references['causal_future']['candidate']['LSE'][256:-256].reshape(common.LSHAPE)
        assert np.array_equal(la[:,:,:1024],lf[:,:,:1024])
        report['causal_gqa_mutations_passed']=True
      with phase('validator_all_bit_negative_controls_A_normalized_candidate') as row:
        select('A','a');poison('A','candidate');returned=invoke('A','candidate')
        check_flags('A','candidate',returned,'a',label='negative_control_clean')
        device.synchronize();expected={};modified={}
        for index,key in enumerate(('O','LSE')):
          whole=references['a']['candidate'][key];ret=whole[256:-256].copy()
          changed,changed_ret,edits=common.controls(whole,ret);modified[key]=(whole,ret,changed,changed_ret)
          assert stores['A','candidate'][index].uop.buffer.base is not returned[index].uop.buffer.base
          patch_words(stores['A','candidate'][index],whole,changed);patch_words(returned[index],ret,changed_ret)
          expected[key]=common.cpu_flags(changed,changed_ret,whole);row[key+'_edits']=edits
        check_flags('A','candidate',returned,'a',expected,label='negative_all_bits',save_arrays=True)
        for index,key in enumerate(('O','LSE')):
          whole,ret,changed,changed_ret=modified[key]
          patch_words(stores['A','candidate'][index],changed,whole);patch_words(returned[index],changed_ret,ret)
        check_flags('A','candidate',returned,'a',label='negative_controls_restored')
        poison('A','candidate');expected={}
        for key,dt,poison_word,guard in [('O',np.uint16,0x7fc1,0x5aa5),('LSE',np.uint32,0x7fc12345,0x5aa55aa5)]:
          whole=references['a']['candidate'][key];poisoned=np.full_like(whole,poison_word);poisoned[:256]=poisoned[-256:]=guard
          expected[key]=common.cpu_flags(poisoned,whole[256:-256],whole)
        check_flags('A','candidate',returned,'a',expected,label='full_GPU_poison',save_arrays=True)
        returned=invoke('A','candidate');check_flags('A','candidate',returned,'a',label='after_poison_recompute')
        report['validator_qualified']=True
      report['ordered_mutable_replay']=[]
      for index,fixture in enumerate(plan['replay_order']):
        with phase(f'ordered_{index:02d}_{fixture}_A_normalized_candidate') as row:
          select('A',fixture);poison('A','candidate');count=runners['A','candidate'].cnt;assert count>=2
          returned=invoke('A','candidate');record=check_flags('A','candidate',returned,fixture,label=f'replay_{index:02d}')
          row.update(index=index,fixture=fixture,jit_count_before=count,exact_full_bit_flags=record)
          report['ordered_mutable_replay'].append({'index':index,'fixture':fixture,'jit_count_after':runners['A','candidate'].cnt,
            'full_output_and_LSE_bitwise_equal':True,'all_finite_poison_redzones_aliases_passed':True})
      assert len(report['ordered_mutable_replay'])==40
      with phase('boundary_B_full_snapshot_normalized_both') as row:
        select('B','a');row['checks']={}
        for arm in ('baseline','candidate'):
          poison('B',arm);returned=invoke('B',arm)
          data,checks=snapshot('B',arm,returned)
          assert all(np.array_equal(data[key],references['a'][arm][key]) for key in data)
          row['checks'][arm]=checks
          poison('B',arm);returned=invoke('B',arm);assert runners['B',arm].cnt==2
          check_flags('B',arm,returned,'a',label='B_captured_conversion')
        tails()
      with phase('capacity_tail_all_bit_negative_controls') as row:
        expected=np.zeros((2,1024),np.uint32);patches=[]
        for input_index in (1,2):
          original=np.full((1,8,2304,128),0x7fc1,np.uint16)
          original[:,:,:2048]=cases['a']['k' if input_index==1 else 'v']
          changed=original.copy();tail=changed[:,:,2048:]
          for bit in range(16):
            # Distinct groups in head zero; all heads and the final tail word also receive controls.
            tail[0,0,bit*2,8]^=np.uint16(1<<bit)
          for head in range(8): tail[0,head,-1,-1]^=np.uint16(1)
          expected[input_index-1]=np.count_nonzero(tail.reshape(-1,256)!=0x7fc1,axis=1).astype(np.uint32)
          patch_words(inputs['B'][input_index],original,changed);patches.append((input_index,original,changed))
        tails(expected,label='tail_negative_all_bits')
        for index,original,changed in patches: patch_words(inputs['B'][index],changed,original)
        tails(label='tail_controls_restored');row['all_bits_and_heads_detected']=True
      report['timings']={}
      for boundary in ('A','B'):
        with phase('timing_'+boundary+'_normalized_both') as row:
          select(boundary,'a')
          for arm in ('baseline','candidate'):
            poison(boundary,arm);returned=invoke(boundary,arm)
            assert runners[boundary,arm].cnt>=2
            check_flags(boundary,arm,returned,'a',label='timing_precheck')
          samples={'baseline':[],'candidate':[]};start_calls=dict(calls)
          for repeat in range(plan['timing']['alternating_samples']):
            for arm in (('baseline','candidate') if repeat%2==0 else ('candidate','baseline')):
              key=boundary,arm;runner=runners[key];xs=(*inputs[boundary],*stores[key]);before=runner.cnt
              report['active_arm']=arm;report['active_boundary']=boundary
              assert [helper.storage_info(t,True) for t in xs]==identities[key]
              device.synchronize();report['in_timing']=True;begin=time.perf_counter()
              for _ in range(plan['timing']['calls_per_sample']): returned=runner(*xs)
              device.synchronize();elapsed=time.perf_counter()-begin;report['in_timing']=False
              assert runner.cnt==before+plan['timing']['calls_per_sample']
              calls[key]+=plan['timing']['calls_per_sample']
              assert next(iter(runner.captured._linears.values()))[0].key.hex()==capture_keys[key]
              samples[arm].append(elapsed*1000/plan['timing']['calls_per_sample'])
          for arm in ('baseline','candidate'):
            expected=plan['timing']['alternating_samples']*plan['timing']['calls_per_sample']
            assert calls[boundary,arm]-start_calls[boundary,arm]==expected
            returned=runners[boundary,arm].captured.ret
            check_flags(boundary,arm,returned,'a',label='timing_postcheck')
          if boundary=='B': tails(label='tail_after_timing')
          stats={arm:{'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),'samples_ms':values}
                 for arm,values in samples.items()}
          stats['median_ratio']=stats['candidate']['median_ms']/stats['baseline']['median_ms']
          stats['p95_ratio']=stats['candidate']['p95_ms']/stats['baseline']['p95_ms']
          report['timings'][boundary]=stats;row['samples']=stats
    with phase('final_source_ABI_binding_gates') as row:
      assert all(k in plan['programs'] for k in report['actual_programs'])
      assert all(r['CU_mode_bit']==r['dispatch_pointer']==0 for r in report['native_argument_records'])
      for record in report['native_argument_records']:
        assert record['arguments'][0]['offset_bytes']==512
        if record['name']=='alpamayo_ck_d128':
          assert record['pointer_offsets']==[0,8,16,24,32] and record['kernarg_bytes']==296
          assert record['arguments'][1]['offset_bytes']==1024 and record['CK_implicit_hex']==adapter.implicit_args().hex()
      for key,xs in identities.items(): assert [helper.storage_info(t,True) for t in (*inputs[key[0]],*stores[key])]==xs
      for boundary in ('A','B'):
        for arm in ('baseline','candidate'):
          names={r['name'] for b in report['hcq_batches'] if '_'+boundary+'_' in b['phase'] for r in b['calls']}
          assert ('alpamayo_ck_d128' if arm=='candidate' else 'agent_attention_rowsoftmax') in names
      for batch in report['hcq_batches']:
        rows=batch['calls']
        for native in [r for r in rows if r['name'] in ('alpamayo_ck_d128','agent_attention_rowsoftmax')]:
          for out_index in native['outs']:
            view=native['arguments'][out_index]
            readers=[r for r in rows[native['tag']+1:] if any(r['arguments'][i]['base_key']==view['base_key'] for i in r['ins'])]
            assert readers and all(native['tag'] in r['dependencies'] for r in readers)
      replay_records=[r for r in report['actual_replay_bindings'] if r['phase'].startswith('ordered_')]
      assert len(replay_records)==40 and len({r['linked_linear_key'] for r in replay_records})==1
      assert len({tuple(r['input_keys']) for r in replay_records})==1
      report['helper_execution_counts']=[{'tag':key[0],'signature':str(key[1]),'count':runner.cnt,
        'captured':runner.captured is not None,'linked_linear_key':helper_keys.get(key)} for key,runner in helpers.items()]
      assert {x['tag'] for x in report['sentinel_full_readbacks']}=={'validate_A_O','validate_B_O','validate_LSE','validate_tails'}
      report['tracked_host_payload_bytes']=report['io']['host_write_bytes']+report['io']['host_read_bytes']
      assert report['tracked_host_payload_bytes']<=plan['budget']['conservative_total_payload_bytes']
      device.synchronize()
      report['speed_screen_passed']=all(v['median_ratio']<=plan['limits']['speed_screen']['both_boundary_median_ratio_max'] and
        v['p95_ratio']<=plan['limits']['speed_screen']['both_boundary_p95_ratio_max'] for v in report['timings'].values())
      report['numerical_micro_gates_passed']=True
      report['completed']=True
      row['compile_seconds_total']=sum(r['seconds'] for r in report['compilation'])
      report['remaining']='Actual 28-layer prefill, complete pipeline, and route quality remain unqualified. Under-100ms goal remains unmet.'
  except BaseException as error:
    report['failure']={'phase':report['active_phase'],'arm':report.get('active_arm'),'boundary':report.get('active_boundary'),
      'type':type(error).__name__,'message':str(error)}
    report['completed']=False;raise
  finally:
    signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--parent-run',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'parent-result.json')
  args=parser.parse_args();plan,cases=verify()
  if not args.parent_run: print(json.dumps({'cpu_preflight':'passed','gpu_executed':False,'protocol':str(PROTOCOL),'budget':plan['budget']}))
  else:
    assert args.output.parent.resolve()==ROOT.resolve() and not args.output.exists()
    with args.output.open('x') as stream: stream.write('{"status":"starting","gpu_executed":false}\n')
    os.environ.update(DEV='USB+AMD:LLVM',CPU='0',AMD='0',ALLOW_DEVICE_USAGE='1',PARALLEL='0',PROFILE='0',BEAM='0',JITBEAM='0',JIT='1',CAPTURING='1',
      LLVMOPT='1',GMMU='0',TC_OPT='2',DEBUG='0',TMPDIR=str(ROOT),CACHEDB=str(ROOT/'cpu-cache.db'),PYTHONDONTWRITEBYTECODE='1',
      OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    sys.path.insert(0,plan['tinygrad']);run(plan,cases,args.output)

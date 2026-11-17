"""CPU producer-to-consumer lifecycle checks using the real pinned storage_info and Buffer APIs."""
import argparse
import ast
from contextlib import contextmanager
import copy
from dataclasses import replace
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

EXP=Path(__file__).parent
spec=importlib.util.spec_from_file_location('ck_storage_overlay',EXP/'agent-ck-storage-overlay.py')
O=importlib.util.module_from_spec(spec);spec.loader.exec_module(O)

def main(output):
  os.environ.update(DEV='USB+AMD:LLVM',ALLOW_DEVICE_USAGE='0',CACHEDB=str(O.ROOT/'cache.db'),TMPDIR=str(O.ROOT/'tmp'),
    PYTHONDONTWRITEBYTECODE='1',PROFILE='0',BEAM='0',JITBEAM='0')
  sys.path.insert(0,'/tmp/opencode/alpamayo-asius-20261114/tinygrad')
  from tinygrad import Tensor,dtypes,Device
  from tinygrad.device import Buffer,BufferSpec,HostAllocator
  from tinygrad.uop.ops import UOp
  def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
      assert not os.fsdecode(args[0]).startswith(('/dev/bus/usb','/dev/dri','/dev/kfd'))
  sys.addaudithook(audit)
  class SimulatedAllocator(HostAllocator):
    device_only=False
    def _alloc(self,size,options):
      storage=super()._alloc(size,options)
      # Address arithmetic uses a distinct simulated GPU address. Nothing dereferences it.
      return replace(storage,buf=storage.buf+(1<<46),host=None if self.device_only else storage.host)
  events=[]
  dev=SimpleNamespace(device='AMD',host='CPU',renderer=SimpleNamespace(compiler=SimpleNamespace()),
                      synchronize=lambda:events.append('synchronize'))
  dev.allocator=SimulatedAllocator(dev)
  def getitem(self,name): assert name=='AMD',('Device access forbidden',name);return dev
  assert not Device._opened_devices
  Device.__class__.__getitem__=getitem
  helper=O.load('agent-ck-output-common');old=O.load('agent-ck-model-common')
  shapes=[helper.QSHAPE,helper.KVSHAPE,helper.KVSHAPE];sizes=[math.prod(s)*2 for s in shapes]
  recorded=json.loads((O.WORK/'agent-ck-model-parent/report.json').read_text())
  assert recorded['failure']=={'type':'KeyError','message':"'offset'",'phase':'live_producer_layout'}
  assert recorded['io']=={'read_bytes':0,'poison_write_bytes':0}
  try: old.storage_gate(recorded['packed_inputs'],sizes)
  except KeyError as error: assert error.args==('offset',)
  else: raise AssertionError('Original failure did not reproduce')
  assert O.storage_gate(recorded['packed_inputs'],sizes)
  assert O.storage_gate([recorded['actual_cache_storage']],[9437184])
  original_catalog=json.loads((O.ORIGINAL_ROOT/'prepared-final/producer-catalog.json').read_text())
  assert recorded['producer_batches'] and all(b==original_catalog['batch'] for b in recorded['producer_batches'])
  assert recorded['producer_programs'] and all(original_catalog['associations'][aid]==item
    for aid,item in recorded['producer_programs'].items())
  proof={'gpu_executed':False,'arithmetic_executed':False,'opened_devices':[], 'gate_delta':O.check_delta(),
    'recorded_failure_reproduced':True,'recorded_real_storage_rows_pass_corrected_gate':True,
    'prior_producer_bindings_preserved':True,'positive_lifecycles':[],'negative_controls':[]}
  tree=ast.parse(O.ORIGINAL_RUNNER.read_text())
  capture=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='capture')
  begin=next(i for i,n in enumerate(capture.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='packed')
  end=next(i for i,n in enumerate(capture.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
           and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='storage_gate')
  fragment=ast.Module(body=capture.body[begin:end+1],type_ignores=[])
  source=ast.unparse(fragment)+'\n'
  (output.parent/(output.stem+'-actual-lifecycle.py')).write_text(source)
  tm=importlib.import_module('tinygrad.tensor')
  def metadata_only_execute(linear,*args,**kwargs):
    events.append('mock_producer_complete')
    for call in linear.src:
      for u in call.src[1:]:
        assert not u.is_variable
        u.buffer.ensure_allocated()
  @contextmanager
  def observed(*args):
    events.append('producer_enter')
    try: yield
    finally: events.append('producer_exit')
  def tensor(buffer,shape): return Tensor(UOp.from_buffer(buffer,dtypes.bfloat16)).reshape(shape)
  def allocation(size): return Buffer('AMD',size,options=BufferSpec(nolru=True))
  for device_only in (False,True):
    dev.allocator.device_only=device_only
    for mode in ('whole','offset_views','composed_views','shared_adjacent_views'):
      buffers=[];expected=[]
      if mode=='shared_adjacent_views':
        base=allocation(sum(sizes)+1024);offset=512
        for size in sizes: buffers.append(base.view(size,offset));expected.append(offset);offset+=size
      else:
        for index,size in enumerate(sizes):
          base=allocation(size if mode=='whole' else size+8192)
          if mode=='whole': buffer=base;offset=0
          elif mode=='offset_views': offset=(index+1)*512;buffer=base.view(size,offset)
          else:
            offset=512+1024;buffer=base.view(size+2048,512).view(size,1024)
          buffers.append(buffer);expected.append(offset)
      inputs=[tensor(b,s) for b,s in zip(buffers,shapes,strict=True)]
      cache=tensor(allocation(9437184),(2,1,8,2304,128))
      before=[helper.storage_info(t,False) for t in inputs]
      assert all(not r['is_allocated'] and 'gpu_address' not in r and 'offset_bytes' in r and 'offset' not in r for r in before)
      # The allocator emulates completion of GPU allocation, without executing any program.
      for buffer in [*buffers,cache.uop.buffer]: buffer.ensure_allocated()
      events.clear();report={}
      namespace=dict(q=inputs[0],k=inputs[1],v=inputs[2],cache=cache,device=dev,report=report,helper=helper,C=O,
        producer=SimpleNamespace(observe=observed),producer_catalog={},folder=output.parent)
      with patch.object(tm,'run_linear',metadata_only_execute):
        exec(compile(fragment,str(O.ORIGINAL_RUNNER),'exec'),namespace)
      rows=report['packed_inputs'];assert [r['offset_bytes'] for r in rows]==expected
      assert all(r['is_allocated'] and 'offset' not in r for r in rows)
      for t,row in zip(namespace['packed'],rows,strict=True):
        b=t.uop.buffer
        assert row['buffer_id']==id(b) and row['base_buffer_id']==id(b.base)
        assert row['bytes']==b.nbytes and row['base_bytes']==b.base.nbytes and row['offset_bytes']==b.offset
        assert row['gpu_address']==int(b.get_storage().buf)==int(b.base.get_storage().buf)+b.offset
        assert (b.get_storage().host is None)==device_only
        if not device_only: assert row['gpu_address']!=b.get_storage().host.addr
      assert events[0]=='producer_enter' and events[-2:]==['producer_exit','synchronize']
      assert O.storage_gate([report['actual_cache_storage']],[9437184])
      proof['positive_lifecycles'].append({'mode':mode,'device_only_capability':device_only,'events':events[:],
        'metadata_before_allocation':before,'metadata_after_producer':rows,'actual_cache_storage':report['actual_cache_storage']})
  good=copy.deepcopy(recorded['packed_inputs'])
  def reject(name,mutate):
    rows=copy.deepcopy(good);mutate(rows)
    try: O.storage_gate(rows,sizes)
    except (AssertionError,KeyError,TypeError): proof['negative_controls'].append(name)
    else: raise AssertionError(('Invalid storage accepted',name))
  reject('missing_offset_bytes',lambda r:r[0].pop('offset_bytes'))
  def legacy_only(rows): rows[0]['offset']=rows[0].pop('offset_bytes')
  reject('legacy_offset_is_not_an_alias_or_default',legacy_only)
  reject('negative_offset',lambda r:r[0].update(offset_bytes=-16,gpu_address=r[0]['base_gpu_address']-16))
  reject('offset_exceeds_extent',lambda r:r[0].update(offset_bytes=16,gpu_address=r[0]['base_gpu_address']+16))
  reject('wrong_pointer_sum',lambda r:r[0].update(gpu_address=r[0]['gpu_address']+16))
  reject('wrong_base_pointer',lambda r:r[0].update(base_gpu_address=r[0]['base_gpu_address']+16))
  reject('wrong_byte_count',lambda r:r[0].update(bytes=r[0]['bytes']-16))
  reject('truncated_base_extent',lambda r:r[0].update(base_bytes=r[0]['base_bytes']-16))
  reject('overlapping_physical_ranges',lambda r:r[1].update(gpu_address=r[0]['gpu_address']+16,base_gpu_address=r[0]['gpu_address']+16))
  reject('unaligned_pointer',lambda r:r[0].update(gpu_address=r[0]['gpu_address']+1,base_gpu_address=r[0]['base_gpu_address']+1))
  for field in ('gpu_address','base_gpu_address','base_bytes','bytes'):
    reject('missing_'+field,lambda r,field=field:r[0].pop(field))
  # Runpy loads all original function bodies. Only the common binding is overlaid.
  namespace=O.load_runner();globals_=namespace['run'].__globals__
  assert globals_['C'].storage_gate is O.storage_gate and globals_['C'].ROOT==O.ROOT
  assert namespace['run'].__code__.co_filename==str(O.ORIGINAL_RUNNER)
  assert namespace['main'].__code__.co_filename==str(O.ORIGINAL_RUNNER)
  proof['frozen_runner_runpy_and_overlay_binding']=True
  assert not Device._opened_devices
  proof['source_sha256']={str(p):O.sha(p) for p in [Path(__file__),EXP/'agent-ck-storage-overlay.py',
    O.ORIGINAL_RUNNER,EXP/'agent-ck-output-common.py',Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad/tinygrad/device.py')]}
  with output.open('x') as stream: stream.write(json.dumps(proof,indent=2,allow_nan=False)+'\n')
  print(json.dumps({'gpu_executed':False,'positive_lifecycles':len(proof['positive_lifecycles']),
    'negative_controls':len(proof['negative_controls']),'output':str(output)}))

if __name__=='__main__':
  parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args();main(args.output)

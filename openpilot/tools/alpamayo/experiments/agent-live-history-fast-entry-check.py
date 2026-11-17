"""Actual canonical Consumer entrypoint, with GPU computation blocked at the established CPU boundary."""
import argparse
import copy
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import runpy
import socket
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('history_entry_run',HERE/'agent-live-history-fast-run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  assert not args.output.exists();C=R.context();C.configure()
  assert os.environ.get('DEV')=='CPU' and os.environ.get('ALLOW_DEVICE_USAGE')=='0'
  from tinygrad import Device
  B=R.load(R.BASE/'agent-live-cpu-integrated-check.py','history_entry_existing_boundary');B.C=C
  root=R.ARTIFACTS/args.output.stem;assert not root.exists();root.mkdir()
  output=root/'consumer/report.json';boundary=B.Boundary('consumer');previous=importlib.util.spec_from_file_location
  modules={name:sys.modules.get(name) for name in ('model','benchmark','inputs','native_warp','stream_inputs')}
  targets=[]
  def observing(frame,event,arg):
    if event=='call' and frame.f_code.co_name in ('run_pipeline','launch') and 'path' in frame.f_locals:targets.append(str(frame.f_locals['path']))
  error=None
  try:
    with ExitStack() as stack:
      stack.enter_context(patch.dict(os.environ));stack.enter_context(patch.object(sys,'path',sys.path[:]))
      stack.enter_context(patch.object(sys,'argv',[str(HERE/'agent-live-history-fast-run.py'),'--cpu-boundary','--output',str(output)]))
      stack.enter_context(patch.object(type(Device),'__getitem__',B.blocked));stack.enter_context(patch.object(socket.socket,'connect',B.blocked))
      stack.enter_context(patch.object(importlib.util,'spec_from_file_location',boundary.observe(previous,stack)))
      old=sys.getprofile();sys.setprofile(observing)
      try:runpy.run_path(sys.argv[0],run_name='__main__')
      except B.ComputeBoundary as exc:error=exc
      finally:sys.setprofile(old)
  finally:
    for name,value in modules.items():
      if value is None:sys.modules.pop(name,None)
      else:sys.modules[name]=value
  assert error is not None and not Device._opened_devices and importlib.util.spec_from_file_location is previous
  assert targets==[str(C.N.LIVE/'experiments/benchmark-pipeline.py')]
  status=R.read(output.parent/'agent-live-history-fast-status.json');original=R.read(output.parent/'agent-live-cpu-integrated-status.json')
  assert not status['passed'] and not status['gpu_executed'] and original['hooks_restored'] and original['constructor_restored']
  info=status['pack_hooks'];assert len(info)==1 and info[0]['cleanup_exact'] and info[0]['consumer_bindings']==1
  assert len(info[0]['calls'])==1 and info[0]['calls'][0]['completed'] and info[0]['calls'][0]['inside_consumer_predict']
  assert info[0]['calls'][0]['packed_images']>0 and info[0]['final']['counts'].get('fallback_images',0)==0
  assert boundary.owner_errors==['duplicate owner rejected','active query rejected']
  assert original['cpu_hook']['cleanup_exact'] and original['cpu_hook']['shadow_after_cleanup']==0
  events=original['storage_events']
  assert [e['event'] for e in events].count('normalize')==1
  assert [e['path'] for e in events if e['event']=='cache_open']==[str(C.N.WEIGHTS/'index.json')]
  assert any(e['event']=='factory_enter' and e['cachedb_constant']==C.N.storage()['CACHEDB'] for e in events)
  assert boundary.consumer._closed and not boundary.consumer._lock.locked()
  # The existing canonical output guard remains active before any runtime construction.
  occupied=root/'occupied/report.json';occupied.parent.mkdir();occupied.with_suffix('.side.json').write_text('{}\n')
  try:R.execute(occupied,'CPU-BOUNDARY-ONLY',cpu_only=True)
  except AssertionError as exc:assert str(exc)=='Preserve canonical no-overwrite guard'
  else:raise AssertionError('Occupied output accepted')
  reference=R.read(R.REFERENCE);comparison=R.compare_selected(C,reference,R.REFERENCE)
  assert comparison['exact'] and comparison['queries']==2 and comparison['reference_sha256']==R.REFERENCE_SHA
  rejections=[]
  for kind in ('token','input','history','speculation','kv','owner','unknown','snapshot_path'):
    changed=copy.deepcopy(reference)
    if kind=='token':changed['rows'][0]['prediction']['generated_token_ids'][0]+=1
    elif kind=='input':changed['rows'][0]['prepared']['unexpected']=1
    elif kind=='history':changed['rows'][0]['history_before']['history'].append([1])
    elif kind=='speculation':changed['rows'][0]['speculation']['accepted']=False
    elif kind=='kv':changed['rows'][0]['defined_kv_prefix_bits'][0]['sha256']='0'*64
    elif kind=='owner':changed['rows'][0]['consumer_protocol']['input_reused']=False
    elif kind=='unknown':changed['rows'][0]['unexpected']=1
    else:changed['consumer_provenance']['initializer']['factory']['source_snapshot']=str(root/'wrong-snapshot')
    try:R.compare_selected(C,changed,R.REFERENCE)
    except (ValueError,AssertionError):rejections.append(kind)
    else:raise AssertionError('Changed selected reference accepted: '+kind)
  result=dict(complete=True,cpu_only=True,gpu_executed=False,actual_runpy_targets=targets,events=boundary.events,
    storage_events=events,pack_hook=info[0],position_digest_hook=original['cpu_hook'],owner_rejections=boundary.owner_errors,
    occupied_output_rejected=True,source_clock_started=False,selected_comparison=comparison,comparison_rejections=rejections,
    mocked='Only the existing GPU factory completion/allocation/startup doubles; actual factory index read, canonical main, normalization, pad hook, Consumer constructor/predict, adapter.prepare and packing execute.',
    source_sha256={str(p):R.sha(p) for p in (Path(__file__),HERE/'agent-live-history-fast-run.py',HERE/'agent-live-history-fast-hook.py',HERE/'agent-live-history-fast-pack.c')})
  R.write(args.output,result);print(dict(complete=True,gpu_executed=False,packed_images=info[0]['calls'][0]['packed_images']),flush=True)

if __name__=='__main__':main()

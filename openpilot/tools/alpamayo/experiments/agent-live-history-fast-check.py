"""Exact CPU adapter integration, chronological timings, and mutation controls."""
import argparse
from contextlib import ExitStack
import copy
import ctypes
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('history_fast_common',HERE/'agent-live-history-fast-profile.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)

def index_checks(handle):
  import numpy as np
  import torch
  by,bx,my,mx,y,x,c=np.indices((10,18,2,2,16,16,3),sparse=True)
  indices=((c*320+by*32+my*16+y)*576+bx*32+mx*16+x).reshape(1,720,768)
  results=[]
  for byte in range(3):
    image=((np.arange(3*320*576,dtype=np.uint32)>>(8*byte))&255).astype(np.uint8).reshape(1,3,320,576)
    actual,h,w=handle.image.patchify(torch.from_numpy(image),16,2,1)
    P.exact(actual.numpy(),image.reshape(-1)[indices]);assert (h,w)==(20,36)
    results.append('exhaustive source-index byte '+str(byte))
  for label,image in [('zeros',np.zeros((1,3,320,576),np.uint8)),('all255',np.full((1,3,320,576),255,np.uint8)),
      ('random',np.random.default_rng(401).integers(0,256,(1,3,320,576),dtype=np.uint8))]:
    protected=np.full(image.size+64,117,np.uint8)
    handle.library.pack_320_576(protected.ctypes.data+32,image.ctypes.data)
    assert np.all(protected[:32]==117) and np.all(protected[-32:]==117)
    P.exact(protected[32:-32].reshape(1,720,768),image.reshape(-1)[indices]);results.append(label+' independent map/redzones')
  return results

def live_controls(run,report):
  import numpy as np
  import torch
  paced=P.load(P.EXP/'agent-paced-replay.py','history_fast_paced')
  trace=paced.Trace(paced.SOURCE/'manifest.json',paced.LOGS)
  root=P.WORK/'warp-fast-pathfix-source';module=P.load(root/'agent-warp-fast-sampler.py','history_fast_sampler')
  module.ARTIFACTS=P.ARTIFACTS/'sampler'
  cfg=json.loads((root/'agent-warp-fast-validated-portable-v6.json').read_text())['variants'][0]
  sampler=module.Sampler(cfg['kind'],cfg['workers'],Path(cfg['provenance']['command'][0]).name,cfg['provenance']['native'],cfg['max_maps'],cfg['max_bytes'])
  assert sampler.lib.provenance['library_sha256']==cfg['provenance']['library_sha256']
  rgbroot=P.WORK/'agent-paced-replay-cache-35-43/rgb';rgb=paced.RGB(rgbroot,trace.frames,verify=False)
  inventory=json.loads((rgbroot.parent/'paced-cache.json').read_text());assert inventory['frame_metadata_sha256']==paced.json_hash(trace.frames)
  class Clock:
    now=trace.start;anchor=None
    def __call__(self):return self.now
  clock=Clock();verified=set()
  def arrived(ref):
    assert ref['release_ns']<=clock();path=rgb.path(ref);key=str(path.relative_to(rgbroot))
    if key not in verified:assert P.sha(path)==inventory['files'][key];verified.add(key)
    return rgb(ref)
  arrived.report=rgb.report
  store=paced.bridge.Store(frame_limit=64,byte_limit=384*1024**2);producer=paced.Producer(trace,store,arrived,clock)
  prep=paced.bridge.Preparer(trace.manifest['source_camera_geometry'],trace.manifest['target_camera_geometry'],warp=sampler.calibrated_image)
  rows=[];last=0;current={}
  def consume(selection,label):
    arrays,detail=prep.prepare(selection);arrays['nav_text']=[trace.nav(selection.query_ns)]
    data={k:v if k=='nav_text' else torch.from_numpy(v) for k,v in arrays.items()}
    a=run(0,data)[0];b=run(1,data)[0];P.exact(a,b)
    rows.append(dict(case=label,query_ns=selection.query_ns,all_arrays_exact=True,actual_timestamps=arrays['absolute_timestamps'].tolist(),
      image_cache_hits=detail['image_cache_hits'],history_ms=detail['history_build_ms'],warp_stack_ms=detail['color_warp_stack_ms']))
    return arrays
  try:
    producer.prime_history()
    for i in range(3):
      clock.now=trace.start+i*400_000_000;producer.pump()
      selection=store.select(clock(),last,current,250_000_000,False)
      assert all(f.ready_ns<=clock() for f in selection.frames)
      consume(selection,'actual original chronological live selection')
      last=selection.query_ns;current={c:max(f.capture_ns for f in selection.frames if f.camera==c) for c in (1,6)}
    original=consume(selection,'original before state mutations')
    def mutate(kind,change):
      state=dict(selection.states);values=list(state[kind])
      for i,entry in enumerate(values):
        if entry.valid and entry.stamp<=selection.query_ns:values[i]=replace(entry,value=change(copy.deepcopy(entry.value)))
      state[kind]=tuple(values);return replace(selection,states=state)
    def field(key,delta):
      def change(value):value[key]+=delta;return value
      return change
    def motion(value):value[0][2]+=.01;value[1][0]+=.25;return value
    def calibration(value):value['rpyCalib'][2]+=.002;return value
    for label,kind,change in [('motion and speed','deviceMotion',motion),('learned steering ratio','vehicleParameters',field('steerRatio',.5)),
        ('car parameters','carParams',field('wheelbase',.1)),('calibration','extrinsicsCalibration',calibration),
        ('car state steering','carState',field('steeringAngleDeg',.5))]:
      arrays=consume(mutate(kind,change),label)
      assert any(arrays[k].tobytes()!=original[k].tobytes() for k in ('ego_history_xyz','ego_history_rot','image_frames'))
    errors=[]
    for kind in ('carState','vehicleParameters','carParams','deviceMotion','extrinsicsCalibration'):
      state=dict(selection.states);state[kind]=tuple([*state[kind],paced.bridge.State(selection.query_ns,selection.query_ns,None,False)])
      try:consume(replace(selection,states=state),'invalid '+kind)
      except paced.bridge.Unavailable as error:errors.append(dict(case='current '+kind+' tombstone',error=str(error)))
      else:raise AssertionError('Tombstone bypassed')
    state=dict(selection.states);state['carState']=()
    try:consume(replace(selection,states=state),'missing carState')
    except paced.bridge.Unavailable as error:errors.append(dict(case='missing carState',error=str(error)))
    else:raise AssertionError('Missing state accepted')
    restored=consume(selection,'restored original states');P.exact(original,restored)
  finally:sampler.close();producer.close()
  report['live_preparer']=dict(rows=rows,rejections=errors,verified_arrived_files=len(verified),future_pixels_read=False,
    selection_and_preparer_unchanged=True,sampler_library_sha256=cfg['provenance']['library_sha256'])

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
  assert args.output.resolve().is_relative_to(P.WORK) and not args.output.exists();P.configure()
  import numpy as np
  import torch
  torch.set_num_threads(4)
  with ExitStack() as stack:
    P.cpu_guards(stack)
    A=P.load(P.EXP/'fast-inputs.py','history_fast_adapter');K=P.load(P.EXP/'compact-inputs.py','history_fast_compact')
    policy=P.load(P.EXP/'agent-speculation-policy.py','history_fast_policy');H=P.load(HERE/'agent-live-history-fast-hook.py','history_fast_hook')
    options=dict(camera_profile='nvidia',allow_missing_cameras=True,direct_byte_inputs=True,cache_resize=True)
    adapters=[A.FastInputs(P.CHECKPOINT,**options) for _ in (0,1)];handle=H.install(adapters[1]);stack.callback(handle.close)
    settings=SimpleNamespace(max_tokens=64,vision_batch=1,byte_pixels=True,cache_rotary_inputs=False)
    def run(arm,data):
      start=time.perf_counter_ns();_,_,prepared=adapters[arm].prepare(data);p=time.perf_counter_ns()
      packed=adapters[arm].native_inputs(prepared);n=time.perf_counter_ns()
      prepared,packed=policy.padded_inputs(prepared,packed,2048,policy.CHECKPOINT_DRAFT_IDS,adapters[arm].config.traj_ids['future_start'])
      arrays=K.CompactInputs.arrays(settings,prepared,packed);end=time.perf_counter_ns()
      return (prepared,packed,arrays),dict(prepare_ms=(p-start)/1e6,native_ms=(n-p)/1e6,pad_arrays_ms=(end-n)/1e6,total_ms=(end-start)/1e6)
    rows=[];schedule_path=P.WORK/'stream-10hz/floor/manifest.json';schedule=json.loads(schedule_path.read_text())['schedule'][:96:4]
    first=None
    for i,row in enumerate(schedule):
      data=P.sample(row)
      if first is None:first=data
      outputs={};timings={}
      for arm in ([0,1] if i%2==0 else [1,0]):outputs[arm],timings[str(arm)]=run(arm,data)
      P.exact(outputs[0],outputs[1])
      rows.append(dict(index=i,query_ns=row['query_ns'],sample=row['sample'],measured=i>=4,exact=True,timing=timings))
    report=dict(complete=False,cpu_only=True,gpu_executed=False,torch_threads=4,rows=rows,controls=[],
      manifest_sha256=P.sha(schedule_path),hook=handle.state(),
      boundary='Original adapter.prepare + native_inputs + pad2048 + CompactInputs.arrays; input reads/comparisons excluded equally. No GPU allocation, upload, or inference.',
      timing={str(a):{key:P.stats([r['timing'][str(a)][key] for r in rows if r['measured']]) for key in rows[0]['timing']['0']} for a in (0,1)})
    cold=[]
    for i in range(4):
      for adapter in adapters:
        adapter.resize_cache.clear();adapter.position_cache.clear()
      outputs={};timings={}
      for arm in ([0,1] if i%2==0 else [1,0]):outputs[arm],timings[str(arm)]=run(arm,first)
      P.exact(outputs[0],outputs[1]);cold.append(dict(timing=timings,exact=True))
    report['empty_cache_rows']=cold
    report['cold_boundary']='Actual base FastInputs resize/position caches empty; global libraries initialized. First process call is retained in rows[0], not used for a cold speed claim.'
    def control(label,data,reject=False):
      outputs=[];errors=[]
      for arm in (0,1):
        try:outputs.append(run(arm,data)[0])
        except Exception as error:errors.append((type(error).__name__,str(error)))
      if reject:assert len(errors)==2 and errors[0]==errors[1],(label,errors)
      else:assert not errors,(label,errors);P.exact(*outputs)
      report['controls'].append(dict(case=label,exact=True,rejected=reject,errors=errors))
    control('restored first query',first)
    control('navigation mutation',dict(first,nav_text=['Take the second exit, then turn right.']))
    control('system prompt mutation',dict(first,system_prompt='Drive carefully.'))
    for key,index,delta in [('ego_history_xyz',(0,4,1),.4),('ego_history_rot',(0,5,0,1),.001),('absolute_timestamps',(0,),17000)]:
      value=first[key].clone();value[index]+=delta;control(key+' mutation',dict(first,**{key:value}))
    for image in (0,4,5,9):
      value=first['image_frames'].clone();version=value._version;value.numpy()[image,0,1,23,17]^=127
      assert value._version==version;control('image '+str(image)+' NumPy version-bypass mutation',dict(first,image_frames=value))
    control('restored source bytes',first)
    for label,data in [('empty navigation',dict(first,nav_text=[''])),('image tombstone',dict(first,image_frames=None)),
        ('nonfinite history',dict(first,ego_history_xyz=torch.full_like(first['ego_history_xyz'],float('nan')))),
        ('missing front camera',dict(first,camera_indices=torch.zeros_like(first['camera_indices']))),
        ('wrong timestamps dtype',dict(first,absolute_timestamps=first['absolute_timestamps'].float()))]:control(label,data,reject=True)
    report['index_checks']=index_checks(handle)
    # Fallback preserves the original processor for every nonselected dtype/shape/layout.
    for image,layout in [(torch.zeros(1,3,64,64,dtype=torch.uint8),(16,2,1)),(torch.zeros(1,3,320,576),(16,2,1)),
        (torch.zeros(1,3,320,576,dtype=torch.uint8),(16,2,2)),(torch.zeros(1,3,576,320,dtype=torch.uint8).transpose(2,3),(16,2,1))]:
      P.exact(handle.original(image,*layout),handle.image.patchify(image,*layout))
    report['fallback_cases']=4
    try:H.install(adapters[1])
    except ValueError:report['duplicate_install_rejected']=True
    else:raise AssertionError('Duplicate hook accepted')
    bad=P.ARTIFACTS/'rejected-library.bin';bad.write_bytes(b'not the qualified packing binary')
    try:H.install(adapters[0],binary=bad)
    except AssertionError:report['changed_library_rejected']=True
    else:raise AssertionError('Changed library accepted')
    live_controls(run,report)
    report['hook']=handle.state();original=handle.original;handle.close()
    assert adapters[1].processor.image_processor.patchify==original and 'patchify' not in vars(handle.image)
    control('original method restored',first)
    paths=[Path(__file__),HERE/'agent-live-history-fast-hook.py',HERE/'agent-live-history-fast-pack.c',H.BINARY,
      P.EXP/'agent-cpu-inputs-adapter.py',P.EXP/'fast-inputs.py',P.EXP/'agent-live-bridge.py',P.EXP/'replay-route.py',P.EXP/'calibration_trials.py']
    report.update(complete=True,hooks_restored=True,source_sha256={str(p):P.sha(p) for p in paths},no_inference_or_live_hz_claim=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,timing=report['timing'],controls=len(report['controls']),live_cases=len(report['live_preparer']['rows']))),flush=True)

if __name__=='__main__':main()

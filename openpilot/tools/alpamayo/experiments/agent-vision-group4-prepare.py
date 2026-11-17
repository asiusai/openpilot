"""CPU-only byte-copy freeze and actual four-query native-input qualification."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('agent_vision_group4_prepare_common',HERE/'agent-vision-group4-common.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)


def main():
  c.require(os.environ.get('DEV')=='CPU','Set DEV=CPU for preparation')
  c.require(not c.PLAN.exists(),'Do not mutate an existing frozen protocol')
  template=c.WORK/'dynamic-vision-protocol/dynamic-vision-native-5hz.json'
  recipe=c.read(template);command=recipe['command'][:]
  first,last=command.index('--samples')+1,command.index('--initial-noise')
  samples=command[first:last][:4];c.require(len(samples)==4,'Require four chronological native inputs')
  command[first:last]=samples
  c.ROOT.mkdir(exist_ok=True);c.SNAP.mkdir(exist_ok=True);(c.SNAP/'experiments').mkdir(exist_ok=True)
  origins={}
  for path in sorted(c.BASE.rglob('*')):
    if path.is_file() and path.suffix in ('.py','.c','.json') and '__pycache__' not in path.parts:
      relative=path.relative_to(c.BASE);out=c.SNAP/relative;out.parent.mkdir(parents=True,exist_ok=True)
      shutil.copyfile(path,out);c.require(c.sha(path)==c.sha(out),'Source copy changed bytes')
      origins[str(relative)]=dict(path=str(path),sha256=c.sha(path))
  for path in sorted(HERE.glob('agent-vision-group4-*.py')):
    out=c.EXP/path.name;shutil.copyfile(path,out);origins[str(out.relative_to(c.SNAP))]=dict(path=str(path),sha256=c.sha(path))
  sys.path[:0]=[str(c.TINY),str(c.SNAP),str(c.EXP)]
  import torch
  from tinygrad import Device
  c.require(Device.DEFAULT=='CPU','Preparation must use CPU')
  pipeline=c.module(c.EXP/'benchmark-pipeline.py','agent_vision_group4_frozen_pipeline')
  initializer=pipeline.initialize
  cpu=c.module(c.EXP/'agent-live-replay.py','agent_vision_group4_cpu_fingerprints')
  # Exact copies use their own relative imports. No path substitution inside sources.
  c.set_option(command,'checkpoint','/tmp/opencode/alpamayo-asius-20261114/model')
  c.set_option(command,'pad-prompt-to',2048);c.set_option(command,'vision-cache-capacity',48)
  c.set_option(command,'warmup',0);c.set_option(command,'runs',4)
  c.set_option(command,'reduced-vocab',c.EXP/'coc_vocab.json')
  command[1]=str(c.EXP/'benchmark-pipeline.py')
  # Compiler-mode controls are recorded, never normalized to a different mode.
  env=copy.deepcopy(recipe['environment']);env['PYTHONPATH']=f'{c.TINY}:{c.SNAP}'
  env['PYTHONDONTWRITEBYTECODE']='1'
  env['CACHEDB']=str(c.ROOT/'agent-vision-group4-runtime-cache.db')
  invocations={};effective={}
  for limit in (2,4):
    arm=f'group{limit}';current=command[:];c.set_option(current,'vision-group-limit',limit)
    c.set_option(current,'output',c.WORK/('agent-vision-group4-'+arm+'-parity4')/'report.json')
    invocation=dict(command=current,environment=env,source_template=str(template),source_template_sha256=c.sha(template),
      outer_speculation_args=dict(mode='candidate',prefill_attention='shuffle-fma',pad_prompt_to=2048,warmup=0,runs=4))
    path=c.ROOT/('agent-vision-group4-'+arm+'-invocation.json')
    path.write_text(json.dumps(invocation,indent=2)+'\n');invocations[arm]=path
    args=pipeline.parse_args(current[2:]);effective[arm]=vars(args).copy()
    for key in ('pad_prompt_to','steps','max_tokens','vision_batch','vision_cache_capacity','vision_group_limit'):
      c.require(getattr(args,key)=={'pad_prompt_to':2048,'steps':3,'max_tokens':64,'vision_batch':1,'vision_cache_capacity':48,'vision_group_limit':limit}[key],f'Effective {key} differs')
    c.require(args.vision_calibration is None and args.front_history_indices is None and args.history_pixel_scale==args.history_feature_scale==args.pixel_scale==1,'Vision input/arithmetic changed')
    c.require(not args.image_gemm and not args.warm_current_shapes and not args.warmup_graphs,'Unexpected alternative hook/startup')
  args=pipeline.parse_args(c.read(invocations['group2'])['command'][2:])
  adapter=c.module(c.EXP/'fast-inputs.py','agent_vision_group4_actual_adapter').FastInputs(args.checkpoint,
    pixel_scale=args.pixel_scale,cache_resize=args.cache_resize,narrow_crop=args.narrow_crop,
    wide_equidistant_focal=args.wide_equidistant_focal,wide_target_fov=args.wide_target_fov,
    camera_profile=args.camera_profile,camera_pixel_scales=args.camera_pixel_scales,
    allow_missing_cameras=args.allow_missing_cameras,history_pixel_scale=args.history_pixel_scale,
    history_feature_scale=args.history_feature_scale,direct_byte_inputs=args.direct_byte_inputs)
  native=c.module(c.EXP/'stream-inputs.py','agent_vision_group4_native').NativeFrames(args.native_manifest,args.native_calibrations,args.native_frame_cache)
  compact=c.module(c.EXP/'compact-inputs.py','agent_vision_group4_compact').CompactInputs
  policy=c.module(c.EXP/'agent-speculation-policy.py','agent_vision_group4_spec_policy')
  draft=policy.validate_draft(adapter.tokenizer.encode(policy.DRAFT_TEXT,add_special_tokens=False),adapter.config.traj_ids['future_start'])
  c.require(draft==policy.CHECKPOINT_DRAFT_IDS,'Checkpoint six-target proposal differs')
  settings=SimpleNamespace(max_tokens=64,vision_batch=1,byte_pixels=True)
  rows=[];input_files={str(template):c.sha(template)};previous=0
  manifest=c.read(args.native_manifest);schedule={Path(row['sample']).name:row for row in manifest['schedule']}
  for sample in samples:
    metadata=schedule[Path(sample).name]
    c.require(metadata['query_ns']>previous,'Queries must strictly advance');previous=metadata['query_ns']
    c.require(len(metadata['frames'])==10 and all(frame['capture_ns']<=metadata['query_ns'] for frame in metadata['frames']),'Future/missing input')
    c.require(sorted(frame['view'] for frame in metadata['frames'])==[1]*5+[6]*5,'Five images/camera required')
    data={p.stem:[str(np.load(p,allow_pickle=False).item())] if p.stem=='nav_text' else torch.from_numpy(np.load(p,allow_pickle=False,mmap_mode='c'))
      for p in Path(sample).glob('*.npy') if p.stem!='image_frames'}
    images,timing=native.prepare(sample);data['image_frames']=torch.from_numpy(images)
    config,tokenizer,prepared=adapter.prepare(data);packed=adapter.native_inputs(prepared)
    prepared,packed=policy.padded_inputs(prepared,packed,2048,draft,adapter.config.traj_ids['future_start'])
    arrays=compact.arrays(settings,prepared,packed)
    c.require(prepared['tokenized_data']['image_grid_thw'].tolist()==[[1,20,36]]*10,'Full720patch grids differ')
    c.require(prepared['actual_prompt_length']+6<=2048 and packed[0].shape==(1,2048),'Actual six-target prompt does not fit')
    width=3*adapter.config.vlm_config.vision_config.patch_size**2
    c.require(width==768 and arrays[0].dtype==np.uint8 and arrays[0].shape==(7200,width),f'Direct full byte patches differ: {arrays[0].shape}/{arrays[0].dtype}')
    rows.append(dict(sample=sample,query_ns=metadata['query_ns'],frames=metadata['frames'],
      prepared=cpu.tree(prepared),packed=cpu.tree(packed),arrays=cpu.tree(arrays),actual_prompt=prepared['actual_prompt_length'],
      native_timing_cpu_only=timing))
    for path in Path(sample).glob('*.npy'):
      if path.stem!='image_frames':input_files[str(path)]=c.sha(path)
    for frame in metadata['frames']:
      raw=args.native_frame_cache/str(frame['view'])/(str(frame['video_index'])+'.npy');input_files[str(raw)]=c.sha(raw)
  c.require(pipeline.initialize is initializer,'Protected initializer was replaced')
  c.require(all(name.split(':')[0] in ('CPU','NPY','PYTHON') for name in Device._opened_devices),'A non-CPU device opened')
  for path in (args.native_manifest,args.native_calibrations,args.prefill_calibration,args.initial_noise,Path(args.checkpoint)/'config.json',Path(args.checkpoint)/'model.safetensors.index.json'):
    input_files[str(path)]=c.sha(path)
  for path in invocations.values():input_files[str(path)]=c.sha(path)
  source_sha={str(path.relative_to(c.SNAP)):c.sha(path) for path in sorted(c.SNAP.rglob('*')) if path.is_file()}
  tiny_files=['tinygrad/engine/realize.py','tinygrad/engine/worker.py','tinygrad/codegen/__init__.py','tinygrad/renderer/llvmir.py','tinygrad/runtime/support/compiler_llvm.py']
  for relative in tiny_files:input_files[str(c.TINY/relative)]=c.sha(c.TINY/relative)
  plan=dict(schema=1,created_cpu_only=True,gpu_executed=False,kind='four-query exact image/network parity; diagnostic timings only',
    sole_candidate_change='vision_group_limit2->4; same dynamic per-image activation scales and original tiled4x4 kernels',
    template_pad2112_explicitly_replaced_in_both_arms_with2048=True,outer_mode='candidate',outer_attention='shuffle-fma',
    source_sha256=source_sha,source_origins=origins,input_files_sha256=input_files,
    tinygrad_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.TINY,text=True).strip(),
    effective_pipeline_args=cpu.tree(effective),cpu_queries=rows,opened_devices=sorted(Device._opened_devices),
    memory_limit_bytes=7*1024**3,memory_limit_excludes='allocator reservations and driver overhead; parent actual peak remains required',
    compiler_modes='Preserve --llvm-wgp and inherited worker scheduling; do not fix known renderer reconstruction CU behavior',
    forbidden_hooks=['vision-tiles','fixed-vision-calibration','nominal-only substitution','history selection changes'],
    parent_gate='Both children must complete exactly4, exact control/per-image/embedding/KV/prediction/history/spec/cache parity; stop on any bit difference',
    no_automatic_timing_or_route=True)
  commands=[]
  for arm in ('group2','group4'):
    commands.append([str(c.PY),'/tmp/opencode/run-alpamayo-speed-trial.py','agent-vision-group4-'+arm+'-parity4',
      '--program',str(c.EXP/'agent-vision-group4-run.py'),'--timeout-seconds','600','--set','arm',arm,'--flag','execute-gpu',
      '--env','PYTHONDONTWRITEBYTECODE','1','--env','PYTHONPATH',f'{c.TINY}:{c.SNAP}'])
  commands_path=c.ROOT/'agent-vision-group4-parent-commands.json'
  c.write(commands_path,commands)
  input_files[str(commands_path)]=c.sha(commands_path)
  c.write(c.PLAN,plan)
  print(json.dumps(dict(plan=str(c.PLAN),plan_sha256=c.sha(c.PLAN),cpu_queries=len(rows),source_files=len(source_sha),gpu_executed=False),indent=2))


if __name__=='__main__':main()

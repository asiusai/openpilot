"""Direct selected-pipeline composition. No hardware work occurs during import."""
from functools import lru_cache
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

EXP=Path(__file__).parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-ck-selected-work';PREPARED=ROOT/'prepared'
BASE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
PY='/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
  with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
  Path(p).parent.mkdir(parents=True,exist_ok=True)
  Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def module(path,name):
  s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

T=module(EXP/'agent-ck-quality64-timing-common.py','ck_selected_lean')
Q=T.Q;TINY=Q.TINY
QUALITY=WORK/'agent-ck-quality64-parent-candidate-v1/report.json'
QUALITY_SHA='170afc214e2719340b8e97b7ff27b0ceb13ad97794a86790a7cfe0b1a689e1d6'
PACED=WORK/'agent-arrival-warp-ownership-paced1/report.json'
PACED_SHA='bbdfdb419f6a3b13dea4a347351233a3e7c0af61b94013850c8f4d8b0099cab2'
TIMING=WORK/'agent-ck-quality64-timing-parent-candidate-v1/report.json'
TIMING_SHA='c6e8ec1134e417548163ce1f62430a2cf470d82f2b7e2886fe3dadd5a1b75988'
REVIEW=WORK/'agent-ck-quality64-review-v1.json'
REVIEW_SHA='ec6d7861ec7290d327a4f7bc45f6dc5a0d1e9659f700cbd12e6c06d7bb09b7c3'

@lru_cache(None)
def selected():
  A=module(EXP/'agent-arrival-warp-ownership-common.py','ck_selected_arrival');C=A.C;N=C.N
  # New caches and artifacts only. Keep all existing source/path checks and original installers.
  A.ARTIFACTS=A.OLD.ARTIFACTS=C.ARTIFACTS=C.S.ARTIFACTS=N.ARTIFACTS=ROOT/'artifacts'
  return A,C,N

def configure():
  # Context defaults must exist before selected.paths imports tinygrad/compiler helpers.
  cpu_only=os.environ.get('ALLOW_DEVICE_USAGE')=='0' and os.environ.get('DEV')=='CPU'
  for key,value in read(PREPARED/'candidate-invocation.json')['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=str(value)
  if cpu_only:os.environ['DEV']='CPU'
  A,C,N=selected();C.configure();return A,C,N

def hook_source(live=False):
  text=T.hook_source()
  # The passed graph visitor checks empty nodes and every real container. Avoid walking arbitrary CALL.arg values here.
  if live:text=Q.replace_once(text,'counts=_outside_flags(self) if before in (0,1,63) else None','counts=_outside_flags(self) if before in (0,1) else None')
  return text

def no_poison(captured):
  from tinygrad.uop.ops import Ops
  from tinygrad.runtime.support.hcq2 import HCQInfo
  for linear in [captured._linear,*[x[0] for x in captured._linears.values()]]:
    for u in linear.toposort():
      if u.op is Ops.PROGRAM:assert not u.src[0].arg.name.startswith('ckq_poison_')
      aux=getattr(u.arg,'aux',None)
      if u.op is Ops.CALL and isinstance(aux,HCQInfo):assert not any(x[1].startswith('ckq_poison_') for x in aux.kernels)

@lru_cache(None)
def hook(live=False):
  m=ModuleType('ck_selected_live_hook' if live else 'ck_selected_quality_hook');m.__file__=str(PREPARED/('live-hook.py' if live else 'quality-hook.py'))
  m._graph=Q.OLD.load('agent-ck-graph-inspection');m._outside_flags=T.outside_flags;m._no_captured_poison=no_poison
  exec(compile(hook_source(live),m.__file__,'exec'),m.__dict__);return m

def load(name,directory=EXP):
  if name=='agent-ck-integrated-hook' and Path(directory)==EXP:return hook()
  return Q.load(name,directory)

def runner_source():
  text=T.runner_source()
  text=Q.replace_once(text,"assert Path(origins['text_layer'])==C.BASE/'experiments/quantize-packed.py'",
    "assert Path(origins['text_layer'])==C.FUSED_SOURCE, 'Selected language fusion missing before CK'")
  text=Q.replace_once(text,"        pipeline=json.loads((directory/'pipeline.json').read_text())",
    "        C.selected_coverage(report)\n        pipeline=json.loads((directory/'pipeline.json').read_text())")
  text=Q.replace_once(text,"scope=H.native_scope(C,report) if arm=='candidate' else nullcontext()","scope=nullcontext()")
  text=Q.replace_once(text,"instance=H.Integration(C,model,self,arm,report);instances.append(instance)",
    "if arm=='candidate': C.native_stack.enter_context(H.native_scope(C,report))\n            instance=H.Integration(C,model,self,arm,report);instances.append(instance)")
  return text

def runner(common):
  m=ModuleType('ck_selected_quality_runner');m.__file__=str(EXP/'agent-ck-integrated-run.py')
  exec(compile(runner_source(),m.__file__,'exec'),m.__dict__);m.C=common
  original=m.run
  def run(*args,**kwargs):
    with ExitStack() as stack:
      common.native_stack=stack
      try:return original(*args,**kwargs)
      finally:del common.native_stack
  m.run=run
  return m

C=SimpleNamespace(**{k:v for k,v in vars(T.C).items() if not k.startswith('__')})
C.BASE=BASE;C.ROOT=ROOT;C.PREPARED=PREPARED;C.load=load;C.runner=runner
C.FUSED_SOURCE=WORK/'agent-fused-mlp-2048-protocol-v2/alpamayo/experiments/agent-fused-mlp.py'

def exact(quality,result):
  T.exact_quality_parity(quality,result)
  old,new=[read(r['pipeline']) for r in (quality,result)]
  equal=module(EXP/'agent-exact-combined-common.py','ck_selected_equal').equal
  for i,(a,b) in enumerate(zip(old['runs'],new['runs'],strict=True)):
    # All prediction fields except measured stage durations. Unknown added fields also fail.
    equal({k:v for k,v in a['prediction'].items() if k!='timings_ms'},
          {k:v for k,v in b['prediction'].items() if k!='timings_ms'},f'prediction.{i}')
    for k in ('sample','input_latest_timestamp_us','vision_hits','vision_misses','uploaded_pixel_bytes','draft'):
      assert (k in a)==(k in b),k
      if k in a:equal(a[k],b[k],f'query.{i}.{k}')
  for k in module(EXP/'agent-exact-combined-common.py','ck_selected_settings').PAIRED_SETTINGS:
    equal(old[k],new[k],k)
  equal(read(Path(quality['pipeline']).with_suffix('.speculation.json')),
        read(Path(result['pipeline']).with_suffix('.speculation.json')),'all_six_target_verifications_and_fallback')
  return dict(exact=True,queries=64,all_prediction_bits_tokens_histories=True,reference_sha256=QUALITY_SHA,
    inherited_quality_limits='Original full64 aggregate and four 16-query windows <=5% ADE/FDE; manual review retained')

def verify():
  T.verify()
  p=read(PREPARED/'protocol.json')
  for path,digest in p['files'].items():assert sha(path)==digest,('Changed selected source/evidence',path)
  proof=read(p['cpu_proof']['path']);assert sha(p['cpu_proof']['path'])==p['cpu_proof']['sha256']
  assert proof['complete'] and not proof['gpu_executed']
  for row in proof['parts'].values():assert sha(row['path'])==row['sha256']
  assert sha(QUALITY)==QUALITY_SHA and sha(PACED)==PACED_SHA and sha(TIMING)==TIMING_SHA
  assert sha(REVIEW)==REVIEW_SHA==read(TIMING)['timing_review_sha256']
  review=read(REVIEW);assert review['reviewed_query_indices']==list(range(64)) and review['text_speed_slowing_turns_reviewed']
  assert sha(review['comparison'])==review['comparison_sha256']==read(TIMING)['quality_comparison_sha256']
  assert read(review['comparison'])['paired_quality64_passed']
  Q.validate_arm(read(QUALITY));T.validate_arm(read(TIMING));assert read(TIMING)['timing_identity_passed']
  assert sha(read(QUALITY)['pipeline'])==read(QUALITY)['pipeline_sha256']
  assert read(PACED)['agent_arrival_warp_integrated']['passed']
  for row in read(PREPARED/'source-equality.json'):
    assert sha(row['old'])==sha(row['current'])==row['sha256']
  for r in read(Q.PREPARED/'inputs.json'):
    for path,digest in r['files'].items():assert sha(path)==digest,path
  assert (PREPARED/'quality-hook.py').read_text()==hook_source()
  assert (PREPARED/'live-hook.py').read_text()==hook_source(True)
  assert (PREPARED/'quality-runner.py').read_text()==runner_source()
  return p

def quality_gate(path):
  r=read(path);assert r['completed'] and r['selected_exact']['exact'] and r['hooks_restored']
  T.validate_arm(r)
  assert r['source_after_exact']
  assert r['selected_constructor_restored'] and r['selected_pv_cleanup']['cleanup_exact'] and r['selected_pv_cleanup']['observer_cleanup_exact']
  assert r['selected_protocol_sha256']==sha(PREPARED/'protocol.json')
  assert sha(r['pipeline'])==r['pipeline_sha256'];exact(read(QUALITY),r)
  return r

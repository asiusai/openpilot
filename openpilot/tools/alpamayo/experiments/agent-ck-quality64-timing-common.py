"""Separate lean mode. Full64 review and exact per-arm prediction parity are mandatory."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from types import ModuleType,SimpleNamespace

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_quality64_common',EXP/'agent-ck-quality64-common.py')
Q=importlib.util.module_from_spec(s);s.loader.exec_module(Q)
ROOT=Q.WORK/'agent-ck-quality64-timing-work';PREPARED=ROOT/'prepared'
QUALITY_PROTOCOL_SHA='03f804fe52e2b9c0d317f1a2798785c72dc12bf3d24e9dd204e43c1073a086ae'
AUDIT_INDICES=(0,1,63)

def outside_flags(owner):
  from tinygrad import Tensor,dtypes
  from tinygrad.engine import realize
  assert not realize.capturing,'Diagnostic audit entered the timed capture'
  H=owner.common.load('agent-ck-integrated-hook');flags=[]
  for i in range(28):
    row=owner.layers[i]
    if owner.arm=='candidate':
      flags += [H.audit(row['stores'][0],row['out']),H.audit(row['stores'][1],row['lse'])]
      flags += [Tensor.stack(*[(p.bitcast(dtypes.uint16)!=owner.model.cache[i][j,:,:,:2048].bitcast(dtypes.uint16)).cast(dtypes.int32).sum()
        for j,p in enumerate(row['packed'][1:])]).realize()]
    else: flags += [(~row['out'].isfinite()).cast(dtypes.int32).sum().reshape(1).realize()]
  counts=owner.download(Tensor.cat(*[f.flatten() for f in flags]).contiguous().realize()).reshape(28,-1)
  assert not counts.any(),('Outside-capture all28 audit',counts.tolist())
  return counts

def no_captured_poison(captured):
  from tinygrad.uop.ops import Ops
  from tinygrad.runtime.support.hcq2 import HCQInfo
  for linear in [captured._linear,*[x[0] for x in captured._linears.values()]]:
    for u in linear.toposort():
      if u.op is Ops.PROGRAM: assert not u.src[0].arg.name.startswith('ckq_poison_')
      if u.op is Ops.CALL and isinstance(u.arg.aux,HCQInfo): assert not any(x[1].startswith('ckq_poison_') for x in u.arg.aux.kernels)

def hook_source():
  text=Q.hook_source()
  old="""      for store in stores: poison(self.common,store)
"""
  text=Q.replace_once(text,old,'')
  marker="""          self.io.allocate('AMD',(LC+2*G,),dtypes.float32))}
"""
  text=Q.replace_once(text,marker,marker+"        for store in self.layers[index]['stores']: poison(self.common,store)\n")
  old="""      prefix=Tensor.stack(*[(p.bitcast(dtypes.uint16)!=cache[j,:,:,:2048].bitcast(dtypes.uint16)).cast(dtypes.int32).sum()
        for j,p in enumerate(packed[1:])]).realize()
      row.update(out=out,lse=lse,packed=packed,flags=(audit(stores[0],out),audit(stores[1],lse),prefix))
"""
  text=Q.replace_once(text,old,"      row.update(out=out,lse=lse,packed=packed,flags=())\n")
  text=Q.replace_once(text,"'flags':((~out.isfinite()).cast(dtypes.int32).sum().reshape(1).realize(),)","'flags':()")
  old="""    self.device.synchronize()
    assert self.device.renderer is self.renderer and self.renderer.compiler is self.compiler
    assert set(self.layers)==set(range(28))
    flags=Tensor.cat(*[f.flatten() for i in range(28) for f in self.layers[i]['flags']]).contiguous().realize()
    counts=self.download(flags).reshape(28,-1)
    assert not counts.any(),('Per-layer finite/poison/redzone/binding audit',phase,counts.tolist())
"""
  new="""    assert self.device.renderer is self.renderer and self.renderer.compiler is self.compiler
    assert set(self.layers)==set(range(28))
    counts=_outside_flags(self) if before in (0,1,63) else None
"""
  text=Q.replace_once(text,old,new)
  text=Q.replace_once(text,'native=_graph.inspect_captured(captured,self.common,self.report,self.arm,phase)',
    'native=_graph.inspect_captured(captured,self.common,self.report,self.arm,phase)\n      _no_captured_poison(captured)')
  text=Q.replace_once(text,"'all28_audit_counts':counts.tolist()","'all28_audit_counts':counts.tolist() if counts is not None else None")
  return text

@lru_cache(None)
def load(name,directory=EXP):
  if name!='agent-ck-integrated-hook' or Path(directory)!=EXP: return Q.load(name,directory)
  m=ModuleType('ck_lean_hook');m.__file__=str(PREPARED/'effective-hook.py')
  m._graph=Q.OLD.load('agent-ck-graph-inspection');m._outside_flags=outside_flags;m._no_captured_poison=no_captured_poison
  exec(compile(hook_source(),m.__file__,'exec'),m.__dict__);return m

def runner_source():
  text=Q.runner_source()
  old="""               before=copy.deepcopy(obj.history);value=original_run(*args,**kwargs)
""".lstrip(' ')
  # Preserve the original indentation while replacing exactly this statement.
  old="              before=copy.deepcopy(obj.history);value=original_run(*args,**kwargs)\n"
  new="""              from tinygrad.engine.realize import runtime_cache
              from tinygrad.codegen import to_program_cache
              before=copy.deepcopy(obj.history);nr,nc=len(runtime_cache),len(to_program_cache)
              value=original_run(*args,**kwargs)
              instance.device.synchronize()
              index=len(report.get('decoder_histories',[]));dr,dc=len(runtime_cache)-nr,len(to_program_cache)-nc
              report.setdefault('timing_query_state',[]).append({'index':index,'new_runtime_entries':dr,'new_compile_entries':dc,
                'diagnostic_audited':index in (0,1,63),'warm_eligible':index not in (0,1,63) and dr==0 and dc==0})
"""
  text=Q.replace_once(text,old,new)
  text=Q.replace_once(text,"pipeline['summary']['profiling_enabled']=True","pipeline['summary']['profiling_enabled']=False")
  text=Q.replace_once(text,'All28 audits/poison remain included; deep mutation controls are reused by exact passed report hash.',
    'Lean capture has no diagnostic poison/audit operations. Queries0,1,63 are audited outside capture and excluded from warm metrics. Strict same-arm full64 parity is still required.')
  return text

def runner(common):
  m=ModuleType('ck_lean_runner');m.__file__=str(EXP/'agent-ck-integrated-run.py')
  exec(compile(runner_source(),m.__file__,'exec'),m.__dict__);m.C=common;return m

def validate_arm(report):
  assert report['completed'] and report['gpu_executed'] and report['indices']==Q.INDICES
  assert report['hooks_restored'] and report['model_renderer_unchanged'] and report['initial_decoder_history']==[]
  assert len(report['prefills'])==len(report['decoder_histories'])==len(report['timing_query_state'])==64
  for i,(p,h,t) in enumerate(zip(report['prefills'],report['decoder_histories'],report['timing_query_state'],strict=True)):
    assert p['jit_count_before']==i and p['jit_count_after']==i+1 and t['index']==i
    assert h['before']==([] if i==0 else report['decoder_histories'][i-1]['after'])
    if i in AUDIT_INDICES:
      assert len(p['all28_audit_counts'])==28 and not any(v for row in p['all28_audit_counts'] for v in row)
      assert not t['warm_eligible']
    else: assert p['all28_audit_counts'] is None and not t['diagnostic_audited']
    assert p['captured_native_calls']==(28 if report['arm']=='candidate' and i else 0)
  assert len(report['capture_inspections'])==63
  return True

def verify():
  assert Q.sha(Q.PREPARED/'protocol.json')==QUALITY_PROTOCOL_SHA;Q.verify()
  p=json.loads((PREPARED/'protocol.json').read_text());assert p['quality_protocol_sha256']==QUALITY_PROTOCOL_SHA
  for path,digest in p['pins'].items(): assert Q.sha(path)==digest,('Changed timing source',path)
  assert (PREPARED/'effective-hook.py').read_text()==hook_source() and (PREPARED/'effective-runner.py').read_text()==runner_source()
  return p

def review_gate(path):
  verify();r=json.loads(Path(path).read_text());comparison=Path(r['comparison'])
  assert Q.sha(comparison)==r['comparison_sha256']
  result=json.loads(comparison.read_text());assert result['paired_quality64_passed']
  assert r['reviewed_query_indices']==Q.INDICES and r['text_speed_slowing_turns_reviewed'] is True
  assert r['decision']=='timing_only_no_adoption'
  reports={}
  for arm in ('baseline','candidate'):
    p=Path(r[arm+'_report']);assert Q.sha(p)==result[arm+'_report_sha256']
    reports[arm]=json.loads(p.read_text());Q.validate_arm(reports[arm])
    assert reports[arm]['source_protocol_sha256']==QUALITY_PROTOCOL_SHA
    assert Q.sha(reports[arm]['pipeline'])==reports[arm]['pipeline_sha256']
  return r,reports

def exact_quality_parity(quality,lean):
  import numpy as np
  assert lean['decoder_histories']==quality['decoder_histories']
  a,b=[json.loads(Path(r['pipeline']).read_text()) for r in (quality,lean)]
  assert len(a['runs'])==len(b['runs'])==64
  for i,(x,y) in enumerate(zip(a['runs'],b['runs'],strict=True)):
    assert x['sample']==y['sample']
    for key in ('generated_token_ids','reasoning'): assert x['prediction'][key]==y['prediction'][key],(i,key)
    for key in ('xyz','rot'):
      left,right=[np.asarray(v['prediction'][key],np.float64) for v in (x,y)]
      assert np.isfinite(left).all() and np.isfinite(right).all()
      np.testing.assert_array_equal(left.view(np.uint64),right.view(np.uint64),err_msg=f'Query{i} {key} timing/quality parity')
  return True

C=SimpleNamespace(**{k:v for k,v in vars(Q).items() if not k.startswith('__')})
C.ROOT=ROOT;C.PREPARED=PREPARED;C.load=load;C.verify=verify;C.runner=runner;C.validate_arm=validate_arm

"""Offline recovery of one immutable reporting failure. Never changes its status."""
import copy
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_selected_recovery_original',EXP/'agent-ck-selected-common.py')
S=importlib.util.module_from_spec(s);s.loader.exec_module(S)
ROOT=S.WORK/'agent-ck-selected-recovery-work'
FAILED=S.WORK/'agent-ck-selected-parent-quality-v1'
FAILED_SHA='61ec8552c58ea78ee55c2c5dd40182686e0c9cd5befc0754c3ea29ce96ae0f35'
ORIGINAL_PROTOCOL='724123dc5a7fb80be6c41fc2265a488bf0b2c15e96fae98a726b28e77b4cff4a'
PINS={
  'report.json':FAILED_SHA,
  'pipeline.json':'69c1232fc5a698d0fafda8fc12fc05e4d70f6aa37e851d1e3837df04ad2c82b9',
  'pipeline.speculation.json':'cc670a597f25571b3507a55355fc9b2a64855765471bdf2755126fe485bd2ed1',
  'pipeline.jsonl':'efd4fdcc98351b089db15bf4d0db21540b7d3aa8af02f0d91df5cd554432b014',
  'invocation.json':'fff1025bf4fa6474ba9af3ff0e8f59533144557845a54b3d9dccfb6122efd80f',
  'wall.json':'88de0a783ff5fab9067239d760013f12cbc3b66d05033fd72ebe1c1a5ab2c43b',
  'run.log':'d7fd91d238d2a2c9be21c76d714bb815fb6145b41a2bfdb8faf85674feedf43d',
}
ORIGINAL_STORAGE=S.selected()[2].storage().copy()

def configure():
  A,C,N=S.selected()
  A.ARTIFACTS=A.OLD.ARTIFACTS=C.ARTIFACTS=C.S.ARTIFACTS=N.ARTIFACTS=ROOT/'artifacts'
  return S.configure()

def active_pv(composition,N):
  """Runtime ownership checks. The unchanged final validator runs after exit."""
  info=composition.pv_handle.info;_,evidence=N.pv()
  assert info['candidate'] is True and info['evidence']==evidence
  assert info['active'] and not info['cleanup_exact'] and not info['observer_cleanup_exact']
  assert composition.session.frame is None and not composition.session.state['prefill']

def quality_module():
  text=(EXP/'agent-ck-selected-quality.py').read_text()
  text=S.Q.replace_once(text,'_,evidence=N.pv();N.validate_pv(composition.pv_handle.info,True,64,evidence)',
    '_recovery.active_pv(composition,N)')
  text=S.Q.replace_once(text,'    execution_ok=True',
    "    _,evidence=N.pv();N.validate_pv(composition.pv_handle.info,True,64,evidence)\n    execution_ok=True")
  m=ModuleType('ck_selected_recovery_quality');m.__file__=str(EXP/'agent-ck-selected-quality.py')
  exec(compile(text,m.__file__,'exec'),m.__dict__);m.S=S
  from types import SimpleNamespace
  m._recovery=SimpleNamespace(active_pv=active_pv)
  return m,text

def recorded_arm(r):
  # Keep every original data assertion. The known failed lifecycle status must remain false.
  # This is an evidence validator, not a successful-run status or a rewritten report.
  source=inspect.getsource(S.T.validate_arm)
  source=S.Q.replace_once(source,"assert report['completed'] and report['gpu_executed']",
    "assert report['completed'] is False and report['gpu_executed']")
  ns=dict(vars(S.T));exec(compile(source,'<saved-arm-evidence-validator>','exec'),ns)
  ns['validate_arm'](r)

def ck_evidence(r):
  sizes=[8388608,131072,8388608,4194304,4194304]
  assert r['checked_native_program_keys']==['b301f96dee9c13b54828f97bc5aab99b42c51be3ec30c869e121e5b3fb030cd7']
  assert len(r['raw_native_pointers'])==len(r['live_layouts'])==56
  for rows in r['raw_native_pointers']:
    S.C.storage_gate(rows,sizes);assert [x['offset_bytes'] for x in rows[:2]]==[512,1024]
  for i,row in enumerate(r['live_layouts']):
    assert row['layer']==i%28 and row['q_shape']==[1,16,2048,128]
    assert row['k_shape']==row['v_shape']==[1,8,2048,128] and row['cache_shape']==[2,1,8,2304,128]
    assert row['cache_store_dependencies'];S.C.storage_gate(row['actual_arguments'],sizes)
    addresses=[x['gpu_address'] for x in row['actual_arguments']]
    assert addresses in [[x['gpu_address'] for x in rows] for rows in r['raw_native_pointers']]
  assert r['native_kernarg_builds']==29 and len(r['native_argument_templates'])==29
  for rows in r['native_argument_templates'].values():
    assert [x['bytes'] for x in rows]==sizes and [x['offset_bytes'] for x in rows[:2]]==[512,1024]
  # One reusable eager launch template, then the complete 28-call captured graph.
  assert [len(b) for b in r['native_dependency_batches']]==[1,28]
  for batch in r['native_dependency_batches']:
    for row in batch:
      assert row['device']=='AMD' and row['queue']=='COMPUTE:0' and row['outs']==[0,1] and row['ins']==[2,3,4]
      assert [x['bytes'] for x in row['arguments']]==sizes
  for i,row in enumerate(r['capture_inspections'],1):
    assert row['phase']==f'screen_{i}_original64_{i}'
    assert [g['graph'] for g in row['graphs']]==['compiled','linked_0']
    for g in row['graphs']:
      assert g['native']==28 and g['hcq_containers']==1 and g['hcq_kernels']==591 and g['direct_programs']==0
  assert r['reused_model_controls']['source_report_sha256']==S.C.CONTROL_SHA
  return dict(prefills=64,graphs_inspected=63,physical_layouts=56,native_calls_per_graph=28,
    program_key=r['checked_native_program_keys'][0],identity=S.read(S.PREPARED/'ck-identity.json'),
    gates='Original runtime rejected unknown containers and changed source/AST/ELF/target/CU/ABI; saved records rechecked offline')

def coverage_evidence(r):
  c=r['selected_coverage'];st=c['language_stats']
  assert set(st['layer_hits'])=={f'vlm.model.language_model.layers.{i}' for i in range(28)}
  assert set(st['layer_hits'].values())=={2}
  assert st['python_trace_fc1']==st['python_trace_fc2']==st['python_trace_fused_quantize']==st['layer_completions']==56
  assert st['failed_text_layers']==0 and st['shape_hits']=={'2048':56}
  assert any(x['fused_fc1']==28 for x in c['language_graphs'])
  assert len(c['vision_graphs'])==2 and all(x['fc1']==x['fc2']==25 for x in c['vision_graphs'])
  for b in (1,2):
    for i in range(2,27):
      for fc in (1,2):assert c['vision_hook']['trace_calls'][f'vlm.model.visual.encoder.layers.{i}.mlp.fc{fc}/batch{b}']>0
  assert c['sampler_calls']==0 and c['sampler'] is None
  assert r['selected_constructor_restored'] and r['hooks_restored'] and r['model_renderer_unchanged']
  assert len(r['selected_restorations'])==14 and all(x['restored'] for x in r['selected_restorations'])
  return dict(all28_mlp=True,vision_batch1_batch2=True,original_coverage_returned=True,
    quantizer_idle='Proved by the assertion immediately before the failing PV call',
    session_cleanup='Proved by the final cleanup conjunction that made hooks_restored true',
    coverage_methods_restored_field_is_preexit_snapshot=st['methods_restored'] is False)

def recover():
  assert S.sha(S.PREPARED/'protocol.json')==ORIGINAL_PROTOCOL
  plan=S.verify()
  for name,digest in PINS.items():assert S.sha(FAILED/name)==digest,name
  r=S.read(FAILED/'report.json');p=S.read(FAILED/'pipeline.json');wall=S.read(FAILED/'wall.json')
  assert r['completed'] is False and wall['returncode']==1 and not wall['timed_out']
  assert r['failure']=={'type':'AssertionError','message':'','phase':'screen_63_original64_63'}
  assert not {'selected_exact','source_after_exact','selected_storage','pipeline','pipeline_sha256'} & r.keys()
  assert r['source_protocol_sha256']==r['selected_protocol_sha256']==ORIGINAL_PROTOCOL
  log=(FAILED/'run.log').read_text()
  assert log.count('Traceback (most recent call last):')==1
  assert "assert info['cleanup_exact'] and info['observer_cleanup_exact'] and not info['active']" in log
  assert 'agent-ck-selected-quality.py", line 24, in coverage' in log
  recorded_arm(r);ck=ck_evidence(r);coverage=coverage_evidence(r)
  _,_,N=S.selected();_,e=N.pv();N.validate_pv(r['selected_pv_cleanup'],True,64,e)
  # Supply only the independently pinned pipeline path. Never set completed or remove failure.
  parity=S.exact(S.read(S.QUALITY),dict(r,pipeline=str(FAILED/'pipeline.json')))
  assert not p['warmup'] and [x['sample'] for x in p['runs']]==S.read(S.PREPARED/'samples.json')
  snapshot=Path(p['source_snapshot']);assert snapshot==FAILED/'pipeline-sources'
  for name,digest in p['source_sha256'].items():
    assert Path(name).name==name and S.sha(snapshot/name)==digest,name
    current=S.BASE/'model.py' if name=='model.py' else S.BASE/'experiments'/name
    assert S.sha(current)==digest,('Snapshot differs from current pinned foundation',name)
  assert S.verify()['files']==plan['files']
  return dict(schema=1,recovery_complete=True,gpu_executed_by_recovery=False,original_completed=False,
    original_failed_report=str(FAILED/'report.json'),original_failed_report_sha256=FAILED_SHA,
    original_protocol_sha256=ORIGINAL_PROTOCOL,artifacts={str(FAILED/n):h for n,h in PINS.items()},
    failure='Final expertPV cleanup validator ran inside active composition; original failure retained',
    exact=parity,ck=ck,selected_coverage=coverage,pv_original_final_validator_passed=True,
    source_verification=dict(current_original_verifiers_passed=True,recorded_pipeline_snapshot_files=len(p['source_sha256']),
      source_after_check_in_failed_process=False,independent_posthoc_verification=True),
    storage=dict(selected_storage_recorded=False,
      missing=['factory event list','individual cache-open event records and modes','actual temporary-directory basename'],
      derived_environment=ORIGINAL_STORAGE,
      derivation='Pinned factory wrapper asserts environment and imported CACHEDB before load, temporary parent during load, and environment after return. Complete predictions plus constructor restoration prove this path returned.',
      no_event_records_reconstructed=True,original_check_storage_not_run_without_events=True,
      paced_requires_fresh_original_storage_validator=True),
    runtime_arm_check='All original validate_arm data assertions retained; only completed=True changed to required immutable failure=False',
    preserved_manual_review_sha256=S.REVIEW_SHA,limitations=['Query24 missed turn/slowing','Query47 changed reasoning/slowing'],
    eligible_for_separate_parent_paced=True,adoption_allowed=False,under100ms_established=False)

def gate(failed,failed_sha,receipt,receipt_sha):
  assert Path(failed).resolve()==FAILED/'report.json' and failed_sha==FAILED_SHA==S.sha(failed)
  assert S.sha(receipt)==receipt_sha
  saved=S.read(receipt);assert saved==recover(),'Recovery receipt no longer matches independently recomputed evidence'
  return saved

def verify_protocol():
  p=S.read(ROOT/'protocol.json')
  for path,digest in p['files'].items():assert S.sha(path)==digest,path
  assert p['original_protocol_sha256']==ORIGINAL_PROTOCOL
  proof=S.read(p['cpu_check']);assert proof['complete'] and not proof['gpu_executed']
  return p

"""One direct source manifest for the chronological CK quality pair."""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType

EXP=Path(__file__).parent
def module(name):
  s=importlib.util.spec_from_file_location(name.replace('-','_'),EXP/(name+'.py'))
  m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
OLD=module('agent-ck-integrated-common')
WORK,BASE,TINY=OLD.WORK,OLD.BASE,OLD.TINY
ROOT=WORK/'agent-ck-quality64-work';PREPARED=ROOT/'prepared';INDICES=list(range(64))
sha,write,storage_gate=OLD.sha,OLD.write,OLD.storage_gate
LIVE,LIVE_SHA,MICRO,MICRO_SHA=OLD.LIVE,OLD.LIVE_SHA,OLD.MICRO,OLD.MICRO_SHA
CONTROL=WORK/'agent-ck-aligned-control-parent-v1/report.json'
CONTROL_SHA='2ff105518b5255640d808fe87fb61f0dff11e28c2f0307c0b0bfe3bdc5279652'
SCREEN=CONTROL.with_name('screen8-comparison.json')
SCREEN_SHA='3f4e5a557784d65dde3ea8403013cd49474da34023e28bec8733b0d8e6571696'
HOOK=WORK/'agent-ck-graph-inspection-work/prepared/effective-hook.py'
HOOK_SHA='dae178df643c305ad348e62e5d7d46080e3c618e9258beaf757c77f9029dd87c'
SOURCE64=WORK/'padding-source/padding2048-quality.json'
TRUTH=WORK/'agent-expert-pv-route-source/inputs/5100176f7e48eaee-manifest.json'

def replace_once(source,old,new):
  assert source.count(old)==1,(old,source.count(old));return source.replace(old,new)

def hook_source():
  assert sha(HOOK)==HOOK_SHA
  return replace_once(HOOK.read_text(),"if arm=='candidate': common.load('agent-ck-integrated-controls').run(owner,ret)",
    "if arm=='candidate': owner.report['reused_model_controls']={'source_report_sha256':common.CONTROL_SHA,'new_negative_controls_executed':False}")

@lru_cache(None)
def load(name,directory=EXP):
  if name!='agent-ck-integrated-hook' or Path(directory)!=EXP: return OLD.load(name,directory)
  m=ModuleType('ck_quality64_hook');m.__file__=str(HOOK);m._graph=OLD.load('agent-ck-graph-inspection')
  exec(compile(hook_source(),m.__file__,'exec'),m.__dict__);return m

def runner_source():
  source=(EXP/'agent-ck-integrated-run.py').read_text()
  edits=(("'--runs','8'","'--runs','64'"),('len(screen)==8 and owner.original_jit.cnt>=8','len(screen)==64 and owner.original_jit.cnt>=64'),
    ("len(report['decoder_histories'])==8","len(report['decoder_histories'])==64"),
    ("report['model_controls']['passed']","report['reused_model_controls']['source_report_sha256']==C.CONTROL_SHA"),
    ("phase='fresh_screen8'","phase='fresh_quality64'"),("phase='screen8_complete'","phase='quality64_complete'"),
    ('Fresh paired 5% aggregate/window comparison and manual text/speed/slowing/turn review; then parent decision.',
     'Fresh paired full64 aggregate/four-window gates and per-query text/speed/slowing/turn review; no adoption.'),
    ('Per-layer audits, pointer observations, and capture-time mutation controls are included.',
     'All28 audits/poison remain included; deep mutation controls are reused by exact passed report hash.'))
  for old,new in edits: source=replace_once(source,old,new)
  return source

def runner(common):
  m=ModuleType('ck_quality64_runner');m.__file__=str(EXP/'agent-ck-integrated-run.py')
  exec(compile(runner_source(),m.__file__,'exec'),m.__dict__);m.C=common;return m

def validate_arm(report):
  assert report['completed'] and report['gpu_executed'] and report['indices']==INDICES
  assert report['hooks_restored'] and report['model_renderer_unchanged'] and report['initial_decoder_history']==[]
  assert len(report['prefills'])==len(report['decoder_histories'])==64
  for i,(p,h) in enumerate(zip(report['prefills'],report['decoder_histories'],strict=True)):
    assert p['phase']==f'screen_{i}_original64_{i}' and p['jit_count_before']==i and p['jit_count_after']==i+1
    assert len(p['all28_audit_counts'])==28 and not any(v for row in p['all28_audit_counts'] for v in row)
    assert h['before']==([] if i==0 else report['decoder_histories'][i-1]['after'])
    assert p['captured_native_calls']==(28 if report['arm']=='candidate' and i else 0)
  assert len(report['capture_inspections'])==63
  if report['arm']=='candidate':
    assert report['reused_model_controls']=={'source_report_sha256':CONTROL_SHA,'new_negative_controls_executed':False}
  return True

def verify():
  p=json.loads((PREPARED/'protocol.json').read_text())
  assert p['indices']==INDICES and p['timeout_seconds']==600 and p['adoption_allowed'] is False
  assert subprocess.check_output(['git','-C',str(TINY),'rev-parse','HEAD'],text=True).strip()==p['tinygrad_commit']
  for path,digest in p['pins'].items(): assert sha(path)==digest,('Changed quality source',path)
  for row in json.loads((PREPARED/'inputs.json').read_text()):
    for path,digest in row['files'].items(): assert sha(path)==digest,('Changed quality input',path)
  assert sha(CONTROL)==CONTROL_SHA and sha(SCREEN)==SCREEN_SHA
  r=json.loads(CONTROL.read_text());assert r['completed'] and r['model_controls']['passed'] and r['hooks_restored']
  assert json.loads(SCREEN.read_text())['paired_screen8_passed']
  assert (PREPARED/'effective-hook.py').read_text()==hook_source()
  assert (PREPARED/'effective-runner.py').read_text()==runner_source()
  return p

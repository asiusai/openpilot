"""Path-only overlays around the complete qualified production wrapper stack."""
import ast
import copy
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
QUALIFIED=WORK/'agent-expert-live-compare-source-v2'
QUALIFIED_PROTOCOL=WORK/'agent-expert-live-compare-protocol-v2.json'
QUALIFIED_SHA='209aebd94565cb99ceff2a806f6fe60bb185087d2e18bbbca74014b260a579ee'
CANDIDATE=WORK/'agent-expert-live-compare-candidate/report.json'
CANDIDATE_SHA='751549150a637df24df3fed6c1782a33bf89b77871f358c2b87b9e06a3269367'
FAILED=WORK/'agent-expert-live-compare-paced'
ROOT=WORK/'agent-expert-paced-stem-source'
ARTIFACTS=WORK/'agent-expert-paced-stem-artifacts'
PROTOCOL=WORK/'agent-expert-paced-stem-protocol.json'
OUTPUT=WORK/'agent-expert-paced-stem-run/report.json'


def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


V=load(QUALIFIED/'agent-expert-live-compare-common.py','stem_qualified_validator')
sha,read,write=V.sha,V.read,V.write
assert sha(QUALIFIED_PROTOCOL)==QUALIFIED_SHA
assert sha(QUALIFIED/'agent-expert-live-compare-common.py')==read(QUALIFIED_PROTOCOL)['files'][str(QUALIFIED/'agent-expert-live-compare-common.py')]
# Separate private common module for new artifact paths; qualification retains its recorded paths.
S=load(V.FROZEN/'agent-expert-live-storage-common.py','stem_storage_runtime')
S.ARTIFACTS=ARTIFACTS;S.N.ARTIFACTS=ARTIFACTS
N=S.N


def configure():
  result=N.configure()
  target=ARTIFACTS/'hooks/agent-exact-combined-install.py'
  assert sha(target)==read(N.MANIFEST)['files'][str(N.EXP/target.name)]
  N.paths();return result


def candidate_admission(report=None):
  assert sha(CANDIDATE)==CANDIDATE_SHA
  data=V.verify(QUALIFIED_PROTOCOL,QUALIFIED_SHA)
  actual=read(CANDIDATE) if report is None else report
  paired=V.paired(actual,data,QUALIFIED_SHA)
  overlay=actual['agent_expert_live_compare'];status=actual['agent_expert_live_storage']
  assert overlay['passed'] and overlay['gpu_executed'] and status['passed'] and status['hooks_restored']
  assert overlay['protocol_sha256']==QUALIFIED_SHA and overlay['runtime_overlay']==data['runtime_overlay']
  assert overlay['overlay_helper_sha256']==data['overlay_helper_sha256'] and overlay['paired_exact']==paired
  if report is None:
    assert overlay==read(CANDIDATE.with_suffix('.agent-expert-live-compare.json'))
    assert status==read(CANDIDATE.with_suffix('.agent-expert-live-storage.json'))
  parent=read(QUALIFIED/'candidate-parent.json')
  assert parent['complete'] and parent['report_sha256']==CANDIDATE_SHA and parent['paired_exact']==paired
  return dict(candidate_sha256=CANDIDATE_SHA,qualified_protocol_sha256=QUALIFIED_SHA,
    recovered_current_admission_sha256=sha(V.ADMISSION),paired_exact=paired,
    numerical_sources_unchanged=True,no_parity_rerun=True)


def metadata(output):
  output=Path(output)
  return {k:output.parent/f'agent-expert-paced-stem-{k}.json' for k in ('storage','compare','overlay','admission','status')}


def validate_paths(output,paths):
  output=Path(output)
  assert output.name=='report.json' and output.is_absolute() and output==output.resolve() and output.is_relative_to(WORK)
  assert paths==metadata(output)
  for path in paths.values():
    assert path.parent==output.parent and path==path.resolve() and not path.name.startswith(output.stem)
  assert len(set(paths.values()))==len(paths)
  N.reserve(output.parent)


def rewrite(source,expected,replacements):
  assert hashlib.sha256(source.encode()).hexdigest()==expected,'Unknown source for artifact overlay'
  old=source
  for before,after in replacements:
    assert source.count(before)==1,'Ambiguous artifact path replacement'
    source=source.replace(before,after,1)
  restored=source
  for before,after in reversed(replacements):
    assert restored.count(after)==1
    restored=restored.replace(after,before,1)
  assert restored==old
  assert ast.dump(ast.parse(restored),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
  return source,dict(input_sha256=expected,output_sha256=hashlib.sha256(source.encode()).hexdigest(),
    replacements=[list(x) for x in replacements],source_reversible=True,ast_reversible=True)


def build(output,*,repair=True):
  """Retain all production bodies and checks; change reviewed artifact paths only."""
  output=Path(output);validate_paths(output,metadata(output))
  data=read(QUALIFIED_PROTOCOL)
  raw=load(V.FROZEN/'agent-expert-live-storage-run.py','stem_frozen_storage_runner')
  qualified_source=inspect.getsource(raw.run)
  for before,after in data['runtime_overlay']['literal_path_replacements']:
    assert qualified_source.count(before)==1;qualified_source=qualified_source.replace(before,after,1)
  assert hashlib.sha256(qualified_source.encode()).hexdigest()==data['runtime_overlay']['transformed_function_sha256']
  inner_paths=[
    ("S.WORK/f'agent-expert-live-compare-{args.arm}/report.json'",'Path('+repr(str(output))+')'),
    ("HERE/'agent-expert-live-compare-check.py'","HERE/'agent-expert-paced-stem-check.py'"),
    ("args.output.parent/'agent-expert-live-storage-admission.json'","args.output.parent/'agent-expert-paced-stem-admission.json'")]
  if repair:inner_paths.append(("args.output.with_suffix('.agent-expert-live-storage.json')","args.output.parent/'agent-expert-paced-stem-storage.json'"))
  inner_source,inner_proof=rewrite(qualified_source,data['runtime_overlay']['transformed_function_sha256'],inner_paths)
  runtime=raw.S;runtime.ARTIFACTS=ARTIFACTS;runtime.N.ARTIFACTS=ARTIFACTS
  runtime.verify=V.verify
  runtime.paired=lambda current,candidate,data,digest:V.paired(candidate,data,digest)
  # Paced only: no candidate qualification or model implementation is replaced.
  namespace=dict(raw.__dict__,S=runtime,N=runtime.N,HERE=Path(__file__).resolve().parent,FROZEN=V.FROZEN)
  exec(compile(inner_source,str(ROOT/'agent-expert-paced-stem-run.py')+':storage_paths','exec'),namespace)
  inner=namespace['run']
  original=load(QUALIFIED/'agent-expert-live-compare-run.py','stem_frozen_compare_runner')
  source=inspect.getsource(original.run)
  expected=hashlib.sha256(source.encode()).hexdigest()
  assert sha(QUALIFIED/'agent-expert-live-compare-run.py')==data['files'][str(QUALIFIED/'agent-expert-live-compare-run.py')]
  outer_paths=[
    ("V.WORK/f'agent-expert-live-compare-{args.arm}/report.json'",'Path('+repr(str(output))+')'),
    ("args.output.parent/'agent-expert-live-compare-overlay.json'","args.output.parent/'agent-expert-paced-stem-overlay.json'"),
    ("args.output.with_suffix('.agent-expert-live-compare.json')","args.output.parent/'agent-expert-paced-stem-compare.json'")]
  outer_source,outer_proof=rewrite(source,expected,outer_paths)
  inner_identity=dict(data['runtime_overlay'],artifact_path_overlay=inner_proof)
  def verified(path,digest):
    assert Path(path)==QUALIFIED_PROTOCOL and digest==QUALIFIED_SHA
    result=V.verify(path,digest)
    return dict(result,runtime_overlay=inner_identity)
  facade=SimpleNamespace(**vars(V))
  facade.configure=configure;facade.verify=verified
  facade.runtime_module=lambda:(inner,runtime,inner_identity)
  namespace=dict(original.__dict__,V=facade)
  exec(compile(outer_source,str(ROOT/'agent-expert-paced-stem-run.py')+':compare_paths','exec'),namespace)
  return namespace['run'],runtime,dict(inner=inner_proof,outer=outer_proof,
    qualified_runtime_overlay=data['runtime_overlay'],qualified_recipe_protocol_sha256=QUALIFIED_SHA,
    only_artifact_paths_changed=True,canonical_paced_source_sha256=sha(N.LIVE/'experiments/agent-paced-replay.py'))


def failed_evidence():
  assert not (FAILED/'report.json').exists()
  wall=read(FAILED/'wall.json');assert wall['returncode']==1 and not wall['timed_out']
  log=(FAILED/'run.log').read_text()
  assert log.rstrip().endswith("FileExistsError: Use a new output stem; retain all existing artifacts")
  assert 'agent-paced-replay.py", line 573' in log
  status=read(FAILED/'report.agent-expert-live-storage.json')
  assert not status['gpu_executed'] and not status['passed'] and status['hooks_restored']
  assert status['storage_events']==[] and not status['startup_spec']
  assert (FAILED/'report.agent-expert-live-compare.json').is_file()
  return dict(preserved_failure=True,model_predictions=0,gpu_executed=False,
    cause='Production storage sidecar occupied the canonical report.* stem before main')


def verify(path,digest):
  assert sha(path)==digest
  data=read(path)
  assert data['qualified_protocol_sha256']==QUALIFIED_SHA and data['candidate_sha256']==CANDIDATE_SHA
  for name,expected in data['files'].items():assert sha(name)==expected,name
  output=Path(data['output']);validate_paths(output,{k:Path(v) for k,v in data['metadata'].items()})
  if data['cpu_only']:
    assert os.environ.get('ALLOW_DEVICE_USAGE')=='0' and os.environ.get('DEV')=='CPU'
    assert output.is_relative_to(ARTIFACTS)
  else:assert output==OUTPUT and data['repair'] is True
  assert candidate_admission()==data['candidate_admission']
  assert failed_evidence()==data['failed_evidence']
  assert build(output,repair=data['repair'])[2]==data['artifact_overlay']
  assert data['gpu_arms']==['paced'] and data['automatic_paced'] is False
  return data

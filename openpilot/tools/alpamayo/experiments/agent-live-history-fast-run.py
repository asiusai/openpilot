"""Optional parent-only two-query Consumer check; no automatic continuous replay."""
import argparse
from contextlib import ExitStack
import copy
from functools import wraps
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType,SimpleNamespace

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=WORK/'agent-live-cpu-integrated-source'
BASE_PROTOCOL=WORK/'agent-live-cpu-integrated-protocol.json'
BASE_SHA='1dfe4f0dbe24147662ed7e009bfd1c062064d104e3a311db5145c625cd7dd96a'
REFERENCE=WORK/'agent-live-cpu-integrated-consumer/report.json'
REFERENCE_SHA='002ea0312a4613e018731e0408bae92a673aebac8782df4d146683754ccc8102'
CPU=WORK/'agent-live-history-fast-report-v3.json'
CPU_SHA='4f9368a4cfba60c963fd4567fbcacaf053e80bb87bb63d6c39a04b7b41ac4b22'
ARTIFACTS=WORK/'agent-live-history-fast-integration-artifacts'
FROZEN=WORK/'agent-live-history-fast-source-v2'
PROTOCOL=WORK/'agent-live-history-fast-protocol-v2.json'
OUTPUT=WORK/'agent-live-history-fast-consumer/report.json'

def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
  sys.modules[name]=module;spec.loader.exec_module(module);return module
def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,data):
  with Path(path).open('w') as stream:json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')

def context():
  C=load(BASE/'agent-live-cpu-integrated-common.py','history_fast_selected_common')
  # Set the new cache before verification imports tinygrad. Validate saved storage
  # records against their original directory, then bind this run's new directory.
  previous_artifacts=C.N.ARTIFACTS;C.N.ARTIFACTS=ARTIFACTS
  C.N.configure();C.N.ARTIFACTS=previous_artifacts
  C.verify(BASE_PROTOCOL,BASE_SHA)
  assert sha(REFERENCE)==REFERENCE_SHA and sha(CPU)==CPU_SHA
  r=read(REFERENCE);C.validate_status(r,r['agent_live_cpu_integrated'],'consumer',REFERENCE)
  proof=read(CPU);assert proof['complete'] and not proof['gpu_executed'] and proof['hooks_restored']
  C.ARTIFACTS=ARTIFACTS;C.S.ARTIFACTS=ARTIFACTS;C.N.ARTIFACTS=ARTIFACTS
  C.output=lambda arm:OUTPUT if arm=='consumer' else (_ for _ in ()).throw(AssertionError('No paced mode'))
  return C

def extend_factory(original,H,records):
  def factory(C):
    Parent=original(C)
    class Hooks(Parent):
      def runtime(self,module):
        super().runtime(module);binding=module.bind_components;owner=self
        @wraps(binding)
        def bind(*a,**kw):
          consumer=binding(*a,**kw);adapter=consumer.adapter
          assert not records and consumer._pending is None
          original_prepare=adapter.prepare
          # Canonical Consumer parity installs its exact ordered-input auditor here.
          # Retain that callable and restore the same instance slot after this run.
          assert vars(adapter).get('prepare') is original_prepare
          assert Path(original_prepare.__code__.co_filename)==C.N.LIVE/'experiments/agent-live-replay.py'
          assert original_prepare.__code__.co_qualname=='Audit.install.<locals>.prepare'
          handle=H.install(adapter);info=dict(consumer_bindings=1,calls=[],cleanup_exact=False,initial=handle.state(),
            original_prepare_audit=original_prepare.__code__.co_qualname)
          records.append(info)
          def measured(this,data):
            assert this is adapter and consumer._pending is not None
            before=handle.counts['packed_images'];start=time.perf_counter_ns()
            row=dict(query_ns=consumer._pending['query_ns'],inside_consumer_predict=True,completed=False)
            try:
              result=original_prepare(data);row['completed']=True;return result
            finally:
              row.update(prepare_ms=(time.perf_counter_ns()-start)/1e6,packed_images=handle.counts['packed_images']-before)
              info['calls'].append(row)
          installed=MethodType(measured,adapter);adapter.prepare=installed
          def close():
            assert adapter.prepare is installed
            adapter.prepare=original_prepare;info['final']=handle.state();handle.close()
            info['cleanup_exact']=adapter.prepare is original_prepare and vars(adapter)['prepare'] is original_prepare and handle.image.patchify==handle.original
            assert info['cleanup_exact'] and 'patchify' not in vars(handle.image)
          owner.stack.callback(close);return consumer
        self.patch(module,'bind_components',bind)
    return Hooks
  return factory

def runtime(records):
  R=load(BASE/'agent-live-cpu-integrated-run.py','history_fast_original_runner');C=context();R.C=C
  H=load(HERE/'agent-live-history-fast-hook.py','history_fast_runtime_hook')
  original_load=C.load
  def importing(path,name):
    module=original_load(path,name)
    if Path(path)==BASE/'agent-live-cpu-integrated-install.py':module.factory=extend_factory(module.factory,H,records)
    return module
  C.load=importing
  def admission(command,**kw):
    assert command[1:3]==[str(BASE/'agent-live-cpu-integrated-check.py'),'--admit']
    return subprocess.run([command[0],str(HERE/'agent-live-history-fast-run.py'),*command[2:]],**kw)
  R.subprocess=SimpleNamespace(run=admission)
  return R,C

def verify(digest):
  assert sha(PROTOCOL)==digest;data=read(PROTOCOL)
  assert data['base_protocol_sha256']==BASE_SHA and data['cpu_report_sha256']==CPU_SHA and data['reference_sha256']==REFERENCE_SHA
  for path,expected in data['files'].items():assert sha(path)==expected,path
  assert data['automatic_paced'] is False and data['timeout_seconds']==1200
  return data

def compare_selected(C,report,output):
  # The inherited comparator hardcodes its original reference directory. Bind this
  # additional comparison to the selected Consumer's actual snapshot directory.
  reference=read(REFERENCE);m,_=C.N.live()
  result=m.compare(dict(reference,mode='reference'),report)
  m.equal(C.V.project(reference,REFERENCE),C.V.project(report,output),'selected_consumer_owner_and_source')
  for i,(a,b) in enumerate(zip(reference['rows'],report['rows'],strict=True)):
    m.equal(a['consumer_protocol'],b['consumer_protocol'],f'selected_owner.{i}')
  return dict(**result,reference_sha256=REFERENCE_SHA,reference_reused=True,reference_gpu_rerun=False)

def execute(output,protocol_sha,cpu_only=False):
  if not cpu_only:verify(protocol_sha);assert output==OUTPUT
  else:assert output.is_relative_to(ARTIFACTS) and os.environ.get('DEV')=='CPU' and os.environ.get('ALLOW_DEVICE_USAGE')=='0'
  assert not any(output.parent.glob(output.stem+'*')),'Preserve canonical no-overwrite guard'
  records=[];R,C=runtime(records);side=output.parent/'agent-live-history-fast-status.json'
  assert not side.exists();C.configure();C.N.reserve(output.parent);output.parent.mkdir(parents=True,exist_ok=True)
  status=dict(passed=False,gpu_executed=False,overlay_protocol_sha256=protocol_sha,reference_sha256=REFERENCE_SHA,
    cpu_report_sha256=CPU_SHA,base_protocol_sha256=BASE_SHA,runtime_overrides=['owned output/storage paths','owned admission entrypoint','post-construction byte packer'],
    automatic_paced=False,pack_hooks=records)
  write(side,status)
  protocol=BASE_PROTOCOL;digest=BASE_SHA
  if cpu_only:
    protocol=output.parent/'agent-live-history-fast-cpu-base.json';write(protocol,dict(read(BASE_PROTOCOL),cpu_only=True));digest=sha(protocol)
  try:
    R.run(SimpleNamespace(arm='consumer',protocol=protocol,protocol_sha=digest,reviewed_parity_sha256=None,output=output))
    report=read(output);status['gpu_executed']=report['gpu_executed']
    C.validate_status(report,report['agent_live_cpu_integrated'],'consumer',output)
    assert len(records)==1 and records[0]['consumer_bindings']==1 and records[0]['cleanup_exact']
    calls=records[0]['calls'];assert [r['query_ns'] for r in calls]==[r['query_ns'] for r in report['rows']]
    assert len(calls)==2 and all(r['completed'] and r['inside_consumer_predict'] for r in calls)
    assert sum(r['packed_images'] for r in calls)>0 and records[0]['final']['counts'].get('fallback_images',0)==0
    status['comparison_to_selected_digest_consumer']=compare_selected(C,report,output)
    verify(protocol_sha);status['passed']=True;report['agent_live_history_fast']=status;write(output,report)
  except BaseException as error:
    status['failure']=dict(type=type(error).__name__,message=str(error));raise
  finally:write(side,status)

def prepare():
  assert not FROZEN.exists() and not PROTOCOL.exists()
  C=context();C.configure()
  revision=hashlib.sha256((sha(Path(__file__))+sha(HERE/'agent-live-history-fast-entry-check.py')).encode()).hexdigest()[:12]
  proof=WORK/f'agent-live-history-fast-entry-proof-{revision}.json';assert not proof.exists()
  env={**os.environ,**C.N.storage(),'DEV':'CPU','ALLOW_DEVICE_USAGE':'0','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(C.N.TINY)}
  with proof.with_suffix('.log').open('x') as log:
    subprocess.run([C.N.PY,str(HERE/'agent-live-history-fast-entry-check.py'),'--output',str(proof)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
  result=read(proof);assert result['complete'] and not result['gpu_executed']
  FROZEN.mkdir()
  files={str(p):sha(p) for p in (CPU,REFERENCE,BASE_PROTOCOL,proof,WORK/'agent-live-history-fast-artifacts/pack.so')}
  for name in ('agent-live-history-fast-run.py','agent-live-history-fast-hook.py','agent-live-history-fast-pack.c','agent-live-history-fast-entry-check.py'):
    p=HERE/name;target=FROZEN/name;target.write_bytes(p.read_bytes());target.chmod(0o444);files[str(target)]=sha(target)
  write(PROTOCOL,dict(schema=1,files=files,base_protocol_sha256=BASE_SHA,cpu_report_sha256=CPU_SHA,reference_sha256=REFERENCE_SHA,
    automatic_paced=False,timeout_seconds=1200,source_overlay_explicit=True));PROTOCOL.chmod(0o444)
  print(json.dumps(dict(ready=True,protocol_sha256=sha(PROTOCOL),gpu_executed=False,command=command(sha(PROTOCOL)))),flush=True)

def command(digest):
  C=context()
  result=[C.N.PY,str(C.N.WRAPPER),'agent-live-history-fast-consumer','--program',str(FROZEN/'agent-live-history-fast-run.py'),
    '--timeout-seconds','1200','--set','protocol-sha',digest,'--env','PYTHONDONTWRITEBYTECODE','1','--env','PYTHONPATH',str(C.N.TINY)]
  for key,value in C.N.storage().items():result+=['--env',key,value]
  return result

def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--admit',action='store_true')
  parser.add_argument('--plan',action='store_true');parser.add_argument('--cpu-boundary',action='store_true');parser.add_argument('--protocol-sha');parser.add_argument('--output',type=Path)
  args=parser.parse_args()
  if args.prepare:prepare()
  elif args.admit:
    C=context();C.configure();B=load(BASE/'agent-live-cpu-integrated-check.py','history_fast_cache_admission');B.C=C
    assert args.output is not None and not args.output.exists();write(args.output,B.admit())
  elif args.plan:verify(args.protocol_sha);print(json.dumps(command(args.protocol_sha)),flush=True)
  elif args.cpu_boundary:execute(args.output,'CPU-BOUNDARY-ONLY',cpu_only=True)
  else:
    assert args.output is not None and args.protocol_sha;execute(args.output,args.protocol_sha)

if __name__=='__main__':main()

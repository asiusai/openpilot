"""Real startup/main/model_run/Output/launcher/snapshot path; mock ONLY model initialization."""
import copy,importlib.util,inspect,json,os,shutil,socket,sys
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nearest_storage_check_run',HERE/'agent-nearest-storage-run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
class FactoryReached(Exception):pass

def main():
  old=run.inherited();run.setup_cpu(old);verified=old.a.verify();run.weights()
  c,plan,policy,startup,dynamic,order,reference,saved,admission=verified
  inherited=old.verify_frozen()
  fixture_protocol=run.CPU/'agent-nearest-storage-fixture-protocol.json'
  run.save(fixture_protocol,dict(cpu_only=True,gpu_executed=False,source_sha256=run.sources()))
  from tinygrad import Device
  def blocked(*args,**kwargs):raise AssertionError('CPU integration prohibits devices and network')
  boundary=[];guards={};actual_output=[]
  def attempt(name,collision=False):
    target=run.CPU/name/'report.json';original_spec=importlib.util.spec_from_file_location
    real_save=old.a.save
    def saved_file(path,value):
      real_save(path,value)
      if collision and Path(path).name=='agent-nearest-storage-admission.json':
        run.save(target.with_suffix('.occupied.json'),{'CPU_fixture':'reserved artifact must stop canonical main'})
    def spec(name,location,*args,**kwargs):
      result=original_spec(name,location,*args,**kwargs)
      path=Path(location).resolve()
      if path not in (run.BASE/'experiments/benchmark-pipeline.py',run.BASE/'experiments/agent-live-runtime.py'):return result
      loader=result.loader
      class Loader:
        def create_module(self,value):return loader.create_module(value)
        def exec_module(self,module):
          loader.exec_module(module)
          if path.name=='agent-live-runtime.py':
            Output=module.Output
            class ObservedOutput(Output):
              def __init__(self,output,storage_ok):
                super().__init__(output,storage_ok)  # ORIGINAL constructor and real disk_usage.
                actual_output.append(dict(path=str(output),filesystem=run.reserve(output.parent),
                  original_constructor_source_sha256=run.sha(module.__file__)))
            module.Output=ObservedOutput
          else:
            def initialize(effective,**kwargs):
              # The real canonical loop and complete model_run prologue already ran.
              stack=[dict(file=frame.filename,function=frame.function,line=frame.lineno) for frame in inspect.stack()]
              assert any(row['file']==str(run.BASE/'experiments/agent-paced-replay.py') and row['function']=='main' and row['line']==574 for row in stack)
              assert any(row['function']=='launch' and 'agent-startup-spec-run.py:model_run' in row['file'] for row in stack)
              assert actual_output[-1]['path']==str(target.with_suffix('.jsonl'))
              assert run.mounted(kwargs['source_dir']).stat().st_dev==Path('/mnt/nvme-500').stat().st_dev
              assert kwargs['sources']==saved['initialized']['source_sha256']
              protocol=run.read(target.with_suffix('.normalized-invocation.json'))
              paths=run.cache_environment(target.parent)
              assert all(protocol['environment'][key]==value for key,value in paths.items())
              assert all(os.environ.get(key)==value for key,value in paths.items())
              rows=[json.loads(line) for line in target.with_suffix('.jsonl').read_text().splitlines()]
              assert len(rows)==1 and rows[0]['kind']=='protocol' and rows[0]['rgb']['sha256']==admission['raw_inventory_sha256']
              assert rows[0]['trace']['frames']==450 and rows[0]['initialization']==saved['qualified_factory_protocol']
              assert rows[0]['requested_hz']==10 and rows[0]['inference_budget_seconds']==180
              assert not any(row.get('kind') in ('dispatch','initialized','model_completion') for row in rows)
              boundary.append(dict(output=str(target),call_path=stack,canonical_output_constructor_passed=True,
                original_source_snapshot_exact=True,numerical_protocol_exact=True,cache_paths=paths,
                real_rgb_inventory_verified=True,real_native_warp_loaded=True,real_speculative_launcher_executed=True,
                original_main_no_overwrite_passed=True,source_clock_not_started=True))
              raise FactoryReached('CPU stop at pipeline.initialize, before loading weights or opening a GPU')
            module.initialize=initialize
      result.loader=Loader();return result
    with patch.object(run,'DEFAULT',target),patch.object(importlib.util,'spec_from_file_location',spec):
      function,_=run.transform(old,old.verify_frozen);function.__globals__['_storage_protocol']=fixture_protocol
      try:
        with patch.dict(os.environ,{'AGENT_NEAREST_PACED_PARENT_TIMEOUT':'1'}),\
             patch.object(sys,'argv',['storage-check','--execute-gpu','--output',str(target)]),\
             patch.object(old.fcntl,'flock',side_effect=BlockingIOError),patch.object(old.a,'save',saved_file):function()
      except (FactoryReached,FileExistsError) as error:return dict(type=type(error).__name__,message=str(error),output=str(target))
    raise AssertionError('CPU model boundary or canonical no-overwrite guard did not stop execution')

  with patch.object(type(Device),'__getitem__',blocked),patch.object(socket.socket,'connect',blocked),patch.object(socket.socket,'connect_ex',blocked):
    passed=attempt('actual-full-path')
    assert passed['type']=='FactoryReached' and len(boundary)==1 and len(actual_output)==1
    # Actual file-system Output guard: root is rejected BEFORE creating a file.
    runtime=c.local('agent-live-runtime');bridge=c.local('agent-live-bridge')
    root_path=run.EXP/'agent-nearest-storage-root-probe.jsonl'
    root_usage=shutil.disk_usage(run.EXP)
    assert not bridge.storage_ok(root_usage.total,root_usage.free,runtime.Output.limit)
    try:runtime.Output(root_path,bridge.storage_ok)
    except RuntimeError as error:assert str(error)=='Output would cross the 10% or 5 GiB reserve'
    else:raise AssertionError('Original root storage guard unexpectedly accepted insufficient space')
    assert not root_path.exists();guards['actual_root_output_rejected']=True
    collision=attempt('reserved-stem',collision=True)
    assert collision['type']=='FileExistsError' and collision['message']=='Use a new output stem; retain all existing artifacts'
    assert len(boundary)==1 and len(actual_output)==1;guards['reserved_stem_rejected_before_output_and_factory']=True
    try:run.validate_output(run.EXP/'nearest-paced-storage-run/report.json')
    except ValueError:guards['root_output_path_rejected']=True
    else:raise AssertionError('Root output path accepted')
    # The original Output exclusive-open guard also rejects an existing file on500GB.
    probe=run.CPU/'agent-nearest-storage-existing.jsonl';out=runtime.Output(probe,bridge.storage_ok);out.close()
    try:runtime.Output(probe,bridge.storage_ok)
    except FileExistsError:guards['original_output_exclusive_open_preserved']=True
    else:raise AssertionError('Existing Output artifact overwritten')
  assert not Device._opened_devices
  run.check_pins(run.PINS);run.inherited();old.a.verify()
  assert not run.DEFAULT.parent.exists()
  _,transformation=run.transform(old,old.verify_frozen)
  report=dict(passed=True,cpu_only=True,gpu_executed=False,opened_devices=[],source_sha256=run.sources(),
    admission=admission,transformation=transformation,actual_filesystem=run.reserve(run.DEFAULT.parent),
    root_filesystem=dict(total_bytes=root_usage.total,free_bytes=root_usage.free),actual_output_constructors=actual_output,
    factory_boundaries=boundary,positive=passed,negative=collision,guards=guards,
    old_sources_and_evidence_unchanged=True,all_new_artifacts_and_caches_on500GB=True,
    limit='Real canonical startup/main/model_run/RGB/native-warp/Output/launcher/snapshot; model initialize is a CPU sentinel; parent lock simulated')
  run.save(run.CHECKS,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()

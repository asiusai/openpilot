"""CPU-only cache-path and unchanged-rejection checks for the optional primer."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

os.environ['DEV']='AMD'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')


def local(path):
  spec=importlib.util.spec_from_file_location(Path(path).stem.replace('-','_'),path)
  value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value)
  return value


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  if args.output.exists(): raise FileExistsError(args.output)
  for name in ('agent-startup-key-repro.py','agent-startup-key-prime.py','agent-startup-key-run.py',Path(__file__).name):
    ast.parse((ROOT/name).read_text())
  run=local(ROOT/'agent-startup-key-run.py')
  frozen,prototype=run.validate()
  original=json.loads((WORK/'agent-startup-cache-trial1-ordinary/report.json').read_text())['startup_cache']
  old=json.loads((WORK/'agent-startup-cache-trial1-candidate/report.json').read_text())['startup_cache']
  try: prototype.compare_signatures(original,old)
  except AssertionError: pass
  else: raise AssertionError('Old differing GPU programs were accepted')
  repro=local(ROOT/'agent-startup-key-repro.py')
  helper=local(ROOT/'agent-startup-key-prime.py')
  from tinygrad import Device, dtypes
  from tinygrad.engine.realize import lower_and_compile
  from tinygrad.uop.ops import Ops, UOp
  def forbidden(*args,**kwargs): raise AssertionError('No device open or compile worker is allowed')
  with patch.object(type(Device),'__getitem__',forbidden):
    renderer=repro.renderer()
    first=helper.prime(renderer)
    calls=[]
    for count in range(2,7):
      calls.append(helper.concat_ast(count).call(*[UOp.new_buffer('AMD',n,dtypes.float32) for n in [count]+[1]*count]))
    linear=UOp(Ops.LINEAR,src=tuple(calls))
    fake=SimpleNamespace(_select_renderer=lambda _:renderer)
    def fake_lookup(self,name):
      assert name=='AMD'
      return fake
    with patch.object(type(Device),'__getitem__',fake_lookup), patch('tinygrad.engine.realize.get_worker_pool',forbidden):
      compiled=lower_and_compile(linear)
    assert [c.body.key.hex() for c in compiled.src]==list(helper.ORDINARY.values())
    repeat=helper.prime(renderer)
    assert all(row['cache_hit'] for row in repeat['rows'])
    try: run.initialized_renderer()
    except (AssertionError,ValueError): pass
    else: raise AssertionError('Uninitialized device was accepted')
    # Exercise the real install failure path with one opaque current-input owner.
    # No fake GPU report or modified signature fixture is used.
    owner=object();original_map={};components=SimpleNamespace(decoder=SimpleNamespace(jits=original_map))
    class ControlledFailure(Exception): pass
    def failed_initialize(got_components,got_owner,**kwargs):
      assert got_components is components and got_owner is owner
      raise ControlledFailure('CPU lifecycle control')
    warm=SimpleNamespace(initialize=failed_initialize,ClosedGraphs=dict)
    def base_local(path):
      assert Path(path)==run.FROZEN/'agent-startup-spec-warm.py'
      return warm
    base=SimpleNamespace(local=base_local)
    def base_main(options): base.local(run.FROZEN/'agent-startup-spec-warm.py').initialize(components,owner)
    base.main=base_main
    original_loader=run.local
    def loader(path):
      return base if Path(path)==run.FROZEN/'agent-startup-spec-run.py' else original_loader(path)
    with tempfile.TemporaryDirectory(prefix='agent-startup-key-cpu-',dir='/tmp/opencode') as temp:
      path=Path(temp)/'control.json'
      with patch.object(run,'local',loader),patch.object(run,'initialized_renderer',lambda:renderer):
        try: run.main(['--audit-only','--task','replay','--mode','consumer','--execute-gpu','--output',str(path)])
        except ControlledFailure: pass
        else: raise AssertionError('Controlled initializer failure did not propagate')
      assert base.local is base_local and warm.initialize is failed_initialize
      assert components.decoder.jits is original_map and not path.exists()
      failure=json.loads(path.with_suffix('.startup-key.json').read_text())
      assert failure['qualified'] is False and 'ControlledFailure' in failure['error']
  assert not Device._opened_devices
  for field in ('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained'):
    fixture={k:True for k in ('ready','full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','warmed_owner_retained')}
    run.check_startup(fixture)
    fixture[field]=False
    try: run.check_startup(fixture)
    except AssertionError: pass
    else: raise AssertionError('Missing restoration gate was accepted: '+field)
  result=dict(cpu_only=True,gpu_executed=False,devices=[],frozen_sources_valid=True,old_gpu_candidate_still_rejected=True,
    primed_five_call_compile_uses_exact_ordinary_programs=True,compile_worker_calls=0,restoration_failure_controls=5,
    uninitialized_device_rejected=True,failed_install_restores_decoder_and_launcher=True,
    failed_initialize_received_original_current_owner=True,prime_first=first,prime_repeat=repeat,
    source_hashes={p.name:run.sha(p) for p in ROOT.glob('agent-startup-key-*.py')})
  with args.output.open('x') as stream: json.dump(result,stream,indent=2);stream.write('\n')
  print(json.dumps({k:v for k,v in result.items() if k not in ('prime_first','prime_repeat','source_hashes')},indent=2))


if __name__=='__main__': main()

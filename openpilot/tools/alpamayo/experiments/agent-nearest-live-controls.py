"""Independent CPU rejection and qualified-startup import-composition checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
BASE=WORK/'agent-paced-dynamic-source-cpu02/alpamayo'
DATA=WORK/'agent-nearest-live-data'
sys.dont_write_bytecode=True


def load(path,name):
  spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value


def main():
  if os.environ.get('DEV')!='CPU':raise ValueError('DEV=CPU required')
  sys.path[:0]=[str(BASE),str(BASE/'experiments'),'/tmp/opencode/openpilot-alpamayo','/home/john/asius/openpilot/opendbc_repo']
  policy=load(HERE/'agent-nearest-live-policy.py','agent_nearest_live_controls_policy')
  startup=load(WORK/'startup-spec-frozen-source/agent-startup-spec-run.py','agent_nearest_live_controls_startup')
  warm=load(WORK/'startup-spec-frozen-source/agent-startup-spec-warm.py','agent_nearest_live_controls_warm')
  dynamic=load(BASE/'experiments/agent-paced-dynamic-policy.py','agent_nearest_live_controls_dynamic')
  dynamic.verify_snapshot();transforms={}
  with policy.imports(BASE),startup.imports(BASE,warm,{},transforms),dynamic.imports():
    paced=dynamic.local('agent-paced-replay');replay=dynamic.local('agent-live-replay');live=dynamic.local('agent-live-consumer')
    store,preparer,consumer=paced.bridge.Store,paced.bridge.Preparer,live.Consumer
    originalStore,originalPreparer,originalConsumer=store.__bases__[0],preparer.__bases__[0],consumer.__bases__[0]
    assert store.add_frame is originalStore.add_frame and store.add_state is originalStore.add_state and store.bounds is originalStore.bounds
    assert preparer.rgb is originalPreparer.rgb
    for name in ('predict','synchronize','_bind_inputs','_check_bounds','_guard_startup','close'):
      assert getattr(consumer,name) is getattr(originalConsumer,name)
    assert set(transforms)=={'paced','replay'}
    audit=json.loads((DATA/'agent-nearest-live-full64.json').read_text());case=audit['parent_gpu_cases'][0]
    arrays=replay.Dataset.load(dict(sample=case['sample']));meta=copy.deepcopy(case['metadata'])
    data=consumer._data(SimpleNamespace(last_query_ns=0),arrays,meta,meta['prepared_ns'])
    assert data['image_frames'].shape==(10,1,3,432,768)
    try:originalConsumer._data(SimpleNamespace(last_query_ns=0),arrays,meta,meta['prepared_ns'])
    except ValueError:pass
    else:raise AssertionError('Store-only override unexpectedly bypassed original FLOOR Consumer guard')
    checks={}
    for name,mutate in (
      ('unreceived_capture',lambda ref,m:ref.update(received_ns=m['selected_ns']+1,ready_ns=m['selected_ns']+1)),
      ('not_ready_capture',lambda ref,m:ref.update(ready_ns=m['selected_ns']+1)),
      ('future_capture',lambda ref,m:ref.update(capture_ns=m['query_ns']+1)),
      ('future_calibration',lambda ref,m:ref.update(image_calibration_ns=ref['capture_ns']+1)),
      ('wrong_policy',lambda ref,m:m.update(history_policy='all-offline-captures'))):
      poisoned=copy.deepcopy(meta);mutate(poisoned['frames'][0],poisoned)
      try:consumer._data(SimpleNamespace(last_query_ns=0),arrays,poisoned,poisoned['prepared_ns'])
      except ValueError:checks[name]=True
      else:raise AssertionError('Poisoned metadata accepted: '+name)
    try:consumer._data(SimpleNamespace(last_query_ns=meta['query_ns']),arrays,meta,meta['prepared_ns'])
    except ValueError:checks['stale_consumer_query']=True
    else:raise AssertionError('Stale query accepted')
  assert paced.bridge.Store is originalStore and paced.bridge.Preparer is originalPreparer and live.Consumer is originalConsumer
  from tinygrad import Device
  assert not Device._opened_devices
  result=dict(cpu_only=True,gpu_executed=False,initialized_devices=[],actual_model_predictions=0,
    source_exact_startup_import_composition=True,transformed_startup_functions=transforms,original_module_aliases_restored=True,
    unchanged_ingestion_state_bounds_and_consumer_ownership_methods=True,original_floor_Consumer_correctly_rejects_nearest=True,
    poisoned_metadata_rejections=checks,adapter_sha256=policy.sha(HERE/'agent-nearest-live-policy.py'),
    actual_CPU64_report_sha256=policy.sha(DATA/'agent-nearest-live-full64.json'),startup_sources=startup.own_sources())
  with (DATA/'agent-nearest-live-controls.json').open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))


if __name__=='__main__':main()

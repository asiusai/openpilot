"""Keep one output-helper/adapter owner for the frozen runner's ABI scope."""
import importlib.util
import json
from pathlib import Path

EXP=Path(__file__).parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-ck-lifetime-work'
PREPARED=ROOT/'prepared'
PROTOCOL=PREPARED/'protocol.json'
OLD_ROOT=WORK/'agent-ck-storage-work'
OLD_PROTOCOL=OLD_ROOT/'prepared/protocol.json'
spec=importlib.util.spec_from_file_location('ck_lifetime_previous',EXP/'agent-ck-storage-overlay.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
sha=previous.sha
PINS={OLD_PROTOCOL:'3a4a2c80f94c10720d646fd7785a41ae6c4d5e02e8061d43e08b276169385df9',
  EXP/'agent-ck-storage-overlay.py':'09046c4ecb79b67a4b6b46ecbbfe90ba7535a4439d302d234ce9225b7777a79c',
  WORK/'agent-ck-storage-parent/report.json':'61d381062b91284705d7f3932184da5dafe652a174dafd46d39bd816399e005d',
  OLD_ROOT/'parent-result.json':'a327ec34d26be6f84aca257cd59bbabb9b078fe3d8e694473cb3cc593afcc1e2',
  WORK/'agent-ck-storage-parent/run.log':'9d4428c8fd5edd0a7afc94ade5f58b450ea5b968d5fd761b979c39f736863c39'}

def bind_output_helper(common):
  original_load=common.load
  helper=original_load('agent-ck-output-common')
  def load(name,directory=EXP):
    if name=='agent-ck-output-common' and Path(directory)==EXP: return helper
    return original_load(name,directory)
  common.load=load
  return helper

def invocation():
  value=json.loads((OLD_ROOT/'prepared/invocation.json').read_text())
  command=value['command'];command[command.index('--output')+1]=str(ROOT/'benchmark-unused.json')
  value['environment'].update(CACHEDB=str(ROOT/'cache.db'),TMPDIR=str(ROOT/'tmp'))
  return value

def plan():
  value=json.loads(OLD_PROTOCOL.read_text());command=value['launcher_command']
  command[command.index('--output')+1]=str(ROOT/'benchmark-unused.json')
  command[command.index('--invocation')+1]=str(PREPARED/'invocation.json')
  return value

def verify():
  assert all(sha(p)==s for p,s in PINS.items())
  current=json.loads(PROTOCOL.read_text());expected=plan()
  for key,value in expected.items():
    if key=='source_sha256': assert all(current[key].get(p)==s for p,s in value.items())
    else: assert current[key]==value, ('Changed frozen protocol field',key)
  assert json.loads((PREPARED/'invocation.json').read_text())==invocation()
  for name in ('catalog.json','producer-catalog.json'): assert (PREPARED/name).read_bytes()==(OLD_ROOT/'prepared'/name).read_bytes()
  for path,digest in current['source_sha256'].items(): assert sha(path)==digest, f'Changed bound file: {path}'
  return current

def load_runner():
  assert all(sha(p)==s for p,s in PINS.items())
  namespace=previous.load_runner();common=namespace['run'].__globals__['C']
  bind_output_helper(common)
  common.ROOT,common.PREPARED,common.PROTOCOL=ROOT,PREPARED,PROTOCOL
  common.verify=verify
  return namespace

"""Remove only the redundant outer alarm; canonical replay owns SIGALRM."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType

EXP=Path(__file__).resolve().parent
WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT=WORK/'agent-ck-paced-deadline-work'
ORIGINAL=EXP/'agent-ck-selected-recovery-paced.py'
WRAPPER=Path('/tmp/opencode/run-alpamayo-speed-trial.py')
CANONICAL=WORK/'agent-paced-dynamic-source-cpu02/alpamayo/experiments/agent-paced-replay.py'
SEED=WORK/'agent-ck-selected-recovery-work/parent-firmware-cache-seed-v1.json'
PINS={ORIGINAL:'840011ad6b14eea20c0107ca3a58ecd1e0c306c743d8c40ad4bf36c7329a3d20',
  WRAPPER:'c931aa3d0f6fa789999927e873c5eb6f3dcd273f8d666339f681510bc4a092d3',
  CANONICAL:'8af0ee0ff255c2c928eca41a706b4d370340d410444723565a424a2a6144eee5',
  SEED:'6fa69ff5134f830183781264d8077dbda8c0335c5780a3113926e1967c79ba12'}
LOCK=Path('/tmp/opencode/alpamayo-chestnut-speed.lock')

def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def dump(node):return ast.dump(node,include_attributes=False)
def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)

def verify_sources():
  for path,digest in PINS.items():assert sha(path)==digest,('Changed deadline foundation',str(path))
  seed=json.loads(SEED.read_text());assert len(seed['files'])==4
  for row in seed['files']:
    path=Path(row['destination']);assert path.is_relative_to(SEED.parent/'artifacts/cache/tinygrad/downloads/fw')
    assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
  wrapper=ast.parse(WRAPPER.read_text())
  expected=ast.parse("""if __name__=='__main__':
  with Path('/tmp/opencode/alpamayo-chestnut-speed.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    main()
""").body[0]
  assert dump(wrapper.body[-1])==dump(expected),'Wrapper no longer enters main under the global lock'
  return wrapper

def transformed():
  verify_sources();tree=ast.parse(ORIGINAL.read_text());before=copy.deepcopy(tree);main=function(tree,'main')
  removed=ast.parse("""def expired(*args):raise TimeoutError('Selected recovery paced child reached 580 seconds')
old=signal.signal(signal.SIGALRM,expired)
signal.alarm(580)
try:execute(a)
finally:
  signal.alarm(0)
  signal.signal(signal.SIGALRM,old)
""").body
  assert [dump(n) for n in main.body[-4:]]==[dump(n) for n in removed], 'Outer alarm source differs'
  main.body[-4:]=ast.parse('_require_wrapper(a)\nexecute(a)\n').body
  assert dump(function(tree,'execute'))==dump(function(before,'execute'))
  restored=copy.deepcopy(tree);function(restored,'main').body[-2:]=removed
  assert dump(restored)==dump(before),'Unexpected change outside the exact outer alarm block'
  assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
    and n.func.value.id=='signal' for n in ast.walk(function(tree,'main')))
  return ast.fix_missing_locations(tree),dict(original_execute_ast_sha256=hashlib.sha256(dump(function(before,'execute')).encode()).hexdigest(),
    execute_ast_unchanged=True,only_outer_alarm_block_replaced=True,canonical_source_sha256=PINS[CANONICAL],
    wrapper_source_sha256=PINS[WRAPPER],parent_timeout_seconds=600,canonical_timeout_seconds=180,
    wrapper_sigint_grace_seconds=5,firmware_seed_unchanged=True)

def option(words,name):
  assert words.count(name)==1,('Require one wrapper argument',name)
  return words[words.index(name)+1]

def parent_contract(words,invocation,output,fd_targets):
  assert len(words)>2 and Path(words[1]).resolve()==WRAPPER,'Use the normal direct parent wrapper'
  assert Path(option(words,'--program')).resolve()==Path(__file__).resolve()
  assert float(option(words,'--timeout-seconds'))==600
  assert words[2]=='agent-ck-selected-recovery-parent-paced-v3'
  assert Path(output).resolve()==WORK/words[2]/'report.json'
  assert invocation['timeout_seconds']==600 and Path(invocation['command'][1]).resolve()==Path(__file__).resolve()
  assert Path(option(invocation['command'],'--output')).resolve()==Path(output).resolve()
  assert '--parent-run' in invocation['command'] and str(LOCK) in fd_targets

def require_wrapper(args):
  verify_sources();parent=Path('/proc')/str(os.getppid())
  words=[x.decode() for x in (parent/'cmdline').read_bytes().split(b'\0') if x]
  targets=[]
  for fd in (parent/'fd').iterdir():
    try:targets.append(os.readlink(fd))
    except FileNotFoundError:pass
  invocation=json.loads((args.output.parent/'invocation.json').read_text())
  parent_contract(words,invocation,args.output,targets)

def build():
  tree,proof=transformed();module=ModuleType('ck_paced_deadline_original');module.__file__=str(ORIGINAL)
  exec(compile(tree,str(ORIGINAL),'exec'),module.__dict__);module._require_wrapper=require_wrapper
  return module,proof

if __name__=='__main__':build()[0].main()

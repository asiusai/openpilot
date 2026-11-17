"""CPU constructor, scoped dispatch, complete PROGRAM, capture, and route gate tests."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', TC_OPT='2', BEAM='0', DEBUG='0', PROFILE='0', CCACHE='0')
  here = Path(__file__).parent
  def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), here/(name+'.py'))
    value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value; spec.loader.exec_module(value)
    return value
  C, H, R = (load(n) for n in ('agent-expert-pv-qualified-common', 'agent-expert-pv-qualified-hook', 'agent-expert-pv-qualified-run'))
  evidence = C.qualification()
  C.paths()
  from tinygrad import Device, dtypes
  from tinygrad.codegen import to_program_cache
  from tinygrad.engine.realize import _get_call_to_compile
  from tinygrad.helpers import PARALLEL
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops, UOp
  from model import EdgeModel
  import amd_attention
  renderer = C.usb().renderers()['parent-wgp']
  class FakeDevice:
    DEFAULT = 'AMD'
    def __class_getitem__(cls, key): return SimpleNamespace(renderer=renderer)
  rejected, installs = [], []
  def reject(label, fn):
    try: fn()
    except (AssertionError, KeyError, ValueError, StopIteration): rejected.append(label)
    else: raise AssertionError('Invalid case passed: '+label)
  def make(candidate, bad=None):
    model = EdgeModel.__new__(EdgeModel)
    model.fp8_wmma, model.dtype, model.weights = True, dtypes.bfloat16, {}
    model.config = {'expert_config': {'llm_config': {'num_hidden_layers': 28}}}
    model.trajectory_jits = {}
    model.cache = [SimpleNamespace(shape=(2, 1, 8, 2304, 128))]*28
    expert = C.load(C.SNAP/'experiments/expert-kernels.py', 'qualified_fake_expert_'+str(candidate))
    expert.install(model)
    if bad == 'dtype': model.dtype = dtypes.float16
    elif bad == 'layers': model.config['expert_config']['llm_config']['num_hidden_layers'] = 27
    elif bad == 'capture': model.trajectory_jits = {'already_captured': object()}
    elif bad == 'other_expert_hook': model.agent_expert_tiles = object()
    before = (model.linear, model.quantize, model.text_layer, model.attention, expert.attention, expert._pack, expert._softmax)
    global_before = renderer, renderer.compiler, PARALLEL.value, dict(to_program_cache)
    with patch('tinygrad.Device', FakeDevice): handle = H.install(model, expert, candidate=candidate, evidence=evidence)
    assert before == (model.linear, model.quantize, model.text_layer, model.attention, expert.attention, expert._pack, expert._softmax)
    assert global_before == (renderer, renderer.compiler, PARALLEL.value, dict(to_program_cache))
    return model, expert, handle
  shapes = C.usb().load('agent-expert-pv-compile').specs(C.SHAPE)
  buffers = [UOp.placeholder(shape, dtypes.bfloat16, i) for i, shape in enumerate(shapes)]
  capture_infos = {}
  for candidate in (False, True):
    model, expert, h = make(candidate)
    original, originals = h.original_matmul, h.originals.copy()
    calls = []
    h.original_matmul = lambda *a, **kw: calls.append(kw) or h.prepared['current']['ast']
    assert expert._matmul(*buffers, shape=C.SHAPE) is h.prepared['current']['ast']
    h.depth = 1
    selected = expert._matmul(*buffers, shape=C.SHAPE)
    assert selected is h.prepared['candidate']['program'] if candidate else selected is h.prepared['current']['ast']
    if candidate:
      assert selected.op is Ops.PROGRAM and selected.src[-1].op is Ops.BINARY
      params = [UOp.param(i, x.dtype, x.numel(), device='AMD') for i, x in enumerate(buffers)]
      assert _get_call_to_compile(selected.call(*params)) is None, 'Candidate would re-enter compiler workers'
    for shape, causal, scaled in ((C.SHAPE, True, False), (C.SHAPE, False, True),
      ((1,16,8,32,128,2368),False,False), ((2,16,8,64,128,2368),False,False),
      ((1,16,8,64,128,2112),False,False), ((1,16,8,64,2368,128),False,True)):
      count = len(calls)
      assert expert._matmul(*buffers, shape=shape, causal=causal, scaled=scaled) is h.prepared['current']['ast']
      assert len(calls) == count+1
    for i in range(3):
      for shape, dtype in ((shapes[i], dtypes.float16), ((1,), dtypes.bfloat16)):
        changed = buffers.copy(); changed[i] = UOp.placeholder(shape, dtype, i)
        count = len(calls)
        assert expert._matmul(*changed, shape=C.SHAPE) is h.prepared['current']['ast'] and len(calls) == count+1
    h.depth = 0
    def trace(*a, **kw):
      for step in range(3):
        for layer in range(28):
          expert._matmul(*buffers, shape=(1,16,8,64,2368,128), scaled=True)
          expert._matmul(*buffers, shape=C.SHAPE)
      return 'trace'
    h.originals['_trajectory'] = trace
    assert model._trajectory(None,None,None,2033,3,6) == 'trace'
    assert h.info['traces'] == [{'pv':84,'qk':84}]
    def fail(*a, **kw): raise RuntimeError('injected')
    h.originals['_trajectory'] = fail
    try: model._trajectory(None,None,None,2033,3,6)
    except RuntimeError: pass
    assert h.depth == 0
    h.originals['_trajectory'] = lambda *a, **kw: 'unqualified'
    assert model._trajectory(None,None,None,2033,2,6) == 'unqualified'
    model.cache[0] = SimpleNamespace(shape=(2,1,8,2048,128))
    assert model._trajectory(None,None,None,2033,3,6) == 'unqualified'
    assert h.info['unqualified_trajectory_calls'] == 2
    model.cache[0] = SimpleNamespace(shape=(2,1,8,2304,128))
    program = h.prepared['candidate' if candidate else 'current']['program']
    params = [UOp.param(i, x.dtype, x.numel(), device='AMD') for i, x in enumerate(buffers)]
    with patch.object(hcq2, '_build_queues', lambda ctx: None):
      previous = hcq2._build_queues
      with h.observe_programs(): hcq2._build_queues(SimpleNamespace(batch=[(program.call(*params),None,None)]))
      assert hcq2._build_queues is previous
    def jit(pv_count=84):
      records = []
      for name, key, count in ((h.expected['name'],program.key,pv_count),
          ('alpamayo_attention_scores',b'q'*32,84),('alpamayo_attention_softmax',b's'*32,84)):
        if key != program.key: h.programs[key.hex()] = {'name':name,'binary_sha256':name,'target':evidence['target']}
        records.extend([('AMD',name,None,None,key,(),((0,),(1,2)))]*count)
      call = SimpleNamespace(without_after=SimpleNamespace(arg=SimpleNamespace(aux=SimpleNamespace(kernels=records))))
      return SimpleNamespace(captured=SimpleNamespace(_linear=SimpleNamespace(src=(call,))))
    captured = jit(); h.inspect_capture(captured); h.inspect_capture(captured)
    assert len(h.info['captures']) == 1
    reject('missing capture PV '+str(candidate),lambda:h.inspect_capture(jit(83)))
    reject('extra capture PV '+str(candidate),lambda:h.inspect_capture(jit(85)))
    real_record = h.programs[program.key.hex()]
    h.programs[program.key.hex()] = {**real_record, 'target':'AMD:LLVM:gfx1200'}
    reject('wrong full target '+str(candidate),lambda:h.inspect_capture(jit()))
    h.programs[program.key.hex()] = real_record
    capture_infos[candidate] = copy.deepcopy(h.info['captures'])
    h.original_matmul, h.originals = original, originals
    h.close(); h.close()
    assert expert._matmul is original and model.trajectory == originals['trajectory'] and model._trajectory == originals['_trajectory']
    assert h.info['cleanup_exact'] and h.info['observer_cleanup_exact'] and not hasattr(model,'agent_expert_pv_qualified')
    installs.append({'candidate':candidate,'full_28_layer_three_euler_trace':True,'noneligible_original_dispatch':True,
      'fp8_linear_quant_attention_pack_softmax_unchanged':True,'global_renderer_parallel_cache_unchanged':True,'cleanup_exact':True})
  # Execute the actual benchmark import namespace under runpy, without main().
  namespace = runpy.run_path(str(C.SNAP/'experiments/benchmark-pipeline.py'),run_name='qualified_cpu_import_probe')
  exec('import wgp_compiler',namespace)
  assert Path(namespace['wgp_compiler'].__file__).resolve() == C.SNAP/'experiments/wgp_compiler.py'
  compiler = namespace['wgp_compiler'].WGPCompiler('gfx1200')
  assert type(pickle.loads(pickle.dumps(compiler))) is type(compiler)
  for bad in ('dtype','layers','capture','other_expert_hook'):
    reject('constructor/'+bad,lambda bad=bad:make(False,bad))
  # Synthetic complete-report controls use actual prior matched route records.
  raw = Path.read_bytes
  sources = {p.name:p for p in [*(C.SNAP/'experiments').glob('*.py'),*(C.SNAP/'experiments').glob('*.c'),C.SNAP/'model.py']}
  def read(path):
    if path.parent.name == 'qualified-synthetic-sources': return raw(sources[path.name])
    return raw(path)
  for phase in R.COUNTS:
    base_path = C.WORK/f'agent-expert-tiles-{phase}-reference/report.json'
    baseline = json.loads(base_path.read_text()); baseline.pop('agent_expert_tiles',None)
    sidecar = json.loads(base_path.with_suffix('.speculation.json').read_text())
    planned = R.plan(C,phase,C.WORK/'qualified-synthetic-report.json')
    reports = []
    for candidate in (False,True):
      report = copy.deepcopy(baseline)
      report['source_sha256'] = {name:C.sha(raw(path)) for name,path in sources.items()}
      report['source_snapshot'] = str(C.WORK/'qualified-synthetic-sources')
      for field,flag in (('manifest','--native-manifest'),('calibrations','--native-calibrations')):
        path = R.original(C).option(planned['terminal_command'],flag)
        report['native_preprocessing'][field]=path
        report['native_preprocessing'][field+'_sha256']=C.sha(raw(Path(path)))
      info = {'candidate':candidate,'active':False,'cleanup_exact':True,'observer_cleanup_exact':True,'evidence':evidence,
        'traces':[{'pv':84,'qk':84}],'captures':copy.deepcopy(capture_infos[candidate]),
        'trajectory_calls':sum(R.COUNTS[phase]),'unqualified_trajectory_calls':0,'outside_scope_selected':0,
        'bound_program_calls':84 if candidate else 0}
      report['agent_expert_pv'] = {'protocol_sha256':'synthetic','variant':'candidate' if candidate else 'current',
        'plan_only':False,'route_completed':True,'coverage_validated':True,'plan':planned,'evidence':evidence,'hooks':[info]}
      with patch.object(Path,'read_bytes',read): R.validate(C,report,sidecar,planned,candidate,evidence,'synthetic')
      reports.append(report)
    R.compare(C,*reports,planned,sidecar,sidecar)
    for field in ('pv83','target','binary','qk','cleanup','unqualified','binding_count','fixed_vision','source',
                  'sixth_target','xyz','rot','tokens','text','input','history','speculation','negative_zero','nonfinite'):
      before,after = copy.deepcopy(reports)
      spec = copy.deepcopy(sidecar)
      info = after['agent_expert_pv']['hooks'][0]
      def change_leaf(value, replacement=None):
        while isinstance(value[0],list): value=value[0]
        value[0] = value[0]+0.001 if replacement is None else replacement
      if field=='pv83': info['captures'][0]['pv_calls']=83
      elif field=='target': info['captures'][0]['pv_full_description']['target']='AMD:LLVM:gfx1200'
      elif field=='binary': info['captures'][0]['binaries'][evidence['candidate']['name']]=['changed']
      elif field=='qk': info['captures'][0]['binaries']['alpamayo_attention_scores']=['changed']
      elif field=='cleanup': info['cleanup_exact']=False
      elif field=='unqualified': info['unqualified_trajectory_calls']=1
      elif field=='binding_count': info['bound_program_calls']=83
      elif field=='fixed_vision': after['vision_calibration_sha256']='changed'
      elif field=='source': after['source_sha256']['expert-kernels.py']='changed'
      elif field=='sixth_target': after['runs'][0]['draft']['prefill_draft']['targets'][5]+=1
      elif field in ('xyz','rot'): change_leaf(after['runs'][0]['prediction'][field])
      elif field=='tokens': after['runs'][0]['prediction']['generated_token_ids'][0]+=1
      elif field=='text': after['runs'][0]['prediction']['reasoning']='changed'
      elif field=='input': after['runs'][0]['sample']='changed'
      elif field=='history': after['native_preprocessing']['manifest_sha256']='changed'
      elif field=='speculation': spec['records'][0]['changed_verification_field']=True
      elif field=='negative_zero':
        before['runs'][0]['signed_zero_control']=0.0; after['runs'][0]['signed_zero_control']=-0.0
      else: change_leaf(after['runs'][0]['prediction']['xyz'],float('nan'))
      def gate():
        with patch.object(Path,'read_bytes',read): R.validate(C,after,spec,planned,True,evidence,'synthetic')
        R.compare(C,before,after,planned,sidecar,spec)
      reject(phase+'/'+field,gate)
  with patch.object(Path,'read_bytes',lambda p: raw(p)+b' ' if p==C.MICRO else raw(p)):
    reject('qualified GPU report mutation',C.qualification)
  code = amd_attention._matmul.__code__
  try:
    amd_attention._matmul.__code__ = (lambda *a,**kw:None).__code__
    reject('runtime original arithmetic mutation',lambda:make(False))
  finally: amd_attention._matmul.__code__ = code
  Q = C.load(Path('/tmp/opencode/queue-alpamayo-expert-pv-qualified.py'),'qualified_queue_cpu')
  queue_checks = Q.sequence_checks()
  assert not Device._opened_devices
  report = {'complete':True,'gpu_executed':False,'installs':installs,'rejected':rejected,
    'actual_runpy_wgp_import_and_pickle_passed':True,'canonical_finite_float_bits_and_signed_zero_checked':True,
    'all_query_spec_input_history_fields_checked':True,'quality_failure_blocks_native':True,'queue_checks':queue_checks,
    'evidence':evidence,'sources':{str(p):C.sha(raw(p)) for p in [*here.glob('agent-expert-pv-qualified-*.py'),
      Path('/tmp/opencode/queue-alpamayo-expert-pv-qualified.py')]}}
  C.usb().old().write(args.output,report)
  print(json.dumps({'complete':True,'gpu_executed':False,'rejections':len(rejected)}))


if __name__=='__main__': main()

"""Fixed compiler-mode quality and native timing protocol; private, optional."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct
import sys

WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-compiler-speed-source'
PRIVATE = ROOT/'private'
SNAP = ROOT/'alpamayo'
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
PY = '/tmp/opencode/alpamayo-asius-20261114/upstream/.venv/bin/python'
CACHE = WORK/'agent-compiler-speed-cache/cache.db'
SCRATCH = WORK/'agent-compiler-speed-cache/tmp'
COUNTS = {'quality': (2, 62), 'native': (12, 24)}
ARMS = ('original', 'consistent-wgp')
ROW_TIMES = {'load_ms', 'preprocess_ms', 'prepared_queue_ms', 'cpu_buffers_ms', 'pipeline_ms', 'service_ms', 'gc_ms', 'native_calibration_ms'}
STAGE_TIMES = {'diffusion', 'input_upload', 'prefill', 'reasoning', 'total', 'trajectory_decode', 'vision'}
TOP_TIMES = {'model_load_ms', 'startup_through_first_prediction_s', 'startup_before_measured_input_s', 'process_elapsed_s'}
SUMMARY_TIMES = {'median_ms', 'p95_ms', 'max_ms', 'throughput_fps', 'measured_wall_seconds', 'inverse_mean_latency_fps',
  'predictions_over_250ms', 'service_median_ms', 'service_p95_ms', 'input_to_output_over_100ms', 'input_to_output_over_200ms',
  'service_over_100ms', 'service_over_200ms'}
CONTRACT = ('name', 'target', 'lowered_ast', 'linear', 'source_sha256', 'program_info', 'params')


def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as f:
    for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
  return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec); sys.modules[name] = module
  spec.loader.exec_module(module)
  return module
def route(): return load(SNAP/'experiments/agent-expert-pv-route-run.py', 'agent_compiler_speed_recipe')
def out(stage, arm): return WORK/f'agent-compiler-speed-{stage}-{arm}'
def metadata_path(stage, arm): return out(stage, arm)/'report.json'


def commands(digest):
  result={}
  for stage in COUNTS:
    for arm in ARMS:
      result[stage+'/'+arm]=[PY,str(PRIVATE/'run-alpamayo-speed-trial.py'),out(stage,arm).name,
        '--program',str(PRIVATE/'agent-compiler-speed-run.py'),'--timeout-seconds','900',
        '--set','protocol',str(ROOT/'protocol.json'),'--set','protocol-sha',digest,'--set','stage',stage,'--set','arm',arm,
        '--env','PYTHONPATH',':'.join(map(str,(PRIVATE,SNAP/'experiments',TINY,SNAP))),
        '--env','CACHEDB',str(CACHE),'--env','TMPDIR',str(SCRATCH),'--env','PYTHONDONTWRITEBYTECODE','1',
        '--env','PARALLEL','32','--env','LLVMOPT','1']
  return result


def verify(protocol_path, digest, inputs=True):
  assert file_sha(protocol_path) == digest
  p = read(protocol_path)
  assert p['timeout_seconds'] == 900 and p['counts'] == {k: list(v) for k,v in COUNTS.items()}
  assert file_sha(ROOT/'source-manifest.json') == p['manifest_sha256']
  m = read(ROOT/'source-manifest.json')
  for name, item in m['files'].items():
    path = ROOT/name
    assert not path.is_symlink() and file_sha(path) == item['sha256'], path
  for group in ('external_pins', 'data_input_pins') if inputs else ('external_pins',):
    for name, item in m[group].items(): assert file_sha(name) == item['sha256'], name
  for name, digest in p['cpu_evidence_sha256'].items(): assert file_sha(name) == digest
  return p


def canonical(value):
  if isinstance(value, dict): return ('dict', tuple((k, canonical(v)) for k,v in sorted(value.items())))
  if isinstance(value, list): return ('list', tuple(map(canonical, value)))
  if isinstance(value, float):
    assert math.isfinite(value), 'Nonfinite canonical result'
    return ('float', struct.pack('<d', value))
  assert value is None or type(value) in (int, str, bool)
  return type(value).__name__, value


def measured(value):
  assert type(value) in (float, int) and math.isfinite(value) and value >= 0
  return '<measured>'


def canonical_row(row):
  value = copy.deepcopy(row)
  for key in ROW_TIMES & value.keys(): value[key] = measured(value[key])
  if 'allocated_bytes_by_device' in value:
    value['allocated_bytes_by_device'] = {k: measured(v) for k,v in value['allocated_bytes_by_device'].items()}
  for key in STAGE_TIMES & value['prediction'].get('timings_ms', {}).keys():
    value['prediction']['timings_ms'][key] = measured(value['prediction']['timings_ms'][key])
  return canonical(value)


def canonical_top(report):
  value = {k: copy.deepcopy(v) for k,v in report.items() if k not in ('warmup', 'runs')}
  for key in TOP_TIMES & value.keys(): value[key] = measured(value[key])
  if 'source_snapshot' in value: value['source_snapshot'] = '<separately hash-verified snapshot>'
  for key in SUMMARY_TIMES & value.get('summary', {}).keys(): value['summary'][key] = measured(value['summary'][key])
  if 'warmup_gc_freeze' in value:
    for key in ('seconds', 'frozen_objects'): value['warmup_gc_freeze'][key] = measured(value['warmup_gc_freeze'][key])
  return canonical(value)


def exact_reports(a, b, sa, sb, stage, samples):
  warm, runs = COUNTS[stage]
  for r,s in ((a,sa),(b,sb)):
    assert len(r['warmup']) == warm and len(r['runs']) == runs
    rows = r['warmup']+r['runs']
    assert [x['sample'] for x in rows] == samples
    assert r['validation'] == [] and r['pad_prompt_to'] == 2048 and r['vision_calibration_sha256'] is None
    assert not r['profiling'] and not r['capture_vision_scales'] and not r['capture_prefill_scales']
    assert r['settings']['diffusion_steps'] == 3 and r['trajectory_solver'] == 'euler' and r['pixel_scale'] == 1
    assert r['summary']['profiling_enabled'] is False and r['summary']['measured_predictions'] == runs
    assert r['summary']['all_inputs_strictly_advance'] and r['summary']['measured_inputs_strictly_advance']
    assert s['queries'] == warm+runs == len(s['records']) and not s['audit'] and s['prefill_attention'] == 'shuffle-fma'
    assert s['padded_prompt_bound'] == 2048
    for row in rows: canonical_row(row)
  for x,y in zip(a['warmup']+a['runs'],b['warmup']+b['runs'],strict=True):
    assert canonical_row(x) == canonical_row(y), f'Canonical prediction/input/history differs: {x["sample"]}'
  assert canonical_top(a) == canonical_top(b), 'Top-level policy/provenance or unknown field differs'
  assert canonical(sa) == canonical(sb), 'Complete speculation/target-verification record differs'
  return {'all_exact': True, 'queries': warm+runs, 'all_canonical_fields': True, 'unknown_fields_compared': True,
          'complete_speculation_records': True, 'finite_float_bits': True, 'row_measurements': sorted(ROW_TIMES)}


def validate_snapshot(model):
  directory = Path(model['source_snapshot'])
  assert directory.is_relative_to(WORK) and directory.parent.name.startswith('agent-compiler-speed-')
  expected = {p.name: file_sha(p) for p in [*(SNAP/'experiments').glob('*.py'), *(SNAP/'experiments').glob('*.c'), SNAP/'model.py']}
  assert model['source_sha256'] == expected
  for name,digest in expected.items(): assert file_sha(directory/name) == digest


def input_map(inventory):
  result = {}
  for row in inventory['compile_records']:
    key = json.dumps([row['input_ast_key'],row['origin'],row['context'],row['parent_renderer']['target']], sort_keys=True)
    result.setdefault(key,set()).add(row['program_key'])
  return result


def origins(inventory):
  kinds = {}
  for row in inventory['compile_records']: kinds.setdefault(row['program_key'],set()).add(row['origin'])
  return kinds


def program_pair(a,b,explicit=False,same=False):
  for field in CONTRACT: assert a[field] == b[field], f'Program contract differs: {field}'
  if same or explicit:
    assert a['key'] == b['key'] and a['binary_sha256'] == b['binary_sha256'], 'Existing PROGRAM binary changed'
  elif a['key'] != b['key']:
    assert b['resources']['workgroup_processor_mode'] == b['resources']['descriptor_wgp_bit'] == 1
  return a['key'] != b['key']


def capture_contract(inventory):
  result = []
  for cap in inventory['captures']:
    c = {k:copy.deepcopy(v) for k,v in cap.items() if k not in ('linear_key','kernels')}
    c['kernels'] = []
    for kernel in cap['kernels']:
      p = inventory['programs'][kernel['program_key']]
      c['kernels'].append({**{k:v for k,v in kernel.items() if k != 'program_key'},
        'program_contract': {k:p[k] for k in CONTRACT}})
    result.append(c)
  return canonical(result)


def compare_warm(a,b,same=False):
  am,bm = input_map(a),input_map(b)
  assert am.keys() == bm.keys(), 'Representative compiler-input set differs'
  changed = 0
  for key in sorted(am):
    assert len(bm[key]) == 1
    for old in am[key]:
      changed += program_pair(a['programs'][old],b['programs'][next(iter(bm[key]))],
        explicit=json.loads(key)[1]=='precompiled_binary',same=same)
  ao,bo = origins(a),origins(b)
  ae = {k for k,v in ao.items() if 'precompiled_binary' in v}
  be = {k for k,v in bo.items() if 'precompiled_binary' in v}
  assert ae == be, 'Representative explicit PROGRAM identities differ'
  assert capture_contract(a) == capture_contract(b), 'Captured order, program contracts, or argument slots differ'
  assert len(a['captures']) and all(bo.get(k) for cap in b['captures'] for k in (x['program_key'] for x in cap['kernels']))
  return {'common_inputs':len(am),'changed':changed,'explicit_programs':len(ae),'explicit_binary_identities_equal':True,
          'capture_order_and_arguments_equal':True,'classification':'Input op and existing BINARY, never source-language guesses'}


def compare_post(a,b):
  assert a['entries'].keys() == b['entries'].keys(), 'Full-run compiler-input identities differ'
  changed = 0
  for key in a['entries']:
    x,y = (r['programs'][r['entries'][key]['program_key']] for r in (a,b))
    changed += program_pair(x,y)
  return {'inputs':len(a['entries']),'changed':changed,'all_input_ast_target_config_keys_equal':True,
    'existing_binary_preservation':'Existing BINARY participates in its input AST key; pinned do_to_program preserves it',
    'origin_scope':'Observed origins are in the warmup inventory; later input kinds are not guessed'}


def checked_result(path,protocol_sha):
  r = read(path)
  assert r['complete'] and r['gpu_executed'] and not r.get('failure')
  assert r['protocol_sha256'] == protocol_sha and r['audit_detached_before_measured_loop']
  assert r['all_hooks_restored'] and r['measured_observer_callbacks'] == 0
  if r['stage']=='native':assert r['amd_cache_entries_added_after_warmup']==0,'New AMD compilation during native measurement'
  for name,digest in r['artifact_sha256'].items(): assert file_sha(name) == digest, name
  return r


def validate_pair(stage,protocol_sha,protocol):
  a,b = [checked_result(metadata_path(stage,arm),protocol_sha) for arm in ARMS]
  assert a['stage'] == b['stage'] == stage and a['arm'] == ARMS[0] and b['arm'] == ARMS[1]
  assert a['runtime']['context'] == b['runtime']['context'] and a['runtime']['environment'] == b['runtime']['environment']
  if stage=='native':
    assert a['quality_gate_sha256_before_run']==b['quality_gate_sha256_before_run']==file_sha(WORK/'agent-compiler-speed-quality-gate.json')
  reports = [read(r['model_report']) for r in (a,b)]
  for r in reports: validate_snapshot(r)
  specs = [read(Path(r['model_report']).with_suffix('.speculation.json')) for r in (a,b)]
  exact = exact_reports(*reports,*specs,stage,protocol['plans'][stage]['scheduled_samples'])
  warm = compare_warm(read(a['warm_inventory']),read(b['warm_inventory']))
  post = compare_post(read(a['post_inventory']),read(b['post_inventory']))
  return {'complete':True,'stage':stage,'protocol_sha256':protocol_sha,'exact':exact,'warm_programs':warm,'post_programs':post,
    'quality_64_qualified':True,'throughput_qualified':stage=='native',
    'result_sha256':{str(metadata_path(stage,arm)):file_sha(metadata_path(stage,arm)) for arm in ARMS},
    'native_throughput': native_summary(*reports) if stage=='native' else None}


def quality_gate(protocol_sha,protocol):
  path = WORK/'agent-compiler-speed-quality-gate.json'
  gate = read(path)
  assert gate == validate_pair('quality',protocol_sha,protocol), 'Quality gate or inputs changed'
  return file_sha(path)


def percentile(values,p):
  xs=sorted(values);x=(len(xs)-1)*p/100;i=int(x)
  return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(x-i)


def native_summary(a,b):
  result={}
  for name,r in zip(ARMS,(a,b),strict=True):
    rows=r['runs'];assert len(rows)==24
    latency=[x['pipeline_ms'] for x in rows];service=[x['service_ms'] for x in rows]
    wall=r['summary']['measured_wall_seconds'];assert wall>0
    assert math.isclose(r['summary']['throughput_fps'],24/wall,rel_tol=1e-12)
    assert math.isclose(r['summary']['median_ms'],statistics.median(latency),rel_tol=1e-12)
    for key,value in {'p95_ms':percentile(latency,95),'max_ms':max(latency),'inverse_mean_latency_fps':1000/statistics.mean(latency),
                      'service_median_ms':statistics.median(service),'service_p95_ms':percentile(service,95)}.items():
      assert math.isclose(r['summary'][key],value,rel_tol=1e-12),key
    for key,values,bound in (('predictions_over_250ms',latency,250),('input_to_output_over_100ms',latency,100),
      ('input_to_output_over_200ms',latency,200),('service_over_100ms',service,100),('service_over_200ms',service,200)):
      assert r['summary'][key]==sum(x>bound for x in values)
    result[name]={'queries':24,'warmup':12,'measured_wall_seconds':wall,'throughput_fps':24/wall,
      'latency_median_ms':statistics.median(latency),'latency_p95_ms':percentile(latency,95),'latency_max_ms':max(latency),
      'service_median_ms':statistics.median(service),'under_100ms_queries':sum(x<100 for x in latency),
      'latency_ms':latency,'service_ms':service}
  result['throughput_change_percent']=(result[ARMS[1]]['throughput_fps']/result[ARMS[0]]['throughput_fps']-1)*100
  result['latency_median_change_percent']=(result[ARMS[1]]['latency_median_ms']/result[ARMS[0]]['latency_median_ms']-1)*100
  result['limit']='One sequential pair; no drift estimate or significance claim. Unpaced native replay, not live arrival cadence.'
  result['boundary']=a['timing_boundary'];assert a['timing_boundary']==b['timing_boundary']
  return result

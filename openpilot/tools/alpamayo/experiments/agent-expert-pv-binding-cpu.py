"""CPU-only proof of renderer serialization and PV compiler-mode divergence."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sqlite3
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  os.environ.update(DEV='CPU', ALLOW_DEVICE_USAGE='0', CCACHE='0', BEAM='0', PROFILE='0', DEBUG='0', TC_OPT='2')
  spec = importlib.util.spec_from_file_location('pv_binding_common', Path(__file__).with_name('agent-expert-pv-binding-common.py'))
  B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
  sys.path.insert(0, str(B.SNAP))
  from tinygrad import Device
  from tinygrad.codegen import to_program_cache, to_program_key
  from tinygrad.helpers import Context, PARALLEL
  result = dict(B.verify(), complete=False, gpu_executed=False, actual_failed_capture_available=False,
                compiler_cases={}, source_sha256={})
  evidence = json.loads((B.ROOT/'evidence/pv-gpu-report.json').read_text())
  renderers = B.renderers()
  assert type(renderers['parent-wgp'].compiler).__name__ == 'WGPCompiler'
  assert type(renderers['worker-cu'].compiler).__name__ == 'AMDLLVMCompiler'
  result['serialization'] = {'before': 'WGPCompiler(-cumode)', 'after': 'AMDLLVMCompiler(+cumode)',
    'renderer_class_preserved': True, 'target_preserved': renderers['parent-wgp'].target == renderers['worker-cu'].target,
    'default_parallel': PARALLEL.value, 'micro_parallel': 0, 'route_parallel_override': None}
  programs = {}
  for case in evidence['cases']:
    shape = tuple(case['shape_b_h_kvh_m_n_k'])
    row = result['compiler_cases'][case['case']] = {}
    for mode, renderer in renderers.items():
      row[mode] = {}
      for label in ('original', '2x2'):
        ast, program = B.compile_program(renderer, shape, label)
        data = B.artifacts(args.output.parent/'agent-expert-pv-binding-compiler', f'{case["case"]}-{mode}-{label}', program, ast)
        row[mode][label] = data
        programs[case['case'], mode, label] = (ast, program)
        if mode == 'parent-wgp':
          for key in ('llvm_sha256', 'binary_sha256'): assert data[key] == case['compiler'][label][key]
        assert data['resources']['wavefront_size'] == 32
        print(json.dumps({'case': case['case'], 'mode': mode, 'label': label,
                          'binary_sha256': data['binary_sha256']}), flush=True)
    for label in ('original', '2x2'):
      a, b = (row[m][label] for m in renderers)
      assert a['input_ast_key'] == b['input_ast_key'] and a['lowered_ast_key'] == b['lowered_ast_key']
      assert a['llvm_sha256'] == b['llvm_sha256'] and a['binary_sha256'] != b['binary_sha256']
      assert a['global_size'] == b['global_size'] and a['local_size'] == b['local_size'] and a['params'] == b['params']
      pa, pb = (programs[case['case'], m, label][1] for m in renderers)
      compiler = B.load('agent-gemm-compile')
      assembly = compiler.disassemble(pa.src[3].arg)
      assert assembly == compiler.disassemble(pb.src[3].arg)
      assert len(pa.src[3].arg) == len(pb.src[3].arg)
      row.setdefault('mode_comparison', {})[label] = {
        'identical_disassembled_instructions': True, 'assembly_sha256': B.sha(assembly.encode()),
        'binary_byte_differences': [{'offset': i, 'wgp': x, 'cu': y}
          for i, (x, y) in enumerate(zip(pa.src[3].arg, pb.src[3].arg, strict=True)) if x != y]}
      assert len(row['mode_comparison'][label]['binary_byte_differences']) == 2
  ast, wgp = programs['expert', 'parent-wgp', 'original']
  _, cu = programs['expert', 'worker-cu', 'original']
  key = to_program_key(ast, renderers['parent-wgp'])
  assert key == to_program_key(ast, renderers['worker-cu'])
  result['in_memory_cache_compiler_collision'] = True
  # Reproduce the actual compiler worker task boundary without opening a device.
  from tinygrad.engine.realize import _compile_kernel
  from tinygrad.codegen import to_program_context
  ctx = {v.key: v.value for v in to_program_context}
  to_program_cache.clear()
  task = pickle.loads(pickle.dumps((0, (ast, renderers['parent-wgp']), ctx)))
  index, replay = _compile_kernel(task)
  assert index == 0 and replay.src[3].arg == cu.src[3].arg and replay.src[2].arg == wgp.src[2].arg
  result['serialized_worker_task_reproduces_cu_binary'] = True
  # Read the existing shared cache, never update it or derive approval from it.
  cache = Path('/tmp/opencode/alpamayo-asius-20261114/cache.db')
  with sqlite3.connect('file:'+str(cache)+'?mode=ro', uri=True) as db:
    db.execute('PRAGMA query_only=ON')
    matches = []
    for table, in db.execute("select name from sqlite_master where type='table' and name like 'compile_llvm_gfx1200%'"):
      rows = db.execute('select val from "'+table+'" where key=?', (wgp.src[2].arg,)).fetchall()
      for raw, in rows:
        value = pickle.loads(raw)
        matches.append({'table': table, 'binary_sha256': B.sha(value), 'matches_cpu_wgp': value == wgp.src[3].arg,
                        'matches_cpu_cu': value == cu.src[3].arg})
  result['readonly_existing_cache_exact_ir_matches'] = matches
  assert any(v['matches_cpu_cu'] for v in matches) and any(v['matches_cpu_wgp'] for v in matches)
  failure = B.WORK/'agent-expert-pv-route-quality-reference'
  failed = json.loads((failure/'report.agent-expert-pv.json').read_text())
  assert not failed['route_completed'] and not failed['exactness_validated'] and failed['hooks'] == []
  assert failed['failure']['type'] == 'AssertionError'
  result['failure_evidence_sha256'] = {str(p): B.sha(p.read_bytes()) for p in (
    failure/'run.log', failure/'wall.json', failure/'report.agent-expert-pv.json', B.ROOT/'queue-results.json')}
  result['failure_metadata_lost_actual_hash'] = True
  result['actual_route_binary_unverified'] = True
  result['candidate_route_equality_established'] = False
  result['source_sha256'] = {str(p): B.sha(p.read_bytes()) for p in Path(__file__).parent.glob('agent-expert-pv-binding-*.py')}
  assert not Device._opened_devices
  result.update(complete=True, parent_wgp_reproduces_passed_micro=True, compiled_shapes=3, compiled_programs=12)
  B.write(args.output, result)


if __name__ == '__main__': main()

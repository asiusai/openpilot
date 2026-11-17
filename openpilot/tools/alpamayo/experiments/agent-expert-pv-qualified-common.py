"""Independent pins for the qualified USB PV route experiment."""
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
SNAP = WORK/'agent-expert-pv-route-source/alpamayo'
ROOT = WORK/'agent-expert-pv-qualified-source'
MICRO = WORK/'agent-expert-pv-usb-kernels/report.json'
MICRO_SHA = '23f7fab3b18db190984249a33b508d6ff95d3a0c3b80f425422c5a9144957a67'
USB_PROTOCOL = WORK/'agent-expert-pv-usb-protocol.json'
USB_SHA = '0d9a8bb330e7a65df7bec22d4ba639944e940bc2ffffda8d1d0a5d7f91db321c'
SHAPE = (1, 16, 8, 64, 128, 2368)
CURRENT_KEY = '9e5a8d0446a8b99e397e91a550d189a9bdab0b6540bbc38832f3c86c2fce01cc'
CANDIDATE_KEY = '60a9c413ed3944c9daa71121b6189e68005ae4296f7bc356adbf02295e1bbe59'


def sha(data): return hashlib.sha256(data).hexdigest()


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  result = importlib.util.module_from_spec(spec); sys.modules[name] = result
  spec.loader.exec_module(result)
  return result


@lru_cache(None)
def local(name): return load(HERE/(name+'.py'), name.replace('-', '_'))


@lru_cache(None)
def usb():
  assert sha(USB_PROTOCOL.read_bytes()) == USB_SHA
  protocol = json.loads(USB_PROTOCOL.read_text())
  for path, digest in protocol['files'].items(): assert sha(Path(path).read_bytes()) == digest, path
  return load(WORK/'agent-expert-pv-usb-source/agent-expert-pv-usb-common.py', 'qualified_frozen_usb')


def paths():
  # runpy does not add the script directory. Benchmark imports wgp_compiler by name.
  sys.path[:0] = [str(SNAP/'experiments'), str(SNAP)]
  module = importlib.import_module('wgp_compiler')
  assert Path(module.__file__).resolve() == SNAP/'experiments/wgp_compiler.py'
  expected = json.loads((SNAP.parent/'source-manifest.json').read_text())['files']['alpamayo/experiments/wgp_compiler.py']['sha256']
  assert sha(Path(module.__file__).read_bytes()) == expected
  return module


def qualification():
  U = usb()
  _, provenance, capture, cpu = U.preflight(USB_PROTOCOL, USB_SHA)
  assert sha(MICRO.read_bytes()) == MICRO_SHA, 'Qualified GPU report changed'
  report = json.loads(MICRO.read_text())
  for key in ('complete', 'gpu_executed', 'arithmetic_qualified', 'micro_full_bit_controls_passed',
              'saved_actual_program_reconstructed_exactly', 'all_12_usb_programs_observed'):
    assert report[key] is True, key
  assert not report.get('failure') and report['model_predictions'] == 0 and not report['numerical_candidate']
  assert not report['route_equality_established'] and not report['timings_valid']
  assert len(report['cases']) == 3 and len(report['executed_bindings']) == 12
  modes = ['parent-wgp/original', 'parent-wgp/2x2', 'worker-cu/original', 'worker-cu/2x2']
  states = ['base', 'a-only', 'b-only-after-a', 'edges', 'restore', 'replay', 'post']
  for case in report['cases']:
    assert case['complete'] and len(case['checks']) == 28
    assert {(c['state'], c['program']) for c in case['checks']} == {(s, m) for s in states for m in modes}
    assert all(c['exact_bf16_bits'] for c in case['checks'])
  expert = next(c for c in report['cases'] if c['case'] == 'expert')
  assert tuple(expert['shape']) == SHAPE and expert['masked_history_value_mutation_exact']
  assert expert['final_fresh_replay_and_readback_exact']
  assert set(expert['serial_40_call_proofs']) == set(modes)
  for proof in expert['serial_40_call_proofs'].values():
    assert proof['captured_calls'] == 40 and proof['ordered_execution_barriers'] == 39 and proof['output_input_slot'] == 2
  assert [s['state'] for s in expert['captured_40_call_states']] == states
  assert all(s['programs'] == modes and s['exact_bf16_bits'] for s in expert['captured_40_call_states'])
  current, candidate = (report['executed_bindings'][k] for k in (CURRENT_KEY, CANDIDATE_KEY))
  assert current == capture['actual_reference'] == cpu['compiler_cases']['expert']['worker-cu']['original']
  assert candidate == cpu['compiler_cases']['expert']['parent-wgp']['2x2']
  for name, row in cpu['compiler_cases'].items():
    for mode in U.MODES:
      for label in U.LABELS:
        expected = row[mode][label]
        assert report['executed_bindings'][expected['program_key']] == expected
  return {'micro_report_sha256': MICRO_SHA, 'usb_protocol_sha256': USB_SHA, 'target': U.TARGET,
          'actual_capture_sha256': U.CAPTURE_SHA, 'current': current, 'candidate': candidate,
          'compiler_context': report['runtime']['compiler_context'], 'parallel': report['runtime']['parallel'],
          'provenance': provenance}


def prepare(evidence):
  U, B = usb(), usb().old()
  renderers = U.renderers()
  result = {}
  for arm, mode, label in (('current', 'worker-cu', 'original'), ('candidate', 'parent-wgp', '2x2')):
    ast, program = U.compile_program(renderers[mode], SHAPE, label)
    U.validate_description(B.describe(program, ast), evidence[arm])
    result[arm] = {'ast': ast, 'program': program}
  return result


def runtime(renderer, evidence):
  from tinygrad.codegen import to_program_context
  from tinygrad.helpers import PARALLEL, Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  expected = paths().WGPCompiler
  assert type(renderer) is AMDLLVMRenderer and renderer.target == Target.parse(evidence['target'])
  compiler = type(renderer.compiler)
  assert compiler.__name__ == 'WGPCompiler' and renderer.compiler.arch == 'gfx1200'
  assert compiler.__init__.__code__ == expected.__init__.__code__ and compiler.__reduce__.__code__ == expected.__reduce__.__code__
  assert {v.key: str(v.value) for v in to_program_context} == evidence['compiler_context']
  assert PARALLEL.value == evidence['parallel'] == 32


def protocol(path, digest):
  assert sha(path.read_bytes()) == digest, 'Qualified route protocol changed'
  data = json.loads(path.read_text())
  for source, expected in data['files'].items(): assert sha(Path(source).read_bytes()) == expected, source
  assert data['micro_report_sha256'] == MICRO_SHA and data['policy'] == 'PV only; dynamic vision; fixed language; pad2048; target-verified NOMINAL; shuffle-FMA; full inputs; 3 Euler'
  return data, qualification()

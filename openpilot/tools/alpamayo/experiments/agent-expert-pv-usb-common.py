"""Exact USB target binding to saved route evidence. No model or kernel changes."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
CAPTURE = WORK/'agent-expert-pv-binding-kernels-pathfix'
STEM = CAPTURE/'agent-expert-pv-binding-captured/agent-expert-pv-binding-actual-reference'
OLD_PROTOCOL = WORK/'agent-expert-pv-binding-protocol-v2.json'
OLD_PROTOCOL_SHA = '04003b2cb8505249d11e88854c70677cf8c87dd2a755b44e4ead605792c6648e'
CAPTURE_SHA = 'fcc0cdce1d32143680cbd907c44d964cbbafe73c6661ae332a602ca2b8b62fa2'
TARGET = 'USB+AMD:LLVM:gfx1200'
MODES = ('parent-wgp', 'worker-cu')
LABELS = ('original', '2x2')


def sha(data): return hashlib.sha256(data).hexdigest()


@lru_cache(None)
def old():
  assert sha(OLD_PROTOCOL.read_bytes()) == OLD_PROTOCOL_SHA
  protocol = json.loads(OLD_PROTOCOL.read_text())
  for path, digest in protocol['new_source_sha256'].items(): assert sha(Path(path).read_bytes()) == digest, path
  for name in ('cpu_report', 'cpu_check_report'):
    assert sha(Path(protocol[name]).read_bytes()) == protocol[name+'_sha256']
  path = WORK/'agent-expert-pv-binding-source-v2/agent-expert-pv-binding-common.py'
  spec = importlib.util.spec_from_file_location('pv_usb_frozen_common', path)
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  sys.path[:0] = [str(module.SNAP/'experiments'), str(module.SNAP)]
  return module


def load(name): return old().load(name)


def validate_description(actual, expected):
  assert set(actual) == set(expected), 'Program fields changed'
  for field in expected: assert actual[field] == expected[field], 'Program field changed: '+field
  assert actual['target'] == TARGET, 'Exact USB target required'


def validate_renderer(renderer, mode):
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  from tinygrad.runtime.support.compiler_llvm import AMDLLVMCompiler
  assert type(renderer) is AMDLLVMRenderer, 'Unexpected renderer class'
  assert renderer.target == Target.parse(TARGET) and str(renderer.target) == TARGET, 'Unexpected full target'
  expected = load('wgp_compiler').WGPCompiler if mode == 'parent-wgp' else AMDLLVMCompiler
  assert mode in MODES and type(renderer.compiler) is expected, 'Unexpected compiler class'
  assert renderer.compiler.arch == 'gfx1200', 'Unexpected compiler architecture'


def renderers():
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  parent = AMDLLVMRenderer(Target.parse(TARGET))
  parent.compiler = load('wgp_compiler').WGPCompiler('gfx1200')
  worker = pickle.loads(pickle.dumps(parent))
  result = dict(zip(MODES, (parent, worker), strict=True))
  for mode, renderer in result.items(): validate_renderer(renderer, mode)
  return result


def verify_capture():
  B = old()
  provenance = B.verify()  # Keep the 953 input, checkpoint metadata, and source pins.
  assert sha((CAPTURE/'report.json').read_bytes()) == CAPTURE_SHA
  report = json.loads((CAPTURE/'report.json').read_text())
  assert report['complete'] is False and report['micro_full_bit_controls_passed'] is False
  assert report['gpu_executed'] and not report['candidate_route_equality_established']
  assert report['failure'] == {'type': 'AssertionError', 'message': 'Captured program differs: target'}
  assert report['runtime']['renderer_target'] == TARGET and report['runtime']['renderer_class'] == 'AMDLLVMRenderer'
  assert report['runtime']['compiler_class'] == 'WGPCompiler' and report['runtime']['parallel'] == 32
  expected = report['actual_reference']
  assert json.loads(STEM.with_suffix('.json').read_text()) == expected
  assert expected['target'] == TARGET and expected['binary_sha256'] == '66dd48e6cd81db1e0397f4cba8268e3da6037a8a81c02f6c1325d896d1ef49b4'
  assert sha(STEM.with_suffix('.hsaco').read_bytes()) == expected['binary_sha256']
  assert sha(STEM.with_suffix('.ll').read_bytes()) == expected['llvm_sha256']
  key = expected['program_key']
  assert report['captured_pv_program_keys'] == [key]
  assert all(report['capture_counts'][name] == 84 for name in (
    'alpamayo_attention_values', 'alpamayo_attention_scores', 'alpamayo_attention_softmax'))
  calls = report['captured_pv_call_arguments']
  assert len(calls) == 84 and all(c == {'program_key': key, 'global_input_slots': [],
    'output_argument_indices': [0], 'input_argument_indices': [1, 2]} for c in calls)
  assert len(report['input_asts']) == 1 and report['input_asts'][0]['shape'] == list(B.SHAPE)
  return provenance, report


def compile_program(renderer, shape, label):
  B = old()
  mode = next((m for m in MODES if type(renderer.compiler).__name__ == ('WGPCompiler' if m == 'parent-wgp' else 'AMDLLVMCompiler')), None)
  validate_renderer(renderer, mode)
  return B.compile_program(renderer, shape, label)


def captured_program(ast, program, capture):
  B = old()
  validate_description(B.describe(program, ast), capture['actual_reference'])
  assert str(ast)+'\n' == STEM.with_suffix('.input-ast.txt').read_text(), 'Saved input AST changed'
  assert str(program.src[0])+'\n' == STEM.with_suffix('.lowered-ast.txt').read_text(), 'Saved lowered AST changed'
  source, binary = STEM.with_suffix('.ll').read_text(), STEM.with_suffix('.hsaco').read_bytes()
  assert program.src[2].arg == source and program.src[3].arg == binary
  # Reload saved bytes, preserving every PROGRAM field and its exact USB key.
  result = program.replace(src=(*program.src[:2], program.src[2].replace(arg=source), program.src[3].replace(arg=binary)))
  assert result.key.hex() == capture['actual_reference']['program_key']
  return result


def elf(binary):
  from tinygrad.runtime.support.elf import elf_loader
  _, sections, relocs = elf_loader(binary)
  assert binary[:6] == b'\x7fELF\x02\x01'
  # The existing object has one relative kernel-descriptor relocation.
  text = next(s for s in sections if s.name == '.text')
  assert relocs == [(len(text.content)+16, 0, 5, 16)], relocs
  return {'class': 64, 'endianness': 'little', 'relocations': [list(r) for r in relocs],
          'sections': {s.name: {'size': len(s.content), 'sha256': sha(s.content)} for s in sections}}


def preflight(protocol_path, digest):
  assert sha(protocol_path.read_bytes()) == digest, 'Protocol changed'
  protocol = json.loads(protocol_path.read_text())
  assert protocol['maximum_model_predictions'] == 0 and protocol['target'] == TARGET
  for path, expected in protocol['files'].items(): assert sha(Path(path).read_bytes()) == expected, 'Pinned file changed: '+path
  provenance, capture = verify_capture()
  cpu = json.loads(Path(protocol['cpu_report']).read_text())
  assert cpu['complete'] and not cpu['gpu_executed'] and cpu['all_actual_capture_fields_exact']
  assert cpu['target'] == TARGET and cpu['negative_tests_passed']
  host = json.loads(Path(protocol['host_report']).read_text())
  assert host['complete'] and not host['gpu_executed'] and host['model_predictions'] == 0
  assert host['runtime_usb_program_observed_exactly'] and host['runtime_native_target_rejected']
  assert host['observer_cleanup_exact']
  for result in (cpu, host):
    for path, expected in result['source_sha256'].items(): assert sha(Path(path).read_bytes()) == expected, path
  return protocol, provenance, capture, cpu

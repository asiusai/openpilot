"""Read-only anchors and compiler controls for the failed PV route binding."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
ROOT = WORK/'agent-expert-pv-route-source'
SNAP = ROOT/'alpamayo'
TINY = Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
QUEUE = Path('/tmp/opencode/queue-alpamayo-expert-pv-route.py')
QUEUE_SHA = '8fbb9c5daa5eceb499fa852201a8537593b7e193f104a0f39c5f4ccd148775a9'
CPU = WORK/'agent-expert-pv-binding-cpu.json'
SHAPE = (1, 16, 8, 64, 128, 2368)


def sha(value): return hashlib.sha256(value).hexdigest()


@lru_cache(None)
def load(name, frozen=True):
  path = (SNAP/'experiments' if frozen else HERE)/(name+'.py')
  spec = importlib.util.spec_from_file_location('pv_binding_'+name.replace('-', '_'), path)
  value = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = value
  spec.loader.exec_module(value)
  return value


def write(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify():
  assert sha(QUEUE.read_bytes()) == QUEUE_SHA
  spec = importlib.util.spec_from_file_location('pv_binding_frozen_queue', QUEUE)
  queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
  manifest = queue.verify_sources()
  return {'frozen_manifest_sha256': sha(queue.MANIFEST.read_bytes()), 'frozen_queue_sha256': QUEUE_SHA,
          'source_and_metadata_files': len(manifest['files']), 'external_pins': len(manifest['external_pins']),
          'input_pins': len(manifest['data_input_pins']), 'passed_micro_report_sha256': load('agent-expert-pv-route-hook').REPORT_SHA}


def renderers():
  import pickle
  from tinygrad.helpers import Target
  from tinygrad.renderer.llvmir import AMDLLVMRenderer
  original = AMDLLVMRenderer(Target('AMD', 'LLVM', 'gfx1200'))
  original.compiler = load('wgp_compiler').WGPCompiler('gfx1200')
  # Exactly the serialization path used by tinygrad's compilation worker pool.
  worker = pickle.loads(pickle.dumps(original))
  return {'parent-wgp': original, 'worker-cu': worker}


def ast_for(renderer, shape, label):
  from tinygrad import dtypes
  from tinygrad.uop.ops import UOp
  compiler = load('agent-expert-pv-compile')
  args = [UOp.placeholder(s, dtypes.bfloat16, i) for i, s in enumerate(compiler.specs(shape))]
  return compiler.functions(renderer, shape)[label](*args)


def compile_program(renderer, shape, label):
  # Bypass the context-blind in-memory cache. This never opens a GPU device.
  from tinygrad.codegen import do_to_program
  ast = ast_for(renderer, shape, label)
  return ast, do_to_program(ast, renderer)


def describe(program, ast=None):
  compiler = load('agent-gemm-compile')
  source, binary = program.src[2].arg, program.src[3].arg
  return {'name': program.src[0].arg.function_name, 'program_key': program.key.hex(),
          'input_ast_key': None if ast is None else ast.key.hex(), 'lowered_ast_key': program.src[0].key.hex(),
          'llvm_sha256': sha(source.encode()), 'binary_sha256': sha(binary),
          'global_size': list(program.arg.global_size), 'local_size': list(program.arg.local_size),
          'target': str(program.arg.target), 'resources': compiler.metadata(binary),
          'params': [{'slot': p.arg.slot, 'dtype': str(p.dtype), 'shape': list(p.max_shape),
                      'addrspace': str(p.arg.addrspace)} for p in program.kernel_params],
          'program_info': repr(program.arg)}


def artifacts(directory, name, program, ast=None):
  directory.mkdir(parents=True, exist_ok=True)
  stem = directory/('agent-expert-pv-binding-'+name)
  stem.with_suffix('.ll').write_text(program.src[2].arg)
  stem.with_suffix('.hsaco').write_bytes(program.src[3].arg)
  stem.with_suffix('.lowered-ast.txt').write_text(str(program.src[0])+'\n')
  if ast is not None: stem.with_suffix('.input-ast.txt').write_text(str(ast)+'\n')
  result = describe(program, ast)
  write(stem.with_suffix('.json'), result)
  return result

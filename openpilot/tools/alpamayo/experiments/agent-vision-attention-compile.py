"""Compile ONE CK specialization offline. No device access or dependency changes."""
import ast, ctypes, hashlib, json, os, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

EXP=Path(__file__).parent
ROOT=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116/agent-vision-attention-artifacts')
CK=EXP/'agent-ck-attention-work/ck'
HEADERS=EXP/'agent-ck-attention-v2'
TINY=Path('/tmp/opencode/alpamayo-asius-20261114/tinygrad')
os.environ.update(CPU='1', AMD='0', DEV='CPU', CACHEDB=str(ROOT/'compiler-cache.db'), TMPDIR=str(ROOT))
sys.path.insert(0,str(TINY))
from tinygrad.runtime.support.compiler_amd import comgr, check, set_options, _get_comgr_data

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def reviewed_function(filename, name):
  """Reuse a pure, reviewed function; do not execute legacy import-time path settings."""
  tree=ast.parse((EXP/filename).read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  space=dict(globals());exec(compile(ast.Module(body=[fn],type_ignores=[]),str(EXP/filename),'exec'),space)
  return space[name]

def main():
  ROOT.mkdir(exist_ok=False)
  # Reject hardware opens even if an imported helper changes later.
  from tinygrad import Device
  def blocked(*a,**k): raise AssertionError('CPU-only compiler: device open forbidden')
  type(Device).__getitem__=blocked
  sys.path.insert(0,str(CK/'example/ck_tile/01_fmha'))
  from codegen.ops.fmha_fwd import get_fwd_blobs
  name=('fmha_fwd_d128_bf16_batch_b64x64x32x128x32x128_r4x1x1_r4x1x1'
        '_w16x16x16_w16x16x16_qr_hpad_vr_psskddv_nlogits_nbias_nmask_lse_ndropout_nskip_nqscale_ntrload_nsink')
  pool,kernels=get_fwd_blobs(['gfx1200'],name,2,[128],'simplified')
  assert len(kernels)==1, (name,len(kernels))
  selected=kernels[0]
  (ROOT/'generator-reference.cpp').write_text(selected.render())
  (ROOT/'dispatcher-reference.txt').write_text(pool.render('vision_only'))
  (ROOT/'selection.json').write_text(json.dumps(asdict(selected),indent=2)+'\n')
  # Reuse the reviewed COMGR pipeline while redirecting ONLY its output paths.
  # Every include/manifest still uses the genuine immutable HIP 6.4 header root.
  path=EXP/'agent-ck-attention-v2-compile.py';tree=ast.parse(path.read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compile_source')
  class OutputPath(ast.NodeTransformer):
    def visit_BinOp(self,node):
      node=self.generic_visit(node)
      if isinstance(node.left,ast.Name) and node.left.id=='ROOT' and isinstance(node.op,ast.Div) and isinstance(node.right,ast.Name) and node.right.id=='name':
        node.left.id='OUTPUT'
      return node
  fn=OutputPath().visit(fn);ast.fix_missing_locations(fn)
  space=dict(globals(),ROOT=HEADERS,OUTPUT=ROOT)
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),space)
  record=space['compile_source'](EXP/'agent-vision-attention-native.cpp','candidate')
  assert record['status']=='compiled_not_executed', record
  inspect(record)

def inspect(record):
  build=ROOT/'candidate';blob=(build/'kernel.hsaco').read_bytes()
  assert hashlib.sha256(blob).hexdigest()==record['binary_sha256']
  assert sha(EXP/'agent-vision-attention-native.cpp')==record['source_sha256']
  action=reviewed_function('agent-ck-attention-v2-inspect.py','comgr_action')
  (build/'disassembly.txt').write_bytes(action(blob,comgr.AMD_COMGR_DATA_KIND_EXECUTABLE,
    comgr.AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE,comgr.AMD_COMGR_DATA_KIND_SOURCE,[]))
  (build/'readelf.txt').write_text(subprocess.check_output(['llvm-readelf-18','--all',str(build/'kernel.hsaco')],text=True))
  from tinygrad.runtime.ops_amd import _amd_program_image
  dev=SimpleNamespace(target=(12,0,0),iface=SimpleNamespace(props={'lds_size_in_kb':64}))
  data,image=_amd_program_image(dev,blob)
  (build/'loader.json').write_text(json.dumps(asdict(data),indent=2)+'\n')
  pins={str(p):sha(p) for p in [Path(__file__),EXP/'agent-vision-attention-native.cpp',EXP/'agent-ck-attention-v2-compile.py',
    EXP/'agent-ck-attention-v2-inspect.py',CK/'LICENSE',HEADERS/'header-manifest.json',HEADERS/'resource-header-manifest.json']}
  (ROOT/'direct-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
  print(json.dumps(asdict(data),indent=2))

if __name__=='__main__':
  if sys.argv[1:]==['--inspect']: inspect(json.loads((ROOT/'candidate/record.json').read_text()))
  else: main()

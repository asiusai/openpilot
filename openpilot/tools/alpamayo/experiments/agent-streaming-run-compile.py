"""Offline gfx1200 compilation only. Opens no AMD or USB device."""
import importlib.util,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stream_compile_common',HERE/'agent-streaming-run-common.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
common.verify_sources();common.prepare_cpu_environment()
from tinygrad import Device,dtypes
from tinygrad.uop.ops import UOp,Ops
from tinygrad.codegen import to_program
from tinygrad.renderer.llvmir import AMDLLVMRenderer
from tinygrad.runtime.support.compiler_llvm import LLVMCompiler
from tinygrad.device import Target
kernels=common.load('agent-streaming-run-kernels')

def main():
  report=common.report_base('offline-gfx1200');report.update(passed=False,kernels=[])
  folder=common.ROOT/'offline-gfx1200';folder.mkdir(exist_ok=True)
  # All buckets possible for 0..8 reused 180-token images on this route.
  shapes=[(17,33)]+[(rows,2048) for rows in (640,832,960,1152,1344,1536,1664,1856,2048)]
  for rows,keys in shapes:
    text,binary=kernels.compiled(16,8,rows,keys)
    name=f'slot-q{rows}-k{keys}';(folder/(name+'.cpp')).write_text(text);(folder/(name+'.co')).write_bytes(binary)
    baseline,_=kernels.selected().compiled(16,8,rows,keys,True,128,8,True,'shuffle-fma')
    assert '__builtin_amdgcn_wmma' in text and '__builtin_amdgcn_wmma' in baseline
    # The complete accumulation body after slot masking must be byte-identical.
    begin='    for(int sub=0;sub<2;sub++) scores[sub]=transpose_probability(scores[sub],lane);'
    end='  for(int i=0;i<8;i++) {\n    int row=tile*16+(lane/16)*8+i;'
    assert text[text.index(begin):text.index(end)]==baseline[baseline.index(begin):baseline.index(end)]
    args=[UOp.placeholder(shape,dtype,slot=i) for i,(shape,dtype) in enumerate([
      ((1,16,rows,128),dtypes.bfloat16),((1,16,rows,128),dtypes.bfloat16),
      ((1,8,keys,128),dtypes.bfloat16),((1,8,keys,128),dtypes.bfloat16),((rows,),dtypes.int32),((2,),dtypes.int32)])]
    bound=kernels.program(*args)
    params={u.arg.slot for u in bound.src[0].toposort() if u.op is Ops.PARAM}
    assert params==set(range(6)),params
    report['kernels'].append(dict(name=name,bytes=len(binary),source_sha256=common.sha(folder/(name+'.cpp')),
      accumulation_body_identical=True,all_six_arguments_in_dependency_graph=True))
  renderer=AMDLLVMRenderer(Target('AMD','LLVM','gfx1200'))
  renderer.compiler=LLVMCompiler('AMDGPU','gfx1200','-cumode')
  def placeholder(shape,index,dtype=dtypes.bfloat16):return UOp.placeholder(shape,dtype,slot=index)
  cases=[('gather',kernels.gather_ast,[((1,1024,2048),None),((1,2048,2048),None),((1024,),dtypes.int32)]),
    ('rope-gather',kernels.gather_ast,[((1,1,1024,128),None),((1,1,2048,128),None),((1024,),dtypes.int32)]),
    ('assemble',kernels.assemble_ast,[((2,1,8,2048,128),None),((2,1,8,1024,128),None),((2,1,8,8640,128),None),
      ((2048,),dtypes.int32),((2048,),dtypes.int32)]),
    ('scatter',kernels.scatter_ast,[((2,1,8,8640,128),None),((2,1,8,1800,128),None),((1800,),dtypes.int32)])]
  for name,build,shapes in cases:
    ast=build(*(placeholder(shape,i,dtype or dtypes.bfloat16) for i,(shape,dtype) in enumerate(shapes)))
    program=to_program(ast,renderer)
    text,binary=program.src[2].arg,program.src[3].arg
    (folder/(name+'.ll')).write_text(text);(folder/(name+'.co')).write_bytes(binary)
    report['kernels'].append(dict(name=name,bytes=len(binary),source_sha256=common.sha(folder/(name+'.ll'))))
  report['opened_devices']=sorted(Device._opened_devices)
  assert not any('AMD' in d or 'USB' in d for d in report['opened_devices'])
  report.update(passed=True,limit='Compiler validation only; no GPU launch, numerical parity, race freedom or performance proof')
  (common.ROOT/'offline-compile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

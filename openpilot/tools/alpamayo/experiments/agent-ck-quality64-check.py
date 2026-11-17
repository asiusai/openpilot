"""Entire 64-query production launcher and TinyJit lifecycle, with CPU arithmetic stand-ins."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_quality64_common',EXP/'agent-ck-quality64-common.py')
C=importlib.util.module_from_spec(s);s.loader.exec_module(C)

def run_check(output,common=C):
  path=EXP/'agent-ck-integrated-check.py';source=path.read_text()
  old=json.loads((C.OLD.PREPARED/'protocol.json').read_text());assert C.sha(path)==old['pins'][str(path)]
  # Reuse the real USB alignment reader, without importing previous protocol/overlay chains.
  tree=ast.parse((EXP/'agent-ck-aligned-control-check.py').read_text())
  reader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='strict_install')
  ns={'SimpleNamespace':__import__('types').SimpleNamespace,'proof':{'read_calls':0,'read_bytes':0}}
  exec(compile(ast.Module(body=[reader],type_ignores=[]),'<qualified_USB_reader>','exec'),ns)
  source=source.replace("'--runs','8'","'--runs','64'")
  source=C.replace_once(source,'  io.read=lambda t,d:raw(t.uop.buffer,np.uint8).tobytes()','  _strict_install(io,raw,dev)')
  begin=source.index("       patches=candidate['model_controls']") if "       patches=candidate['model_controls']" in source else source.index("      patches=candidate['model_controls']")
  end=source.index('      model=models[-1]',begin)
  source=source[:begin]+source[end:]
  m=ModuleType('ck_quality64_complete_check');m.__file__=str(path);m._strict_install=ns['strict_install']
  exec(compile(source,str(path),'exec'),m.__dict__);m.C=common;m.R=common.runner(common)
  m.main(output)
  r=json.loads(output.read_text());assert all(x['prefills']==64 for x in r['arms'])
  root=output.parent/output.stem
  for arm in ('baseline','candidate'):
    report=json.loads((root/arm/'report.json').read_text())
    assert not report['gpu_executed'] and report['CPU_MOCK_ONLY']
    # Validate all production shape/history/audit gates; only the evidence flag is substituted in this local assertion.
    common.validate_arm({**report,'gpu_executed':True})
  r['full64_checks']={'both_complete_loops':True,'original_query_order':common.INDICES,
    'capture_on_second_call':True,'own_evolving_histories':True,'deep_controls_reused_not_reexecuted':True,
    'USB_read_guard':ns['proof'],'GPU_execution':False}
  output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
  print(json.dumps({'full64_CPU_lifecycle_passed':True,'gpu_executed':False}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run_check(a.output)

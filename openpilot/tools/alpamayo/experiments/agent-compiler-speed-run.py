"""One frozen parent-only run. Audit warmup; detach every observer before timing."""
import argparse
import builtins
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

C = importlib.import_module('agent-compiler-speed-common')


def hooks():
  from tinygrad.engine import realize,jit
  from tinygrad.runtime.support import hcq2
  return realize.lower_and_compile,hcq2.lower_and_compile,hcq2._build_queues,jit._TinyJit.__call__


def save_program(program,directory):
  P = importlib.import_module('agent-compiler-mode-proof')
  I = importlib.import_module('agent-compiler-mode-inventory')
  row=P.describe(program)
  row.update(name=program.src[0].arg.function_name,target=repr(program.arg.target),resources=I.metadata(program.src[3].arg),
    params=[{'slot':p.arg.slot,'dtype':str(p.dtype),'shape':list(p.max_shape),'addrspace':str(p.arg.addrspace)} for p in program.kernel_params],
    queue_occurrences=0)
  directory.mkdir(parents=True,exist_ok=True)
  stem=directory/('agent-compiler-speed-'+row['key'])
  for suffix,data in (('.source.txt',program.src[2].arg.encode()),('.hsaco',program.src[3].arg)):
    path=stem.with_suffix(suffix)
    if path.exists(): assert path.read_bytes()==data
    else: path.write_bytes(data)
  return row


def audit_class():
  I=importlib.import_module('agent-compiler-mode-inventory')
  class Audit(I.Inventory):
    callbacks=0
    def record(self,program):
      from tinygrad.uop.ops import Ops
      if program.op is not Ops.PROGRAM or program.arg.target.device!='AMD':return None
      key=program.key.hex()
      if key not in self.programs:self.programs[key]=save_program(program,self.directory)
      return key
    def before_after_compile(self,*a,**kw):
      self.callbacks+=1
      return super().before_after_compile(*a,**kw)
    def build(self,*a,**kw):
      self.callbacks+=1
      return super().build(*a,**kw)
    def captured(self,*a,**kw):
      self.callbacks+=1
      return super().captured(*a,**kw)
    def release(self):
      self.programs.clear();self.compile_records.clear();self.captures.clear();self._captures_seen.clear()
  return Audit


def full_cache_inventory(directory):
  from tinygrad.codegen import to_program_cache
  result={'entries':{},'programs':{},'scope':'Host compiler cache after all queries; no GPU reads or additional executions'}
  for key,program in to_program_cache.items():
    if key[2].device!='AMD':continue
    identity=json.dumps([key[0].hex(),repr(key[2]),*key[3:]],sort_keys=True)
    assert identity not in result['entries'], 'Multiple renderer namespaces used during one arm'
    row=save_program(program,directory)
    result['entries'][identity]={'program_key':row['key'],'renderer_class':key[1].__module__+'.'+key[1].__name__}
    result['programs'][row['key']]=row
  assert result['entries']
  return result


def boundary(original_print,filename,warmup,finish):
  def emit(*values,**kwargs):
    original_print(*values,**kwargs)
    caller=sys._getframe(1)
    if caller.f_code.co_filename!=str(filename) or caller.f_code.co_name!='main':return
    if not values or not isinstance(values[0],str) or not values[0].startswith(f'Iteration {warmup-1}: '):return
    assert caller.f_locals['iteration']==warmup-1
    assert len(caller.f_locals['report']['warmup'])==warmup and caller.f_locals['report']['runs']==[]
    assert caller.f_locals['measured_begin'] is None
    # Both selected recipes suppress prefetch of the first measured input.
    assert caller.f_locals['pending'] is None
    finish()
  return emit


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True)
  p.add_argument('--stage',choices=C.COUNTS,required=True);p.add_argument('--arm',choices=C.ARMS,required=True)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args();assert not sys.flags.optimize
  protocol=C.verify(args.protocol,args.protocol_sha)
  assert args.output==C.metadata_path(args.stage,args.arm) and not args.output.exists()
  args.output.parent.mkdir(parents=True,exist_ok=True)
  gate_sha=C.quality_gate(args.protocol_sha,protocol) if args.stage=='native' else None
  if args.stage=='quality' and args.arm=='consistent-wgp':
    C.checked_result(C.metadata_path('quality','original'),args.protocol_sha)
  assert 'tinygrad' not in sys.modules
  invocation=C.read(C.ROOT/'invocations'/f'{args.stage}-0.json')
  for key,value in invocation['environment'].items():
    if value is None:os.environ.pop(key,None)
    else:os.environ[key]=value
  os.environ.update(PYTHONDONTWRITEBYTECODE='1',PARALLEL='32',LLVMOPT='1',PROFILE='0',TMPDIR=str(C.SCRATCH))
  assert os.environ['CACHEDB']==str(C.CACHE)
  assert os.environ.get('BEAM','0')==os.environ.get('JITBEAM','0')=='0'
  sys.path[:0]=list(map(str,(C.PRIVATE,C.SNAP/'experiments',C.SNAP,C.TINY)))
  route=C.route()
  model_path=args.output.parent/'agent-compiler-speed-model.json'
  planned=route.plan(C.ROOT/'invocations'/f'{args.stage}-0.json',model_path)
  expected=protocol['plans'][args.stage]
  assert planned['pair_identity_sha256']==expected['pair_identity_sha256']
  command=planned['effective_command']
  from tinygrad import Device
  from tinygrad.codegen import to_program_context
  from tinygrad.helpers import PARALLEL
  assert not Device._opened_devices and PARALLEL.value==32
  fixed=importlib.import_module('agent-compiler-mode-renderer')
  shared=importlib.import_module('wgp_compiler')
  assert Path(fixed.__file__).resolve()==C.PRIVATE/'agent-compiler-mode-renderer.py'
  assert Path(shared.__file__).resolve()==C.SNAP/'experiments/wgp_compiler.py'
  P=importlib.import_module('agent-compiler-mode-proof')
  record={'complete':False,'gpu_executed':False,'stage':args.stage,'arm':args.arm,'protocol_sha256':args.protocol_sha,
    'quality_gate_sha256_before_run':gate_sha,'model_report':str(model_path),'actual_command':command,
    'warm_inventory':str(args.output.parent/'agent-compiler-speed-warm.json'),
    'post_inventory':str(args.output.parent/'agent-compiler-speed-post.json'),
    'artifact_directory':str(args.output.parent/'agent-compiler-speed-artifacts'),
    'audit_detached_before_measured_loop':False,'throughput_qualified':False}
  def save():C.write(args.output,record)
  old_install,old_print,old_argv=shared.install,builtins.print,sys.argv[:]
  original_hooks=hooks()
  audit=audit_class()(Path(record['artifact_directory']))
  observer=audit.observe();observer.__enter__()
  def install(device):
    assert not record.get('installed') and not audit.programs and device.arch=='gfx1200' and device.is_usb
    if args.arm=='original':old_install(device)
    else:record['installation']=fixed.install(device)
    record['installed']=True
    record['runtime']={'renderer':P.identity(device.renderer),'parallel':PARALLEL.value,
      'context':{v.key:v.value for v in to_program_context},
      'environment':{k:os.environ.get(k) for k in sorted(set(invocation['environment'])|{'PARALLEL','LLVMOPT','TMPDIR','BEAM','JITBEAM'})}}
    save()
  shared.install=install
  def detach():
    assert not record['audit_detached_before_measured_loop']
    observer.__exit__(None,None,None)
    builtins.print=old_print
    assert hooks()==original_hooks
    data=audit.export();kind=C.origins(data)
    for key,row in data['programs'].items():
      assert key in kind,'Unclassified warmup PROGRAM'
      row['observed_input_origins']=sorted(kind[key])
      if args.arm=='consistent-wgp' and 'renderer_compilation' in kind[key]:
        assert row['resources']['workgroup_processor_mode']==row['resources']['descriptor_wgp_bit']==1
    assert data['programs'] and len(data['captures'])>=4
    if args.stage=='quality':
      anchor=C.read(C.ROOT/'evidence'/f'mode-{args.arm}.json')
      record['actual_two_query_program_binding']=C.compare_warm(anchor['inventory'],data,same=True)
    C.write(record['warm_inventory'],data)
    record['callbacks_at_detach']=audit.callbacks
    from tinygrad.codegen import to_program_cache
    record['amd_cache_keys_at_detach']=sorted(json.dumps([k[0].hex(),repr(k[2]),*k[3:]],sort_keys=True)
      for k in to_program_cache if k[2].device=='AMD')
    record['audit_detached_before_measured_loop']=True
    record['detach_boundary']={'after_warmup_queries':C.COUNTS[args.stage][0],'measured_queries_completed':0,
      'before_measured_clock':True,'first_measured_prefetch_pending':False,'gpu_readbacks_added':0}
    audit.release();del data
    save()
  builtins.print=boundary(old_print,C.SNAP/'experiments/benchmark-pipeline.py',C.COUNTS[args.stage][0],detach)
  started=time.monotonic();save()
  try:
    sys.argv=command;record['gpu_executed']=True
    runpy.run_path(command[0],run_name='__main__')
    assert record['audit_detached_before_measured_loop'] and hooks()==original_hooks and builtins.print is old_print
    record['measured_observer_callbacks']=audit.callbacks-record['callbacks_at_detach']
    assert record['measured_observer_callbacks']==0
    model=C.read(model_path);spec=C.read(model_path.with_suffix('.speculation.json'))
    C.validate_snapshot(model)
    record['self_validation']=C.exact_reports(model,model,spec,spec,args.stage,planned['scheduled_samples'])
    if args.stage=='quality':
      anchor=C.read(C.ROOT/'evidence'/f'mode-{args.arm}-model.json')
      assert [C.canonical_row(x) for x in model['warmup']]==[C.canonical_row(x) for x in anchor['warmup']]
      record['actual_two_query_output_binding']=True
    post=full_cache_inventory(Path(record['artifact_directory']))
    expected_class='tinygrad.renderer.llvmir.AMDLLVMRenderer' if args.arm=='original' else 'agent-compiler-mode-renderer.AMDLLVMRenderer'
    assert all(x['renderer_class']==expected_class for x in post['entries'].values())
    C.write(record['post_inventory'],post)
    record['amd_cache_entries_added_after_warmup']=len(set(post['entries'])-set(record['amd_cache_keys_at_detach']))
    files=[model_path,model_path.with_suffix('.speculation.json'),model_path.with_suffix('.jsonl'),
      Path(record['warm_inventory']),Path(record['post_inventory']),*Path(record['artifact_directory']).glob('*')]
    record['artifact_sha256']={str(p):C.file_sha(p) for p in files}
    record['all_hooks_restored']=True
    record['complete']=True
  except BaseException as exc:
    record['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
  finally:
    observer.__exit__(None,None,None)
    shared.install,builtins.print,sys.argv=old_install,old_print,old_argv
    record['elapsed_seconds']=time.monotonic()-started
    record['all_hooks_restored']=hooks()==original_hooks
    save()


if __name__=='__main__':main()

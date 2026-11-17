"""Complete old CPU runner, plus HCQ containers built from its actual captured calls.

Only the CPU inspection boundary gets shaped wrappers. CPU execution remains the existing
full Tensor/TinyJit mock. No AMD queue is encoded, linked, uploaded, or executed.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

EXP=Path(__file__).parent
s=importlib.util.spec_from_file_location('ck_graph_overlay',EXP/'agent-ck-graph-inspection-overlay.py')
O=importlib.util.module_from_spec(s);s.loader.exec_module(O)

def main(output):
  check=O.module('agent-ck-integrated-check');check.R=O.R;check.C=O.C
  original=O.GRAPH.inspect_captured
  proof={'old_failure_reproduced':False,'both_second_calls_checked':False,'phases':[],
    'metadata_builder':'Pinned hcq2._finalize_batch on actual captured PROGRAM/copy CALLs',
    'limitation':'AFTER STORE/BUFFER wrappers and the empty compiled CPU hcq_submit body are shape fixtures. HCQ metadata comes from real captured calls. No GPU queue lowering or execution.',
    'negative_cases':[]}
  cache={};seen=set()
  def shaped(linear,linked):
    from tinygrad import dtypes
    from tinygrad.uop.ops import UOp,Ops,KernelInfo
    from tinygrad.runtime.support import hcq2
    from tinygrad.codegen import do_to_program
    from tinygrad.renderer.llvmir import CPULLVMRenderer
    from tinygrad.helpers import Target
    key=(linear.key,linked)
    if key in cache: return cache[key]
    calls=[u.without_after for u in linear.src]
    assert calls and all(u.op is Ops.CALL and u.body.op in (Ops.PROGRAM,Ops.STORE) for u in calls)
    # Real batching gives keys, output/input indices, estimates, and input slots from these actual calls.
    call=hcq2._finalize_batch(hcq2.BatchCtx([(u,'AMD','COMPUTE:0' if u.body.op is Ops.PROGRAM else 'COPY:0') for u in calls],False))
    linkword=UOp.alloc((1,),dtypes.uint32,device='AMD').index(0).store(UOp.const(0,dtypes.uint32))
    dependency=next(u for u in linear.toposort() if u.op is Ops.BUFFER) if linked else linkword
    # Real compiled PROGRAM form, with a deliberately empty CPU submitter. It is never executed.
    # The tested launch rows still come unchanged from the actual captured attention/utility calls.
    host=do_to_program(UOp.sink(arg=KernelInfo('hcq_submit')),CPULLVMRenderer(Target('CPU','LLVM','x86_64,generic')))
    compiled_call=call.replace(src=(host,*call.src[1:]))
    wrapped=compiled_call.after(dependency)
    assert wrapped.op is Ops.AFTER and wrapped.arg is None
    try: wrapped.arg.aux
    except AttributeError: proof['old_failure_reproduced']=True
    else: raise AssertionError('Old graph-inspection error did not reproduce')
    result=UOp(Ops.LINEAR,src=(UOp(Ops.LINEAR),UOp(Ops.NOOP),UOp(Ops.LINEAR,src=(wrapped,))))
    cache[key]=(result,call,calls)
    return cache[key]
  def inspect(captured,common,report,arm,phase):
    from tinygrad.uop.ops import UOp,Ops,CallInfo
    graph,hcq,calls=shaped(captured._linear,False)
    linears={k:(shaped(v[0],True)[0],v[1]) for k,v in captured._linears.items()}
    value=original(SimpleNamespace(_linear=graph,_linears=linears),common,report,arm,phase)
    assert O.GRAPH.inspect_linear(hcq,common,report)['native']==value
    proof['phases'].append({'arm':arm,'phase':phase,'native':value,'linked_graphs':len(linears),
      'actual_raw_calls':len(calls),'hcq_rows':len(hcq.arg.aux.kernels)})
    if phase=='screen_1_original64_8': seen.add(arm)
    if arm=='candidate' and not proof['negative_cases']:
      def reject(name,linear):
        try: O.GRAPH.inspect_linear(linear,common,report)
        except (AssertionError,KeyError,TypeError): proof['negative_cases'].append(name)
        else: raise AssertionError(('Graph negative accepted',name))
      native=[c for c in calls if c.body.op is Ops.PROGRAM and c.body.src[0].arg.name=='alpamayo_ck_d128']
      assert len(native)==28
      direct=UOp(Ops.LINEAR,src=tuple(native))
      assert O.GRAPH.inspect_linear(direct,common,report)['native']==28
      proof['direct_programs_checked']=True
      info=hcq.arg.aux;idx=next(i for i,k in enumerate(info.kernels) if k[1]=='alpamayo_ck_d128')
      def hcq_graph(rows): return UOp(Ops.LINEAR,src=(hcq.replace(arg=replace(hcq.arg,aux=replace(info,kernels=tuple(rows)))),))
      for name,column,replacement in (('HCQ_wrong_program_key',4,b'\0'*32),('HCQ_wrong_native_name',1,'renamed'),
        ('HCQ_missing_LSE_output',6,((0,),(2,3,4))),('HCQ_wrong_device',0,'CPU')):
        rows=list(info.kernels);row=list(rows[idx]);row[column]=replacement;rows[idx]=tuple(row)
        reject(name,hcq_graph(rows))
      for name,rows in (('missing_native',info.kernels[:idx]+info.kernels[idx+1:]),('duplicate_native',info.kernels+(info.kernels[idx],))):
        linear=hcq_graph(rows)
        try: original(SimpleNamespace(_linear=linear,_linears={(): (linear,{})}),common,report,arm,name)
        except AssertionError: proof['negative_cases'].append(name)
        else: raise AssertionError(('Native count negative accepted',name))
      reject('unknown_schedule_node',UOp(Ops.LINEAR,src=(UOp.sink(),)))
      reject('unknown_auxiliary_container',hcq.replace(arg=replace(hcq.arg,aux=('unknown',))))
      reject('missing_CallInfo',hcq.replace(arg=None))
      reject('unknown_HCQ_body',hcq.replace(src=(UOp(Ops.LINEAR),*hcq.src[1:])))
      reject('unknown_call_body',UOp(Ops.LINEAR).call())
      reject('unknown_custom_function',UOp.custom_function('unrecognized').call())
      reject('executable_after_dependency',hcq.after(native[0]))
      reject('hidden_call_in_after_store',hcq.after(UOp(Ops.STORE,src=(native[0],))))
      p=native[0].body
      bad=p.replace(src=(*p.src[:2],UOp(Ops.SOURCE,arg=p.src[2].arg+'\n'),p.src[3]))
      reject('direct_same_ELF_changed_source',native[0].replace(src=(bad,*native[0].src[1:])))
      # Empty/nonkernel nodes cannot accidentally become native launches.
      blank=UOp(Ops.LINEAR,src=(UOp(Ops.LINEAR),UOp(Ops.NOOP)))
      assert O.GRAPH.inspect_linear(blank,common,report)['native']==0
      assert O.GRAPH.inspect_linear(hcq.after(UOp(Ops.NOOP)),common,report)['native']==28
      assert O.GRAPH.inspect_linear(UOp.custom_function('validate').call(),common,report)['custom_functions']==1
      assert O.GRAPH.inspect_linear(UOp.custom_function('encdec').call(),common,report)['custom_functions']==1
      assert O.GRAPH.inspect_linear(native[0].src[1].store_call(native[0].src[1]),common,report)['copies']==1
      proof['nested_empty_and_noop_checked']=True
      proof['both_pre_lower_SINK_and_compiled_PROGRAM_submitters_checked']=True
    return value
  O.GRAPH.inspect_captured=inspect
  try: check.main(output)
  finally: O.GRAPH.inspect_captured=original
  proof['both_second_calls_checked']=seen=={'baseline','candidate'};assert proof['both_second_calls_checked']
  result=json.loads(output.read_text());result['graph_shape_checks']=proof
  output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  print(json.dumps({'both_second_calls_checked':True,'graph_negatives':len(proof['negative_cases']),'gpu_executed':False}))

if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)

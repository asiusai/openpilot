"""Observe original serial lengths 2..6 from one current prefill. Never synthesize a plan."""
import ast
import copy
import gc
import json
import os
from pathlib import Path
import resource
import time

EXP=Path(__file__).resolve().parent

def install(frozen,contract,protocol,started):
  """Retain the qualified current-input setup and restoration verbatim as AST."""
  text=(EXP/'agent-plan-bind-run.py').read_text()
  original=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='capture_one')
  start=next(i for i,n in enumerate(original.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='key' for t in n.targets))
  final=next(n for n in original.body[start:] if isinstance(n,ast.Try)).finalbody
  prefix=copy.deepcopy(original.body[:start])
  body=ast.parse('family_capture_session(locals())').body
  after=ast.parse("report['plan_family']['restored']={k:report[k] for k in ('full_kv_bits_restored','actual_first_logits_bits_restored','target_payload_bits_restored','current_owner_retained','generation_history_unchanged')}").body
  derived=copy.deepcopy(original);derived.body=prefix+[ast.Try(body=body,handlers=[],orelse=[],finalbody=copy.deepcopy(final))]+after
  namespace=frozen.__dict__
  namespace['family_capture_session']=lambda context:capture_session(context,contract,protocol,started)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[derived],type_ignores=[])),str(EXP/'agent-plan-family-capture.py')+' [qualified-prefix/finally]','exec'),namespace)
  return dict(prefix_ast_sha256=contract.hashlib.sha256(ast.dump(ast.Module(body=prefix,type_ignores=[])).encode()).hexdigest(),
    restoration_ast_sha256=contract.hashlib.sha256(ast.dump(ast.Module(body=final,type_ignores=[])).encode()).hexdigest())

def capture_session(context,contract,protocol,started):
  from tinygrad import Tensor
  from tinygrad.engine.jit import _prepare_jit_inputs
  from tinygrad.helpers import GlobalCounters
  from tinygrad.uop.ops import UOp
  model,decoder,logits,dc,ds=(context[k] for k in ('model','decoder','logits','dc','ds'))
  prompt,reset,exact=(context[k] for k in ('prompt','reset_kv','exact'))
  report,output,args=(context[k] for k in ('report','output','args'))
  audit=contract.local(EXP/'agent-plan-bind-audit.py');expected=contract.signatures()
  root=output.parent/'agent-plan-family-capture';root.mkdir()
  result=dict(schema=1,counts=protocol['counts'],current_inputs=1,future_inputs=0,count1_reexecuted=False,
    qualified=False,executable=False,ready_for_live_use=False,plans={},stages=[],limits=protocol['limits'],
    snapshot_bytes=sum(t.numel()*t.dtype.itemsize for t in context['snapshots']))
  report['plan_family']=result
  report['capture']=dict(qualified_numerical_identities_equal=False)
  def event(name,count):
    model.synchronize()
    row=dict(name=name,count=count,elapsed_ms=(time.perf_counter()-started)*1000,
             live_tensor_bytes=GlobalCounters.mem_used,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['stages'].append(row);print(json.dumps({'agent_plan_family_stage':row}),flush=True)
  for count in protocol['counts']:
    if time.perf_counter()-started>=protocol['limits']['start_next_count_before_seconds']:
      raise TimeoutError('Family capture budget exhausted before next length; completed artifacts are preserved')
    if GlobalCounters.mem_used>=protocol['limits']['live_tensor_bytes']: raise MemoryError('Live tensor allocation budget exceeded')
    key=(dc.shape,count,2304)
    if key in decoder.jits: raise ValueError('Family member must start without an existing original JIT')
    reset();eager=decoder.decode(logits,dc,ds,prompt,0,count);Tensor.realize(*eager);model.synchronize()
    saved=tuple(t.clone().realize() for t in eager);saved_kv=tuple(t.clone().realize() for t in model.cache)
    event('eager_reference',count);reset()
    ranges=contract.bounds(count)
    pos=UOp.variable('block_position',0,ranges['block_position']['max']).bind(prompt)
    step=UOp.variable('block_step',0,ranges['block_step']['max']).bind(0)
    inputs,*_=_prepare_jit_inputs((logits,dc,ds,pos,step),{})
    with audit.capture_boundary(inputs) as record: captured=decoder.decode(logits,dc,ds,prompt,0,count)
    Tensor.realize(*captured);model.synchronize()
    def compare(values):
      for a,b in zip(values,saved,strict=True): exact(a,b)
      for a,b in zip(model.cache,saved_kv,strict=True): exact(a,b)
    compare(captured);reset()
    replayed=decoder.decode(logits,dc,ds,prompt,0,count);Tensor.realize(*replayed);compare(replayed)
    reset();aliased=replayed[0];aliased.assign(context['first']).realize()
    alias_values=decoder.decode(aliased,dc,ds,prompt,0,count);Tensor.realize(*alias_values);compare(alias_values)
    jit=decoder.jits[key]
    if len(jit.captured._linears)!=2: raise AssertionError('Both alias variants must be linked and compared')
    registry=audit.model_registry(model,(logits,dc,ds),jit.captured.ret,
      bulk_index=Path(os.environ['ALPAMAYO_FP8_CACHE'])/'index.json',
      vocab=args.reduced_vocab,checkpoint=args.checkpoint)
    summary=audit.export_capture(record,jit,registry,root/str(count),expected[str(count)])
    views={name:contract.view_recipe(t) for name,t in zip(('return/logits','return/token'),jit.captured.ret,strict=True)}
    doc=json.loads(Path(summary['audit']).read_text())
    member=contract.validate_document(doc,count,expected[str(count)],views)
    sidecar=dict(schema=1,count=count,audit_sha256=summary['audit_sha256'],output_views=views,contract=member,
      eager_capture_replay_alias_output_bits_equal=True,all28_KV_bits_equal=True,source_protocol_sha256=report['family_protocol_sha256'])
    sidepath=root/str(count)/'family.json';sidepath.write_text(json.dumps(sidecar,indent=2)+'\n')
    result['plans'][str(count)]=dict(**summary,sidecar=str(sidepath),sidecar_sha256=contract.sha(sidepath),contract=member)
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if total>protocol['limits']['artifact_bytes']: raise ValueError('Family artifact budget exceeded')
    result['artifact_bytes']=total;event('original_capture_replay_and_alias_exact',count)
    # No JIT is installed for normal driving by this diagnostic. Release one
    # member before capturing the next; closed model/current input owners stay.
    del decoder.jits[key]
    del eager,saved,saved_kv,captured,replayed,alias_values,aliased,jit,registry,record,doc,compare
    gc.collect()
  if decoder.jits: raise AssertionError('Unexpected serial plans survived isolated capture')
  result.update(capture_completed=True,original_numerical_identities_exact=True,
                scratch_proof_pending=True,fresh_process_reload_pending=True)
  report['capture']=dict(qualified_numerical_identities_equal=True,family_counts=protocol['counts'])

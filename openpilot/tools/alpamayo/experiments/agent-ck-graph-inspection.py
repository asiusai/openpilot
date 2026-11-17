"""Inspect scheduled calls, not metadata ASTs. No device lookup or execution."""
import hashlib
import json


def inspect_linear(linear, common, report):
  from tinygrad.uop.ops import UOp, Ops, CallInfo, ProgramInfo, KernelInfo
  from tinygrad.runtime.support.hcq2 import HCQInfo, _is_link_patch
  from tinygrad.renderer import Estimates
  expected=json.loads((common.PREPARED/'ck-identity.json').read_text())
  checked=report.get('checked_native_program_keys',[])
  stats={'native':0,'hcq_containers':0,'hcq_kernels':0,'direct_programs':0,
    'copies':0,'custom_functions':0,'after_wrappers':0,'link_stores':0,'buffer_refs':0,'noops':0,'empty_linears':0}
  def metadata(dep):
    # lower_call hoists link-time STOREs; hcq_link retains BUFFER references and removes written NOOPs.
    assert dep.op in (Ops.STORE,Ops.BUFFER,Ops.NOOP),('Unknown AFTER dependency',dep.op)
    assert _is_link_patch(dep),'Executable operation hidden in AFTER metadata'
    stats[{Ops.STORE:'link_stores',Ops.BUFFER:'buffer_refs',Ops.NOOP:'noops'}[dep.op]]+=1
  def program(p):
    assert isinstance(p.arg,ProgramInfo) and len(p.src)==4 and p.src[3].op is Ops.BINARY,'Uncompiled/unknown PROGRAM'
    name=p.src[0].arg.name
    native=name==expected['name'] or hashlib.sha256(p.src[3].arg).hexdigest()==expected['binary_sha256']
    if native:
      assert common.load('agent-ck-source-identity').identity(p)==expected,'Changed direct CK identity'
      assert p.key.hex() in checked,'CK program was not checked by the active native scope'
    return native
  def visit(node):
    assert isinstance(node,UOp)
    if node.op is Ops.AFTER:
      assert node.arg is None and node.src
      stats['after_wrappers']+=1
      for dep in node.src[1:]: metadata(dep)
      visit(node.src[0]);return
    if node.op is Ops.LINEAR:
      if not node.src: stats['empty_linears']+=1
      for child in node.src: visit(child)
      return
    if node.op is Ops.NOOP:
      assert not node.src
      stats['noops']+=1;return
    assert node.op is Ops.CALL and isinstance(node.arg,CallInfo),('Unknown schedule container',node.op,type(node.arg).__name__)
    body,aux=node.body,node.arg.aux
    if isinstance(aux,HCQInfo):
      assert aux.device and all(isinstance(d,str) for d in aux.device)
      assert body.op in (Ops.SINK,Ops.PROGRAM),('Unknown HCQ body',body.op)
      if body.op is Ops.PROGRAM:
        assert isinstance(body.arg,ProgramInfo) and body.src[0].arg.name=='hcq_submit'
      else: assert isinstance(body.arg,KernelInfo) and body.arg.name=='hcq_submit'
      stats['hcq_containers']+=1
      # HCQInfo already records launches. The host submitter's AST is not another launch schedule.
      for row in aux.kernels:
        assert isinstance(row,tuple) and len(row)==7,'Unknown HCQ kernel tuple'
        device,name,estimates,stamps,key,slots,io=row
        assert device in aux.device and isinstance(name,str) and isinstance(estimates,Estimates)
        assert isinstance(stamps,tuple) and isinstance(slots,tuple) and isinstance(io,tuple) and len(io)==2
        assert all(isinstance(x,tuple) and all(isinstance(i,int) and i>=0 for i in x) for x in (stamps,slots,*io))
        assert key is None or isinstance(key,bytes) and len(key)==32
        if key is None:
          # These are the only non-PROGRAM operations emitted by the pinned HCQ batch builder.
          from tinygrad.helpers import ansistrip
          assert ansistrip(name).startswith(('copy ','enc/dec ')),('Unknown nonkernel HCQ entry',name)
        is_checked=key is not None and key.hex() in checked
        if name==expected['name'] or is_checked:
          assert name==expected['name'] and is_checked,'Changed HCQ CK name/program-key association'
          assert device=='AMD' and io==((0,1),(2,3,4)),'Changed HCQ CK device/output/input ABI'
          stats['native']+=1
        stats['hcq_kernels']+=1
      return
    assert aux is None,('Unknown CALL auxiliary container',type(aux).__name__)
    if body.op is Ops.PROGRAM:
      stats['direct_programs']+=1;stats['native']+=int(program(body));return
    if body.op is Ops.STORE:
      assert len(node.src)==3,'Unknown copy CALL signature'
      stats['copies']+=1;return
    if body.op is Ops.CUSTOM_FUNCTION:
      assert body.arg.name in ('validate','encdec'),('Unknown CUSTOM_FUNCTION',body.arg.name)
      stats['custom_functions']+=1;return
    raise AssertionError(('Unknown CALL body',body.op))
  visit(linear)
  return stats


def inspect_captured(captured, common, report, arm, phase):
  expected=28 if arm=='candidate' else 0
  graphs=[('compiled',captured._linear)]+[(f'linked_{i}',row[0]) for i,row in enumerate(captured._linears.values())]
  assert len(graphs)>=2,'Capture returned without a linked executable graph'
  rows=[]
  for name,linear in graphs:
    row=inspect_linear(linear,common,report)
    assert row['native']==expected,('Captured CK coverage',phase,name,row['native'],expected)
    rows.append({'graph':name,'key':linear.key.hex(),**row})
  report.setdefault('capture_inspections',[]).append({'phase':phase,'graphs':rows})
  return expected

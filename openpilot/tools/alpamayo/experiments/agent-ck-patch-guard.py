"""Keep every frozen compute guard; independently authorize only active dword copy transports."""
from contextlib import contextmanager
import importlib
from pathlib import Path

transport=importlib.import_module('agent-ck-patch-transport')
identity=importlib.import_module('agent-ck-patch-identity')
compute=importlib.import_module('agent-ck-source-guard')
record_programs=compute.record_programs


@contextmanager
def bindings(helper,plan,report):
  from tinygrad.engine import realize
  from tinygrad.runtime.support import hcq2
  from tinygrad.uop.ops import Ops,graph_rewrite
  base_queues,base_run,base_compile=hcq2._build_queues,realize.run_linear,realize.compile_linear
  with compute.bindings(helper,plan,report):
    guarded_compute=hcq2._build_queues
    def run(linear,*args,**kwargs):
      if (active:=transport.ACTIVE) is not None:
        assert not active['raw_checked'] and linear is active['linear']
        assert len(linear.src)==1 and linear.src[0].body.op is Ops.STORE and not args
        assert kwargs=={'update_stats':False}
        row=active['record']
        destination=[row['logical_buffer']['bytes'],row['logical_buffer']['offset_bytes'],row['offset'],row['bytes']]
        assert destination in plan['copy_destinations'],('Unprepared physical patch range',destination)
        active['raw_checked']=True
        inputs=[]
        template=graph_rewrite(linear,hcq2.pm_replace_buffers,ctx=(inputs,{}),walk=True)
        signature=identity.copy_call(template.src[0]);cid=identity.key(signature)
        assert cid in plan['copy_calls'] and plan['copy_calls'][cid]==signature,('Unprepared copy view',cid,signature)
        active.update(template=template,copy_id=cid)
        active['record']['copy_association']=cid
        report.setdefault('copy_invocations',[]).append({'phase':report['active_phase'],'association':cid,'signature':signature,
          'actual_target':transport.info(active['target']),'actual_staging':transport.info(active['staging']),
          'payload_sha256':active['record']['payload_sha256']})
      return base_run(linear,*args,**kwargs)
    def queues(ctx):
      if not any(c.body.op is Ops.STORE for c,_,_ in ctx.batch): return guarded_compute(ctx)
      active=transport.ACTIVE
      assert active is not None and active['raw_checked'] and len(ctx.batch)==1
      call,dev,queue=ctx.batch[0]
      assert (dev,queue)==('AMD','COPY:0') and call is active['template'].src[0]
      signature=identity.copy_call(call);assert identity.key(signature)==active['copy_id']
      row={'phase':report['active_phase'],'association':active['copy_id'],'signature':signature,
        'actual_target':transport.info(active['target']),'actual_staging':transport.info(active['staging'])}
      report.setdefault('copy_batches',[]).append(row)
      return base_queues(ctx)
    def compile_copy(linear,*args,**kwargs):
      result=base_compile(linear,*args,**kwargs)
      if transport.ACTIVE is not None:
        programs=[u for u in result.toposort(enter_calls=True) if u.op is Ops.PROGRAM]
        assert programs and all(p.arg.target.device=='CPU' for p in programs)
        for program in programs:
          item=identity.host_program(program);pid=identity.key(item)
          assert pid in plan['copy_host_programs'] and plan['copy_host_programs'][pid]==item,('Unprepared copy host program',pid,item)
          records=report.setdefault('copy_host_programs',{})
          if pid not in records:
            records[pid]=item
            folder=Path(plan['patch_output_root'])/'parent-programs'
            (folder/(pid+'.copy-host.source')).write_text(program.src[2].arg)
            (folder/(pid+'.copy-host.elf')).write_bytes(program.src[3].arg)
          transport.ACTIVE['record'].setdefault('host_program_associations',[]).append(pid)
      return result
    hcq2._build_queues,realize.run_linear,realize.compile_linear=queues,run,compile_copy
    try: yield
    finally:
      assert hcq2._build_queues is queues and realize.run_linear is run and realize.compile_linear is compile_copy
      hcq2._build_queues,realize.run_linear,realize.compile_linear=guarded_compute,base_run,base_compile

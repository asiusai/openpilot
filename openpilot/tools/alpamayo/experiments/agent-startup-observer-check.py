"""Bounded CPU benchmark: real captured ELF parsing, HCQ linking, and host writes."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import threading
import time
import types
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;WORK=Path('/mnt/nvme-500/tmp/alpamayo-speed-20261116')
sys.dont_write_bytecode=True;sys.path.insert(0,'/tmp/opencode/alpamayo-asius-20261114/tinygrad')
spec=importlib.util.spec_from_file_location('observer_check_run',HERE/'agent-startup-observer-run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run);m=run.m

def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
  for path in (args.output,run.PROTOCOL,*[Path(os.environ.get(k,'')) for k in ('CACHEDB','TMPDIR','XDG_CACHE_HOME','TORCH_HOME')]): m.storage(path)
  if args.output.exists() or run.PROTOCOL.exists(): raise FileExistsError('Fresh observer CPU evidence/protocol required')
  run.c.gate(m.read(run.BASE));observer=run.local(HERE/'agent-startup-observer.py')
  from tinygrad import Device,dtypes
  from tinygrad.device import Buffer
  from tinygrad.uop.ops import UOp,Ops
  original=type(Device).__getitem__
  def forbidden(*a,**kw): raise AssertionError('GPU/network forbidden')
  def cpu(owner,key):
    if key is None or key.split(':')[0] not in ('CPU','NPY'): raise AssertionError('GPU forbidden')
    return original(owner,key)
  with patch.object(type(Device),'__getitem__',cpu),patch.object(socket.socket,'connect',forbidden):
    from tinygrad.runtime.support import hcq2
    from tinygrad.runtime.support.elf import elf_loader
    buf=Buffer('CPU',256,initial_value=bytes(256));ref=UOp.from_buffer(buf,dtypes.uint8)
    def link():
      arena=UOp.alloc((256,),dtypes.uint8,device='CPU')
      result=hcq2.hcq_link(UOp(Ops.LINEAR,src=(arena,)),input_uops=[])
      assert result.op is Ops.LINEAR and any(u.op is Ops.BUFFER for u in result.toposort())
      return result
    link();hcq2._write_host(ref,[(3,b'abc')])
    identities={n:getattr(hcq2,n) for n in ('hcq_link','bufferize_buf','_write_host')}
    with observer.LinkObserver(m,256) as seen:
      link();result=hcq2._write_host(buf=ref,writes=[(3,b'xyz')])
      assert result.op is Ops.NOOP
      # PatternMatcher clones the function object but retains its exact code object.
      clone=types.FunctionType(hcq2.bufferize_buf.__code__,hcq2.bufferize_buf.__globals__)
      cloned=clone(hcq2.LinkCtx([]),UOp.alloc((256,),dtypes.uint8,device='CPU').base)
      assert cloned.op is Ops.BUFFER
    counts={name:sum(e['name']==name for e in seen.events) for name in identities}
    assert counts=={'hcq_link':1,'bufferize_buf':2,'_write_host':1},counts
    assert [e['bytes'] for e in seen.events if e['name']=='_write_host']==[3]
    assert all(e['proven_arena_extent'] for e in seen.events if e['name']=='bufferize_buf')
    assert bytes(buf.host.view(fmt='B')[3:6])==b'xyz'
    assert sum(e['inclusive_ms'] for e in seen.events if e['name']=='hcq_link')>=0
    marker=RuntimeError('exact exception identity')
    def blocked(*a): raise marker
    with observer.LinkObserver(m,256) as failed,patch.object(type(Device),'__getitem__',blocked):
      try: hcq2.bufferize_buf(hcq2.LinkCtx([]),UOp.alloc((256,),dtypes.uint8,device='CPU',name='exception-test').base)
      except RuntimeError as error: assert error is marker
      else: raise AssertionError('Exception was swallowed')
    assert len(failed.events)==1 and failed.events[0]['outcome']=='exception:RuntimeError' and not failed.active
    # Unwinding the observer context also preserves the original exception and releases its slot.
    try:
      with observer.LinkObserver(m,256): raise marker
    except RuntimeError as error: assert error is marker
    with observer.LinkObserver(m,256) as threaded:
      errors=[]
      def worker():
        try: link();hcq2._write_host(ref,[(0,b't')])
        except BaseException as e: errors.append(repr(e))
      thread=threading.Thread(target=worker);thread.start();thread.join(5)
      assert not thread.is_alive() and not errors and not threaded.events
      try:
        with observer.LinkObserver(m,256): pass
      except ValueError: pass
      else: raise AssertionError('Nested observer accepted')
    assert all(getattr(hcq2,n) is value for n,value in identities.items()) and sys.getprofile() is None
    # Real immutable gfx1200 binaries from the qualified count1 plan. No program executes.
    audit=WORK/'agent-plan-bind-capture1/agent-plan-bind-capture/audit.json';doc=m.read(audit)
    binaries={h:size for row in doc['programs'].values() for h,size in row['binaries'].items()}
    chosen=sorted(binaries,key=binaries.get,reverse=True)[:3];blobs=[]
    for h in chosen:
      blob=(audit.parent/'programs'/(h+'.elf')).read_bytes()
      assert len(blob)==binaries[h] and hashlib.sha256(blob).hexdigest()==h;blobs.append(blob)
    def work():
      digest=hashlib.sha256()
      for _ in range(30):
        for blob in blobs:
          image,sections,relocations=elf_loader(blob)
          digest.update(image);digest.update(repr(([(s.name,s.header.sh_addr,s.header.sh_size) for s in sections],relocations)).encode())
        link();hcq2._write_host(ref,[(0,b'qualified CPU host write')])
      return digest.hexdigest()
    expected=work();timings={'baseline':[],'observer':[]};contexts=[]
    for _ in range(3):
      for name in ('baseline','observer'):
        context=nullcontext() if name=='baseline' else observer.LinkObserver(m,256)
        start=time.perf_counter()
        with context: actual=work()
        timings[name].append((time.perf_counter()-start)*1000)
        assert actual==expected
        if name=='observer':
          contexts.append(context)
          assert {n:sum(e['name']==n for e in context.events) for n in identities}=={n:30 for n in identities}
    old=m.load(HERE/'agent-startup-persist-install.py').LinkObserver
    start=time.perf_counter()
    with old(m,256): assert work()==expected
    old_ms=(time.perf_counter()-start)*1000
    baseline=statistics.median(timings['baseline']);observed=statistics.median(timings['observer'])
    if observed>baseline*1.25+1: raise AssertionError(f'Observer overhead too large: {baseline:.3f} -> {observed:.3f} ms')
    assert sys.getprofile() is None and all(sys.monitoring.get_tool(i) is None for i in (3,4))
  report=dict(passed=True,gpu_executed=False,sources=run.sources(),devices=sorted(Device._opened_devices),
    actual_elf_sha256=chosen,ELF_parses_per_block=90,HCQ_links_per_block=30,host_writes_per_block=30,
    exact_image_sections_relocations_digest=expected,all_results_exact=True,
    timings_ms=timings,baseline_median_ms=baseline,observer_median_ms=observed,observer_ratio=observed/baseline,
    original_global_profiler_ms=old_ms,original_global_profiler_ratio=old_ms/baseline,
    return_and_kwargs_preserved=True,exception_identity_preserved=True,thread_scope_checked=True,
    cloned_pattern_function_observed=True,inclusive_events_and_bytes_checked=True,runtime_function_identities_restored=True,
    monitoring_slots_restored=True,causal_limit='CPU benchmark measures observer overhead; it does not attribute the entire GPU startup timeout')
  m.save(args.output,report)
  protocol=dict(base_sha256=run.BASE_SHA,sources=run.sources(),cpu_evidence=str(args.output),cpu_sha256=m.sha(args.output),child_seconds=450)
  m.save(run.PROTOCOL,protocol);run.gate(protocol)
  print(json.dumps({k:report[k] for k in ('passed','devices','baseline_median_ms','observer_median_ms','observer_ratio',
    'original_global_profiler_ms','original_global_profiler_ratio')},indent=2))

if __name__=='__main__': main()

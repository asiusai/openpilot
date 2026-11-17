"""Code-specific CPython3.12 monitoring for three unchanged HCQ functions."""
from contextlib import contextmanager, ExitStack
import importlib.util
import inspect
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch

_ACTIVE=threading.local()

def target_codes(hcq,path):
  def body(fn,name):
    pending=[fn];seen=set();found=[]
    while pending:
      value=pending.pop()
      if not inspect.isfunction(value) or id(value) in seen: continue
      seen.add(id(value))
      if value.__code__.co_name==name and value.__code__.co_filename==path: found.append(value.__code__)
      pending.extend(cell.cell_contents for cell in value.__closure__ or ())
      if len(seen)>32: raise ValueError('Unreviewed HCQ decorator chain')
    if len(found)!=1: raise ValueError('Cannot identify the exact HCQ function body: '+name)
    return found[0]
  return {body(getattr(hcq,name),name):name for name in ('hcq_link','_write_host','bufferize_buf')}

class LinkObserver:
  def __init__(self,m,arena_bytes):
    from tinygrad.runtime.support import hcq2
    if sys.version_info[:2]!=(3,12): raise ValueError('Pinned CPython3.12 monitoring required')
    self.codes=target_codes(hcq2,str(m.TG/'tinygrad/runtime/support/hcq2.py'))
    self.hcq,self.arena_bytes,self.events,self.active=hcq2,arena_bytes,[],{}
    self.identities={name:getattr(hcq2,name) for name in self.codes.values()}
    self.errors=[];self.tool=None;self.thread=None

  def _start(self,code,offset):
    if threading.get_ident()!=self.thread: return
    frame=sys._getframe(1)
    if frame.f_code is not code: self.errors.append('Unexpected start frame');return
    self.active[id(frame)]=time.perf_counter()

  def _finish(self,code,frame,outcome):
    if frame.f_code is not code: self.errors.append('Unexpected finish frame');return
    started=self.active.pop(id(frame),None)
    if started is None: self.errors.append('Missing start event');return
    row=dict(name=self.codes[code],inclusive_ms=(time.perf_counter()-started)*1000,outcome=outcome)
    try:
      if row['name']=='_write_host': row['bytes']=sum(len(data) for _,data in frame.f_locals['writes'])
      if row['name']=='bufferize_buf':
        size=frame.f_locals['b'].nbytes();row.update(bytes=size,proven_arena_extent=size==self.arena_bytes)
    except Exception as error: self.errors.append(type(error).__name__+': '+str(error))
    self.events.append(row)

  def _return(self,code,offset,result):
    if threading.get_ident()==self.thread: self._finish(code,sys._getframe(1),'return')

  def _unwind(self,code,offset,error):
    # Python3.12 has no local PY_UNWIND mask. Only exception unwinds reach this filter.
    if code in self.codes and threading.get_ident()==self.thread:
      self._finish(code,sys._getframe(1),'exception:'+type(error).__name__)

  def __enter__(self):
    if sys.getprofile() is not None or getattr(_ACTIVE,'held',False): raise ValueError('Existing or nested observer rejected')
    mon=sys.monitoring
    for tool in (3,4):
      if mon.get_tool(tool) is None:
        try: mon.use_tool_id(tool,'asius-startup-observer')
        except ValueError: continue
        self.tool=tool;break
    if self.tool is None: raise RuntimeError('No free monitoring slot')
    self.thread=threading.get_ident();_ACTIVE.held=True
    try:
      for event,fn in ((mon.events.PY_START,self._start),(mon.events.PY_RETURN,self._return),(mon.events.PY_UNWIND,self._unwind)):
        if mon.register_callback(self.tool,event,fn) is not None: raise RuntimeError('Occupied monitoring callback')
      for code in self.codes: mon.set_local_events(self.tool,code,mon.events.PY_START|mon.events.PY_RETURN)
      mon.set_events(self.tool,mon.events.PY_UNWIND)
      return self
    except BaseException:
      self._close();raise

  def _close(self):
    if self.tool is not None:
      mon=sys.monitoring;mon.set_events(self.tool,0)
      for code in self.codes: mon.set_local_events(self.tool,code,0)
      for event in (mon.events.PY_START,mon.events.PY_RETURN,mon.events.PY_UNWIND): mon.register_callback(self.tool,event,None)
      mon.free_tool_id(self.tool);self.tool=None
    _ACTIVE.held=False

  def __exit__(self,kind,error,tb):
    self._close()
    if any(getattr(self.hcq,name) is not value for name,value in self.identities.items()): self.errors.append('Runtime function identity changed')
    if self.active: self.errors.append('Unfinished observed function')
    if kind is None and self.errors: raise AssertionError('; '.join(self.errors))
    return False

@contextmanager
def imports(m):
  previous=importlib.util.spec_from_file_location;stack=ExitStack();state={'modules':0,'observers':[]}
  class RecordedObserver(LinkObserver):
    def __init__(self,*args): super().__init__(*args);state['observers'].append(self)
  def spec(name,location,*args,**kwargs):
    result=previous(name,location,*args,**kwargs)
    if result is None or location is None or Path(location).resolve()!=m.EXP/'agent-startup-persist-install.py': return result
    loader=result.loader
    class Loader:
      def create_module(self,spec): return loader.create_module(spec)
      def exec_module(self,module):
        loader.exec_module(module);before=module.LinkObserver;module.LinkObserver=RecordedObserver;state['modules']+=1
        def restore():
          if module.LinkObserver is not RecordedObserver: raise RuntimeError('Observer overlay ownership changed')
          module.LinkObserver=before
        stack.callback(restore)
    result.loader=Loader();return result
  try:
    with patch.object(importlib.util,'spec_from_file_location',spec): yield state
  finally: stack.close()

def summary(state):
  if any(o.tool is not None or o.active or o.errors for o in state['observers']): raise AssertionError('Observer cleanup/evidence incomplete')
  events=[e for o in state['observers'] for e in o.events]
  return dict(modules=state['modules'],contexts=len(state['observers']),events=len(events),
    counts={name:sum(e['name']==name for e in events) for name in ('hcq_link','_write_host','bufferize_buf')},
    host_write_bytes=sum(e.get('bytes',0) for e in events if e['name']=='_write_host'),
    arena_allocation_events=sum(bool(e.get('proven_arena_extent')) for e in events),
    runtime_functions_unchanged=True,monitoring_restored=True,
    scope='Three code-specific start/return events; filtered exception-unwind events; initiating thread only',
    timing='Inclusive host intervals overlap. Not exclusive GPU/driver timing.')

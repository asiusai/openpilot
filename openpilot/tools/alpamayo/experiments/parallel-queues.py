"""Scoped experiment: route scratch-free online attention to a second AMD queue.

This owns one extra queue on the existing USB GPU. Captured graphs from this
scope must not run after exit. Public tinygrad files remain unchanged.
"""
from contextlib import ExitStack
import inspect
import textwrap
from types import MethodType

from tinygrad.device import Compiled
from tinygrad.helpers import PROFILE, wait_cond
from tinygrad.runtime import ops_amd
from tinygrad.runtime.support import hcq2
from tinygrad.uop.ops import Ops


def clone(function, replacements, extra=None):
  function=inspect.unwrap(function)
  # This pinned rewrite_group decorator does not expose __wrapped__.
  while function.__name__=='__wrapper' and function.__module__=='tinygrad.uop.ops':
    cells=dict(zip(function.__code__.co_freevars,function.__closure__ or (),strict=True))
    function=cells['func'].cell_contents
  source=textwrap.dedent(inspect.getsource(function))
  for before,after in replacements:
    assert source.count(before)==1, f'Pinned source changed: {before}'
    source=source.replace(before,after)
  namespace=dict(function.__globals__)
  namespace.update(extra or {})
  exec(compile(source,f'<alpamayo-{function.__name__}>','exec'),namespace)
  return namespace[function.__name__]


class SecondComputeQueue:
  def __init__(self,device):
    self.device=device
    self.stack=ExitStack()
    self.queue=None
    self.selected=0
    self.scratch_checks=0
    self.drained=False

  def replace(self,obj,name,value):
    missing=object()
    previous=obj.__dict__.get(name,missing)
    def restore():
      if previous is missing: delattr(obj,name)
      else: setattr(obj,name,previous)
    self.stack.callback(restore)
    setattr(obj,name,value)

  def route(self,call):
    if self.device.device not in hcq2.get_enqueue_devs(call): return False
    selected=any(u.op is Ops.SINK and getattr(u.arg,'name',None)=='alpamayo_online' for u in call.body.src)
    if selected:self.selected+=1
    return selected

  def check_scratch(self,queue,data):
    if queue.dev is self.device and queue.queue=='COMPUTE:1':
      assert data.private_segment_size==0,'Secondary queue cannot share scratch storage'
      self.scratch_checks+=1

  def __enter__(self):
    dev=self.device
    assert dev.is_usb and dev.is_am() and dev.arch=='gfx1200' and not dev.is_aql and not PROFILE
    assert not hasattr(dev,'_alpamayo_secondary_queue')
    dev.synchronize()
    dev.compute_queue
    gfx=dev.iface.dev_impl.gfx
    assert gfx.xccs==1 and len(gfx.mqd_mc)>=2
    # Construct every source patch before allocating or enabling hardware.
    setup=clone(type(gfx).setup_ring,[
      ('else am.AMDGPU_NAVI10_DOORBELL_MEC_RING0','else am.AMDGPU_NAVI10_DOORBELL_MEC_RING0 + idx')])
    create=clone(type(dev.iface).create_queue,[
      ('is_aql:=(queue_type==kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL), is_aql',
       'idx, queue_type==kfd.KFD_IOC_QUEUE_TYPE_COMPUTE_AQL')])
    submit=clone(ops_amd.AMDComputeQueue.submit,[
      ('_queue_args(self, self.dev.compute_queue)',
       '_queue_args(self, self.dev._alpamayo_secondary_queue if self.queue=="COMPUTE:1" else self.dev.compute_queue)')])
    execute=clone(ops_amd.AMDComputeQueue.exec,[
      ('data, lib = amd_build_program(self.dev, prg, self.devs)',
       'data, lib = amd_build_program(self.dev, prg, self.devs)\n  _alpamayo_check(self, data)')],
      {'_alpamayo_check':self.check_scratch})
    schedule=clone(hcq2.sched_batches,[
      ('if c.body.op is Ops.PROGRAM: return "COMPUTE:0"',
       'if c.body.op is Ops.PROGRAM: return "COMPUTE:1" if _alpamayo_route(c) else "COMPUTE:0"')],
      {'_alpamayo_route':self.route})
    gfx._grbm_select(me=1,pipe=0,queue=1,inst=0)
    try: assert not (gfx.adev.regCP_HQD_ACTIVE.read(inst=0)&1),'Queue 1 is already active'
    finally:gfx._grbm_select(inst=0)
    try:
      self.replace(dev,'can_recover',False)
      self.replace(gfx,'setup_ring',MethodType(setup,gfx))
      self.replace(dev.iface,'create_queue',MethodType(create,dev.iface))
      self.queue=dev.create_queue(ops_amd.kfd.KFD_IOC_QUEUE_TYPE_COMPUTE,1<<20,eop_buffer_size=0x1000,idx=1)
      assert self.queue.doorbell.options.external_ptr!=dev.compute_queue.doorbell.options.external_ptr
      self.replace(dev,'_alpamayo_secondary_queue',self.queue)
      for name in ('ring','write_ptr','doorbell','put_value'):
        key=ops_amd.to_name(dev.device,name,'COMPUTE:1')
        assert key not in Compiled.link_bufs
        Compiled.link_bufs[key]=lambda b,name=name:getattr(self.queue,name)
        self.stack.callback(Compiled.link_bufs.pop,key)
      self.replace(ops_amd.AMDComputeQueue,'submit',submit)
      self.replace(ops_amd.AMDComputeQueue,'exec',execute)
      self.replace(hcq2,'sched_batches',schedule)
      # Eager templates include queue assignments; keep the two scopes separate.
      saved_cache=dict(hcq2.hcq_compile_cache)
      def restore_cache():
        hcq2.hcq_compile_cache.clear();hcq2.hcq_compile_cache.update(saved_cache)
      self.stack.callback(restore_cache)
      hcq2.hcq_compile_cache.clear()
      return self
    except BaseException:
      self.__exit__(None,None,None)
      raise

  def __exit__(self,kind,value,traceback):
    try:
      if self.queue is not None:
        self.device.synchronize()
        gfx=self.device.iface.dev_impl.gfx
        gfx._grbm_select(me=1,pipe=0,queue=1,inst=0)
        try:
          if gfx.adev.regCP_HQD_ACTIVE.read(inst=0)&1:
            gfx.adev.regCP_HQD_DEQUEUE_REQUEST.write(1,inst=0)
            wait_cond(lambda:gfx.adev.regCP_HQD_ACTIVE.read(inst=0)&1,value=0,timeout_ms=2000,msg='Drain secondary queue')
          self.drained=True
        finally:gfx._grbm_select(inst=0)
    finally:self.stack.close()
    return False

"""Actual KFD handles: live and cumulative accounting, including owned host unmaps."""
import os
from pathlib import Path
import time

class Ledger:
  def __init__(self,p,availability,own_pid=None):
    self.p=p;self.availability=availability;self.pid=os.getpid() if own_pid is None else own_pid;self.handles={};self.started=time.monotonic()
    self.report=dict(calls=0,requested_bytes=0,live_bytes=0,peak_live_bytes=0,maximum_request_bytes=0,
      free_calls=0,owned_unmap_calls=0,phases=[],
      live_limit=p['live_allocation_limit'],cumulative_limit=p['cumulative_allocation_limit'],
      single_buffer_limit=p['max_single_buffer'],selected_free_reserve=p['free_reserve'],
      accounting='All KFD allocations, including host mappings; actual handle identity through free and owned unmap. Cumulative never resets per step.')
  def phase(self,name):
    self.report['phases'].append(dict(name=name,requested_bytes_at_start=self.report['requested_bytes'],
      live_bytes_at_start=self.report['live_bytes'],peak_live_bytes=self.report['live_bytes'],seconds_since_start=time.monotonic()-self.started))
  def validate(self,iface):
    assert iface.gpu_id==self.p['selected_gpu_id'] and Path(iface.dev_sysfs_path).resolve().name==self.p['selected_pci']
  def available(self,upcoming):
    a=self.availability();assert a['selected_node']['gpu_id']==self.p['selected_gpu_id'] and a['selected_node']['pci']==self.p['selected_pci']
    assert a['kfd_proc_root_read'] and all(r['complete'] for r in a['kfd_processes'] if r['pid']!=self.pid)
    assert not(set(a['observed_selected_kfd_owners'])-{self.pid})
    assert a['vram_free_bytes']>=self.p['free_reserve']+upcoming
  def alloc(self,original):
    def bounded(iface,size,*args,**kwargs):
      self.validate(iface);r=self.report
      assert 0<size<=self.p['max_single_buffer'] and r['live_bytes']+size<=self.p['live_allocation_limit']
      assert r['requested_bytes']+size<=self.p['cumulative_allocation_limit'];self.available(size)
      r['calls']+=1;r['requested_bytes']+=size;r['maximum_request_bytes']=max(r['maximum_request_bytes'],size)
      value=original(iface,size,*args,**kwargs);handle=int(value.meta.handle)
      assert handle not in self.handles and int(value.meta.size)==size
      self.handles[handle]=size;r['live_bytes']+=size;r['peak_live_bytes']=max(r['peak_live_bytes'],r['live_bytes'])
      if r['phases']:r['phases'][-1]['peak_live_bytes']=max(r['phases'][-1]['peak_live_bytes'],r['live_bytes'])
      return value
    return bounded
  def release(self,handle):
    self.report['live_bytes']-=self.handles.pop(int(handle));assert self.report['live_bytes']>=0
  def free(self,original):
    def bounded(iface,value):
      self.validate(iface);handle=int(value.meta.handle);assert handle in self.handles
      result=original(iface,value);self.release(handle);self.report['free_calls']+=1;return result
    return bounded
  def unmap(self,original):
    def bounded(iface,value):
      self.validate(iface);handle,owned=value.meta
      if owned:assert int(handle) in self.handles
      result=original(iface,value)
      if owned:self.release(handle);self.report['owned_unmap_calls']+=1
      return result
    return bounded

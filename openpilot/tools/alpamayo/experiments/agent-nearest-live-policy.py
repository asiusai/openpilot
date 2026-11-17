"""Optional received-only nearest history. Original classes/methods remain immutable."""
from bisect import bisect_right
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import textwrap
from unittest.mock import patch

POLICY='nearest-received-history-past-tie-v1'
PINS={'agent-live-bridge.py':'4a0baefed467b2a384ebf0a6992b4dc9832141d8c58bc09dc4a2523adfeb4225',
      'agent-live-consumer.py':'65db37fd614e5f378edc56afc34b947c2ac58151f2b9019c771b0a67cbfceccf'}


def sha(path):
  with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def nearest_index(times,desired,query):
  """Only already admitted Store frames enter times. Current capture stays FLOOR."""
  end=bisect_right(times,query)
  if desired>query or not end:return -1
  if desired==query:return end-1
  after=min(bisect_right(times,desired),end)
  candidates=[i for i in (after-1,after) if 0<=i<end]
  return min(candidates,key=lambda i:(abs(times[i]-desired),times[i]))


def timestamp_valid(meta,ref,desired,query):
  if meta.get('history_policy')!=POLICY:return False
  stamp=ref['capture_ns']
  if not (0<stamp<=ref['eof_ns']<=ref['received_ns']<=ref['ready_ns']<=meta['selected_ns'] and stamp<=query):return False
  return (0<=desired-stamp<=150_000_000) if desired==query else abs(desired-stamp)<=150_000_000


def transformed(module,method,changes,extra):
  path=Path(module.__file__)
  if sha(path)!=PINS[path.name]:raise ValueError('Unreviewed original source: '+str(path))
  source=textwrap.dedent(inspect.getsource(method))
  for before,after in changes:
    if source.count(before)!=1:raise ValueError('Ambiguous nearest policy substitution')
    source=source.replace(before,after,1)
  namespace=dict(module.__dict__,**extra)
  exec(compile(source,str(Path(__file__).resolve()),'exec'),namespace)
  return namespace[method.__name__],hashlib.sha256(source.encode()).hexdigest()


def store_classes(bridge):
  original=bridge.Store
  select,digest=transformed(bridge,original.select,[
    ('index = bisect_right(times, desired)-1','index = _nearest_index(times, desired, query)'),
    ('desired-times[index] > 150_000_000','abs(desired-times[index]) > 150_000_000')],{'_nearest_index':nearest_index})
  class Store(original):pass
  Store.select=select
  class Preparer(bridge.Preparer):
    def prepare(self,selection):
      arrays,detail=super().prepare(selection)
      detail=dict(detail,history_policy=POLICY,nearest_availability='Only Store-admitted frames with ready_ns<=selected_ns and capture_ns<=query_ns')
      return arrays,detail
  return SimpleNamespace(Store=Store,Preparer=Preparer,transformed_select_sha256=digest,original=original)


def consumer_class(live):
  original=live.Consumer
  data,digest=transformed(live,original._data,[
    ('not 0 <= desired-stamp <= 150_000_000','not _nearest_timestamp_valid(meta, ref, desired, q)')],
    {'_nearest_timestamp_valid':timestamp_valid})
  class Consumer(original):
    def __init__(self,*args,**kwargs):
      super().__init__(*args,**kwargs)
      self.provenance['nearest_live_policy']=dict(policy=POLICY,numerical_input_change=True,source_sha256=sha(__file__),
        transformed_data_sha256=digest,original_consumer_sha256=PINS['agent-live-consumer.py'],
        availability='received and ready by dispatch; never all offline captures',current_latest='unchanged causal FLOOR')
      self.provenance_sha256=live.digest(self.provenance)
  Consumer._data=data
  return Consumer


@contextmanager
def imports(root):
  """Compose outside the unchanged qualified startup importer. No original method edits."""
  exp=Path(root).resolve()/'experiments';original=importlib.util.spec_from_file_location
  restored=[]
  def spec(name,location,*args,**kwargs):
    result=original(name,location,*args,**kwargs);path=Path(location).resolve()
    if path.parent!=exp or path.name not in PINS:return result
    loader=result.loader
    class Loader:
      def create_module(self,value):return loader.create_module(value)
      def exec_module(self,value):
        loader.exec_module(value)
        if path.name=='agent-live-bridge.py':
          classes=store_classes(value)
          for key in ('Store','Preparer'):
            restored.append((value,key,getattr(value,key)));setattr(value,key,getattr(classes,key))
        else:
          restored.append((value,'Consumer',value.Consumer));value.Consumer=consumer_class(value)
    result.loader=Loader();return result
  try:
    with patch.object(importlib.util,'spec_from_file_location',spec):yield
  finally:
    for module,name,value in reversed(restored):setattr(module,name,value)

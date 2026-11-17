"""Check the exact owned closure, then run every original coverage guard unchanged."""
import importlib.util
import threading
from types import CodeType, MethodType


def cells(fn):
  return dict(zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=True))


def factory(common):
  old = common.original_load(common.HERE/'agent-arrival-warp-integrated-install.py', 'arrival_ownership_frozen_install')
  Base = old.factory(common)
  expected_code, = [c for c in Base.__enter__.__code__.co_consts if isinstance(c, CodeType) and c.co_name == 'spec']
  expected_method = Base.spec
  expected_method_code = expected_method.__code__

  class Hooks(Base):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.ownership = dict(entered=False, checks=0, completed=0, scoped_restorations=0,
        all_original_coverage_retained=True, finally_restored=False)
      common.ownership.append(self.ownership)

    def __enter__(self):
      result = super().__enter__()
      self._ownership_thread = threading.get_ident()
      self._ownership_outer = importlib.util.spec_from_file_location
      self._ownership_cells = cells(self._ownership_outer)
      self._ownership_bound = self._ownership_cells['previous'].cell_contents
      self.check_chain()
      self.ownership['entered'] = True
      return result

    def check_chain(self):
      outer, bound = self._ownership_outer, self._ownership_bound
      if threading.get_ident() != self._ownership_thread:
        raise RuntimeError('Coverage owner thread changed')
      if importlib.util.spec_from_file_location is not outer or outer.__code__ is not expected_code:
        raise RuntimeError('Unexpected arrival outer hook identity')
      actual = cells(outer)
      if set(actual) != {'C','hooks','previous'} or any(actual[k] is not self._ownership_cells[k] for k in actual):
        raise RuntimeError('Arrival closure cells changed')
      if actual['C'].cell_contents is not common.C or actual['hooks'].cell_contents is not self or actual['previous'].cell_contents is not bound:
        raise RuntimeError('Arrival closure ownership changed')
      if not isinstance(bound, MethodType) or bound.__self__ is not self or bound.__func__ is not expected_method:
        raise RuntimeError('Selected bound hook owner changed')
      if expected_method.__code__ is not expected_method_code or self.spec != bound:
        raise RuntimeError('Selected import hook was rebound')
      return outer, bound

    def coverage(self):
      outer, bound = self.check_chain()
      producer = self.arrival.producer
      if producer is None or (producer.thread is not None and producer.thread.is_alive()):
        raise RuntimeError('Coverage requires a stopped source producer')
      self.ownership['checks'] += 1
      importlib.util.spec_from_file_location = bound
      try:
        # Original arrival -> CPU digest -> storage -> exact live -> numerical
        # composition coverage. The original bound-owner guard executes normally.
        result = super().coverage()
      finally:
        intact = importlib.util.spec_from_file_location is bound
        importlib.util.spec_from_file_location = outer
        self.ownership['scoped_restorations'] += 1
        self.check_chain()
        if not intact: raise RuntimeError('Import hook changed inside original coverage')
      self.ownership['completed'] += 1
      return result

    def __exit__(self, *error):
      try:
        self.check_chain()
      finally:
        result = super().__exit__(*error)
        self.ownership['finally_restored'] = importlib.util.spec_from_file_location is self.composition.original_spec
      return result
  return Hooks
